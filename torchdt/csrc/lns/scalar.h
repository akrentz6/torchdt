#pragma once
// Tensor-free arithmetic shared by CPU kernels and the CUDA compile probe.
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef __CUDACC__
#define TORCHDT_HD __host__ __device__
#else
#define TORCHDT_HD
#endif

namespace torchdt::native {
struct ScalarConfig {
    int precision;
    double base;
    double log_base;
};

struct Math {
    TORCHDT_HD static double log(double x) { return ::log(x); }
    TORCHDT_HD static double pow(double x, double y) { return ::pow(x, y); }
    TORCHDT_HD static double abs(double x) { return ::fabs(x); }
    // Independent of the process floating-point rounding mode.
    TORCHDT_HD static double round(double x) {
        if (!(x >= -std::numeric_limits<double>::max() &&
              x <= std::numeric_limits<double>::max())) return x;
        double lo = ::floor(x), f = x - lo;
        return f < 0.5 ? lo : f > 0.5 ? lo + 1.0 :
            (::fmod(lo, 2.0) == 0.0 ? lo : lo + 1.0);
    }
};

template<int Bits> struct LnsTraits {
    static_assert(Bits == 16 || Bits == 32 || Bits == 64);
    using S = std::conditional_t<Bits == 16, int16_t,
              std::conditional_t<Bits == 32, int32_t, int64_t>>;
    using U = std::make_unsigned_t<S>;
    static constexpr S zero = std::numeric_limits<S>::min();
    static constexpr S pos_inf = std::numeric_limits<S>::max() - 1;
    static constexpr S neg_inf = std::numeric_limits<S>::max();
    static constexpr S min_log = zero / 2;
    static constexpr S max_log = pos_inf / 2;

    // Defined two's-complement bit interpretation, including INT64_MIN.
    TORCHDT_HD static S signed_bits(U u) {
        return u <= U(std::numeric_limits<S>::max()) ? S(u) : S(-1 - S(U(~u)));
    }
    TORCHDT_HD static S wrap_add(S a, S b) { return signed_bits(U(U(a) + U(b))); }
    TORCHDT_HD static S wrap_neg(S a) { return signed_bits(U(U(0) - U(a))); }
    TORCHDT_HD static S shl(S a) { return signed_bits(U(U(a) << 1)); }
    TORCHDT_HD static S shr(S a) {
        U u = U(a) >> 1;
        if (a < 0) u |= U(U(1) << (Bits - 1));
        return signed_bits(u);
    }
    TORCHDT_HD static S clear_sign(S a) { return signed_bits(U(U(a) & ~U(1))); }
    TORCHDT_HD static S toggle_sign(S a) { return signed_bits(U(U(a) ^ U(1))); }
    // PyTorch CPU floating -> integer casts: small carriers narrow an int64
    // conversion, while out-of-range int32/int64 conversions use INT_MIN.
    // Explicit checks avoid undefined C++ floating-to-integer conversions.
    TORCHDT_HD static S cast(double x) {
        if constexpr (Bits == 16) {
            if (!(x >= -0x1p63 && x < 0x1p63)) return 0;
            return signed_bits(U(int64_t(x)));
        } else {
            constexpr double limit = Bits == 32 ? 0x1p31 : 0x1p63;
            if (!(x >= -limit && x < limit)) return zero;
            return S(x);
        }
    }
};

template<int Bits, class M = Math> struct LnsScalar : LnsTraits<Bits> {
    using T = LnsTraits<Bits>;
    using S = typename T::S;
    using T::zero; using T::pos_inf; using T::neg_inf;
    using T::min_log; using T::max_log;
    using T::wrap_add; using T::wrap_neg; using T::shl; using T::shr;
    using T::clear_sign; using T::toggle_sign; using T::cast;
    TORCHDT_HD explicit LnsScalar(ScalarConfig c) : config(c) {}
    ScalarConfig config;
    const S* table = nullptr;
    int64_t table_size = 0;
    int64_t table_ez = 0;

    TORCHDT_HD S checked_add(S a, S b, S sign) const {
        S r = wrap_add(a, b);
        if (a < 0 && b < 0 && r >= 0) return zero;
        if (a >= 0 && b >= 0 && r < 0) return (sign & 1) ? neg_inf : pos_inf;
        return r;
    }
    TORCHDT_HD S from_float(double x) const {
        if (x == 0) return zero;
        double r = M::round(M::log(M::abs(x)) / config.log_base);
        if (r >= double(max_log)) return x < 0 ? neg_inf : pos_inf;
        if (r <= double(min_log)) return zero;
        return T::signed_bits(typename T::U(typename T::U(shl(cast(r))) | (x < 0 ? 1 : 0)));
    }
    TORCHDT_HD double to_float(S x) const {
        if (x == zero) return 0.;
        if (x == pos_inf) return std::numeric_limits<double>::infinity();
        if (x == neg_inf) return -std::numeric_limits<double>::infinity();
        return ((x & 1) ? -1. : 1.) * M::pow(config.base, double(shr(x)));
    }
    TORCHDT_HD S neg(S x) const { return x == zero ? x : toggle_sign(x); }
    TORCHDT_HD S abs(S x) const { return clear_sign(x); }
    TORCHDT_HD S sign(S x) const { return x == zero ? zero : S(x & 1); }
    TORCHDT_HD S add(S x, S y) const {
        if (x == zero) return y;
        if (y == zero) return x;
        if (x == neg(y)) return zero;
        S maximum = x > y ? x : y;
        // Exponents span only Bits-1 bits; their difference fits the carrier.
        S difference = wrap_add(shr(x), wrap_neg(shr(y)));
        S distance = difference < 0 ? wrap_neg(difference) : difference;
        S correction;
        if (table) {
            int64_t z = -int64_t(distance);
            int64_t column = z == 0 ? -1 : z;
            if (column < table_ez) column = table_ez;
            correction = table[((x ^ y) & 1) * table_size + table_size + column];
            return wrap_add(maximum, correction);
        }
        double power = M::pow(config.base, -double(distance));
        double magnitude = M::abs(1.0 - 2.0 * ((x ^ y) & 1) + power);
        correction = shl(cast(M::round(M::log(magnitude) / config.log_base)));
        return checked_add(maximum, correction, maximum & 1);
    }
    TORCHDT_HD S sub(S x, S y) const { return add(x, neg(y)); }
    TORCHDT_HD S mul(S x, S y) const {
        if (x == zero || y == zero) return zero;
        S r = checked_add(x, clear_sign(y), x & 1);
        return r == zero ? zero : ((y & 1) ? toggle_sign(r) : r);
    }
    TORCHDT_HD S div(S x, S y) const {
        if (x == zero) return zero;
        if (y == zero) return (x & 1) ? neg_inf : pos_inf;
        S r = checked_add(x, wrap_neg(clear_sign(y)), x & 1);
        return r == zero ? zero : ((y & 1) ? toggle_sign(r) : r);
    }
    TORCHDT_HD S sqrt(S x) const {
        if (x == zero || x == pos_inf) return x;
        return clear_sign(shr(clear_sign(x)));
    }
    TORCHDT_HD S pow(S x, S y) const {
        double exponent = to_float(y);
        if constexpr (Bits != 16) {
            return clear_sign(cast(double(clear_sign(x)) * exponent));
        } else {
            if (y == zero) return 0;
            if (x == zero) return exponent < 0 ? pos_inf : zero;
            if (x == pos_inf) return exponent < 0 ? zero : pos_inf;
            double r = M::round(shr(x) == 0 ? 0. : double(shr(x)) * exponent);
            if (r <= double(min_log)) return zero;
            if (r >= double(max_log)) return pos_inf;
            return shl(cast(r));
        }
    }
    TORCHDT_HD bool gt(S x, S y) const {
        if ((x & 1) != (y & 1)) return !(x & 1);
        return (x & 1) ? shr(x) < shr(y) : shr(x) > shr(y);
    }
    TORCHDT_HD bool ge(S x, S y) const { return x == y || gt(x, y); }
    TORCHDT_HD bool lt(S x, S y) const { return gt(y, x); }
    TORCHDT_HD bool le(S x, S y) const { return ge(y, x); }
};
} // namespace torchdt::native
