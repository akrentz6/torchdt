#include <torchdt/registry.h>

int16_t zero = -32768;
int16_t pos_inf = 32766;
int16_t neg_inf = 32767;
double base = std::pow(2.0, std::pow(2.0, -10));

int16_t neg(int16_t x) {
    return x ^ 1;
}

int16_t add16(int16_t x, int16_t y) {
    if ((x | 1LL) == zero) return y;
    if ((y | 1LL) == zero) return x;
    if ((neg(x) == y)) return zero;

    const int16_t max_operand = std::max(x, y);
    const int16_t abs_diff = std::abs((x >> 1) - (y >> 1));
    const int16_t sign_diff = (x ^ y) & 1;

    double power_term = std::pow(base, -abs_diff);
    double magnitude = std::abs(1.0 - 2.0 * sign_diff + power_term);
    double log_term = std::log(magnitude) / std::log(base);
    double rounded_value = std::clamp(
        std::round(log_term),
        (double)(std::numeric_limits<int16_t>::min()),
        (double)(std::numeric_limits<int16_t>::max())
    );

    return max_operand + (static_cast<int16_t>(rounded_value) << 1);
}

int16_t sub16(int16_t x, int16_t y) {
    return add16(x, neg(y));
}

int16_t mul16(int16_t x, int16_t y) {
    if ((x | 1LL) == zero || (y | 1LL) == zero) return zero;
    return (x + y - (y & 1)) ^ (y & 1);
}

int16_t div16(int16_t x, int16_t y) {
    if ((x | 1LL) == zero) return zero;
    if ((y | 1LL) == zero) throw std::runtime_error("Division by zero");
    return (x - y + (y & 1)) ^ (y & 1);
}

Ops<16> ops16 = []{
    Ops<16> o;
    o.add = add16;
    o.sub = sub16;
    o.mul = mul16;
    o.div = div16;
    return o;
}();

Ops<16> get_lns16_ops() {
    return ops16;
}