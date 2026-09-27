#include <torchdt/cpu/kernels.h>
#include "scalar.h"
#include "config.h"

namespace torchdt::native {
// LNS-only host compatibility and configuration. Tensor composition belongs to
// CPUKernels, shared with other datatypes through this arithmetic policy.
template<int Bits> struct LnsCPUArithmetic : CPUArithmetic<LnsScalar<Bits>> {
    using Scalar = LnsScalar<Bits>;
    using S = typename Scalar::S;
    using CPUArithmetic<Scalar>::scalar;
    using CPUArithmetic<Scalar>::storage;
    Tensor table_owner;

    explicit LnsCPUArithmetic(const LnsConfig& config)
        : CPUArithmetic<Scalar>(Scalar{{int(config.precision), config.base, config.log_base}}) {
        TORCH_CHECK(config.precision >= 1 && config.precision <= 50,
                    "Precision must be between 1 and 50");
        TORCH_CHECK(std::isfinite(config.base) && config.base > 1 &&
                    std::isfinite(config.log_base) && config.log_base > 0, "Invalid LNS base");
        if (config.table) {
            TORCH_CHECK(config.precision <= 20, "Table-based LNS only supports precision up to 20");
            TORCH_CHECK(config.table->device().is_cpu() && config.table->layout() == at::kStrided &&
                        config.table->scalar_type() == storage, "Invalid LNS correction table storage");
            TORCH_CHECK(config.table->dim() == 2 && config.table->size(0) == 2 &&
                        config.table_ez < 0 && config.table_ez == -config.table->size(1),
                        "Invalid LNS correction table");
            table_owner = config.table->contiguous().clone();
            scalar.table = table_owner.template data_ptr<S>();
            scalar.table_size = table_owner.size(1);
            scalar.table_ez = config.table_ez;
        }
    }

    bool use_batch_add() const {
        return !scalar.table && ((Bits == 16 && scalar.config.precision >= 12) ||
                                 (Bits == 32 && scalar.config.precision >= 27));
    }
    void validate_conversion(const Tensor& input) const {
        TORCH_CHECK_VALUE(!at::isnan(input).any().item<bool>(), "LNS cannot encode NaN values");
    }
    template<class Unary>
    OptionalTensor binary_override(const Tensor& x, const Tensor& y,
                                   const std::string& op, Unary unary) const {
        // At high precisions Python casts corrections outside the carrier's
        // representable range. Those conversions depend on ATen's CPU vector
        // implementation. Keep that exceptional conversion and its expression
        // in ATen; the usual precision range uses the fused scalar kernel.
        if (use_batch_add() && (op == "add" || op == "sub")) {
            auto rhs = op == "sub" ? unary(y, "neg") : y;
            auto maximum = at::maximum(x, rhs);
            auto distance = at::abs(at::bitwise_right_shift(x, 1) - at::bitwise_right_shift(rhs, 1));
            auto signs = at::bitwise_and(at::bitwise_xor(x, rhs), 1);
            auto power = at::pow(at::scalar_tensor(scalar.config.base, x.options().dtype(at::kDouble)), -distance);
            auto magnitude = at::abs(1.0 - 2.0 * signs + power);
            auto correction = at::bitwise_left_shift(at::round(at::log(magnitude) / scalar.config.log_base).to(storage), 1);
            auto result = maximum + correction;
            auto under = (maximum < 0) & (correction < 0) & (result >= 0);
            auto over = (maximum >= 0) & (correction >= 0) & (result < 0);
            auto infinity = at::where(at::bitwise_and(maximum, 1) == 0, Scalar::pos_inf, Scalar::neg_inf).to(storage);
            result = at::where(under, Scalar::zero, at::where(over, infinity, result));
            return at::where(x == Scalar::zero, rhs,
                   at::where(rhs == Scalar::zero, x,
                   at::where(x == unary(rhs, "neg"), Scalar::zero, result)));
        }
        // Python's wide pow deliberately casts unbounded floating values.
        // ATen defines the effective platform/vector conversion behavior; keep
        // that conversion in ATen instead of invoking undefined C++ casts.
        if constexpr (Bits != 16) {
            if (op == "pow") {
                auto exponent = unary(y, "to_float");
                return at::bitwise_and((at::bitwise_and(x, -2) * exponent).to(storage), -2);
            }
        }
        return c10::nullopt;
    }
};

void register_lns_cpu() {
#define REGISTER(BITS) Registry::instance().register_factory("lns", BITS, "cpu", [](const Config& c) { \
    return std::make_shared<CPUKernels<LnsCPUArithmetic<BITS>>>(LnsCPUArithmetic<BITS>(LnsConfig(c))); });
    REGISTER(16) REGISTER(32) REGISTER(64)
#undef REGISTER
}
} // namespace torchdt::native
