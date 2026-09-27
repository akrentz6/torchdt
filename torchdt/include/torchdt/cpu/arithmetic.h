#pragma once
#include <torchdt/registry.h>
#include <utility>

namespace torchdt::native {
// Default policy for a datatype providing scalar primitives. Specialize hooks
// only when batch arithmetic must differ from scalar arithmetic on this host.
template<class ScalarOps> struct CPUArithmetic {
    using Scalar = ScalarOps;
    using S = typename Scalar::S;
    static constexpr auto storage = c10::CppTypeToScalarType<S>::value;
    Scalar scalar;
    explicit CPUArithmetic(Scalar value) : scalar(std::move(value)) {}
    bool use_batch_add() const { return false; }
    void validate_conversion(const Tensor&) const {}
    template<class Unary>
    OptionalTensor binary_override(const Tensor&, const Tensor&, const std::string&, Unary) const {
        return c10::nullopt;
    }
};
} // namespace torchdt::native
