#include "ops_kernels.h"
#include <ATen/native/cpu/Loops.h>

template <size_t bitwidth>
constexpr torch::ScalarType int_type_from_bitwidth() {
    if constexpr (bitwidth == 8)  return torch::kInt8;
    if constexpr (bitwidth == 16) return torch::kInt16;
    if constexpr (bitwidth == 32) return torch::kInt32;
    if constexpr (bitwidth == 64) return torch::kInt64;

    static_assert(bitwidth == 8 || bitwidth == 16 ||
                  bitwidth == 32 || bitwidth == 64,
                  "Unsupported bitwidth");
}

template <size_t bitwidth>
template <typename F>
torch::Tensor OpsImpl<bitwidth>::run_unary_kernel(const torch::Tensor& x, F f) const {
    auto out = at::empty_like(x);

    auto iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .build();

    at::native::cpu_kernel(iter, [f](StorageT a) -> StorageT {
        return f(a);
    });

    return out;
}

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::from_float(const torch::Tensor& x) const {
    auto out = at::empty_like(x, x.options().dtype(int_type_from_bitwidth<bitwidth>()));

    auto iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .check_all_same_dtype(false)
        .build();

    at::native::cpu_kernel(iter, [this](float a) -> StorageT {
        return ops.from_float(a);
    });

    return out;
}

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::to_float(const torch::Tensor& x) const {
    auto out = at::empty_like(x, x.options().dtype(torch::kFloat32));

    auto iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .check_all_same_dtype(false)
        .build();

    at::native::cpu_kernel(iter, [this](StorageT a) -> float {
        return ops.to_float(a);
    });

    return out;
}

template <size_t bitwidth>
template <typename F>
torch::Tensor OpsImpl<bitwidth>::run_binary_kernel(const torch::Tensor& x, const torch::Tensor& y, F f) const {
    auto out = at::empty_like(x);

    auto iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .add_input(y)
        .build();

    at::native::cpu_kernel(iter, [f](StorageT a, StorageT b) -> StorageT {
        return f(a, b);
    });

    return out;
}

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::add(const torch::Tensor& x, const torch::Tensor& y) const {
    return run_binary_kernel(x, y, ops.add);
}

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::sub(const torch::Tensor& x, const torch::Tensor& y) const {
    return run_binary_kernel(x, y, ops.sub);
}

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::mul(const torch::Tensor& x, const torch::Tensor& y) const {
    return run_binary_kernel(x, y, ops.mul);
}

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::div(const torch::Tensor& x, const torch::Tensor& y) const {
    return run_binary_kernel(x, y, ops.div);
}

// Explicit template instantiation
template struct OpsImpl<8>;
template struct OpsImpl<16>;
template struct OpsImpl<32>;
template struct OpsImpl<64>;