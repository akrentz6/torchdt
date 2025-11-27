#include "ops_kernels.h"
#include <ATen/native/cpu/Loops.h>

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