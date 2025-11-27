#ifndef OPS_KERNELS_H
#define OPS_KERNELS_H

#include <torch/extension.h>
#include <torchdt/registry.h>

template <size_t bitwidth>
struct OpsImpl : public OpsBase {
    using StorageT = typename StorageFor<bitwidth>::type;
    using BinOp = StorageT(*)(StorageT, StorageT);

    Ops<bitwidth> ops;

    OpsImpl(const Ops<bitwidth>& o) : ops(o) {}

    template<typename F>
    torch::Tensor run_binary_kernel(const torch::Tensor& x, const torch::Tensor& y, F f) const;

    torch::Tensor add(const torch::Tensor& x, const torch::Tensor& y) const;
    torch::Tensor sub(const torch::Tensor& x, const torch::Tensor& y) const;
    torch::Tensor mul(const torch::Tensor& x, const torch::Tensor& y) const;
    torch::Tensor div(const torch::Tensor& x, const torch::Tensor& y) const;

};

#endif // OPS_KERNELS_H