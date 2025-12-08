#ifndef OPS_IMPL_H
#define OPS_IMPL_H

#include <torch/extension.h>
#include <torchdt/registry.h>

struct OpsBase {
    virtual ~OpsBase() = default;

    virtual torch::Tensor from_float(const torch::Tensor& x) const = 0;
    virtual torch::Tensor to_float(const torch::Tensor& x) const = 0;

    virtual torch::Tensor add(const torch::Tensor& x, const torch::Tensor& y) const = 0;
    virtual torch::Tensor sub(const torch::Tensor& x, const torch::Tensor& y) const = 0;
    virtual torch::Tensor mul(const torch::Tensor& x, const torch::Tensor& y) const = 0;
    virtual torch::Tensor div(const torch::Tensor& x, const torch::Tensor& y) const = 0;

    virtual torch::Tensor ge(const torch::Tensor& x, const torch::Tensor& y) const = 0;
    virtual torch::Tensor gt(const torch::Tensor& x, const torch::Tensor& y) const = 0;
    virtual torch::Tensor le(const torch::Tensor& x, const torch::Tensor& y) const = 0;
    virtual torch::Tensor lt(const torch::Tensor& x, const torch::Tensor& y) const = 0;

    virtual torch::Tensor sum(
        const torch::Tensor& x,
        c10::optional<std::vector<int64_t>> dim,
        bool keepdim
    ) const = 0;

    virtual torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B) const = 0;
    virtual std::vector<torch::Tensor> matmul_backward(
        const torch::Tensor& grad_out,
        const torch::Tensor& A,
        const torch::Tensor& B
    ) const = 0;

    virtual torch::Tensor conv2d(
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const c10::optional<torch::Tensor>& bias,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation,
        int64_t groups
    ) const = 0;
    virtual std::vector<torch::Tensor> conv2d_backward(
        const torch::Tensor& grad_out,
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation,
        bool has_bias,
        int64_t groups
    ) const = 0;
};

template<size_t bitwidth>
struct OpsImpl : public OpsBase {
    using StorageT = typename StorageFor<bitwidth>::type;
    using OpsT = Ops<bitwidth>;

    OpsT ops;

    explicit OpsImpl(const OpsT& ops_) : ops(ops_) {}

    template <typename F>
    torch::Tensor run_unary_kernel(const torch::Tensor& x, F f) const;

    template<typename F>
    torch::Tensor run_binary_kernel(const torch::Tensor& x, const torch::Tensor& y, F f) const;

    template<typename F>
    torch::Tensor run_binary_bool_kernel(const torch::Tensor& x, const torch::Tensor& y, F f) const;

    torch::Tensor from_float(const torch::Tensor& x) const;
    torch::Tensor to_float(const torch::Tensor& x) const;

    torch::Tensor add(const torch::Tensor& x, const torch::Tensor& y) const;
    torch::Tensor sub(const torch::Tensor& x, const torch::Tensor& y) const;
    torch::Tensor mul(const torch::Tensor& x, const torch::Tensor& y) const;
    torch::Tensor div(const torch::Tensor& x, const torch::Tensor& y) const;

    torch::Tensor ge(const torch::Tensor& x, const torch::Tensor& y) const;
    torch::Tensor gt(const torch::Tensor& x, const torch::Tensor& y) const;
    torch::Tensor le(const torch::Tensor& x, const torch::Tensor& y) const;
    torch::Tensor lt(const torch::Tensor& x, const torch::Tensor& y) const;

    torch::Tensor sum(const torch::Tensor& x, c10::optional<std::vector<int64_t>> dim, bool keepdim) const;

    torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B) const;
    std::vector<torch::Tensor> matmul_backward(
        const torch::Tensor& grad_out,
        const torch::Tensor& A,
        const torch::Tensor& B
    ) const;

    torch::Tensor conv2d(
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const c10::optional<torch::Tensor>& bias,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation,
        int64_t groups
    ) const;
    std::vector<torch::Tensor> conv2d_backward(
        const torch::Tensor& grad_out,
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation,
        bool has_bias,
        int64_t groups
    ) const;
};

#endif // OPS_IMPL_H