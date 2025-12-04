#include "ops_kernels.h"
#include <torch/extension.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/ReduceOpsUtils.h>

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
template <typename F>
torch::Tensor OpsImpl<bitwidth>::run_binary_bool_kernel(const torch::Tensor& x, const torch::Tensor& y, F f) const {
    auto out = at::empty_like(x, x.options().dtype(torch::kBool));

    auto iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .add_input(y)
        .check_all_same_dtype(false)
        .build();

    at::native::cpu_kernel(iter, [f](StorageT a, StorageT b) -> bool {
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

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::ge(const torch::Tensor& x, const torch::Tensor& y) const {
    return run_binary_bool_kernel(x, y, ops.ge);
}

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::gt(const torch::Tensor& x, const torch::Tensor& y) const {
    return run_binary_bool_kernel(x, y, ops.gt);
}

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::le(const torch::Tensor& x, const torch::Tensor& y) const {
    return run_binary_bool_kernel(x, y, ops.le);
}

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::lt(const torch::Tensor& x, const torch::Tensor& y) const {
    return run_binary_bool_kernel(x, y, ops.lt);
}

static inline std::vector<int64_t> _canonicalize_dims(std::vector<int64_t> dims, int64_t ndim) {
    std::vector<int64_t> out;
    out.reserve(dims.size());

    for (int64_t dim : dims) {
        dim = dim < 0 ? dim + ndim : dim;
        TORCH_CHECK(dim >= 0 && dim < ndim, "Dimension ", dim, " out of range for tensor of dim ", ndim);
        out.push_back(dim);
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static inline std::vector<int64_t> _reduced_sizes(
    const torch::Tensor& x,
    const std::vector<int64_t>& reduce_dims,
    bool keepdim
) {
    std::vector<int64_t> sizes = x.sizes().vec();

    if (keepdim)
        for (int64_t dim : reduce_dims)
            sizes[dim] = 1;

    else
        for (auto it = reduce_dims.rbegin(); it != reduce_dims.rend(); ++it)
            sizes.erase(sizes.begin() + *it);

    return sizes;
}

template <size_t bitwidth>
struct SumOps {
    using StorageT = typename Ops<bitwidth>::StorageT;
    const Ops<bitwidth>* ops;

    StorageT reduce(StorageT a, StorageT b, int64_t /*index*/) {
        return ops->add(a, b);
    }

    StorageT combine(StorageT a, StorageT b) {
        return ops->add(a, b);
    }

    StorageT project(StorageT acc) {
        return acc;
    }

    StorageT translate_idx(StorageT acc, int64_t /*index*/) {
        return acc;
    }
};

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::sum(
    const torch::Tensor& x,
    c10::optional<std::vector<int64_t>> dim,
    bool keepdim
) const {
    torch::Tensor src = x.is_contiguous() ? x : x.contiguous();

    if (!dim.has_value() || dim.value().empty()) {
        const StorageT* src_ptr = x.data_ptr<StorageT>();
        const int64_t numel = x.numel();

        constexpr int kUnroll = 4;
        const int64_t grain = 1 << 13; // ~8k elements per thread

        StorageT global_acc = at::parallel_reduce(
            /*begin*/ int64_t{0},
            /*end*/ numel,
            /*grain*/ grain,
            /*identity*/ ops.from_float(0.0f),
            /*body*/ [&](int64_t begin, int64_t end, StorageT /*identity*/) -> StorageT {
                const StorageT* ptr = src_ptr + begin;
                int64_t len = end - begin;
                StorageT local = ops.from_float(0.0f);

                int64_t i = 0;
                for (; i <= len - kUnroll; i += kUnroll) {
                    StorageT t0 = ptr[i    ];
                    StorageT t1 = ptr[i + 1];
                    StorageT t2 = ptr[i + 2];
                    StorageT t3 = ptr[i + 3];

                    StorageT block = ops.add(ops.add(t0, t1), ops.add(t2, t3));
                    local = ops.add(local, block);
                }

                // tail
                for (; i < len; ++i)
                    local = ops.add(local, ptr[i]);

                return local;
            },
            /*reduce*/ [this](StorageT a, StorageT b) -> StorageT {
                return ops.add(a, b);
            });

        return at::scalar_tensor(global_acc, x.options());
    }

    std::vector<int64_t> reduce_dims = _canonicalize_dims(dim.value(), src.dim());
    if (reduce_dims.empty())
        return keepdim ? src.clone() : src;

    torch::Tensor out = at::empty(_reduced_sizes(src, reduce_dims, /*keepdim*/ true), src.options());
    auto iter = at::meta::make_reduction(src, out, reduce_dims, /*keepdim*/ true, int_type_from_bitwidth<bitwidth>());

    if (iter.numel() == 0)
        out.fill_(ops.from_float(0.0f));

    else
        at::native::binary_kernel_reduce(
            iter,
            /*ops*/SumOps<bitwidth>{&ops},
            /*identity*/ ops.from_float(0.0f)
        );

    return (keepdim) ? out : out.squeeze(reduce_dims);
}

template<size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::matmul(const torch::Tensor& A, const torch::Tensor& B) const {

    if (A.dim() == 1 && B.dim() == 1) {
        TORCH_CHECK(A.size(0) == B.size(0), "dot product: size mismatch");
        // return sum(mul(A, B));

        const int64_t K = A.size(0);
        const StorageT* A_ptr = A.data_ptr<StorageT>();
        const StorageT* B_ptr = B.data_ptr<StorageT>();

        StorageT acc = ops.from_float(0.0f);
        for (int64_t k = 0; k < K; ++k)
            acc = ops.add(acc, ops.mul(A_ptr[k], B_ptr[k]));

        return at::scalar_tensor(acc, A.options());
    }

    bool A_was_1d = A.dim() == 1;
    bool B_was_1d = B.dim() == 1;

    torch::Tensor A_prep = A_was_1d ? A.unsqueeze(0) : A;
    torch::Tensor B_prep = B_was_1d ? B.unsqueeze(-1) : B;

    TORCH_CHECK(A.size(-1) == B.size(-2), "matmul: size mismatch");

    auto A_batch = A_prep.sizes().slice(0, A_prep.dim() - 2);
    auto B_batch = B_prep.sizes().slice(0, B_prep.dim() - 2);
    std::vector<int64_t> out_batch = at::infer_size(A_batch, B_batch); // throws if not broadcastable

    auto expand_to = [&](const torch::Tensor& t) {
        std::vector<int64_t> s(out_batch.begin(), out_batch.end());
        s.push_back(t.size(-2));
        s.push_back(t.size(-1));
        return t.expand(s);
    };

    A_prep = expand_to(A_prep);
    B_prep = expand_to(B_prep);

    const int64_t M = A_prep.size(-2);
    const int64_t K = A_prep.size(-1);
    const int64_t N = B_prep.size(-1);

    std::vector<int64_t> out_shape(out_batch.begin(), out_batch.end());
    out_shape.push_back(M);
    out_shape.push_back(N);

    torch::Tensor out = torch::full(out_shape, ops.from_float(0.0f), A_prep.options());

    for (int64_t k = 0; k < K; ++k) {
        auto A_slice = A_prep.select(-1, k).unsqueeze(-1);  // ... × M × 1
        auto B_slice = B_prep.select(-2, k).unsqueeze(-2);  // ... × 1 × N

        torch::Tensor term = mul(A_slice, B_slice);
        out = add(out, term);
    }

    if (A_was_1d) out = out.squeeze(-2);
    if (B_was_1d) out = out.squeeze(-1);

    return out;
}

static torch::Tensor _reduce_like(
    const torch::Tensor& grad_expanded,
    const torch::Tensor& original_view
) {
    const int64_t gdim = grad_expanded.dim();
    const int64_t odim = original_view.dim();
    const int64_t offset = gdim - odim;

    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < grad_expanded.dim(); ++d) {
        int64_t o_d = d - offset;
        int64_t o_size = (o_d >= 0) ? original_view.size(o_d) : 1;

        if (o_size == 1 && grad_expanded.size(d) > 1)
            reduce_dims.push_back(d);
    }

    if (reduce_dims.empty())
        return grad_expanded;

    return sum(grad_expanded, reduce_dims, /*keepdim=*/ true);
}

template <size_t bitwidth>
std::vector<torch::Tensor> OpsImpl<bitwidth>::matmul_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& A,
    const torch::Tensor& B
) const {

    if (A.dim() == 1 && B.dim() == 1) {
        TORCH_CHECK(grad_out.dim() == 0, "grad_out for dot product must be a scalar");

        const int64_t K = A.size(0);
        const StorageT go = grad_out.item<StorageT>();

        torch::Tensor dA = at::empty_like(A);
        torch::Tensor dB = at::empty_like(B);

        const StorageT* ap = A.data_ptr<StorageT>();
        const StorageT* bp = B.data_ptr<StorageT>();
        StorageT* dap = dA.data_ptr<StorageT>();
        StorageT* dbp = dB.data_ptr<StorageT>();

        at::parallel_for(
            /*begin*/ int64_t{0},
            /*end*/ K,
            /*grain_size*/ 64,
            /*body*/ [&](int64_t begin, int64_t end) {
                for (int64_t k = begin; k < end; ++k) {
                    dap[k] = ops.mul(go, bp[k]);
                    dbp[k] = ops.mul(go, ap[k]);
                }
            });

        return {dA, dB};
    }

    bool A_was_1d = A.dim() == 1;
    bool B_was_1d = B.dim() == 1;

    torch::Tensor A_prep = A_was_1d ? A.unsqueeze(0) : A;
    torch::Tensor B_prep = B_was_1d ? B.unsqueeze(-1) : B;

    auto A_batch = A_prep.sizes().slice(0, A_prep.dim() - 2);
    auto B_batch = B_prep.sizes().slice(0, B_prep.dim() - 2);
    std::vector<int64_t> batch_shape = at::infer_size(A_batch, B_batch);

    auto expand_to = [&](const torch::Tensor& t) {
        std::vector<int64_t> s(batch_shape.begin(), batch_shape.end());
        s.push_back(t.size(-2));
        s.push_back(t.size(-1));
        return t.expand(s);
    };

    A_prep = expand_to(A_prep);
    B_prep = expand_to(B_prep);
    torch::Tensor grad_out_prep = expand_to(grad_out);

    torch::Tensor dA = matmul(grad_out_prep, B_prep.transpose(-2, -1).contiguous());
    torch::Tensor dB = matmul(A_prep.transpose(-2, -1).contiguous(), grad_out_prep);
    dA = _reduce_like(dA, A);
    dB = _reduce_like(dB, B);

    if (A_was_1d) dA = dA.squeeze(0);
    if (B_was_1d) dB = dB.squeeze(-1);

    return {dA, dB};
}

// Explicit template instantiation
template struct OpsImpl<8>;
template struct OpsImpl<16>;
template struct OpsImpl<32>;
template struct OpsImpl<64>;