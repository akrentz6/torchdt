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

// Explicit template instantiation
template struct OpsImpl<8>;
template struct OpsImpl<16>;
template struct OpsImpl<32>;
template struct OpsImpl<64>;