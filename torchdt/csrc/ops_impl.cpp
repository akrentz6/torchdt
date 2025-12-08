#include <torch/extension.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/ReduceOpsUtils.h>

#include "ops_impl.h"

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

template <size_t bitwidth>
torch::Tensor OpsImpl<bitwidth>::conv2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups
) const {

    torch::Tensor input_prepped;
    bool squeeze_batch = false;
    if (input.dim() == 3) {
        input_prepped = input.unsqueeze(0);
        squeeze_batch = true;
    }
    else {
        input_prepped = input;
    }

    TORCH_CHECK(input_prepped.dim() == 4 && weight.dim() == 4,
        "Expected 4-D input (N, C_in, H, W) and 4-D weight "
        "(C_out, C_in/groups, K_h, K_w)");

    const int64_t N = input_prepped.size(0);
    const int64_t Cin = input_prepped.size(1);
    const int64_t Hin = input_prepped.size(2);
    const int64_t Win = input_prepped.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t Kh = weight.size(2);
    const int64_t Kw = weight.size(3);

    TORCH_CHECK(Cin % groups == 0 && Cout % groups == 0, "C_in and C_out must be divisible by groups");

    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;

    TORCH_CHECK(weight.size(1) == Cin_g, "weight second dim must equal C_in / groups");

    if (bias.has_value())
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == Cout,
                    "bias must be 1-D with length C_out");

    const int64_t stride_h = stride[0], stride_w = stride[1];
    const int64_t pad_h = padding[0], pad_w = padding[1];
    const int64_t dil_h = dilation[0], dil_w = dilation[1];

    const int64_t Hout = (Hin + 2 * pad_h - dil_h * (Kh - 1) - 1) / stride_h + 1;
    const int64_t Wout = (Win + 2 * pad_w - dil_w * (Kw - 1) - 1) / stride_w + 1;

    TORCH_CHECK(Hout > 0 && Wout > 0, "Output size is non-positive");

    torch::Tensor output = at::empty({N, Cout, Hout, Wout}, input_prepped.options());

    const StorageT* in = input_prepped.data_ptr<StorageT>();
    const StorageT* w = weight.data_ptr<StorageT>();
    StorageT* out = output.data_ptr<StorageT>();
    const StorageT* b = bias.has_value() ? bias->data_ptr<StorageT>() : nullptr;

    const int64_t in_stride_N = Cin * Hin * Win;
    const int64_t in_stride_C = Hin * Win;
    const int64_t in_stride_H = Win;

    const int64_t w_stride_Cout = Cin_g * Kh * Kw;
    const int64_t w_stride_Cin = Kh * Kw;
    const int64_t w_stride_Kh = Kw;

    const int64_t out_stride_N = Cout * Hout * Wout;
    const int64_t out_stride_C = Hout * Wout;

    const int64_t work_items = N * Cout;
    const int64_t grain = 16;
    StorageT zero = ops.from_float(0.0f);

    at::parallel_for(
        /*begin*/ 0,
        /*end*/   work_items,
        /*grain*/ grain,
        /*body*/ [&](int64_t begin, int64_t end) {

            for (int64_t linear = begin; linear < end; ++linear) {

                const int64_t n = linear / Cout; // batch index
                const int64_t oc = linear % Cout; // output channel index

                const int64_t g = oc / Cout_g;
                const int64_t oc_in_g = oc % Cout_g;

                const StorageT* w_g = w + (g * Cout_g + oc_in_g) * w_stride_Cout;
                const StorageT* in_g = in + n * in_stride_N + g * Cin_g * Hin * Win;
                StorageT* out_g = out + n * out_stride_N + oc * out_stride_C;

                for (int64_t y_out = 0; y_out < Hout; ++y_out) {
                    const int64_t y_in0 = y_out * stride_h - pad_h;

                    const int64_t kh_min = std::max<int64_t>(0, (-y_in0 + dil_h - 1) / dil_h);
                    const int64_t kh_max = std::min<int64_t>(Kh, (Hin - y_in0 + dil_h - 1) / dil_h);

                    for (int64_t x_out = 0; x_out < Wout; ++x_out) {
                        const int64_t x_in0 = x_out * stride_w - pad_w;

                        const int64_t kw_min = std::max<int64_t>(0, (-x_in0 + dil_w - 1) / dil_w);
                        const int64_t kw_max = std::min<int64_t>(Kw, (Win - x_in0 + dil_w - 1) / dil_w);

                        StorageT acc = b ? b[oc] : zero;
                        for (int64_t ic_g = 0; ic_g < Cin_g; ++ic_g) {

                            const StorageT* in_c = in_g + ic_g * in_stride_C;
                            const StorageT* w_c  = w_g + ic_g * w_stride_Cin;

                            for (int64_t kh = kh_min; kh < kh_max; ++kh) {
                                const int64_t in_row_offset = (y_in0 + kh * dil_h) * in_stride_H;
                                const StorageT* in_row = in_c + in_row_offset;
                                const StorageT* w_row = w_c + kh * w_stride_Kh;

                                for (int64_t kw = kw_min; kw < kw_max; ++kw) {
                                    const StorageT in_val = in_row[x_in0 + kw * dil_w];
                                    const StorageT w_val = w_row[kw];

                                    acc = ops.add(acc, ops.mul(in_val, w_val));
                                }
                            }
                        }

                        out_g[y_out * Wout + x_out] = acc;
                    }
                }
            }

    });

    return squeeze_batch ? output.squeeze(0) : output;
}

template <size_t bitwidth>
std::vector<torch::Tensor> OpsImpl<bitwidth>::conv2d_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    bool has_bias,
    int64_t groups
) const {

    torch::Tensor input_prepped;
    torch::Tensor grad_out_prepped;
    bool squeeze_batch = false;
    if (input.dim() == 3) {
        input_prepped = input.unsqueeze(0);
        grad_out_prepped = grad_out.unsqueeze(0);
        squeeze_batch = true;
    }
    else {
        input_prepped = input;
        grad_out_prepped = grad_out;
    }

    TORCH_CHECK(input_prepped.dim() == 4 && weight.dim() == 4, "conv2d_backward expects 4-D input and weight");

    const int64_t N = input_prepped.size(0);
    const int64_t Cin = input_prepped.size(1);
    const int64_t Hin = input_prepped.size(2);
    const int64_t Win = input_prepped.size(3);

    const int64_t Cout = weight.size(0);
    const int64_t Kh = weight.size(2);
    const int64_t Kw = weight.size(3);

    TORCH_CHECK(Cin % groups == 0 && Cout % groups == 0, "C_in and C_out must be divisible by groups");

    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;

    const int64_t stride_h = stride[0], stride_w = stride[1];
    const int64_t pad_h = padding[0], pad_w = padding[1];
    const int64_t dil_h = dilation[0], dil_w = dilation[1];

    const int64_t Hout = grad_out_prepped.size(2);
    const int64_t Wout = grad_out_prepped.size(3);

    torch::Tensor grad_input = at::empty_like(input_prepped);
    torch::Tensor grad_weight = at::empty_like(weight);
    torch::Tensor grad_bias = has_bias ? at::empty({Cout}, input_prepped.options()) : torch::Tensor();

    StorageT zero = ops.from_float(0.0f);

    grad_input.fill_(zero);
    grad_weight.fill_(zero);
    if (has_bias) grad_bias.fill_(zero);

    const StorageT* in_ptr = input_prepped.data_ptr<StorageT>();
    const StorageT* w_ptr = weight.data_ptr<StorageT>();
    const StorageT* go_ptr = grad_out_prepped.data_ptr<StorageT>();

    StorageT* gi_ptr = grad_input.data_ptr<StorageT>();

    const int64_t in_stride_N = Cin * Hin * Win;
    const int64_t in_stride_C = Hin * Win;
    const int64_t in_stride_H = Win;

    const int64_t w_stride_Cout = Cin_g * Kh * Kw;
    const int64_t w_stride_Cin = Kh * Kw;
    const int64_t w_stride_Kh = Kw;

    const int64_t go_stride_N = Cout * Hout * Wout;
    const int64_t go_stride_C = Hout * Wout;
    const int64_t go_stride_H = Wout;

    // grad_input strides equal input strides    

    const int64_t work_items = N * Cout;
    const int64_t grain = 16;

    at::parallel_for(
        /*begin*/ 0,
        /*end*/ work_items,
        /*grain_size*/ grain,
        /*body*/ [&](int64_t begin, int64_t end) {

            // Thread-local scratch copies
            torch::Tensor gw_private = at::empty_like(weight);
            gw_private.fill_(zero);

            torch::Tensor gb_private;
            if (has_bias) {
                gb_private = at::empty({Cout}, input_prepped.options());
                gb_private.fill_(zero);
            }

            StorageT* gw_p = gw_private.data_ptr<StorageT>();
            StorageT* gb_p = has_bias ? gb_private.data_ptr<StorageT>() : nullptr;

            for (int64_t linear = begin; linear < end; ++linear) {

                const int64_t n = linear / Cout; // batch idx
                const int64_t co = linear % Cout; // output channel idx

                const int64_t g = co / Cout_g;
                const int64_t co_in_group = co % Cout_g;
                const int64_t ci_group_begin = g * Cin_g;

                const int64_t go_base = n * go_stride_N + co * go_stride_C;
                const int64_t in_base = n * in_stride_N;
                const int64_t w_base = (g * Cout_g + co_in_group) * w_stride_Cout;

                for (int64_t y_out = 0; y_out < Hout; ++y_out) {
                    const int64_t go_row_base = go_base + y_out * go_stride_H;
                    const int64_t y_in_origin = y_out * stride_h - pad_h;

                    for (int64_t x_out = 0; x_out < Wout; ++x_out) {
                        const int64_t go_offset = go_row_base + x_out;
                        const StorageT go_val = go_ptr[go_offset];

                        if (has_bias)
                            gb_p[co] = ops.add(gb_p[co], go_val);

                        for (int64_t kh = 0; kh < Kh; ++kh) {
                            const int64_t y_in = y_in_origin + kh * dil_h;
                            if (y_in < 0 || y_in >= Hin) continue;

                            const int64_t w_row_base = w_base + kh * w_stride_Kh;
                            const int64_t in_row_base = in_base + (y_in * in_stride_H);

                            for (int64_t kw = 0; kw < Kw; ++kw) {
                                const int64_t x_in = x_out * stride_w - pad_w + kw * dil_w;
                                if (x_in < 0 || x_in >= Win) continue;

                                const int64_t w_col_offset = w_row_base + kw;
                                const int64_t in_col_base = in_row_base + x_in;

                                for (int64_t ci_rel = 0; ci_rel < Cin_g; ++ci_rel) {
                                    const int64_t ci = ci_group_begin + ci_rel;

                                    const int64_t in_idx = in_col_base + ci * in_stride_C;
                                    const int64_t w_idx = w_col_offset + ci_rel * w_stride_Cin;

                                    const StorageT prod_in = ops.mul(go_val, w_ptr[w_idx]);
                                    gi_ptr[in_idx] = ops.add(gi_ptr[in_idx], prod_in);

                                    const StorageT prod_w  = ops.mul(go_val, in_ptr[in_idx]);
                                    gw_p[w_idx] = ops.add(gw_p[w_idx], prod_w);
                                }
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            {
                grad_weight.copy_(add(grad_weight, gw_private));
                if (has_bias)
                    grad_bias.copy_(add(grad_bias, gb_private));
            }

        });

    if (squeeze_batch) {
        grad_input = grad_input.squeeze(0);
    }

    return {grad_input, grad_weight, grad_bias};
}

// const int64_t tile_cols = 1024;

// template <size_t bitwidth>
// torch::Tensor OpsImpl<bitwidth>::im2col_tile(
//     const StorageT* __restrict in_ptr, // pointer to (C, H, W) for the sample
//     int64_t C_per_group,
//     int64_t H, int64_t W,
//     int64_t KH, int64_t KW,
//     int64_t dil_h, int64_t dil_w,
//     int64_t pad_h, int64_t pad_w,
//     int64_t stride_h, int64_t stride_w,
//     int64_t tile_col_start,
//     int64_t tile_col_end,
//     int64_t OH, int64_t OW
// ) const {
//     const int64_t tile_cols = tile_col_end - tile_col_start;
//     const int64_t Ksize = C_per_group * KH * KW;

//     torch::Tensor out = at::empty({Ksize, tile_cols}, at::device(at::kCPU).dtype(int_type_from_bitwidth<bitwidth>()));
//     StorageT* out_ptr = out.data_ptr<StorageT>();

//     // for each output column index in the tile, compute input coordinates
//     for (int64_t col = 0; col < tile_cols; ++col) {
//         int64_t idx = tile_col_start + col; // linear index in [0, OH*OW)
//         const int64_t oh = idx / OW;
//         const int64_t ow = idx % OW;

//         const int64_t ih0 = oh * stride_h - pad_h;
//         const int64_t iw0 = ow * stride_w - pad_w;

//         int64_t out_offset = col; // column index, we will write at out_ptr[row * tile_cols + col]

//         int64_t k = 0;
//         StorageT zero = ops.from_float(0.0f);
//         for (int64_t ic = 0; ic < C_per_group; ++ic) {
//             const int64_t in_c_offset = ic * H * W; // index into in_ptr for start of channel
//             for (int64_t kh = 0; kh < KH; ++kh) {
//                 const int64_t ih = ih0 + kh * dil_h;
//                 if (ih < 0 || ih >= H) {
//                     for (int64_t kw = 0; kw < KW; ++kw) {
//                         out_ptr[k * tile_cols + out_offset] = zero;
//                         ++k;
//                     }
//                     continue;
//                 }
//                 const int64_t in_row = ih * W;
//                 for (int64_t kw = 0; kw < KW; ++kw) {
//                     const int64_t iw = iw0 + kw * dil_w;
//                     StorageT v = zero;
//                     if (iw >= 0 && iw < W) {
//                         v = in_ptr[in_c_offset + in_row + iw];
//                     }
//                     out_ptr[k * tile_cols + out_offset] = v;
//                     ++k;
//                 }
//             }
//         }
//     }

//     return out;
// }

// template <size_t bitwidth>
// torch::Tensor OpsImpl<bitwidth>::conv2d(
//     const torch::Tensor& input,
//     const torch::Tensor& weight,
//     const c10::optional<torch::Tensor>& bias,
//     const std::vector<int64_t>& stride,
//     const std::vector<int64_t>& padding,
//     const std::vector<int64_t>& dilation,
//     int64_t groups
// ) const {
//     TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "input and weight must be 4D");

//     const int64_t N = input.size(0);
//     const int64_t C = input.size(1);
//     const int64_t H = input.size(2);
//     const int64_t W = input.size(3);

//     const int64_t OC = weight.size(0);
//     const int64_t KC = weight.size(1); // == C / groups
//     const int64_t KH = weight.size(2);
//     const int64_t KW = weight.size(3);

//     TORCH_CHECK(C % groups == 0, "C must be divisible by groups");
//     TORCH_CHECK(OC % groups == 0, "OC must be divisible by groups");

//     const int64_t C_per_group = C / groups;
//     const int64_t OC_per_group = OC / groups;

//     const int64_t stride_h = stride[0], stride_w = stride[1];
//     const int64_t pad_h = padding[0], pad_w = padding[1];
//     const int64_t dil_h = dilation[0], dil_w = dilation[1];

//     const int64_t OH = (H + 2*pad_h - dil_h * (KH - 1) - 1) / stride_h + 1;
//     const int64_t OW = (W + 2*pad_w - dil_w * (KW - 1) - 1) / stride_w + 1;
//     const int64_t L = OH * OW;
//     const int64_t Ksize = C_per_group * KH * KW;

//     torch::Tensor out = at::empty({N, OC, OH, OW}, input.options());
//     torch::Tensor weight_contig = weight.contiguous();
//     torch::Tensor bias_contig;
//     bool has_bias = bias.has_value();
//     if (has_bias) {
//         bias_contig = bias.value().contiguous();
//     }

//     const int64_t num_tiles = (L + tile_cols - 1) / tile_cols;
//     StorageT zero = ops.from_float(0.0f);

//     at::parallel_for(0, N * groups, 0, [&](int64_t begin, int64_t end) {
//         for (int64_t ng = begin; ng < end; ++ng) {
//             const int64_t n = ng / groups;
//             const int64_t g = ng % groups;

//             torch::Tensor in_n = input[n];
//             in_n = in_n.contiguous();
//             const StorageT* in_ptr = in_n.data_ptr<StorageT>();

//             torch::Tensor w_g = weight_contig.slice(0, g * OC_per_group, (g + 1) * OC_per_group).contiguous();
//             torch::Tensor w_g_flat = w_g.view({OC_per_group, Ksize});

//             torch::Tensor bias_g;
//             if (has_bias) bias_g = bias_contig.slice(0, g * OC_per_group, (g + 1) * OC_per_group).contiguous();

//             torch::Tensor out_n = out[n];
//             torch::Tensor out_buf = at::empty({OC_per_group, L}, out_n.options());
//             StorageT* out_buf_ptr = out_buf.data_ptr<StorageT>();

//             if (has_bias) {
//                 for (int64_t oc = 0; oc < OC_per_group; ++oc) {
//                     StorageT b = bias_g.data_ptr<StorageT>()[oc];
//                     StorageT* row_ptr = out_buf_ptr + oc * L;
//                     for (int64_t l = 0; l < L; ++l) row_ptr[l] = b;
//                 }
//             }
//             else {
//                 std::fill(out_buf_ptr, out_buf_ptr + (OC_per_group * L), zero);
//             }

//             for (int64_t t = 0; t < num_tiles; ++t) {
//                 int64_t col_s = t * tile_cols;
//                 int64_t col_e = std::min(col_s + tile_cols, L);
//                 int64_t cur_tile = col_e - col_s;

//                 torch::Tensor X_tile = im2col_tile(
//                     in_ptr + g * C_per_group * H * W,
//                     C_per_group, H, W, KH, KW,
//                     dil_h, dil_w, pad_h, pad_w, stride_h, stride_w,
//                     col_s, col_e, OH, OW
//                 );

//                 torch::Tensor Y_tile = matmul(w_g_flat, X_tile);
//                 const StorageT* yptr = Y_tile.data_ptr<StorageT>();
//                 for (int64_t m = 0; m < OC_per_group; ++m) {
//                     StorageT* dst = out_buf_ptr + m * L + col_s;
//                     const StorageT* src = yptr + m * cur_tile;
//                     std::copy(src, src + cur_tile, dst);
//                 }
//             }

//             StorageT* out_data = out.data_ptr<StorageT>();
//             const int64_t out_stride_n = OC * OH * OW;
//             const int64_t out_stride_c = OH * OW;
//             const int64_t base_channel = g * OC_per_group;

//             for (int64_t oc = 0; oc < OC_per_group; ++oc) {
//                 const StorageT* src = out_buf_ptr + oc * L;
//                 // pointer directly into out storage, no temporaries
//                 StorageT* dst = out_data + (n * out_stride_n) + ((base_channel + oc) * out_stride_c);

//                 // copy entire (OH * OW) region row-major
//                 std::copy(src, src + L, dst);
//             }
//         }
//     });

//     return out;
// }

// template <size_t bitwidth>
// torch::Tensor OpsImpl<bitwidth>::col2im_accumulate(
//     const torch::Tensor& cols,
//     int64_t C_per_group,
//     int64_t H, int64_t W,
//     int64_t KH, int64_t KW,
//     int64_t dil_h, int64_t dil_w,
//     int64_t pad_h, int64_t pad_w,
//     int64_t stride_h, int64_t stride_w,
//     int64_t OH, int64_t OW
// ) const {
//     const int64_t L = OH * OW;
//     const StorageT* col_ptr = cols.data_ptr<StorageT>();

//     torch::Tensor out = at::zeros({C_per_group, H, W}, cols.options());
//     StorageT* out_ptr = out.data_ptr<StorageT>();

//     for (int64_t l = 0; l < L; ++l) {
//         int64_t oh = l / OW;
//         int64_t ow = l % OW;
//         int64_t ih0 = oh * stride_h - pad_h;
//         int64_t iw0 = ow * stride_w - pad_w;

//         int64_t k = 0;
//         for (int64_t ic = 0; ic < C_per_group; ++ic) {
//             StorageT* out_c = out_ptr + ic * H * W;
//             for (int64_t kh = 0; kh < KH; ++kh) {
//                 int64_t ih = ih0 + kh * dil_h;
//                 if (ih < 0 || ih >= H) {
//                     k += KW;
//                     continue;
//                 }
//                 for (int64_t kw = 0; kw < KW; ++kw) {
//                     int64_t iw = iw0 + kw * dil_w;
//                     StorageT v = col_ptr[k * L + l];
//                     if (iw >= 0 && iw < W) {
//                         out_c[ih * W + iw] = ops.add(out_c[ih * W + iw], v);
//                     }
//                     ++k;
//                 }
//             }
//         }
//     }

//     return out;
// }

// template <size_t bitwidth>
// std::vector<torch::Tensor> OpsImpl<bitwidth>::conv2d_backward(
//     const torch::Tensor& grad_out,
//     const torch::Tensor& input,
//     const torch::Tensor& weight,
//     const std::vector<int64_t>& stride,
//     const std::vector<int64_t>& padding,
//     const std::vector<int64_t>& dilation,
//     bool has_bias,
//     const int64_t groups
// ) const {
//     TORCH_CHECK(input.dim() == 4 && weight.dim() == 4 && grad_out.dim() == 4, "conv2d_backward expects 4D tensors");

//     const int64_t N = input.size(0);
//     const int64_t C = input.size(1);
//     const int64_t H = input.size(2);
//     const int64_t W = input.size(3);

//     const int64_t OC = weight.size(0);
//     const int64_t Cg = weight.size(1);
//     const int64_t KH = weight.size(2);
//     const int64_t KW = weight.size(3);

//     TORCH_CHECK(C % groups == 0, "C must be divisible by groups");
//     TORCH_CHECK(OC % groups == 0, "OC must be divisible by groups");

//     const int64_t C_per_group = C / groups;
//     const int64_t OC_per_group = OC / groups;

//     const int64_t stride_h = stride[0], stride_w = stride[1];
//     const int64_t pad_h = padding[0], pad_w = padding[1];
//     const int64_t dil_h = dilation[0], dil_w = dilation[1];

//     const int64_t OH = grad_out.size(2);
//     const int64_t OW = grad_out.size(3);
//     const int64_t L = OH * OW;
//     const int64_t Ksize = C_per_group * KH * KW;

//     torch::Tensor grad_input = at::zeros_like(input);
//     torch::Tensor grad_weight = at::zeros_like(weight);
//     torch::Tensor grad_bias = at::empty({OC}, grad_out.options());

//     if (has_bias) {
//         for (int64_t oc = 0; oc < OC; ++oc) {
//             grad_bias[oc] = sum(grad_out.select(1, oc), /*dim*/ {}, /*keepdim*/ false);
//         }
//     }

//     torch::Tensor grad_out_contig = grad_out.contiguous();
//     torch::Tensor input_contig = input.contiguous();
//     torch::Tensor weight_contig = weight.contiguous();

//     StorageT zero = ops.from_float(0.0f);

//     at::parallel_for(0, N * groups, 0, [&](int64_t begin, int64_t end){
//         for (int64_t ng = begin; ng < end; ++ng) {
//             int64_t n = ng / groups;
//             int64_t g = ng % groups;

//             torch::Tensor in_ng = input_contig[n];
//             torch::Tensor gout_ng = grad_out_contig[n];

//             const StorageT* in_ptr = in_ng.data_ptr<StorageT>() + g * C_per_group * H * W;
//             const StorageT* gout_ptr = gout_ng.data_ptr<StorageT>() + g * OC_per_group * OH * OW;

//             torch::Tensor col = at::empty({Ksize, L}, input.options());
//             StorageT* col_ptr = col.data_ptr<StorageT>();

//             int64_t idx = 0;
//             for (int64_t oh = 0; oh < OH; ++oh) {
//                 for (int64_t ow = 0; ow < OW; ++ow, ++idx) {
//                     int64_t ih0 = oh * stride_h - pad_h;
//                     int64_t iw0 = ow * stride_w - pad_w;
//                     int64_t k = 0;
//                     for (int64_t ic = 0; ic < C_per_group; ++ic) {
//                         const StorageT* in_c = in_ptr + ic * H * W;
//                         for (int64_t kh = 0; kh < KH; ++kh) {
//                             int64_t ih = ih0 + kh * dil_h;
//                             if ((unsigned)ih >= (unsigned)H) {
//                                 for (int64_t kw = 0; kw < KW; ++kw) {
//                                     col_ptr[k * L + idx] = 0;
//                                     ++k;
//                                 }
//                                 continue;
//                             }
//                             const int64_t base = ih * W;
//                             for (int64_t kw = 0; kw < KW; ++kw) {
//                                 int64_t iw = iw0 + kw * dil_w;
//                                 col_ptr[k * L + idx] = (unsigned)iw < (unsigned)W ? in_c[base + iw] : zero;
//                                 ++k;
//                             }
//                         }
//                     }
//                 }
//             }

//             torch::Tensor gout_mat = gout_ng.slice(0, g * OC_per_group, (g+1) * OC_per_group)
//                                             .reshape({OC_per_group, L});
//             torch::Tensor dW_g = matmul(gout_mat, col.transpose(0, 1).contiguous());

//             grad_weight = add(
//                 grad_weight.slice(0, g * OC_per_group, (g+1)*OC_per_group),
//                 dW_g.reshape({OC_per_group, C_per_group, KH, KW})
//             );

//             torch::Tensor Wg = weight_contig.slice(0, g*OC_per_group, (g+1)*OC_per_group)
//                                             .reshape({OC_per_group, Ksize});
//             torch::Tensor dcol = matmul(Wg.transpose(0,1).contiguous(), gout_mat);

//             torch::Tensor din_g = col2im_accumulate(
//                 dcol,
//                 C_per_group,
//                 H, W,
//                 KH, KW,
//                 dil_h, dil_w,
//                 pad_h, pad_w,
//                 stride_h, stride_w,
//                 OH, OW
//             );

//             grad_input[n] = add(grad_input[n].narrow(0, g * C_per_group, C_per_group), din_g);
//         }
//     });

//     return {grad_input, grad_weight, grad_bias};
// }

// Explicit template instantiation
template struct OpsImpl<8>;
template struct OpsImpl<16>;
template struct OpsImpl<32>;
template struct OpsImpl<64>;