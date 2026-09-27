#pragma once
#include <torchdt/registry.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <array>
#include <algorithm>
#include <numeric>
#include <torchdt/cpu/arithmetic.h>
#include <limits>
#include <utility>

namespace torchdt::native {
constexpr int64_t reduction_chunk = 256;

// Streaming equivalent of adjacent-pair reduction with odd tails carried.
// Fixed tree, bounded workspace, no thread-scheduling-dependent accumulation.
template<class Scalar, class Value>
auto tree_reduce(const Scalar& scalar, int64_t count, Value value) -> typename Scalar::S {
    using S = typename Scalar::S;
    std::array<S, 64> levels{};
    uint64_t occupied = 0;
    for (int64_t i = 0; i < count; ++i) {
        S current = value(i);
        unsigned level = 0;
        while (occupied & (uint64_t(1) << level)) {
            current = scalar.add(levels[level], current);
            occupied &= ~(uint64_t(1) << level);
            ++level;
        }
        levels[level] = current;
        occupied |= uint64_t(1) << level;
    }
    S result = Scalar::zero;
    bool first = true;
    for (unsigned level = 0; level < 64; ++level) {
        if (occupied & (uint64_t(1) << level)) {
            result = first ? levels[level] : scalar.add(levels[level], result);
            first = false;
        }
    }
    return result;
}

// Arithmetic is a value-owned policy. All inner-loop scalar calls are statically
// dispatched; no datatype tests or virtual calls occur per arithmetic operation.
template<class Arithmetic> class CPUKernels final : public TensorKernels {
    using Scalar = typename Arithmetic::Scalar;
    using S = typename Scalar::S;
    static constexpr auto storage = Arithmetic::storage;
    const Arithmetic arithmetic;

    Tensor pairwise_tensor(Tensor values) const {
        while (values.size(-1) > 1) {
            int64_t n = values.size(-1), pairs = n/2;
            auto next = binary(values.slice(-1,0,2*pairs,2), values.slice(-1,1,2*pairs,2), "add");
            values = n%2 ? at::cat({next, values.slice(-1,n-1,n)}, -1) : next;
        }
        return values.select(-1,0);
    }
    Tensor tree_tensor(const Tensor& values) const {
        int64_t count = values.size(-1);
        if (count == 0) {
            auto shape = values.sizes().vec(); shape.pop_back();
            return at::full(shape, Scalar::zero, values.options());
        }
        std::vector<Tensor> chunks;
        for (int64_t offset=0; offset<count; offset+=reduction_chunk)
            chunks.push_back(pairwise_tensor(values.slice(-1,offset,std::min(count,offset+reduction_chunk))));
        return pairwise_tensor(at::stack(chunks,-1));
    }
    template<class Value> S reduce_values(int64_t count, Value value) const {
        if (!arithmetic.use_batch_add()) return tree_reduce(arithmetic.scalar,count,value);
        // Policies with batch addition use the same reduction tree via tensor
        // primitives. Bound scratch to a chunk plus chunk totals.
        std::vector<Tensor> chunks;
        for (int64_t offset=0; offset<count; offset+=reduction_chunk) {
            auto values = at::empty({std::min(reduction_chunk,count-offset)}, at::TensorOptions().dtype(storage));
            auto ptr = values.template data_ptr<S>();
            for (int64_t j=0;j<values.numel();++j) ptr[j]=value(offset+j);
            chunks.push_back(pairwise_tensor(values));
        }
        return chunks.empty() ? Scalar::zero : pairwise_tensor(at::stack(chunks,-1)).template item<S>();
    }

    template<class F>
    void parallel_outputs(int64_t begin, int64_t end, int64_t grain, const F& work) const {
        // Tensor operations require the caller's TLS. Batch arithmetic stays
        // on that thread; fused scalar paths parallelize.
        if (arithmetic.use_batch_add()) work(begin,end);
        else at::parallel_for(begin,end,grain,work);
    }

    void check(const Tensor& x) const {
        TORCH_CHECK(x.device().is_cpu(), "Native CPU kernel requires CPU tensors");
        TORCH_CHECK(x.layout() == at::kStrided, "Native kernel requires strided tensors");
        TORCH_CHECK(x.scalar_type() == storage, "Incorrect native storage dtype");
    }
    template<class In, class Out, class F>
    Tensor unary_kernel(const Tensor& x, at::ScalarType output_type, F f) const {
        auto out = at::empty({0}, x.options().dtype(output_type));
        auto iter = at::TensorIteratorConfig().check_all_same_dtype(false)
            .add_output(out).add_const_input(x).build();
        at::native::cpu_kernel(iter, [f](In a) -> Out { return f(a); });
        return out;
    }
    template<class Out, class F>
    Tensor binary_kernel(const Tensor& x, const Tensor& y, at::ScalarType output_type, F f) const {
        auto out = at::empty({0}, x.options().dtype(output_type));
        auto iter = at::TensorIteratorConfig().check_all_same_dtype(false)
            .add_output(out).add_const_input(x).add_const_input(y).build();
        at::native::cpu_kernel(iter, [f](S a, S b) -> Out { return f(a, b); });
        return out;
    }
    Tensor reduce_like(const Tensor& x, const Tensor& target) const {
        Dims dims;
        int64_t extra = x.dim() - target.dim();
        for (int64_t i = 0; i < x.dim(); ++i)
            if (i < extra || (target.size(i-extra) == 1 && x.size(i) != 1)) dims.push_back(i);
        return (dims.empty() ? x : sum(x, dims, true)).reshape(target.sizes());
    }

    struct Conv {
        Tensor x, w;
        bool unbatched;
        int64_t n, ci, h, wi, co, kh, kw, cg, og, ho, wo, sh, sw, ph, pw, dh, dw;
    };
    Conv prepare_conv(const Tensor& x, const Tensor& w, const Dims& stride,
                      const Dims& padding, const Dims& dilation, int64_t groups) const {
        check(x); check(w);
        TORCH_CHECK((x.dim() == 3 || x.dim() == 4) && w.dim() == 4,
                    "conv2d requires a 3D/4D input and 4D weight");
        TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
                    "stride, padding and dilation require two entries");
        TORCH_CHECK(groups > 0 && stride[0] > 0 && stride[1] > 0 &&
                    dilation[0] > 0 && dilation[1] > 0 && padding[0] >= 0 && padding[1] >= 0,
                    "Invalid convolution parameters");
        Conv c;
        c.unbatched = x.dim() == 3;
        c.x = (c.unbatched ? x.unsqueeze(0) : x).contiguous();
        c.w = w.contiguous();
        c.n = c.x.size(0); c.ci = c.x.size(1); c.h = c.x.size(2); c.wi = c.x.size(3);
        c.co = w.size(0); c.kh = w.size(2); c.kw = w.size(3);
        TORCH_CHECK(c.ci > 0 && c.co > 0 && c.kh > 0 && c.kw > 0 &&
                    c.ci % groups == 0 && c.co % groups == 0 && w.size(1) == c.ci / groups,
                    "Invalid grouped convolution weight shape");
        c.cg = c.ci / groups; c.og = c.co / groups;
        c.sh = stride[0]; c.sw = stride[1]; c.ph = padding[0]; c.pw = padding[1];
        c.dh = dilation[0]; c.dw = dilation[1];
        // Validate before division (C++ integer division truncates toward zero).
        const int64_t limit = std::numeric_limits<int64_t>::max();
        TORCH_CHECK(c.ph <= (limit-c.h)/2 && c.pw <= (limit-c.wi)/2 &&
                    c.kh-1 <= limit/c.dh && c.kw-1 <= limit/c.dw,
                    "Convolution parameters exceed the indexing range");
        int64_t nh = c.h + 2*c.ph - c.dh*(c.kh-1) - 1;
        int64_t nw = c.wi + 2*c.pw - c.dw*(c.kw-1) - 1;
        TORCH_CHECK(nh >= 0 && nw >= 0, "Convolution output size is non-positive");
        c.ho = nh/c.sh + 1; c.wo = nw/c.sw + 1;
        return c;
    }
public:
    explicit CPUKernels(Arithmetic policy) : arithmetic(std::move(policy)) {}
    std::vector<std::string> capabilities() const override {
        return {"from_float", "to_float", "neg", "abs", "sign", "sqrt", "pow",
                "add", "sub", "mul", "div", "ge", "gt", "le", "lt", "sum",
                "matmul", "matmul_backward", "conv2d", "conv2d_backward"};
    }
    Tensor unary(const Tensor& x, const std::string& op) const override {
        if (op == "from_float") {
            TORCH_CHECK(x.device().is_cpu() && x.layout() == at::kStrided,
                        "Native conversion requires a strided CPU tensor");
            auto input = x.to(at::kDouble);
            arithmetic.validate_conversion(input);
            return unary_kernel<double, S>(input, storage, [s=arithmetic.scalar](double a) { return s.from_float(a); });
        }
        check(x);
        if (op == "to_float") return unary_kernel<S, double>(x, at::kDouble, [s=arithmetic.scalar](S a) { return s.to_float(a); });
#define UNARY(NAME) if (op == #NAME) return unary_kernel<S, S>(x, storage, [s=arithmetic.scalar](S a) { return s.NAME(a); });
        UNARY(neg) UNARY(abs) UNARY(sign) UNARY(sqrt)
#undef UNARY
        TORCH_CHECK(false, "Unknown native unary operation: ", op);
    }
    Tensor binary(const Tensor& x, const Tensor& y, const std::string& op) const override {
        check(x); check(y);
        if (auto result = arithmetic.binary_override(x, y, op,
                [this](const Tensor& value, const std::string& name) { return unary(value, name); }))
            return *result;
#define BINARY(NAME) if (op == #NAME) return binary_kernel<S>(x, y, storage, [s=arithmetic.scalar](S a, S b) { return s.NAME(a, b); });
        BINARY(add) BINARY(sub) BINARY(mul) BINARY(div) BINARY(pow)
#undef BINARY
#define COMPARE(NAME) if (op == #NAME) return binary_kernel<bool>(x, y, at::kBool, [s=arithmetic.scalar](S a, S b) { return s.NAME(a, b); });
        COMPARE(ge) COMPARE(gt) COMPARE(le) COMPARE(lt)
#undef COMPARE
        TORCH_CHECK(false, "Unknown native binary operation: ", op);
    }
    Tensor sum(const Tensor& x, c10::optional<Dims> dimensions, bool keepdim) const override {
        check(x);
        Dims dims;
        if (dimensions) dims = *dimensions;
        else { dims.resize(x.dim()); std::iota(dims.begin(), dims.end(), 0); }
        for (auto& d : dims) {
            if (x.dim() == 0) {
                TORCH_CHECK_INDEX(d == 0 || d == -1, "Dimension out of range for a scalar tensor");
                d = 0;
            } else d = (d % x.dim() + x.dim()) % x.dim(); // Python reference wraps dims.
        }
        std::sort(dims.begin(), dims.end());
        TORCH_CHECK(std::adjacent_find(dims.begin(), dims.end()) == dims.end(), "dim appears multiple times in the list of dims");
        if (dims.empty() || x.dim() == 0) return x.clone();
        Dims perm, shape;
        int64_t rows = 1, count = 1;
        for (int64_t d = 0; d < x.dim(); ++d) {
            bool reduced = std::binary_search(dims.begin(), dims.end(), d);
            if (!reduced) { perm.push_back(d); rows *= x.size(d); shape.push_back(x.size(d)); }
            else { count *= x.size(d); if (keepdim) shape.push_back(1); }
        }
        perm.insert(perm.end(), dims.begin(), dims.end());
        auto src = x.permute(perm).contiguous();
        if (arithmetic.use_batch_add()) return tree_tensor(src.reshape({rows,count})).reshape(shape);
        auto out = at::empty(shape, x.options());
        const S* input = src.template data_ptr<S>(); S* output = out.template data_ptr<S>();
        int64_t chunks = (count + reduction_chunk - 1) / reduction_chunk;
        auto partial = at::empty({rows, chunks}, x.options());
        S* p = partial.template data_ptr<S>();
        parallel_outputs(0, rows*chunks, 1, [&](int64_t begin, int64_t end) {
            for (auto i = begin; i < end; ++i) {
                int64_t row = i/chunks, offset = (i%chunks)*reduction_chunk;
                p[i] = reduce_values(std::min(reduction_chunk, count-offset),
                    [&](int64_t j) { return input[row*count+offset+j]; });
            }
        });
        parallel_outputs(0, rows, 1, [&](int64_t begin, int64_t end) {
            for (auto i = begin; i < end; ++i)
                output[i] = reduce_values(chunks, [&](int64_t j) { return p[i*chunks+j]; });
        });
        return out;
    }
    Tensor matmul(const Tensor& a, const Tensor& b) const override {
        check(a); check(b);
        TORCH_CHECK(a.dim() > 0 && b.dim() > 0, "matmul expects tensors with at least one dimension");
        bool av = a.dim() == 1, bv = b.dim() == 1;
        auto aa = av ? a.unsqueeze(0) : a;
        auto bb = bv ? b.unsqueeze(1) : b;
        int64_t m = aa.size(-2), k = aa.size(-1), n = bb.size(-1);
        TORCH_CHECK(k == bb.size(-2), "matmul: size mismatch");
        auto batch = at::infer_size(aa.sizes().slice(0, aa.dim()-2), bb.sizes().slice(0, bb.dim()-2));
        Dims ashape(batch.begin(), batch.end()), bshape = ashape, oshape = ashape;
        ashape.insert(ashape.end(), {m,k}); bshape.insert(bshape.end(), {k,n}); oshape.insert(oshape.end(), {m,n});
        // Expand is a view: retain batch strides instead of replicating broadcast data.
        aa = aa.contiguous().expand(ashape); bb = bb.contiguous().expand(bshape);
        if (arithmetic.use_batch_add()) {
            std::vector<Tensor> partials;
            for (int64_t offset=0; offset<k; offset+=reduction_chunk) {
                int64_t end=std::min(k,offset+reduction_chunk);
                auto products = binary(aa.slice(-1,offset,end).unsqueeze(-1),
                                       bb.slice(-2,offset,end).unsqueeze(-3), "mul");
                partials.push_back(pairwise_tensor(products.transpose(-1,-2)));
            }
            auto result = partials.empty() ? at::full(oshape,Scalar::zero,a.options()) :
                          pairwise_tensor(at::stack(partials,-1));
            if (av) result=result.squeeze(-2);
            if (bv) result=result.squeeze(-1);
            return result;
        }
        auto out = at::empty(oshape, a.options());
        auto ap = aa.template data_ptr<S>(), bp = bb.template data_ptr<S>(); auto op = out.template data_ptr<S>();
        int64_t batches = 1; for (auto s : batch) batches *= s;
        constexpr int64_t tile = 16;
        int64_t mt = m/tile + (m%tile != 0), nt = n/tile + (n%tile != 0);
        parallel_outputs(0, batches*mt*nt, 1, [&](int64_t begin, int64_t end) {
            for (auto work = begin; work < end; ++work) {
                int64_t batch_id = work/(mt*nt), rem = batch_id, ao = 0, bo = 0;
                for (int64_t d = int64_t(batch.size())-1; d >= 0; --d) {
                    int64_t idx = rem % batch[d]; rem /= batch[d];
                    ao += idx*aa.stride(d); bo += idx*bb.stride(d);
                }
                int64_t row0 = ((work/nt)%mt)*tile, col0 = (work%nt)*tile;
                for (int64_t i=row0; i<row0+std::min(tile,m-row0); ++i)
                    for (int64_t j=col0; j<col0+std::min(tile,n-col0); ++j)
                        op[(batch_id*m+i)*n+j] = reduce_values(k, [&](int64_t q) {
                            return arithmetic.scalar.mul(ap[ao+i*k+q], bp[bo+q*n+j]);
                        });
            }
        });
        if (av) out = out.squeeze(-2);
        if (bv) out = out.squeeze(-1);
        return out;
    }
    std::vector<Tensor> matmul_backward(const Tensor& grad, const Tensor& a, const Tensor& b) const override {
        check(grad); check(a); check(b);
        TORCH_CHECK(a.dim() > 0 && b.dim() > 0, "matmul expects non-scalar operands");
        bool av = a.dim() == 1, bv = b.dim() == 1;
        auto aa = av ? a.unsqueeze(0) : a;
        auto bb = bv ? b.unsqueeze(1) : b;
        TORCH_CHECK(aa.size(-1) == bb.size(-2), "matmul: size mismatch");
        auto batch = at::infer_size(aa.sizes().slice(0, aa.dim()-2), bb.sizes().slice(0, bb.dim()-2));
        Dims expected(batch.begin(), batch.end());
        if (!av) expected.push_back(aa.size(-2));
        if (!bv) expected.push_back(bb.size(-1));
        TORCH_CHECK(grad.sizes().vec() == expected, "Incorrect matmul gradient shape");
        auto gg = grad;
        if (bv) gg = gg.unsqueeze(-1);
        if (av) gg = gg.unsqueeze(-2);
        auto ga = reduce_like(matmul(gg, bb.transpose(-2,-1)), aa);
        auto gb = reduce_like(matmul(aa.transpose(-2,-1), gg), bb);
        return {ga.reshape(a.sizes()), gb.reshape(b.sizes())};
    }
    Tensor conv2d(const Tensor& x, const Tensor& w, const OptionalTensor& bias,
                  const Dims& stride, const Dims& padding, const Dims& dilation, int64_t groups) const override {
        auto c = prepare_conv(x,w,stride,padding,dilation,groups);
        Tensor bc;
        if (bias) { check(*bias); TORCH_CHECK(bias->dim()==1 && bias->size(0)==c.co, "Incorrect bias shape"); bc = bias->contiguous(); }
        auto out = at::empty({c.n,c.co,c.ho,c.wo}, x.options());
        auto xp=c.x.template data_ptr<S>(), wp=c.w.template data_ptr<S>(); auto yp=out.template data_ptr<S>();
        const S* bp = bias ? bc.template data_ptr<S>() : nullptr;
        parallel_outputs(0, out.numel(), 16, [&](int64_t begin, int64_t end) {
            for (int64_t idx=begin; idx<end; ++idx) {
                int64_t ow=idx%c.wo, oh=(idx/c.wo)%c.ho, oc=(idx/(c.wo*c.ho))%c.co;
                int64_t batch=idx/(c.wo*c.ho*c.co), group=oc/c.og;
                S result = reduce_values(c.cg*c.kh*c.kw, [&](int64_t q) {
                    int64_t kw=q%c.kw, kh=(q/c.kw)%c.kh, ic=group*c.cg+q/(c.kh*c.kw);
                    int64_t ih=oh*c.sh-c.ph+kh*c.dh, iw=ow*c.sw-c.pw+kw*c.dw;
                    S input = ih>=0 && ih<c.h && iw>=0 && iw<c.wi ? xp[((batch*c.ci+ic)*c.h+ih)*c.wi+iw] : Scalar::zero;
                    return arithmetic.scalar.mul(input, wp[oc*c.cg*c.kh*c.kw+q]);
                });
                if (bp && arithmetic.use_batch_add()) {
                    yp[idx] = binary(at::scalar_tensor(result,x.options()),
                                     at::scalar_tensor(bp[oc],x.options()), "add").template item<S>();
                } else yp[idx] = bp ? arithmetic.scalar.add(result,bp[oc]) : result;
            }
        });
        return c.unbatched ? out.squeeze(0) : out;
    }
    std::vector<Tensor> conv2d_backward(const Tensor& grad, const Tensor& x, const Tensor& w,
                   const Dims& stride, const Dims& padding, const Dims& dilation, bool has_bias, int64_t groups) const override {
        auto c=prepare_conv(x,w,stride,padding,dilation,groups); check(grad);
        auto g=(c.unbatched ? grad.unsqueeze(0) : grad).contiguous();
        TORCH_CHECK(g.sizes().vec() == Dims({c.n,c.co,c.ho,c.wo}), "Incorrect convolution gradient shape");
        auto gx=at::empty(c.x.sizes(), x.options()), gw=at::empty(w.sizes(), w.options());
        auto gb=at::empty({has_bias ? c.co : 0}, x.options());
        auto xp=c.x.template data_ptr<S>(), wp=c.w.template data_ptr<S>(), gp=g.template data_ptr<S>();
        auto dx=gx.template data_ptr<S>(), dw=gw.template data_ptr<S>(), db=gb.template data_ptr<S>();
        // Gather into independently owned outputs: no atomics or per-thread full gradients.
        parallel_outputs(0,gx.numel(),16,[&](int64_t begin,int64_t end) {
            for (auto idx=begin;idx<end;++idx) {
                int64_t iw=idx%c.wi, ih=(idx/c.wi)%c.h, ic=(idx/(c.wi*c.h))%c.ci;
                int64_t batch=idx/(c.wi*c.h*c.ci), group=ic/c.cg;
                dx[idx]=reduce_values(c.og*c.kh*c.kw,[&](int64_t q) {
                    int64_t kw=q%c.kw, kh=(q/c.kw)%c.kh, oc=group*c.og+q/(c.kh*c.kw);
                    int64_t oh=ih+c.ph-kh*c.dh, ow=iw+c.pw-kw*c.dw;
                    if (oh<0 || ow<0 || oh%c.sh || ow%c.sw) return Scalar::zero;
                    oh/=c.sh; ow/=c.sw;
                    if (oh>=c.ho || ow>=c.wo) return Scalar::zero;
                    return arithmetic.scalar.mul(gp[((batch*c.co+oc)*c.ho+oh)*c.wo+ow],
                                      wp[((oc*c.cg+ic%c.cg)*c.kh+kh)*c.kw+kw]);
                });
            }
        });
        parallel_outputs(0,gw.numel(),16,[&](int64_t begin,int64_t end) {
            for (auto idx=begin;idx<end;++idx) {
                int64_t kw=idx%c.kw, kh=(idx/c.kw)%c.kh, local=(idx/(c.kw*c.kh))%c.cg;
                int64_t oc=idx/(c.kw*c.kh*c.cg), ic=(oc/c.og)*c.cg+local;
                dw[idx]=reduce_values(c.n*c.ho*c.wo,[&](int64_t q) {
                    int64_t ow=q%c.wo, oh=(q/c.wo)%c.ho, batch=q/(c.wo*c.ho);
                    int64_t ih=oh*c.sh-c.ph+kh*c.dh, iw=ow*c.sw-c.pw+kw*c.dw;
                    if (ih<0 || ih>=c.h || iw<0 || iw>=c.wi) return Scalar::zero;
                    return arithmetic.scalar.mul(gp[((batch*c.co+oc)*c.ho+oh)*c.wo+ow],
                                      xp[((batch*c.ci+ic)*c.h+ih)*c.wi+iw]);
                });
            }
        });
        if (has_bias) parallel_outputs(0,c.co,1,[&](int64_t begin,int64_t end) {
            for (auto oc=begin;oc<end;++oc)
                db[oc]=reduce_values(c.n*c.ho*c.wo,[&](int64_t q) {
                    int64_t batch=q/(c.ho*c.wo), pixel=q%(c.ho*c.wo);
                    return gp[(batch*c.co+oc)*c.ho*c.wo+pixel];
                });
        });
        return {c.unbatched ? gx.squeeze(0) : gx, gw, gb};
    }
};

} // namespace torchdt::native
