#pragma once
#include <torchdt/registry.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <numeric>
#include <torchdt/cuda/reduce.cuh>

namespace torchdt::native {
// Arithmetic supplies an immutable scalar snapshot for this device and stream.
// All per-element calls are statically dispatched. Tensor code is datatype-free.
template<class Arithmetic> class CUDAKernels final : public TensorKernels {
    using Scalar = typename Arithmetic::Scalar;
    using S = typename Scalar::S;
    static constexpr auto storage = Arithmetic::storage;
    const Arithmetic arithmetic;
    void check(const Tensor& x) const {
        TORCH_CHECK(x.is_cuda() && x.layout()==at::kStrided, "Native CUDA requires strided CUDA tensors");
        TORCH_CHECK(x.scalar_type()==storage, "Incorrect native storage dtype");
    }
    template<class In, class Out, class F>
    Tensor unary_kernel(const Tensor& x, at::ScalarType type, F f) const {
        auto out=at::empty({0},x.options().dtype(type));
        auto iter=at::TensorIteratorConfig().check_all_same_dtype(false)
            .add_output(out).add_const_input(x).build();
        at::native::gpu_kernel(iter, f);
        return out;
    }
    template<class Out, class F>
    Tensor binary_kernel(const Tensor& x,const Tensor& y,at::ScalarType type,F f) const {
        auto out=at::empty({0},x.options().dtype(type));
        auto iter=at::TensorIteratorConfig().check_all_same_dtype(false)
            .add_output(out).add_const_input(x).add_const_input(y).build();
        at::native::gpu_kernel(iter,f);
        return out;
    }
    Tensor reduce_like(const Tensor& x, const Tensor& target) const {
        Dims dims;
        int64_t extra = x.dim() - target.dim();
        for (int64_t i = 0; i < x.dim(); ++i)
            if (i < extra || (target.size(i-extra) == 1 && x.size(i) != 1)) dims.push_back(i);
        return (dims.empty() ? x : sum(x, dims, true)).reshape(target.sizes());
    }

public:
    struct Batch { int rank; int64_t sizes[64], as[64], bs[64]; };
    struct Conv {
        bool unbatched;
        int64_t n, ci, h, wi, co, kh, kw, cg, og, ho, wo, sh, sw, ph, pw, dh, dw;
    };
private:
    Conv prepare_conv(const Tensor& x, const Tensor& w, const Dims& stride,
                      const Dims& padding, const Dims& dilation, int64_t groups) const {
        check(x); check(w);
        TORCH_CHECK(x.device()==w.device(),"Native operands must be on the same device");
        TORCH_CHECK((x.dim() == 3 || x.dim() == 4) && w.dim() == 4,
                    "conv2d requires a 3D/4D input and 4D weight");
        TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
                    "stride, padding and dilation require two entries");
        TORCH_CHECK(groups > 0 && stride[0] > 0 && stride[1] > 0 &&
                    dilation[0] > 0 && dilation[1] > 0 && padding[0] >= 0 && padding[1] >= 0,
                    "Invalid convolution parameters");
        Conv c;
        c.unbatched = x.dim() == 3;
        auto xx = c.unbatched ? x.unsqueeze(0) : x;
        c.n = xx.size(0); c.ci = xx.size(1); c.h = xx.size(2); c.wi = xx.size(3);
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
    explicit CUDAKernels(Arithmetic policy): arithmetic(std::move(policy)) {}
    std::vector<std::string> capabilities() const override {
        return {"from_float","to_float","neg","abs","sign","sqrt","pow",
                "add","sub","mul","div","ge","gt","le","lt","sum","matmul","matmul_backward","conv2d","conv2d_backward"};
    }
    Tensor unary(const Tensor& x,const std::string& op) const override {
        TORCH_CHECK(x.is_cuda() && x.layout()==at::kStrided,"Native CUDA requires strided CUDA tensors");
        c10::cuda::CUDAGuard guard(x.device());
        auto s=arithmetic.prepare(x);
        if(op=="from_float") {
            auto input=x.to(at::kDouble);
            arithmetic.validate_conversion(input);
            return unary_kernel<double,S>(input,storage,[s] GPU_LAMBDA(double a)->S {return s.from_float(a);});
        }
        check(x);
        if(op=="to_float") return unary_kernel<S,double>(x,at::kDouble,[s] GPU_LAMBDA(S a)->double {return s.to_float(a);});
#define UNARY(NAME) if(op==#NAME) return unary_kernel<S,S>(x,storage,[s] GPU_LAMBDA(S a)->S {return s.NAME(a);});
        UNARY(neg) UNARY(abs) UNARY(sign) UNARY(sqrt)
#undef UNARY
        TORCH_CHECK(false,"Unknown native unary operation: ",op);
    }
    Tensor binary(const Tensor& x,const Tensor& y,const std::string& op) const override {
        check(x); check(y);
        TORCH_CHECK(x.device()==y.device(),"Native operands must be on the same device");
        c10::cuda::CUDAGuard guard(x.device());
        auto s=arithmetic.prepare(x);
#define BINARY(NAME) if(op==#NAME) return binary_kernel<S>(x,y,storage,[s] GPU_LAMBDA(S a,S b)->S {return s.NAME(a,b);});
        BINARY(add) BINARY(sub) BINARY(mul) BINARY(div) BINARY(pow)
#undef BINARY
#define COMPARE(NAME) if(op==#NAME) return binary_kernel<bool>(x,y,at::kBool,[s] GPU_LAMBDA(S a,S b)->bool {return s.NAME(a,b);});
        COMPARE(ge) COMPARE(gt) COMPARE(le) COMPARE(lt)
#undef COMPARE
        TORCH_CHECK(false,"Unknown native binary operation: ",op);
    }
    Tensor sum(const Tensor& x, c10::optional<Dims> dimensions, bool keepdim) const override {
        check(x);
        c10::cuda::CUDAGuard guard(x.device());
        auto scalar=arithmetic.prepare(x);
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
        int64_t count = 1;
        for (int64_t d = 0; d < x.dim(); ++d) {
            bool reduced = std::binary_search(dims.begin(), dims.end(), d);
            if (!reduced) { perm.push_back(d); shape.push_back(x.size(d)); }
            else { count *= x.size(d); if (keepdim) shape.push_back(1); }
        }
        perm.insert(perm.end(), dims.begin(), dims.end());
        auto src = x.permute(perm).contiguous();
        auto out=at::empty(shape,x.options());
        auto input=src.template data_ptr<S>();
        if (count <= 256) {
            launch_reduce(out,count,scalar,[=] __device__(int64_t row,int64_t q)->S {return input[row*count+q];});
        } else {
            // Independent chunk blocks expose parallelism even for a scalar
            // output. Scratch holds only one input-width value per 256 leaves.
            int64_t chunks=count/256+(count%256!=0);
            auto partial=at::empty({out.numel(),chunks},x.options());
            launch_reduce(partial,256,scalar,[=] __device__(int64_t i,int64_t q)->S {
                auto row=i/chunks, offset=(i%chunks)*256+q;
                return offset<count ? input[row*count+offset] : Scalar::zero;
            });
            auto values=partial.template data_ptr<S>();
            launch_reduce(out,chunks,scalar,[=] __device__(int64_t row,int64_t q)->S {return values[row*chunks+q];});
        }
        return out;
    }
    Tensor matmul(const Tensor& a, const Tensor& b) const override {
        check(a); check(b);
        TORCH_CHECK(a.device()==b.device(),"Native operands must be on the same device");
        c10::cuda::CUDAGuard guard(a.device());
        auto scalar=arithmetic.prepare(a);
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
        aa = aa.expand(ashape); bb = bb.expand(bshape);
        auto ams=aa.stride(-2),aks=aa.stride(-1),bks=bb.stride(-2),bns=bb.stride(-1);
        // Pass fixed-size stride metadata by value; broadcast operands remain views.
        Batch meta{};
        TORCH_CHECK(batch.size()<=64,"Too many matmul batch dimensions");
        meta.rank=batch.size();
        for(int d=0;d<meta.rank;++d) {meta.sizes[d]=batch[d];meta.as[d]=aa.stride(d);meta.bs[d]=bb.stride(d);}
        auto out=at::empty(oshape,a.options());
        auto ap=aa.template data_ptr<S>(),bp=bb.template data_ptr<S>();
        launch_reduce(out,k,scalar,[=] __device__(int64_t i,int64_t q)->S {
            int64_t row=(i/n)%m,col=i%n,rem=i/(m*n),ao=0,bo=0;
            for(int d=meta.rank-1;d>=0;--d) {
                auto idx=rem%meta.sizes[d];rem/=meta.sizes[d];
                ao+=idx*meta.as[d];bo+=idx*meta.bs[d];
            }
            return scalar.mul(ap[ao+row*ams+q*aks],bp[bo+q*bks+col*bns]);
        });
        if(av) out=out.squeeze(-2);
        if(bv) out=out.squeeze(-1);
        return out;
    }
    std::vector<Tensor> matmul_backward(const Tensor& grad, const Tensor& a, const Tensor& b) const override {
        check(grad); check(a); check(b);
        TORCH_CHECK(grad.device()==a.device() && a.device()==b.device(),"Native operands must be on the same device");
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

    Tensor conv2d(const Tensor& x,const Tensor& w,const OptionalTensor& bias,
                  const Dims& stride,const Dims& padding,const Dims& dilation,int64_t groups) const override {
        check(x);
        c10::cuda::CUDAGuard guard(x.device());
        auto c=prepare_conv(x,w,stride,padding,dilation,groups);
        auto scalar=arithmetic.prepare(x);
        auto xx=(c.unbatched ? x.unsqueeze(0) : x).contiguous(),ww=w.contiguous();
        Tensor bc;
        if(bias) {
            check(*bias);
            TORCH_CHECK(bias->device()==x.device(),"Native operands must be on the same device");
            TORCH_CHECK(bias->dim()==1 && bias->size(0)==c.co,"Incorrect bias shape");
            bc=bias->contiguous();
        }
        auto out=at::empty({c.n,c.co,c.ho,c.wo},x.options());
        auto xp=xx.template data_ptr<S>(),wp=ww.template data_ptr<S>();
        const S* bp=bias ? bc.template data_ptr<S>() : nullptr;
        launch_reduce(out,c.cg*c.kh*c.kw,scalar,[=] __device__(int64_t idx,int64_t q)->S {
            int64_t ow=idx%c.wo,oh=(idx/c.wo)%c.ho,oc=(idx/(c.wo*c.ho))%c.co;
            int64_t batch=idx/(c.wo*c.ho*c.co),group=oc/c.og;
            int64_t kw=q%c.kw,kh=(q/c.kw)%c.kh,ic=group*c.cg+q/(c.kh*c.kw);
            int64_t ih=oh*c.sh-c.ph+kh*c.dh,iw=ow*c.sw-c.pw+kw*c.dw;
            S input=ih>=0 && ih<c.h && iw>=0 && iw<c.wi ? xp[((batch*c.ci+ic)*c.h+ih)*c.wi+iw] : Scalar::zero;
            return scalar.mul(input,wp[oc*c.cg*c.kh*c.kw+q]);
        },[=] __device__(int64_t idx,S result)->S {
            auto oc=(idx/(c.wo*c.ho))%c.co;
            return bp ? scalar.add(result,bp[oc]) : result;
        });
        return c.unbatched ? out.squeeze(0) : out;
    }
    std::vector<Tensor> conv2d_backward(const Tensor& grad,const Tensor& x,const Tensor& w,
                   const Dims& stride,const Dims& padding,const Dims& dilation,bool has_bias,int64_t groups) const override {
        check(x); check(grad);
        TORCH_CHECK(grad.device()==x.device(),"Native operands must be on the same device");
        c10::cuda::CUDAGuard guard(x.device());
        auto c=prepare_conv(x,w,stride,padding,dilation,groups);
        auto scalar=arithmetic.prepare(x);
        auto xx=(c.unbatched ? x.unsqueeze(0) : x).contiguous(),ww=w.contiguous();
        auto g=(c.unbatched ? grad.unsqueeze(0) : grad).contiguous();
        TORCH_CHECK(g.sizes().vec()==Dims({c.n,c.co,c.ho,c.wo}),"Incorrect convolution gradient shape");
        auto gx=at::empty(xx.sizes(),x.options()),gw=at::empty(w.sizes(),w.options());
        auto gb=at::empty({has_bias ? c.co : 0},x.options());
        auto xp=xx.template data_ptr<S>(),wp=ww.template data_ptr<S>(),gp=g.template data_ptr<S>();
        // Gather each gradient independently, with no floating or integer atomics.
        launch_reduce(gx,c.og*c.kh*c.kw,scalar,[=] __device__(int64_t idx,int64_t q)->S {
            int64_t iw=idx%c.wi,ih=(idx/c.wi)%c.h,ic=(idx/(c.wi*c.h))%c.ci;
            int64_t batch=idx/(c.wi*c.h*c.ci),group=ic/c.cg;
            int64_t kw=q%c.kw,kh=(q/c.kw)%c.kh,oc=group*c.og+q/(c.kh*c.kw);
            int64_t oh=ih+c.ph-kh*c.dh,ow=iw+c.pw-kw*c.dw;
            if(oh<0 || ow<0 || oh%c.sh || ow%c.sw) return Scalar::zero;
            oh/=c.sh; ow/=c.sw;
            if(oh>=c.ho || ow>=c.wo) return Scalar::zero;
            return scalar.mul(gp[((batch*c.co+oc)*c.ho+oh)*c.wo+ow],wp[((oc*c.cg+ic%c.cg)*c.kh+kh)*c.kw+kw]);
        });
        launch_reduce(gw,c.n*c.ho*c.wo,scalar,[=] __device__(int64_t idx,int64_t q)->S {
            int64_t kw=idx%c.kw,kh=(idx/c.kw)%c.kh,local=(idx/(c.kw*c.kh))%c.cg;
            int64_t oc=idx/(c.kw*c.kh*c.cg),ic=(oc/c.og)*c.cg+local;
            int64_t ow=q%c.wo,oh=(q/c.wo)%c.ho,batch=q/(c.wo*c.ho);
            int64_t ih=oh*c.sh-c.ph+kh*c.dh,iw=ow*c.sw-c.pw+kw*c.dw;
            if(ih<0 || ih>=c.h || iw<0 || iw>=c.wi) return Scalar::zero;
            return scalar.mul(gp[((batch*c.co+oc)*c.ho+oh)*c.wo+ow],xp[((batch*c.ci+ic)*c.h+ih)*c.wi+iw]);
        });
        if(has_bias) launch_reduce(gb,c.n*c.ho*c.wo,scalar,[=] __device__(int64_t oc,int64_t q)->S {
            int64_t batch=q/(c.ho*c.wo),pixel=q%(c.ho*c.wo);
            return gp[(batch*c.co+oc)*c.ho*c.wo+pixel];
        });
        return {c.unbatched ? gx.squeeze(0) : gx,gw,gb};
    }

};
}
