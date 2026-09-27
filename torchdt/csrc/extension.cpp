#include <torch/extension.h>
#include <torch/library.h>
#include "context.h"
namespace torchdt::native { void register_lns_cpu(); }
namespace n = torchdt::native;

TORCH_LIBRARY(torchdt_native, m) {
    n::register_lns_cpu();
    m.class_<n::Context>("Context")
        .def(torch::init<std::string, int64_t, c10::Dict<std::string, c10::IValue>>())
        .def("capabilities", &n::Context::capabilities);
    m.def("unary(Tensor x, __torch__.torch.classes.torchdt_native.Context context, str op) -> Tensor");
    m.def("binary(Tensor x, Tensor y, __torch__.torch.classes.torchdt_native.Context context, str op) -> Tensor");
    m.def("sum(Tensor x, __torch__.torch.classes.torchdt_native.Context context, int[]? dim=None, bool keepdim=False) -> Tensor");
    m.def("matmul(Tensor a, Tensor b, __torch__.torch.classes.torchdt_native.Context context) -> Tensor");
    m.def("matmul_backward(Tensor grad, Tensor a, Tensor b, __torch__.torch.classes.torchdt_native.Context context) -> Tensor[]");
    m.def("conv2d(Tensor x, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, __torch__.torch.classes.torchdt_native.Context context) -> Tensor");
    m.def("conv2d_backward(Tensor grad, Tensor x, Tensor weight, int[] stride, int[] padding, int[] dilation, bool has_bias, int groups, __torch__.torch.classes.torchdt_native.Context context) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torchdt_native, CPU, m) {
    m.impl("unary", [](const at::Tensor& x, const n::ContextPtr& c, const std::string& op) { return c->kernels->unary(x, op); });
    m.impl("binary", [](const at::Tensor& x, const at::Tensor& y, const n::ContextPtr& c, const std::string& op) { return c->kernels->binary(x, y, op); });
    m.impl("sum", [](const at::Tensor& x, const n::ContextPtr& c, c10::optional<n::Dims> dim, bool keepdim) { return c->kernels->sum(x, dim, keepdim); });
    m.impl("matmul", [](const at::Tensor& a, const at::Tensor& b, const n::ContextPtr& c) { return c->kernels->matmul(a, b); });
    m.impl("matmul_backward", [](const at::Tensor& g, const at::Tensor& a, const at::Tensor& b, const n::ContextPtr& c) { return c->kernels->matmul_backward(g, a, b); });
    m.impl("conv2d", [](const at::Tensor& x, const at::Tensor& w, const n::OptionalTensor& b, n::Dims s, n::Dims p, n::Dims d, int64_t groups, const n::ContextPtr& c) { return c->kernels->conv2d(x, w, b, s, p, d, groups); });
    m.impl("conv2d_backward", [](const at::Tensor& g, const at::Tensor& x, const at::Tensor& w, n::Dims s, n::Dims p, n::Dims d, bool bias, int64_t groups, const n::ContextPtr& c) { return c->kernels->conv2d_backward(g, x, w, s, p, d, bias, groups); });
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("has_cuda_kernels") = false;
#if defined(_OPENMP) || AT_PARALLEL_NATIVE
    m.attr("cpu_parallel_enabled") = true;
#else
    m.attr("cpu_parallel_enabled") = false;
#endif
}
