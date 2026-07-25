#include <torch/extension.h>
#include <torchdt/registry.h>

#include "dispatcher.h"
#include "lns16.h"

class BackendHandle {
public:
    BackendHandle(const std::string& name, size_t bitwidth)
        : ops_(get_ops_impl(name, bitwidth)) {
        if (!ops_) {
            throw std::runtime_error(
                "No ops registered for dtype " + name + " with bitwidth " +
                std::to_string(bitwidth)
            );
        }
    }

    torch::Tensor from_float(const torch::Tensor& x) const { return ops_->from_float(x); }
    torch::Tensor to_float(const torch::Tensor& x) const { return ops_->to_float(x); }
    torch::Tensor add(const torch::Tensor& x, const torch::Tensor& y) const { return ops_->add(x, y); }
    torch::Tensor sub(const torch::Tensor& x, const torch::Tensor& y) const { return ops_->sub(x, y); }
    torch::Tensor mul(const torch::Tensor& x, const torch::Tensor& y) const { return ops_->mul(x, y); }
    torch::Tensor div(const torch::Tensor& x, const torch::Tensor& y) const { return ops_->div(x, y); }
    torch::Tensor ge(const torch::Tensor& x, const torch::Tensor& y) const { return ops_->ge(x, y); }
    torch::Tensor gt(const torch::Tensor& x, const torch::Tensor& y) const { return ops_->gt(x, y); }
    torch::Tensor le(const torch::Tensor& x, const torch::Tensor& y) const { return ops_->le(x, y); }
    torch::Tensor lt(const torch::Tensor& x, const torch::Tensor& y) const { return ops_->lt(x, y); }
    torch::Tensor sum(
        const torch::Tensor& x,
        c10::optional<std::vector<int64_t>> dim,
        bool keepdim
    ) const { return ops_->sum(x, std::move(dim), keepdim); }
    torch::Tensor matmul(const torch::Tensor& a, const torch::Tensor& b) const {
        return ops_->matmul(a, b);
    }
    std::vector<torch::Tensor> matmul_backward(
        const torch::Tensor& grad_out,
        const torch::Tensor& a,
        const torch::Tensor& b
    ) const { return ops_->matmul_backward(grad_out, a, b); }
    torch::Tensor conv2d(
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const c10::optional<torch::Tensor>& bias,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation,
        int64_t groups
    ) const { return ops_->conv2d(input, weight, bias, stride, padding, dilation, groups); }
    std::vector<torch::Tensor> conv2d_backward(
        const torch::Tensor& grad_out,
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& dilation,
        bool has_bias,
        int64_t groups
    ) const {
        return ops_->conv2d_backward(
            grad_out, input, weight, stride, padding, dilation, has_bias, groups
        );
    }

private:
    OpsBase* ops_;
};

torch::Tensor dispatch_from_float(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->from_float(x);
}

torch::Tensor dispatch_to_float(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->to_float(x);
}

torch::Tensor dispatch_add(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->add(x, y);
}

torch::Tensor dispatch_sub(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->sub(x, y);
}

torch::Tensor dispatch_mul(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->mul(x, y);
}

torch::Tensor dispatch_div(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->div(x, y);
}

torch::Tensor dispatch_ge(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->ge(x, y);
}

torch::Tensor dispatch_gt(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->gt(x, y);
}

torch::Tensor dispatch_le(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->le(x, y);
}

torch::Tensor dispatch_lt(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->lt(x, y);
}

torch::Tensor dispatch_sum(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    c10::optional<std::vector<int64_t>> dim,
    bool keepdim
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->sum(x, dim, keepdim);
}

torch::Tensor dispatch_matmul(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->matmul(A, B);
}

std::vector<torch::Tensor> dispatch_matmul_backward(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& grad_out,
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->matmul_backward(grad_out, A, B);
}

torch::Tensor dispatch_conv2d(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->conv2d(
        input, weight, bias, stride, padding, dilation, groups
    );
}

std::vector<torch::Tensor> dispatch_conv2d_backward(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& grad_out,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    bool has_bias,
    int64_t groups
) {
    OpsBase* ops = get_ops_impl(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered for dtype " + dtype_name + " with bitwidth " + std::to_string(bitwidth));
    return ops->conv2d_backward(
        grad_out, input, weight, stride, padding, dilation, has_bias, groups
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BackendHandle>(m, "BackendHandle")
        .def("from_float", &BackendHandle::from_float, py::call_guard<py::gil_scoped_release>())
        .def("to_float", &BackendHandle::to_float, py::call_guard<py::gil_scoped_release>())
        .def("add", &BackendHandle::add, py::call_guard<py::gil_scoped_release>())
        .def("sub", &BackendHandle::sub, py::call_guard<py::gil_scoped_release>())
        .def("mul", &BackendHandle::mul, py::call_guard<py::gil_scoped_release>())
        .def("div", &BackendHandle::div, py::call_guard<py::gil_scoped_release>())
        .def("ge", &BackendHandle::ge, py::call_guard<py::gil_scoped_release>())
        .def("gt", &BackendHandle::gt, py::call_guard<py::gil_scoped_release>())
        .def("le", &BackendHandle::le, py::call_guard<py::gil_scoped_release>())
        .def("lt", &BackendHandle::lt, py::call_guard<py::gil_scoped_release>())
        .def("sum", &BackendHandle::sum, py::arg("x"), py::arg("dim") = c10::nullopt,
             py::arg("keepdim") = false, py::call_guard<py::gil_scoped_release>())
        .def("matmul", &BackendHandle::matmul, py::call_guard<py::gil_scoped_release>())
        .def("matmul_backward", &BackendHandle::matmul_backward,
             py::call_guard<py::gil_scoped_release>())
        .def("conv2d", &BackendHandle::conv2d, py::call_guard<py::gil_scoped_release>())
        .def("conv2d_backward", &BackendHandle::conv2d_backward,
             py::call_guard<py::gil_scoped_release>());
    m.def("get_backend", [](const std::string& name, size_t bitwidth) {
        return BackendHandle(name, bitwidth);
    });
    m.def(
        "from_float", &dispatch_from_float,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"),
        "Convert from float to custom dtype"
    );
    m.def(
        "to_float", &dispatch_to_float,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"),
        "Convert from custom dtype to float"
    );
    m.def(
        "add", &dispatch_add,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"), py::arg("y"),
        "Addition for custom dtypes"
    );
    m.def(
        "sub", &dispatch_sub,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"), py::arg("y"),
        "Subtraction for custom dtypes"
    );
    m.def(
        "mul", &dispatch_mul,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"), py::arg("y"),
        "Multiplication for custom dtypes"
    );
    m.def(
        "div", &dispatch_div,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"), py::arg("y"),
        "Division for custom dtypes"
    );
    m.def(
        "ge", &dispatch_ge,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"), py::arg("y"),
        "Greater-than-or-equal comparison for custom dtypes"
    );
    m.def(
        "gt", &dispatch_gt,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"), py::arg("y"),
        "Greater-than comparison for custom dtypes"
    );
    m.def(
        "le", &dispatch_le,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"), py::arg("y"),
        "Less-than-or-equal comparison for custom dtypes"
    );
    m.def(
        "lt", &dispatch_lt,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"), py::arg("y"),
        "Less-than comparison for custom dtypes"
    );
    m.def(
        "sum", &dispatch_sum,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"),
        py::arg("dim") = c10::nullopt, py::arg("keepdim") = false,
        "Summation for custom dtypes"
    );
    m.def(
        "matmul", &dispatch_matmul,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("A"), py::arg("B"),
        "Matrix multiplication for custom dtypes"
    );
    m.def(
        "matmul_backward", &dispatch_matmul_backward,
        py::arg("dtype_name"), py::arg("bitwidth"),
        py::arg("grad_out"), py::arg("A"), py::arg("B"),
        "Matrix multiplication backward for custom dtypes"
    );
    m.def(
        "conv2d", &dispatch_conv2d,
        py::arg("dtype_name"), py::arg("bitwidth"),
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("stride"), py::arg("padding"), py::arg("dilation"),
        py::arg("groups"),
        "2D convolution forward for custom dtypes"
    );
    m.def(
        "conv2d_backward", &dispatch_conv2d_backward,
        py::arg("dtype_name"), py::arg("bitwidth"),
        py::arg("grad_out"), py::arg("input"), py::arg("weight"),
        py::arg("stride"), py::arg("padding"), py::arg("dilation"),
        py::arg("has_bias"), py::arg("groups"),
        "2D convolution backward for custom dtypes"
    );
}

REGISTER_DTYPE("lns", 16, get_lns16_ops());
