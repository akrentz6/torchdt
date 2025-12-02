#include <torch/extension.h>
#include <torchdt/registry.h>

#include "ops_kernels.h"
#include "lns16.h"

torch::Tensor dispatch_from_float(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x
) {
    OpsBase* ops = Registry::instance().get_ops_base(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered");
    return ops->from_float(x);
}

torch::Tensor dispatch_to_float(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x
) {
    OpsBase* ops = Registry::instance().get_ops_base(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered");
    return ops->to_float(x);
}

torch::Tensor dispatch_add(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = Registry::instance().get_ops_base(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered");
    return ops->add(x, y);
}

torch::Tensor dispatch_sub(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = Registry::instance().get_ops_base(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered");
    return ops->sub(x, y);
}

torch::Tensor dispatch_mul(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = Registry::instance().get_ops_base(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered");
    return ops->mul(x, y);
}

torch::Tensor dispatch_div(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {
    OpsBase* ops = Registry::instance().get_ops_base(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered");
    return ops->div(x, y);
}

torch::Tensor dispatch_sum(
    const std::string& dtype_name,
    size_t bitwidth,
    const torch::Tensor& x,
    c10::optional<std::vector<int64_t>> dim,
    bool keepdim
) {
    OpsBase* ops = Registry::instance().get_ops_base(dtype_name, bitwidth);
    if (!ops) throw std::runtime_error("No ops registered");
    return ops->sum(x, dim, keepdim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
        "sum", &dispatch_sum,
        py::arg("dtype_name"), py::arg("bitwidth"), py::arg("x"),
        py::arg("dim") = c10::nullopt, py::arg("keepdim") = false,
        "Summation for custom dtypes"
    );
}

REGISTER_DTYPE("lns", 16, get_lns16_ops());