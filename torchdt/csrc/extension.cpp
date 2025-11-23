#include <torch/extension.h>
#include <torchdt/registry.h>

#include "binary_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Addition for custom dtypes");
    m.def("mul", &mul, "Multiplication for custom dtypes");
}