#include <torch/extension.h>
#include <torchdt/registry.h>

// for now, no functions are exposed
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}