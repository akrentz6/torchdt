#pragma once
#include <torch/custom_class.h>
#include <torchdt/registry.h>
namespace torchdt::native {
struct Context : torch::CustomClassHolder {
    const std::shared_ptr<const TensorKernels> kernels;
    Context(std::string name, int64_t bits, c10::Dict<std::string, c10::IValue> options)
        : kernels(Registry::instance().create(name, "cpu", Config{bits, options})) {}
    std::vector<std::string> capabilities() const { return kernels->capabilities(); }
};
using ContextPtr = c10::intrusive_ptr<Context>;
}
