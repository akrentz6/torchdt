#include <torch/extension.h>
#include <ATen/native/cpu/Loops.h>
#include <torchdt/registry.h>

template<size_t bitwidth>
torch::Tensor add_dispatch(
    const std::string& dtype_name,
    const torch::Tensor& x,
    const torch::Tensor& y
) {

    using scalar_t = typename StorageFor<bitwidth>::type;
    auto *ops = Registry::instance().get_ops_typed<bitwidth>(dtype_name);
    if (!ops) throw std::runtime_error("No registered ops for " + dtype_name + " with bitwidth " + std::to_string(bitwidth));

    auto result_sizes = at::infer_size(x.sizes(), y.sizes());
    auto out = torch::empty(result_sizes, x.options());
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .add_input(y)
        .build();

    at::native::cpu_kernel(
        iter,
        [ops](scalar_t a, scalar_t b) -> scalar_t {
            return ops->add(a, b);
        }
    );

    return out;

}

torch::Tensor add(
    const std::string& dtype_name,
    const size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {

    if (bitwidth == 8) return add_dispatch<8>(dtype_name, x, y);
    else if (bitwidth == 16) return add_dispatch<16>(dtype_name, x, y);
    else if (bitwidth == 32) return add_dispatch<32>(dtype_name, x, y);
    else if (bitwidth == 64) return add_dispatch<64>(dtype_name, x, y);
    else throw std::runtime_error("No registered ops for " + dtype_name);

}

template<size_t bitwidth>
torch::Tensor sub_dispatch(
    const std::string& dtype_name,
    const torch::Tensor& x,
    const torch::Tensor& y
) {

    using scalar_t = typename StorageFor<bitwidth>::type;
    auto *ops = Registry::instance().get_ops_typed<bitwidth>(dtype_name);
    if (!ops) throw std::runtime_error("No registered ops for " + dtype_name + " with bitwidth " + std::to_string(bitwidth));

    auto result_sizes = at::infer_size(x.sizes(), y.sizes());
    auto out = torch::empty(result_sizes, x.options());
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .add_input(y)
        .build();

    at::native::cpu_kernel(
        iter,
        [ops](scalar_t a, scalar_t b) -> scalar_t {
            return ops->sub(a, b);
        }
    );

    return out;

}

torch::Tensor sub(
    const std::string& dtype_name,
    const size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {

    if (bitwidth == 8) return sub_dispatch<8>(dtype_name, x, y);
    else if (bitwidth == 16) return sub_dispatch<16>(dtype_name, x, y);
    else if (bitwidth == 32) return sub_dispatch<32>(dtype_name, x, y);
    else if (bitwidth == 64) return sub_dispatch<64>(dtype_name, x, y);
    else throw std::runtime_error("No registered ops for " + dtype_name);

}

template<size_t bitwidth>
torch::Tensor mul_dispatch(
    const std::string& dtype_name,
    const torch::Tensor& x,
    const torch::Tensor& y
) {

    using scalar_t = typename StorageFor<bitwidth>::type;
    auto *ops = Registry::instance().get_ops_typed<bitwidth>(dtype_name);
    if (!ops) throw std::runtime_error("No registered ops for " + dtype_name + " with bitwidth " + std::to_string(bitwidth));

    auto result_sizes = at::infer_size(x.sizes(), y.sizes());
    auto out = torch::empty(result_sizes, x.options());
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .add_input(y)
        .build();

    at::native::cpu_kernel(
        iter,
        [ops](scalar_t a, scalar_t b) -> scalar_t {
            return ops->mul(a, b);
        }
    );

    return out;

}

torch::Tensor mul(
    const std::string& dtype_name,
    const size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {

    if (bitwidth == 8) return mul_dispatch<8>(dtype_name, x, y);
    else if (bitwidth == 16) return mul_dispatch<16>(dtype_name, x, y);
    else if (bitwidth == 32) return mul_dispatch<32>(dtype_name, x, y);
    else if (bitwidth == 64) return mul_dispatch<64>(dtype_name, x, y);
    else throw std::runtime_error("No registered ops for " + dtype_name);

}

template<size_t bitwidth>
torch::Tensor div_dispatch(
    const std::string& dtype_name,
    const torch::Tensor& x,
    const torch::Tensor& y
) {

    using scalar_t = typename StorageFor<bitwidth>::type;
    auto *ops = Registry::instance().get_ops_typed<bitwidth>(dtype_name);
    if (!ops) throw std::runtime_error("No registered ops for " + dtype_name + " with bitwidth " + std::to_string(bitwidth));

    auto result_sizes = at::infer_size(x.sizes(), y.sizes());
    auto out = torch::empty(result_sizes, x.options());
    at::TensorIterator iter = at::TensorIteratorConfig()
        .add_output(out)
        .add_input(x)
        .add_input(y)
        .build();

    at::native::cpu_kernel(
        iter,
        [ops](scalar_t a, scalar_t b) -> scalar_t {
            return ops->div(a, b);
        }
    );

    return out;

}

torch::Tensor div(
    const std::string& dtype_name,
    const size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
) {

    if (bitwidth == 8) return div_dispatch<8>(dtype_name, x, y);
    else if (bitwidth == 16) return div_dispatch<16>(dtype_name, x, y);
    else if (bitwidth == 32) return div_dispatch<32>(dtype_name, x, y);
    else if (bitwidth == 64) return div_dispatch<64>(dtype_name, x, y);
    else throw std::runtime_error("No registered ops for " + dtype_name);

}