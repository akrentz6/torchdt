#ifndef LNS_ADDITION_H
#define LNS_ADDITION_H

#include <torch/extension.h>

torch::Tensor add(
    const std::string& dtype_name,
    const size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
);

torch::Tensor sub(
    const std::string& dtype_name,
    const size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
);

torch::Tensor mul(
    const std::string& dtype_name,
    const size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
);

torch::Tensor div(
    const std::string& dtype_name,
    const size_t bitwidth,
    const torch::Tensor& x,
    const torch::Tensor& y
);

#endif // LNS_ADDITION_H