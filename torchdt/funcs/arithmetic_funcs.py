import torch
from torchdt import DType
from torchdt.ops.arithmetic_ops import (
    DTAddFunction,
    DTSubFunction,
    DTSumFunction,
    DTMulFunction,
    DTDivFunction,
    DTPowFunction,
    DTSquareFunction,
)

@DType.register_func(torch.add, torch.Tensor.add)
def dt_add(input, other, *, alpha=1, out=None):
    if alpha != 1:
        other = torch.mul(other, alpha)
    result = DTAddFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.sub, torch.Tensor.sub)
def dt_sub(input, other, *, alpha=1, out=None):
    if alpha != 1:
        other = torch.mul(other, alpha)
    result = DTSubFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.sum, torch.Tensor.sum)
def dt_sum(input, dim=None, keepdim=False, *, out=None):
    result = DTSumFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.mul, torch.Tensor.mul)
def dt_mul(input, other, *, out=None):
    result = DTMulFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.div, torch.Tensor.div)
def dt_div(input, other, *, out=None):
    result = DTDivFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.pow, torch.Tensor.pow)
def dt_pow(input, exponent, *, out=None):
    result = DTPowFunction.apply(input, exponent)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.square, torch.Tensor.square)
def dt_square(input, *, out=None):
    result = DTSquareFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result