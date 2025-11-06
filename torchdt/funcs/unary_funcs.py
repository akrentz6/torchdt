import torch
from torchdt import DType
from torchdt.ops.unary_ops import (
    DTSignFunction,
    DTNegFunction,
    DTAbsFunction,
)

@DType.register_func(torch.sign, torch.Tensor.sign,
                     cast=("input",))
def dt_sign(input, *, out=None):
    result = DTSignFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.neg, torch.Tensor.neg,
                     cast=("input",))
def dt_neg(input, *, out=None):
    result = DTNegFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.abs, torch.Tensor.abs,
                     cast=("input",))
def dt_abs(input, *, out=None):
    result = DTAbsFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.positive, torch.Tensor.positive,
                     cast=("input",))
def dt_positive(input, *, out=None):
    if out is not None:
        return out.copy_(input)
    return input