import torch
from torchdt import DType
from torchdt.ops.comparison_ops import (
    DTGeFunction,
    DTGtFunction,
    DTLeFunction,
    DTLtFunction,
)

@DType.register_func(torch.ge, torch.Tensor.ge,
                     cast=("input", "other"))
def dt_ge(input, other, *, out=None):
    result = DTGeFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.gt, torch.Tensor.gt,
                     cast=("input", "other"))
def dt_gt(input, other, *, out=None):
    result = DTGtFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.le, torch.Tensor.le,
                     cast=("input", "other"))
def dt_le(input, other, *, out=None):
    result = DTLeFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.lt, torch.Tensor.lt,
                     cast=("input", "other"))
def dt_lt(input, other, *, out=None):
    result = DTLtFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result