import torch
from torchdt import DType
from torchdt.ops.comparison_ops import (
    DTGeFunction,
    DTGtFunction,
    DTLeFunction,
    DTLtFunction,
)

@DType.register_func(torch.ge, torch.Tensor.ge)
def dt_ge(input, other, *, out=None):
    result = DTGeFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.gt, torch.Tensor.gt)
def dt_gt(input, other, *, out=None):
    result = DTGtFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.le, torch.Tensor.le)
def dt_le(input, other, *, out=None):
    result = DTLeFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.lt, torch.Tensor.lt)
def dt_lt(input, other, *, out=None):
    result = DTLtFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result