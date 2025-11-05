import torch
from torchdt import DType
from torchdt.ops.unary_ops import (
    DTSignFunction,
    DTNegFunction,
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