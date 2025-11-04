import torch
from torchdt import DType
from torchdt.ops.arithmetic_ops import (
    DTAddFunction,
    DTMulFunction,
)

@DType.register_func(torch.add, torch.Tensor.add)
def dt_add(x, y):
    return DTAddFunction.apply(x, y)

@DType.register_func(torch.mul, torch.Tensor.mul)
def dt_mul(x, y):
    return DTMulFunction.apply(x, y)