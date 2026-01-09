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
    DTSqrtFunction,
    DTReciprocalFunction,
    DTExpFunction,
    DTLogFunction,
    DTProdFunction,
    DTMeanFunction,
    DTVarFunction,
    DTMatmulFunction,
    DTTransposeFunction,
)

@DType.register_func(torch.add, torch.Tensor.add,
                     cast=("input", "other"))
def dt_add(input, other, *, alpha=1, out=None):
    if alpha != 1:
        other = torch.mul(other, alpha)
    result = DTAddFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.Tensor.__radd__,
                     cast=("self", "other"))
def dt_radd(self, other):
    return torch.add(other, self)

@DType.register_func(torch.sub, torch.Tensor.sub,
                     cast=("input", "other"))
def dt_sub(input, other, *, alpha=1, out=None):
    if alpha != 1:
        other = torch.mul(other, alpha)
    result = DTSubFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.Tensor.__rsub__,
                     cast=("self", "other"))
def dt_rsub(self, other):
    return torch.sub(other, self)

@DType.register_func(torch.sum, torch.Tensor.sum,
                     cast=("input",))
def dt_sum(input, dim=None, keepdim=False, *, out=None):
    result = DTSumFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.mul, torch.Tensor.mul,
                     cast=("input", "other"))
def dt_mul(input, other, *, out=None):
    result = DTMulFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.Tensor.__rmul__,
                     cast=("self", "other"))
def dt_rmul(self, other):
    return torch.mul(other, self)

@DType.register_func(torch.div, torch.Tensor.div,
                     cast=("input", "other"))
def dt_div(input, other, *, out=None):
    result = DTDivFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.Tensor.__rdiv__,
                     cast=("self", "other"))
def dt_rdiv(self, other):
    return torch.div(other, self)

@DType.register_func(torch.pow, torch.Tensor.pow, torch.Tensor.__pow__,
                     cast=("input", "exponent"))
def dt_pow(input, exponent, *, out=None):
    result = DTPowFunction.apply(input, exponent)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.Tensor.__rpow__,
                     cast=("self", "other"))
def dt_rpow(self, other):
    return torch.pow(other, self)

@DType.register_func(torch.square, torch.Tensor.square,
                     cast=("input"))
def dt_square(input, *, out=None):
    result = DTSquareFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.sqrt, torch.Tensor.sqrt,
                     cast=("input",))
def dt_sqrt(input, *, out=None):
    result = DTSqrtFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.reciprocal, torch.Tensor.reciprocal,
                     cast=("input",))
def dt_reciprocal(input, *, out=None):
    result = DTReciprocalFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.exp, torch.Tensor.exp,
                     cast=("input",))
def dt_exp(input, *, out=None):
    result = DTExpFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.log, torch.Tensor.log,
                     cast=("input",))
def dt_log(input, *, out=None):
    result = DTLogFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.prod, torch.Tensor.prod,
                     cast=("input",))
def dt_prod(input, dim=None, keepdim=False, *, out=None):
    result = DTProdFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.mean, torch.Tensor.mean,
                     cast=("input",))
def dt_mean(input, dim=None, keepdim=False, *, out=None):
    result = DTMeanFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.var, torch.Tensor.var,
                     cast=("input", "correction"))
def dt_var(input, dim=None, *, correction=1, keepdim=False, out=None):
    result = DTVarFunction.apply(input, correction, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.matmul, torch.Tensor.matmul,
                     cast=("input", "other"))
def dt_matmul(input, other, *, out=None):
    result = DTMatmulFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.transpose, torch.Tensor.transpose,
                     cast=("input",))
def dt_transpose(input, dim0, dim1, *, out=None):
    result = DTTransposeFunction.apply(input, dim0, dim1)

    if out is not None:
        return out.copy_(result)
    return result
