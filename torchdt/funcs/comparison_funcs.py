import torch
from torchdt import DType
from torchdt.ops.comparison_ops import (
    DTEqualFunction,
    DTEqFunction,
    DTNeFunction,
    DTGeFunction,
    DTGtFunction,
    DTLeFunction,
    DTLtFunction,
    DTIscloseFunction,
    DTAllcloseFunction,
    DTAnyFunction,
    DTAllFunction,
    DTIsinFunction,
    DTMaximumFunction,
    DTMinimumFunction,
    DTMaxFunction,
    DTMinFunction,
    DTArgmaxFunction,
    DTArgminFunction,
    DTClampFunction,
)

@DType.register_func(torch.equal, torch.Tensor.equal,
                     cast=("input", "other"))
def dt_equal(input, other, *, out=None):
    result = DTEqualFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.eq, torch.Tensor.eq,
                     cast=("input", "other"))
def dt_eq(input, other, *, out=None):
    result = DTEqFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.ne, torch.Tensor.ne,
                     cast=("input", "other"))
def dt_ne(input, other, *, out=None):
    result = DTNeFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

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

@DType.register_func(torch.isclose, torch.Tensor.isclose,
                     cast=("input", "other", "rtol", "atol"))
def dt_isclose(input, other, rtol=1e-05, atol=1e-08, *, out=None):
    result = DTIscloseFunction.apply(input, other, rtol, atol)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.allclose, torch.Tensor.allclose,
                     cast=("input", "other", "rtol", "atol"))
def dt_allclose(input, other, rtol=1e-05, atol=1e-08, *, out=None):
    result = DTAllcloseFunction.apply(input, other, rtol, atol)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.any, torch.Tensor.any,
                     cast=("input",))
def dt_any(input, dim=None, keepdim=False, *, out=None):
    result = DTAnyFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.all, torch.Tensor.all,
                     cast=("input",))
def dt_all(input, dim=None, keepdim=False, *, out=None):
    result = DTAllFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.isin,
                     cast=("elements", "test_elements"))
def dt_isin(elements, test_elements, *, out=None):
    result = DTIsinFunction.apply(elements, test_elements)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.maximum, torch.Tensor.maximum,
                     cast=("input", "other"))
def dt_maximum(input, other, *, out=None):
    result = DTMaximumFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.minimum, torch.Tensor.minimum,
                     cast=("input", "other"))
def dt_minimum(input, other, *, out=None):
    result = DTMinimumFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.max, torch.Tensor.max,
                     cast=("input",))
def dt_max(input, dim=None, keepdim=False, *, out=None):
    result = DTMaxFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.min, torch.Tensor.min,
                     cast=("input",))
def dt_min(input, dim=None, keepdim=False, *, out=None):
    result = DTMinFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.argmax, torch.Tensor.argmax,
                     cast=("input",))
def dt_argmax(input, dim=None, keepdim=False, *, out=None):
    result = DTArgmaxFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.argmin, torch.Tensor.argmin,
                     cast=("input",))
def dt_argmin(input, dim=None, keepdim=False, *, out=None):
    result = DTArgminFunction.apply(input, dim, keepdim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.clamp, torch.Tensor.clamp,
                     cast=("input", "min", "max"))
def dt_clamp(input, min=None, max=None, *, out=None):
    result = DTClampFunction.apply(input, min, max)

    if out is not None:
        return out.copy_(result)
    return result
