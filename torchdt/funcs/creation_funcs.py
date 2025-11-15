import torch
from torchdt import DType

@DType.register_func(torch.zeros)
def dt_zeros(*size, out=None, dtype=None, device=None, requires_grad=False):
    if not issubclass(dtype, DType):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    if isinstance(size[0], tuple):
        size = size[0]

    result = dtype(torch.zeros(size), device=device, requires_grad=requires_grad)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.zeros_like)
def dt_zeros_like(input, *, dtype=None, device=None, requires_grad=False):
    if isinstance(dtype, type) and issubclass(dtype, DType):
        return dtype(torch.zeros(input.size()), device=device, requires_grad=requires_grad)
    return torch.zeros(input.size(), dtype=dtype, device=device, requires_grad=requires_grad)

@DType.register_func(torch.ones)
def dt_ones(*size, out=None, dtype=None, device=None, requires_grad=False):
    if not issubclass(dtype, DType):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    if isinstance(size[0], tuple):
        size = size[0]

    result = dtype(torch.ones(size), device=device, requires_grad=requires_grad)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.ones_like)
def dt_ones_like(input, *, dtype=None, device=None, requires_grad=False):
    if isinstance(dtype, type) and issubclass(dtype, DType):
        return dtype(torch.ones(input.size()), device=device, requires_grad=requires_grad)
    return torch.ones(input.size(), dtype=dtype, device=device, requires_grad=requires_grad)

@DType.register_func(torch.full)
def dt_full(size, fill_value, *, out=None, dtype=None, device=None, requires_grad=False):
    if not isinstance(out, DType) and not issubclass(dtype, DType):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    if isinstance(size[0], tuple):
        size = size[0]

    result = dtype(torch.full(size, fill_value), device=device, requires_grad=requires_grad)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.full_like)
def dt_full_like(input, fill_value, *, dtype=None, device=None, requires_grad=False):
    if isinstance(dtype, type) and issubclass(dtype, DType):
        return dtype(torch.full(input.size(), fill_value), device=device, requires_grad=requires_grad)
    return torch.full(input.size(), fill_value, dtype=dtype, device=device, requires_grad=requires_grad)

@DType.register_func(torch.rand)
def dt_rand(*size, out=None, dtype=None, device=None, requires_grad=False):
    if not issubclass(dtype, DType):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    if isinstance(size[0], tuple):
        size = size[0]

    result = dtype(torch.rand(size), device=device, requires_grad=requires_grad)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.rand_like)
def dt_rand_like(input, *, dtype=None, device=None, requires_grad=False):
    if isinstance(dtype, type) and issubclass(dtype, DType):
        return dtype(torch.rand(input.size()), device=device, requires_grad=requires_grad)
    return torch.rand(input.size(), dtype=dtype, device=device, requires_grad=requires_grad)

@DType.register_func(torch.randn)
def dt_randn(*size, out=None, dtype=None, device=None, requires_grad=False):
    if not issubclass(dtype, DType):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    if isinstance(size[0], tuple):
        size = size[0]

    result = dtype(torch.randn(size), device=device, requires_grad=requires_grad)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.randn_like)
def dt_randn_like(input, *, dtype=None, device=None, requires_grad=False):
    if isinstance(dtype, type) and issubclass(dtype, DType):
        return dtype(torch.randn(input.size()), device=device, requires_grad=requires_grad)
    return torch.randn(input.size(), dtype=dtype, device=device, requires_grad=requires_grad)

@DType.register_func(torch.empty)
def dt_empty(*size, out=None, dtype=None, device=None, requires_grad=False):
    if not issubclass(dtype, DType):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    if isinstance(size[0], tuple):
        size = size[0]

    result = dtype(torch.empty(size), device=device, requires_grad=requires_grad)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.empty_like)
def dt_empty_like(input, *, dtype=None, device=None, requires_grad=False):
    if isinstance(dtype, type) and issubclass(dtype, DType):
        return dtype(torch.empty(input.size()), device=device, requires_grad=requires_grad)
    return torch.empty(input.size(), dtype=dtype, device=device, requires_grad=requires_grad)