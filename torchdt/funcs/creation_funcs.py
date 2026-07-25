import torch
from torchdt import DType


def _is_dtype(value):
    return isinstance(value, type) and issubclass(value, DType)


def _size_tuple(size):
    return size[0] if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size


def _custom_full(
    dtype, size, fill_value, device, requires_grad,
    layout=torch.strided, pin_memory=False, memory_format=None,
):
    target = torch.device(device) if device is not None else None
    ops = dtype.ops.direct_for_device(target)
    kwargs = dict(dtype=dtype.int_dtype, device=target, layout=layout, pin_memory=pin_memory)
    if memory_format is not None:
        kwargs["memory_format"] = memory_format
    internal = torch.empty(size, **kwargs)
    internal.fill_(ops.encoded_scalar(fill_value))
    return dtype(internal, internal=True, requires_grad=requires_grad)


def _custom_full_like(dtype, input, fill_value, device, requires_grad, memory_format):
    target = torch.device(device) if device is not None else input.device
    ops = dtype.ops.direct_for_device(target)
    source = input._int if isinstance(input, DType) else input
    internal = torch.empty_like(
        source, dtype=dtype.int_dtype, device=target, memory_format=memory_format
    )
    internal.fill_(ops.encoded_scalar(fill_value))
    return dtype(internal, internal=True, requires_grad=requires_grad)


def _custom_random(
    dtype, size, device, requires_grad, normal=False, generator=None,
    layout=torch.strided, pin_memory=False,
):
    factory = torch.randn if normal else torch.rand
    values = factory(
        size, generator=generator, dtype=torch.float32, device=device,
        layout=layout, pin_memory=pin_memory,
    )
    return dtype(values, requires_grad=requires_grad)


def _custom_random_like(
    dtype, input, device, requires_grad, normal, generator, memory_format,
):
    factory = torch.randn_like if normal else torch.rand_like
    source = input._float if isinstance(input, DType) else input
    values = factory(
        source, dtype=torch.float32,
        device=device if device is not None else input.device,
        requires_grad=False, memory_format=memory_format,
        **({"generator": generator} if generator is not None else {}),
    )
    return dtype(values, requires_grad=requires_grad)

@DType.register_func(torch.zeros)
def dt_zeros(*size, out=None, dtype=None, layout=torch.strided, device=None,
             requires_grad=False, pin_memory=False):
    if not _is_dtype(dtype):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    result = _custom_full(
        dtype, _size_tuple(size), 0.0, device, requires_grad, layout, pin_memory
    )

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.zeros_like)
def dt_zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False,
                  memory_format=torch.preserve_format):
    if dtype is None and isinstance(input, DType):
        dtype = input.__class__
    if _is_dtype(dtype):
        return _custom_full_like(dtype, input, 0.0, device, requires_grad, memory_format)
    source = input._float if isinstance(input, DType) else input
    return torch.zeros_like(
        source, dtype=dtype, layout=layout, device=device,
        requires_grad=requires_grad, memory_format=memory_format,
    )

@DType.register_func(torch.ones)
def dt_ones(*size, out=None, dtype=None, layout=torch.strided, device=None,
            requires_grad=False, pin_memory=False):
    if not _is_dtype(dtype):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    result = _custom_full(
        dtype, _size_tuple(size), 1.0, device, requires_grad, layout, pin_memory
    )

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.ones_like)
def dt_ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False,
                 memory_format=torch.preserve_format):
    if dtype is None and isinstance(input, DType):
        dtype = input.__class__
    if _is_dtype(dtype):
        return _custom_full_like(dtype, input, 1.0, device, requires_grad, memory_format)
    source = input._float if isinstance(input, DType) else input
    return torch.ones_like(
        source, dtype=dtype, layout=layout, device=device,
        requires_grad=requires_grad, memory_format=memory_format,
    )

@DType.register_func(torch.full)
def dt_full(size, fill_value, *, out=None, dtype=None, layout=torch.strided,
            device=None, requires_grad=False, pin_memory=False):
    if not isinstance(out, DType) and not _is_dtype(dtype):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    if dtype is None:
        dtype = out.__class__
    result = _custom_full(
        dtype, size, fill_value, device, requires_grad, layout, pin_memory
    )

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.full_like)
def dt_full_like(input, fill_value, *, dtype=None, layout=None, device=None,
                 requires_grad=False, memory_format=torch.preserve_format):
    if dtype is None and isinstance(input, DType):
        dtype = input.__class__
    if _is_dtype(dtype):
        return _custom_full_like(dtype, input, fill_value, device, requires_grad, memory_format)
    source = input._float if isinstance(input, DType) else input
    return torch.full_like(
        source, fill_value, dtype=dtype, layout=layout, device=device,
        requires_grad=requires_grad, memory_format=memory_format,
    )

@DType.register_func(torch.rand)
def dt_rand(*size, generator=None, out=None, dtype=None, layout=torch.strided,
            device=None, requires_grad=False, pin_memory=False):
    if not _is_dtype(dtype):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    result = _custom_random(
        dtype, _size_tuple(size), device, requires_grad,
        generator=generator, layout=layout, pin_memory=pin_memory,
    )

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.rand_like)
def dt_rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False,
                 memory_format=torch.preserve_format, generator=None):
    if dtype is None and isinstance(input, DType):
        dtype = input.__class__
    if _is_dtype(dtype):
        return _custom_random_like(
            dtype, input, device, requires_grad, False, generator, memory_format
        )
    source = input._float if isinstance(input, DType) else input
    return torch.rand_like(
        source, dtype=dtype, layout=layout, device=device,
        requires_grad=requires_grad, memory_format=memory_format,
    )

@DType.register_func(torch.randn)
def dt_randn(*size, generator=None, out=None, dtype=None, layout=torch.strided,
             device=None, requires_grad=False, pin_memory=False):
    if not _is_dtype(dtype):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    result = _custom_random(
        dtype, _size_tuple(size), device, requires_grad, normal=True,
        generator=generator, layout=layout, pin_memory=pin_memory,
    )

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.randn_like)
def dt_randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False,
                  memory_format=torch.preserve_format, generator=None):
    if dtype is None and isinstance(input, DType):
        dtype = input.__class__
    if _is_dtype(dtype):
        return _custom_random_like(
            dtype, input, device, requires_grad, True, generator, memory_format
        )
    source = input._float if isinstance(input, DType) else input
    return torch.randn_like(
        source, dtype=dtype, layout=layout, device=device,
        requires_grad=requires_grad, memory_format=memory_format,
    )

@DType.register_func(torch.empty)
def dt_empty(*size, out=None, dtype=None, layout=torch.strided, device=None,
             requires_grad=False, pin_memory=False, memory_format=None):
    if not _is_dtype(dtype):
        raise TypeError(f"dtype must be a subclass of DType, got {dtype}")
    kwargs = dict(
        dtype=dtype.int_dtype, layout=layout, device=device, pin_memory=pin_memory
    )
    if memory_format is not None:
        kwargs["memory_format"] = memory_format
    internal = torch.empty(_size_tuple(size), **kwargs)
    result = dtype(internal, internal=True, requires_grad=requires_grad)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.empty_like)
def dt_empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False,
                  memory_format=torch.preserve_format):
    if dtype is None and isinstance(input, DType):
        dtype = input.__class__
    if _is_dtype(dtype):
        target = device if device is not None else input.device
        source = input._int if isinstance(input, DType) else input
        internal = torch.empty_like(
            source, dtype=dtype.int_dtype, device=target, memory_format=memory_format
        )
        return dtype(internal, internal=True, requires_grad=requires_grad)
    source = input._float if isinstance(input, DType) else input
    return torch.empty_like(
        source, dtype=dtype, layout=layout, device=device,
        requires_grad=requires_grad, memory_format=memory_format,
    )
