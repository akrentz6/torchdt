import torch
from torchdt import DType
from torchdt.ops.misc_ops import (
    DTBroadcastToFunction,
    DTCloneFunction,
    DTSqueezeFunction,
    DTUnsqueezeFunction,
    DTStackFunction,
    DTCatFunction,
    DTChunkFunction,
    DTWhereFunction,
    DTPadFunction,
)

@DType.register_func(torch.broadcast_to, torch.Tensor.expand,
                     cast=("input",))
def dt_broadcast_to(input, shape):
    return DTBroadcastToFunction.apply(input, shape)

@DType.register_func(torch.clone, torch.Tensor.clone,
                     cast=("input",))
def dt_clone(input, memory_format=torch.preserve_format):
    return DTCloneFunction.apply(input, memory_format)

@DType.register_func(torch.squeeze, torch.Tensor.squeeze,
                     cast=("input",))
def dt_squeeze(input, dim=None):
    return DTSqueezeFunction.apply(input, dim)

@DType.register_func(torch.unsqueeze, torch.Tensor.unsqueeze,
                     cast=("input",))
def dt_unsqueeze(input, dim):
    return DTUnsqueezeFunction.apply(input, dim)

@DType.register_func(torch.stack,
                     cast=("tensors",))
def dt_stack(tensors, dim=0, *, out=None):
    result = DTStackFunction.apply(dim, *tensors)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.cat,
                     cast=("tensors",))
def dt_cat(tensors, dim=0, *, out=None):
    result = DTCatFunction.apply(dim=dim, *tensors)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.chunk, torch.Tensor.chunk,
                     cast=("input",))
def dt_chunk(input, chunks, dim=0):
    return DTChunkFunction.apply(input, chunks, dim)

@DType.register_func(torch.where, torch.Tensor.where,
                     cast=("input", "other"))
def dt_where(condition, input, other, *, out=None):
    result = DTWhereFunction.apply(condition, input, other)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.nn.functional.pad,
                     cast=("input", "value"))
def dt_pad(input, pad, mode="constant", value=0):
    return DTPadFunction.apply(input, pad, mode, value)