import math
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
    DTGetItemFunction,
    DTSetItemFunction,
    DTToFunction,
    DTViewFunction,
    DTContiguousFunction,
    DTRepeatFunction,
    DTFlattenFunction,
    DTReshapeFunction,
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
    result = DTCatFunction.apply(dim, *tensors)

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

@DType.register_func(torch.Tensor.__getitem__,
                     cast=("input",))
def dt_getitem(input, index):
    return DTGetItemFunction.apply(input, index)

# turns an index into a set of index tensors for each dimension
def _make_index_tensors(index, shape):
    flat_count = math.prod(shape)
    labels = torch.arange(flat_count).reshape(shape)
    flat_idx = labels[index].reshape(-1)
    return torch.unravel_index(flat_idx, shape)

@DType.register_func(torch.Tensor.__setitem__,
                     cast=("input", "value"))
def dt_setitem(input, index, value):
    index = _make_index_tensors(index, input.shape)
    return DTSetItemFunction.apply(input, index, value)

@DType.register_func(torch.Tensor.to,
                     cast=("input",))
def dt_to(input, device=None):
    return DTToFunction.apply(input, device)

@DType.register_func(torch.Tensor.view,
                     cast=("input",))
def dt_view(input, *shape):
    return DTViewFunction.apply(input, shape)

@DType.register_func(torch.Tensor.contiguous,
                     cast=("input",))
def dt_contiguous(input, memory_format=torch.preserve_format):
    return DTContiguousFunction.apply(input, memory_format)

@DType.register_func(torch.Tensor.repeat,
                     cast=("input",))
def dt_repeat(input, *repeats):
    return DTRepeatFunction.apply(input, repeats)

@DType.register_func(torch.flatten, torch.Tensor.flatten,
                     cast=("input",))
def dt_flatten(input, start_dim=0, end_dim=-1):
    return DTFlattenFunction.apply(input, start_dim, end_dim)

@DType.register_func(torch.reshape, torch.Tensor.reshape,
                     cast=("input",))
def dt_reshape(input, shape):
    return DTReshapeFunction.apply(input, shape)

@DType.register_func(torch.Tensor.item,
                     cast=("input",))
def dt_item(input):
    return input.to_float().item()