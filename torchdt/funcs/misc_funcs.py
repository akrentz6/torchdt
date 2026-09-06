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
    DTPermuteFunction,
    DTSelectFunction,
    DTNarrowFunction,
    DTSplitFunction,
    DTUnbindFunction,
    DTIndexSelectFunction,
    DTGatherFunction,
    DTTakeAlongDimFunction,
    DTIndexAddFunction,
    DTScatterAddFunction,
    DTMaskedFillFunction,
)

@DType.register_func(torch.broadcast_to, torch.Tensor.broadcast_to, torch.Tensor.expand,
                     cast=("input",))
def dt_broadcast_to(input, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = shape[0]
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

@DType.register_func(torch.Tensor.__setitem__,
                     cast=("input", "value"))
def dt_setitem(input, index, value):
    if torch.is_grad_enabled() and (input.requires_grad or value.requires_grad):
        raise NotImplementedError(
            "autograd through in-place indexed assignment is not supported; "
            "use index_add or scatter_add"
        )
    result = DTSetItemFunction.apply(input, index, value)
    input.copy_(result)
    return None

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
def dt_reshape(input, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = tuple(shape[0])
    return DTReshapeFunction.apply(input, tuple(shape))

@DType.register_func(torch.Tensor.item,
                     cast=("input",))
def dt_item(input):
    return input.__class__.ops.scalar_to_float(input._int)


@DType.register_func(torch.permute, torch.Tensor.permute, cast=("input",))
def dt_permute(input, *dims):
    if len(dims) == 1 and isinstance(dims[0], (tuple, list, torch.Size)):
        dims = tuple(dims[0])
    dims = _normalise_dims(dims, input.dim())
    if len(dims) != input.dim():
        raise RuntimeError(
            f"permute expected {input.dim()} dimensions, got {len(dims)}"
        )
    return DTPermuteFunction.apply(input, dims)


def _normalise_dims(dims, ndim):
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    result = tuple(dim % ndim for dim in dims)
    if len(set(result)) != len(result):
        raise RuntimeError("repeated dim in dim list")
    return result


def _movedim_order(ndim, source, destination):
    source = _normalise_dims(source, ndim)
    destination = _normalise_dims(destination, ndim)
    if len(source) != len(destination):
        raise RuntimeError("source and destination must have the same number of dimensions")
    order = [dim for dim in range(ndim) if dim not in source]
    for destination_dim, source_dim in sorted(zip(destination, source)):
        order.insert(destination_dim, source_dim)
    return tuple(order)


@DType.register_func(torch.movedim, torch.moveaxis,
                     torch.Tensor.movedim, torch.Tensor.moveaxis,
                     cast=("input",))
def dt_movedim(input, source, destination):
    return DTPermuteFunction.apply(input, _movedim_order(input.dim(), source, destination))


@DType.register_func(torch.swapdims, torch.swapaxes,
                     torch.Tensor.swapdims, torch.Tensor.swapaxes,
                     cast=("input",))
def dt_swapdims(input, dim0, dim1):
    dims = list(range(input.dim()))
    dim0 %= input.dim()
    dim1 %= input.dim()
    dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return DTPermuteFunction.apply(input, tuple(dims))


@DType.register_func(torch.select, torch.Tensor.select, cast=("input",))
def dt_select(input, dim, index):
    return DTSelectFunction.apply(input, dim, index)


@DType.register_func(torch.narrow, torch.Tensor.narrow, cast=("input",))
def dt_narrow(input, dim, start, length):
    if isinstance(start, torch.Tensor):
        start = int(start.item())
    return DTNarrowFunction.apply(input, dim, start, length)


@DType.register_func(torch.split, torch.Tensor.split, cast=("tensor",))
def dt_split(tensor, split_size_or_sections, dim=0):
    return DTSplitFunction.apply(tensor, split_size_or_sections, dim)


@DType.register_func(torch.unbind, torch.Tensor.unbind, cast=("input",))
def dt_unbind(input, dim=0):
    return DTUnbindFunction.apply(input, dim)


@DType.register_func(torch.index_select, torch.Tensor.index_select, cast=("input",))
def dt_index_select(input, dim, index, *, out=None):
    result = DTIndexSelectFunction.apply(input, dim, index)
    return out.copy_(result) if out is not None else result


@DType.register_func(torch.gather, torch.Tensor.gather, cast=("input",))
def dt_gather(input, dim, index, *, sparse_grad=False, out=None):
    if sparse_grad:
        raise NotImplementedError("sparse DType gradients are not supported")
    result = DTGatherFunction.apply(input, dim, index)
    return out.copy_(result) if out is not None else result


@DType.register_func(torch.take_along_dim, cast=("input",))
def dt_take_along_dim(input, indices, dim=None, *, out=None):
    result = DTTakeAlongDimFunction.apply(input, indices, dim)
    return out.copy_(result) if out is not None else result


@DType.register_func(torch.index_add, torch.Tensor.index_add,
                     cast=("input", "source"))
def dt_index_add(input, dim, index, source, *, alpha=1, out=None):
    result = DTIndexAddFunction.apply(input, dim, index, source, alpha)
    return out.copy_(result) if out is not None else result


@DType.register_func(torch.scatter_add, torch.Tensor.scatter_add,
                     cast=("input", "src"))
def dt_scatter_add(input, dim, index, src, *, out=None):
    result = DTScatterAddFunction.apply(input, dim, index, src)
    return out.copy_(result) if out is not None else result


@DType.register_func(torch.Tensor.masked_fill, cast=("self", "value"))
def dt_masked_fill(self, mask, value):
    return DTMaskedFillFunction.apply(self, mask, value)
