import itertools
import math
import torch
from torchdt.autograd import DTFunction
from torchdt.ops import register_base_op

@register_base_op("broadcast_to")
def dt_broadcast_to(ops, x, size):
    return torch.broadcast_to(x, size)

class DTBroadcastToFunction(DTFunction):

    @staticmethod
    def forward(ops, x, shape):
        return ops.broadcast_to(x, shape)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, shape = inputs
        ctx.x_shape = x.shape
        ctx.shape = shape

    @staticmethod
    def backward(ctx, ops, grad_output):
        # Sum over the broadcasted dimensions
        # First, handle prepended dimensions (when original tensor had fewer dims)
        ndims_added = grad_output.ndim - len(ctx.x_shape)
        grad_x = grad_output
        for i in range(ndims_added):
            grad_x = ops.sum(grad_x, dim=0, keepdim=False)

        # Then, handle expanded dimensions (where original dim was 1)
        for i, (orig_size, grad_size) in enumerate(zip(ctx.x_shape, grad_x.shape)):
            if orig_size == 1 and grad_size > 1:
                grad_x = ops.sum(grad_x, dim=i, keepdim=True)

        return grad_x, None

@register_base_op("clone")
def dt_clone(ops, x, memory_format=torch.preserve_format):
    return x.clone()

class DTCloneFunction(DTFunction):

    @staticmethod
    def forward(ops, x, memory_format=torch.preserve_format):
        return ops.clone(x, memory_format)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        pass

    @staticmethod
    def backward(ctx, ops, grad_output):
        return grad_output, None

@register_base_op("squeeze")
def dt_squeeze(ops, x, dim=None):
    return torch.squeeze(x, dim=dim)

class DTSqueezeFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=None):
        return ops.squeeze(x, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim = inputs
        ctx.x_shape = x.shape

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = grad_output.view(ctx.x_shape)
        return grad_x, None

@register_base_op("unsqueeze")
def dt_unsqueeze(ops, x, dim):
    return torch.unsqueeze(x, dim=dim)

class DTUnsqueezeFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim):
        return ops.unsqueeze(x, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim = inputs
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = grad_output.squeeze(dim=ctx.dim)
        return grad_x, None

@register_base_op("stack")
def dt_stack(ops, tensors, dim=0):
    return torch.stack(tensors, dim=dim)

class DTStackFunction(DTFunction):

    @staticmethod
    def forward(ops, dim, *tensors):
        return ops.stack(tensors, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        dim, *tensors = inputs
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        return None, *grad_output.unbind(ctx.dim)

@register_base_op("cat")
def dt_cat(ops, tensors, dim=0):
    return torch.cat(tensors, dim=dim)

class DTCatFunction(DTFunction):

    @staticmethod
    def forward(ops, dim, *tensors):
        return ops.cat(tensors, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        dim, *tensors = inputs
        ctx.sizes = [t.size(dim) for t in tensors]
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_tensors = torch.split(grad_output, ctx.sizes, dim=ctx.dim)
        return None, *grad_tensors

@register_base_op("chunk")
def dt_chunk(ops, x, chunks, dim=0):
    return torch.chunk(x, chunks, dim=dim)

class DTChunkFunction(DTFunction):

    @staticmethod
    def forward(ops, x, chunks, dim=0):
        return ops.chunk(x, chunks, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, _, dim = inputs
        ctx.dim = dim
        ctx.out_shapes = [o.shape for o in output]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, ops, *grad_outputs):
        parts = []
        device = next((g.device for g in grad_outputs if g is not None), None)
        for g, shape in zip(grad_outputs, ctx.out_shapes):
            if g is None:
                g = ops.zeros(shape, device=device)
            parts.append(g)

        return torch.cat(parts, dim=ctx.dim), None, None

@register_base_op("where")
def dt_where(ops, condition, x, y):
    return torch.where(condition, x, y)

class DTWhereFunction(DTFunction):

    @staticmethod
    def forward(ops, condition, x, y):
        return ops.where(condition, x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        condition, x, y = inputs
        ctx.save_for_backward(condition, x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        condition, x, y = ctx.saved_tensors
        zero = ops.scalar_from_float(0.0, device=grad_output.device)
        grad_x = ops.sum_to_size(torch.where(condition, grad_output, zero), x.shape)
        grad_y = ops.sum_to_size(torch.where(condition, zero, grad_output), y.shape)
        return None, grad_x, grad_y

def _unpad_along_dim(ops, g, left, right, dim, mode):
    if left == right == 0:
        return g
    if mode == "constant":
        return g.narrow(dim, left, g.size(dim) - left - right).clone()

    interior_len = g.size(dim) - left - right
    grad_x = g.narrow(dim, left, interior_len).clone()
    first = 0
    last = interior_len - 1

    if mode == "replicate":
        if left:
            grad_x.select(dim, first).copy_(ops.add(
                grad_x.select(dim, first),
                ops.sum(g.narrow(dim, 0, left), dim=dim),
            ))
        if right:
            grad_x.select(dim, last).copy_(ops.add(
                grad_x.select(dim, last),
                ops.sum(g.narrow(dim, g.size(dim) - right, right), dim=dim),
            ))

    elif mode == "reflect":
        for i in range(left):
            target = left - i
            grad_x.select(dim, target).copy_(ops.add(
                grad_x.select(dim, target),
                g.select(dim, i),
            ))
        for i in range(right):
            target = last - 1 - i
            grad_x.select(dim, target).copy_(ops.add(
                grad_x.select(dim, target),
                g.select(dim, g.size(dim) - 1 - i),
            ))

    elif mode == "circular":
        if left:
            grad_x.narrow(dim, interior_len-left, left).copy_(ops.add(
                grad_x.narrow(dim, interior_len - left, left),
                g.narrow(dim, 0, left),
            ))
        if right:
            grad_x.narrow(dim, 0, right).copy_(ops.add(
                grad_x.narrow(dim, 0, right),
                g.narrow(dim, g.size(dim) - right, right),
            ))

    return grad_x

@register_base_op("pad")
def dt_pad(ops, input, pad, mode="constant", value=None):
    return torch.nn.functional.pad(input, pad, mode=mode, value=value)

class DTPadFunction(DTFunction):

    @staticmethod
    def forward(ops, input, pad, mode="constant", value=None):
        return ops.pad(input, pad, mode=mode, value=value)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, pad, mode, _ = inputs
        ctx.pad = pad
        ctx.mode = mode

    @staticmethod
    def backward(ctx, ops, grad_output):
        ndim_pad = len(ctx.pad) // 2
        grad_x = grad_output

        for i in range(ndim_pad):
            left = ctx.pad[2 * i]
            right = ctx.pad[2 * i + 1]
            dim = grad_output.dim() - 1 - i
            grad_x = _unpad_along_dim(ops, grad_x, left, right, dim, ctx.mode)

        return grad_x, None, None, None

@register_base_op("getitem")
def dt_getitem(ops, x, index):
    return x[index]

class DTGetItemFunction(DTFunction):

    @staticmethod
    def forward(ops, x, index):
        return ops.getitem(x, index)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, idx = inputs
        ctx.x_shape = x.shape
        ctx.x_device = x.device
        ctx.is_idx_tensor = torch.is_tensor(idx)
        if ctx.is_idx_tensor:
            ctx.save_for_backward(idx)
        else:
            ctx.idx = idx

    @staticmethod
    def backward(ctx, ops, grad_output):
        if ctx.is_idx_tensor:
            idx, = ctx.saved_tensors
        else:
            idx = ctx.idx

        grad_x = ops.zeros(ctx.x_shape, device=ctx.x_device)
        positions = torch.arange(
            math.prod(ctx.x_shape), device=ctx.x_device
        ).reshape(ctx.x_shape)[idx]
        values = grad_output.expand(positions.shape).reshape(-1)
        flat_grad = grad_x.reshape(-1)
        for position, value in zip(positions.reshape(-1).tolist(), values):
            flat_grad[position] = ops.add(flat_grad[position], value)
        return grad_x, None

@register_base_op("setitem")
def dt_setitem(ops, x, index, value):
    result = x.clone()
    result[index] = value
    return result

class DTSetItemFunction(DTFunction):

    @staticmethod
    def forward(ops, x, index, value):
        return ops.setitem(x, index, value)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, idx, value = inputs
        ctx.value_shape = value.shape
        ctx.is_idx_tensor = torch.is_tensor(idx)
        positions = torch.arange(
            math.prod(x.shape), device=x.device
        ).reshape(x.shape)[idx]
        if ctx.is_idx_tensor:
            ctx.save_for_backward(idx, positions)
        else:
            ctx.idx = idx
            ctx.save_for_backward(positions)

    @staticmethod
    def backward(ctx, ops, grad_output):
        if ctx.is_idx_tensor:
            idx, positions = ctx.saved_tensors
        else:
            idx = ctx.idx
            positions, = ctx.saved_tensors

        grad_x = grad_output.clone()
        grad_x[idx] = ops.scalar_from_float(0.0, device=grad_x.device)
        grad_value = grad_output[idx]
        flat_positions = positions.reshape(-1).tolist()
        keep = torch.zeros(len(flat_positions), dtype=torch.bool, device=grad_output.device)
        seen = set()
        for offset in range(len(flat_positions) - 1, -1, -1):
            position = flat_positions[offset]
            if position not in seen:
                keep[offset] = True
                seen.add(position)
        keep = keep.reshape(positions.shape)
        zero = ops.scalar_from_float(0.0, device=grad_output.device)
        grad_value = torch.where(keep, grad_value, zero)
        grad_value = ops.sum_to_size(grad_value, ctx.value_shape)

        return grad_x, None, grad_value

@register_base_op("to")
def dt_to(ops, x, device):
    return x.to(device=device)

class DTToFunction(DTFunction):

    @staticmethod
    def forward(ops, x, device):
        return ops.to(x, device)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, _ = inputs
        ctx.orig_device = x.device

    @staticmethod
    def backward(ctx, ops, grad_output):
        return grad_output.to(ctx.orig_device), None

@register_base_op("view")
def dt_view(ops, x, shape):
    return x.view(shape)

class DTViewFunction(DTFunction):

    @staticmethod
    def forward(ops, x, shape):
        return ops.view(x, shape)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, shape = inputs
        ctx.original_shape = x.shape
        ctx.n_shape = len(shape)

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = grad_output.contiguous().view(ctx.original_shape)
        return grad_x, None

@register_base_op("contiguous")
def dt_contiguous(ops, x, memory_format):
    return x.contiguous(memory_format=memory_format)

class DTContiguousFunction(DTFunction):

    @staticmethod
    def forward(ops, x, memory_format=torch.preserve_format):
        return ops.contiguous(x, memory_format)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, _ = inputs
        ctx.original_shape = x.shape
        ctx.original_strides = x.stride()

    @staticmethod
    def backward(ctx, ops, grad_output):
        return torch.as_strided(grad_output, ctx.original_shape, ctx.original_strides), None

@register_base_op("repeat")
def dt_repeat(ops, x, repeats):
    return x.repeat(*repeats)

class DTRepeatFunction(DTFunction):

    @staticmethod
    def forward(ops, x, repeats):
        return ops.repeat(x, repeats)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, repeats = inputs
        ctx.input_shape = tuple(x.shape)
        ctx.repeats = tuple(repeats)

    @staticmethod
    def backward(ctx, ops, grad_output):
        padded_shape = (1,) * (len(ctx.repeats) - len(ctx.input_shape)) + ctx.input_shape
        interleaved = []
        for repeat, size in zip(ctx.repeats, padded_shape):
            interleaved.extend((repeat, size))
        grad_x = grad_output.reshape(interleaved)
        for dim in range(2 * len(ctx.repeats) - 2, -1, -2):
            grad_x = ops.sum(grad_x, dim=dim)
        return grad_x.reshape(ctx.input_shape), None

@register_base_op("flatten")
def dt_flatten(ops, x, start_dim=0, end_dim=-1):
    return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)

class DTFlattenFunction(DTFunction):

    @staticmethod
    def forward(ops, x, start_dim=0, end_dim=-1):
        return ops.flatten(x, start_dim, end_dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, _, _ = inputs
        ctx.original_shape = x.shape

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = grad_output.reshape(ctx.original_shape)
        return grad_x, None, None

@register_base_op("reshape")
def dt_reshape(ops, x, shape):
    return torch.reshape(x, shape)

class DTReshapeFunction(DTFunction):

    @staticmethod
    def forward(ops, x, shape):
        return ops.reshape(x, shape)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, _ = inputs
        ctx.original_shape = x.shape

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = grad_output.reshape(ctx.original_shape)
        return grad_x, None


@register_base_op("permute")
def dt_permute(ops, x, dims):
    return x.permute(dims)


class DTPermuteFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dims):
        return ops.permute(x, dims)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dims = inputs
        ctx.inverse = tuple(sorted(range(len(dims)), key=dims.__getitem__))

    @staticmethod
    def backward(ctx, ops, grad_output):
        return grad_output.permute(ctx.inverse), None


@register_base_op("select")
def dt_select(ops, x, dim, index):
    return x.select(dim, index)


class DTSelectFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim, index):
        return ops.select(x, dim, index)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, index = inputs
        ctx.shape = x.shape
        ctx.device = x.device
        ctx.dim = dim
        ctx.index = index

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = ops.zeros(ctx.shape, device=ctx.device)
        grad_x.select(ctx.dim, ctx.index).copy_(grad_output)
        return grad_x, None, None


@register_base_op("narrow")
def dt_narrow(ops, x, dim, start, length):
    return x.narrow(dim, start, length)


class DTNarrowFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim, start, length):
        return ops.narrow(x, dim, start, length)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, start, length = inputs
        ctx.shape = x.shape
        ctx.device = x.device
        ctx.dim = dim
        ctx.start = start
        ctx.length = length

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = ops.zeros(ctx.shape, device=ctx.device)
        grad_x.narrow(ctx.dim, ctx.start, ctx.length).copy_(grad_output)
        return grad_x, None, None, None


@register_base_op("split")
def dt_split(ops, x, split_size_or_sections, dim=0):
    return torch.split(x, split_size_or_sections, dim=dim)


class DTSplitFunction(DTFunction):

    @staticmethod
    def forward(ops, x, split_size_or_sections, dim=0):
        return ops.split(x, split_size_or_sections, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, _, dim = inputs
        ctx.dim = dim
        ctx.shapes = [part.shape for part in output]
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, ops, *grad_outputs):
        device = next((grad.device for grad in grad_outputs if grad is not None), None)
        parts = [
            grad if grad is not None else ops.zeros(shape, device=device)
            for grad, shape in zip(grad_outputs, ctx.shapes)
        ]
        return torch.cat(parts, dim=ctx.dim), None, None


@register_base_op("unbind")
def dt_unbind(ops, x, dim=0):
    return torch.unbind(x, dim=dim)


class DTUnbindFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=0):
        return ops.unbind(x, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim = inputs
        ctx.dim = dim
        ctx.part_shape = output[0].shape if output else x.shape[:dim] + x.shape[dim + 1:]
        ctx.device = x.device
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, ops, *grad_outputs):
        parts = [
            grad if grad is not None else ops.zeros(ctx.part_shape, device=ctx.device)
            for grad in grad_outputs
        ]
        return torch.stack(parts, dim=ctx.dim), None


def _scatter_add_along_dim(ops, destination, dim, index, source):
    dim %= destination.dim()
    result = destination.clone()
    if index.dim() != result.dim() or source.dim() != result.dim():
        raise RuntimeError("index, source, and input must have the same number of dimensions")
    if index.dtype != torch.int64:
        raise RuntimeError("scatter_add index must have dtype torch.int64")
    if any(index.shape[d] > source.shape[d] for d in range(index.dim())):
        raise RuntimeError("scatter_add index shape must not exceed source shape")
    if any(
        index.shape[d] > result.shape[d]
        for d in range(index.dim()) if d != dim
    ):
        raise RuntimeError("scatter_add index shape must not exceed input shape outside dim")
    if index.numel() and torch.any((index < 0) | (index >= result.shape[dim])):
        raise RuntimeError("index out of bounds in scatter_add")
    for index_coord in itertools.product(*(range(size) for size in index.shape)):
        target_coord = list(index_coord)
        target_coord[dim] = int(index[index_coord].item())
        target_coord = tuple(target_coord)
        result[target_coord] = ops.add(result[target_coord], source[index_coord])
    return result


@register_base_op("index_select")
def dt_index_select(ops, x, dim, index):
    return torch.index_select(x, dim, index)


class DTIndexSelectFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim, index):
        return ops.index_select(x, dim, index)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, index = inputs
        ctx.shape = x.shape
        ctx.device = x.device
        ctx.dim = dim
        ctx.save_for_backward(index)

    @staticmethod
    def backward(ctx, ops, grad_output):
        index, = ctx.saved_tensors
        grad_x = ops.zeros(ctx.shape, device=ctx.device)
        grad_x = ops.index_add(grad_x, ctx.dim, index, grad_output)
        return grad_x, None, None


@register_base_op("gather")
def dt_gather(ops, x, dim, index):
    return torch.gather(x, dim, index)


class DTGatherFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim, index):
        return ops.gather(x, dim, index)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, index = inputs
        ctx.shape = x.shape
        ctx.device = x.device
        ctx.dim = dim
        ctx.save_for_backward(index)

    @staticmethod
    def backward(ctx, ops, grad_output):
        index, = ctx.saved_tensors
        grad_x = ops.zeros(ctx.shape, device=ctx.device)
        grad_x = ops.scatter_add(grad_x, ctx.dim, index, grad_output)
        return grad_x, None, None


@register_base_op("take_along_dim")
def dt_take_along_dim(ops, x, indices, dim=None):
    return torch.take_along_dim(x, indices, dim=dim)


class DTTakeAlongDimFunction(DTFunction):

    @staticmethod
    def forward(ops, x, indices, dim=None):
        return ops.take_along_dim(x, indices, dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, indices, dim = inputs
        ctx.shape = x.shape
        ctx.device = x.device
        ctx.dim = dim
        ctx.save_for_backward(indices)

    @staticmethod
    def backward(ctx, ops, grad_output):
        indices, = ctx.saved_tensors
        if ctx.dim is None:
            flat = ops.zeros((math.prod(ctx.shape),), device=ctx.device)
            flat = ops.scatter_add(flat, 0, indices.reshape(-1), grad_output.reshape(-1))
            grad_x = flat.reshape(ctx.shape)
        else:
            grad_x = ops.zeros(ctx.shape, device=ctx.device)
            grad_x = ops.scatter_add(grad_x, ctx.dim, indices, grad_output)
        return grad_x, None, None


@register_base_op("index_add")
def dt_index_add(ops, x, dim, index, source, alpha=1):
    if alpha != 1:
        raise NotImplementedError("DType index_add currently requires alpha=1")
    dim %= x.dim()
    if index.dim() != 1 or index.dtype not in (torch.int64, torch.int32):
        raise RuntimeError("index_add index must be a one-dimensional integer tensor")
    if index.numel() and torch.any((index < 0) | (index >= x.shape[dim])):
        raise IndexError("index out of range in index_add")
    expected = list(x.shape)
    expected[dim] = index.numel()
    if tuple(source.shape) != tuple(expected):
        raise RuntimeError(f"source shape must be {tuple(expected)}, got {tuple(source.shape)}")
    expanded_index = index.reshape(
        *([1] * dim), index.numel(), *([1] * (x.dim() - dim - 1))
    ).expand(source.shape)
    return _scatter_add_along_dim(ops, x, dim, expanded_index, source)


class DTIndexAddFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim, index, source, alpha=1):
        return ops.index_add(x, dim, index, source, alpha)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim, index, _, alpha = inputs
        ctx.dim = dim
        ctx.alpha = alpha
        ctx.save_for_backward(index)

    @staticmethod
    def backward(ctx, ops, grad_output):
        index, = ctx.saved_tensors
        grad_source = ops.index_select(grad_output, ctx.dim, index)
        return grad_output, None, None, grad_source, None


@register_base_op("scatter_add")
def dt_scatter_add(ops, x, dim, index, source):
    return _scatter_add_along_dim(ops, x, dim, index, source)


class DTScatterAddFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim, index, source):
        return ops.scatter_add(x, dim, index, source)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim, index, _ = inputs
        ctx.dim = dim
        ctx.save_for_backward(index)

    @staticmethod
    def backward(ctx, ops, grad_output):
        index, = ctx.saved_tensors
        grad_source = ops.gather(grad_output, ctx.dim, index)
        return grad_output, None, None, grad_source


@register_base_op("masked_fill")
def dt_masked_fill(ops, x, mask, value):
    if mask.dtype is not torch.bool:
        raise TypeError("masked_fill mask must be boolean")
    return torch.where(mask, value, x)


class DTMaskedFillFunction(DTFunction):

    @staticmethod
    def forward(ops, x, mask, value):
        return ops.masked_fill(x, mask, value)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, mask, _ = inputs
        ctx.save_for_backward(mask)

    @staticmethod
    def backward(ctx, ops, grad_output):
        mask, = ctx.saved_tensors
        zero = ops.scalar_from_float(0.0, device=grad_output.device)
        return torch.where(mask, zero, grad_output), None, None
