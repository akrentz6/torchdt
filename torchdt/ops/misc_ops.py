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
        ctx.save_for_backward(x)
        ctx.shape = shape

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        # Sum over the broadcasted dimensions
        # First, handle prepended dimensions (when original tensor had fewer dims)
        ndims_added = grad_output.ndim - len(x.shape)
        grad_x = grad_output
        for i in range(ndims_added):
            grad_x = ops.sum(grad_x, dim=0, keepdim=False)

        # Then, handle expanded dimensions (where original dim was 1)
        for i, (orig_size, grad_size) in enumerate(zip(x.shape, grad_x.shape)):
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
        ctx.save_for_backward(x)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.view(x.shape)
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
        x, dim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
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
        ctx.save_for_backward(*tensors)
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
        for g, shape in zip(grad_outputs, ctx.out_shapes):
            if g is None:
                g = ops.zeros(*shape)
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
        grad_x = ops.sum_to_size(torch.where(condition, grad_output, ops.from_float(0.0)), x.shape)
        grad_y = ops.sum_to_size(torch.where(condition, ops.from_float(0.0), grad_output), y.shape)
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
