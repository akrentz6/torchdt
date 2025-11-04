import torch
from torchdt.autograd import DTFunction
from torchdt.ops import register_base_op

class DTAddFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.add(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.sum_to_size(grad_output, x.shape)
        grad_y = ops.sum_to_size(grad_output, y.shape)

        return grad_x, grad_y

class DTSubFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.sub(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.sum_to_size(grad_output, x.shape)
        grad_y = ops.sum_to_size(ops.neg(grad_output), y.shape)

        return grad_x, grad_y

@register_base_op("sum")
def dt_sum(ops, x, dim=None, keepdim=False):
    if dim is None:
        flat = x.reshape(-1)
        out = flat[0]

        for i in range(1, flat.numel()):
            out = ops.add(out, flat[i])

        if keepdim:
            out = out.reshape([1] * x.dim())

        return out

    red_dims = (dim,) if isinstance(dim, int) else tuple(dim)
    red_dims = tuple(sorted(d % x.dim() for d in red_dims))

    permute_order = [d for d in range(x.dim()) if d not in red_dims] + list(red_dims)
    transposed = x.permute(*permute_order)

    outer_shape = transposed.shape[:-len(red_dims)]
    transposed = transposed.reshape(*outer_shape, -1)

    out = transposed[..., 0]
    for i in range(1, transposed.shape[-1]):
        out = ops.add(out, transposed[..., i])

    if keepdim:
        for d in red_dims:
            out = out.unsqueeze(d)

    return out

class DTSumFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return ops.sum(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        grad_x = grad_output
        if ctx.dim is None:
            grad_x = grad_x.expand(x.shape)

        else:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(d % x.dim() for d in red_dims)

            if not ctx.keepdim:
                for d in sorted(red_dims):
                    grad_x = grad_x.unsqueeze(d)

            grad_x = grad_x.expand(x.shape)

        return grad_x, None, None

class DTMulFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.mul(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.sum_to_size(ops.mul(grad_output, y), x.shape)
        grad_y = ops.sum_to_size(ops.mul(grad_output, x), y.shape)

        return grad_x, grad_y

class DTDivFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.div(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.sum_to_size(ops.div(grad_output, y), x.shape)
        grad_y = ops.sum_to_size(ops.neg(ops.div(ops.mul(grad_output, x), ops.mul(y, y))), y.shape)

        return grad_x, grad_y

class DTPowFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.pow(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y, output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, output = ctx.saved_tensors

        grad_x = ops.mul(grad_output, ops.mul(ops.div(output, x), y))
        grad_y = ops.mul(grad_output, ops.mul(output, ops.log(x)))

        return grad_x, grad_y

@register_base_op("square")
def dt_square(ops, x):
    return ops.mul(x, x)

class DTSquareFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.square(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        grad_x = ops.mul(grad_output, ops.mul(ops.from_float(2), x))
        return grad_x