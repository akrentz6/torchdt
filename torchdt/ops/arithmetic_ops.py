import torch
from torchdt.autograd import DTFunction
from torchdt.ops import register_base_op

class DTAddFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.add(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        pass

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = grad_output
        grad_y = grad_output

        return grad_x, grad_y

class DTSubFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.sub(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        pass

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = grad_output
        grad_y = ops.neg(grad_output)

        return grad_x, grad_y

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

        grad_x = ops.mul(grad_output, y)
        grad_y = ops.mul(grad_output, x)

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

        grad_x = ops.div(grad_output, y)
        grad_y = ops.neg(ops.div(ops.mul(grad_output, x), ops.mul(y, y)))

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