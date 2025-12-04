import torch
from torchdt.autograd import DTFunction
from torchdt.ops import register_base_op

class DTSignFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.sign(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        pass

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = ops.zeros_like(grad_output)
        return grad_x

@register_base_op("neg")
def dt_neg(ops, x):
    return ops.mul(ops.scalar_from_float(-1.0), x)

class DTNegFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.neg(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        pass

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = ops.neg(grad_output)
        return grad_x

@register_base_op("abs")
def dt_abs(ops, x):
    return ops.mul(ops.sign(x), x)

class DTAbsFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.abs(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        x_less_than_zero = ops.lt(x, ops.scalar_from_float(0.0))
        grad_x = torch.where(x_less_than_zero, ops.neg(grad_output), grad_output)

        return grad_x