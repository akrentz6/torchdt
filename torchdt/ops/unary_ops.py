from torchdt.autograd import DTFunction

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