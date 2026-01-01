import torch
from torchdt.autograd import DTFunction
from torchdt.ops import register_base_op

@register_base_op("relu")
def dt_relu(ops, x):
    return torch.where(
        ops.lt(x, ops.scalar_from_float(0.0)),
        ops.scalar_from_float(0.0), x
    )

class DTReLUFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.relu(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        grad_x = torch.where(ops.eq(output, ops.scalar_from_float(0.0)), ops.scalar_from_float(0.0), grad_output)
        return grad_x

@register_base_op("leaky_relu")
def dt_leaky_relu(ops, x, negative_slope):
    return torch.where(
        ops.lt(x, ops.scalar_from_float(0.0)),
        ops.mul(x, negative_slope), x
    )

class DTLeakyReLUFunction(DTFunction):

    @staticmethod
    def forward(ops, x, negative_slope):
        return ops.leaky_relu(x, negative_slope)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, negative_slope = inputs
        ctx.save_for_backward(x, negative_slope)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, negative_slope = ctx.saved_tensors
        grad_x = torch.where(ops.lt(x, ops.scalar_from_float(0.0)), ops.mul(grad_output, negative_slope), grad_output)
        return grad_x, None

@register_base_op("threshold")
def dt_threshold(ops, x, threshold, value):
    return torch.where(
        ops.gt(x, threshold),
        x, value
    )

class DTThresholdFunction(DTFunction):

    @staticmethod
    def forward(ops, x, threshold, value):
        return ops.threshold(x, threshold, value)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, threshold, _ = inputs
        ctx.save_for_backward(x, threshold)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = torch.where(ops.gt(x, threshold), grad_output, ops.scalar_from_float(0.0))
        return grad_x, None, None

@register_base_op("tanh")
def dt_tanh(ops, x):
    exp_x = ops.exp(x)
    exp_neg_x = ops.exp(ops.neg(x))
    return ops.div(ops.sub(exp_x, exp_neg_x), ops.add(exp_x, exp_neg_x))

class DTTanhFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.tanh(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        grad_x = ops.mul(grad_output, ops.sub(ops.scalar_from_float(1.0), ops.square(output)))
        return grad_x

@register_base_op("sigmoid")
def dt_sigmoid(ops, x):
    exp_neg_x = ops.exp(ops.neg(x))
    return ops.div(ops.scalar_from_float(1.0), ops.add(ops.scalar_from_float(1.0), exp_neg_x))

class DTSigmoidFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.sigmoid(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        grad_x = ops.mul(grad_output, ops.mul(output, ops.sub(ops.scalar_from_float(1.0), output)))
        return grad_x

@register_base_op("logsigmoid")
def dt_logsigmoid(ops, x):
    return ops.log(ops.sigmoid(x))

class DTLogSigmoidFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.logsigmoid(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        grad_x = ops.mul(grad_output, ops.sub(ops.scalar_from_float(1.0), ops.sigmoid(x)))
        return grad_x

@register_base_op("softmin")
def dt_softmin(ops, x, dim=-1):
    exp_neg_x = ops.exp(ops.neg(x))
    sum_exp_neg_x = ops.sum(exp_neg_x, dim=dim, keepdim=True)
    return ops.div(exp_neg_x, sum_exp_neg_x)

class DTSoftminFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=None):
        return ops.softmin(x, dim=dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim = inputs
        ctx.save_for_backward(output)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        dot_product = ops.sum(ops.mul(grad_output, output), dim=ctx.dim, keepdim=True)
        grad_x = ops.mul(output, ops.sub(grad_output, dot_product))
        return grad_x, None

@register_base_op("softmax")
def dt_softmax(ops, x, dim=None):
    exp_x = ops.exp(x)
    sum_exp_x = ops.sum(exp_x, dim=dim, keepdim=True)
    return ops.div(exp_x, sum_exp_x)

class DTSoftmaxFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=None):
        return ops.softmax(x, dim=dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim = inputs
        ctx.save_for_backward(output)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        dot_product = ops.sum(ops.mul(grad_output, output), dim=ctx.dim, keepdim=True)
        grad_x = ops.mul(output, ops.sub(grad_output, dot_product))
        return grad_x, None

@register_base_op("log_softmax")
def dt_log_softmax(ops, x, dim=None):
    if dim is None:
        m = ops.max(x)
    else:
        m = ops.max(x, dim=dim, keepdim=True)[0] # discard indices

    # subtract the max to prevent overflow (logsumexp trick)
    x_sub_m = ops.sub(x, m)
    log_sum_exp_x_sub_m = ops.log(ops.sum(ops.exp(x_sub_m), dim=dim, keepdim=True))
    return ops.sub(x_sub_m, log_sum_exp_x_sub_m)

class DTLogSoftmaxFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=None):
        return ops.log_softmax(x, dim=dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim = inputs
        ctx.save_for_backward(output)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        sum_grad = ops.sum(grad_output, dim=ctx.dim, keepdim=True)
        grad_x = ops.sub(grad_output, ops.mul(ops.exp(output), sum_grad))
        return grad_x, None

@register_base_op("hardtanh")
def dt_hardtanh(ops, x, min_val=-1.0, max_val=1.0):
    result = torch.where(ops.lt(x, min_val), min_val, x)
    result = torch.where(ops.gt(result, max_val), max_val, result)
    return result

class DTHardtanhFunction(DTFunction):

    @staticmethod
    def forward(ops, x, min_val, max_val):
        return ops.hardtanh(x, min_val, max_val)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, min_val, max_val = inputs
        ctx.save_for_backward(x, min_val, max_val)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, min_val, max_val = ctx.saved_tensors
        grad_x = torch.where(
            ops.le(x, min_val) | ops.ge(x, max_val),
            ops.scalar_from_float(0.0), grad_output)
        return grad_x, None, None

@register_base_op("glu")
def dt_glu(ops, x, dim=-1):
    half_size = x.size(dim) // 2
    a = x.narrow(dim, 0, half_size)
    b = x.narrow(dim, half_size, half_size)
    return ops.mul(a, ops.sigmoid(b))

class DTGluFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=-1):
        return ops.glu(x, dim=dim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        half_size = x.size(ctx.dim) // 2
        a = x.narrow(ctx.dim, 0, half_size)
        b = x.narrow(ctx.dim, half_size, half_size)

        sigmoid_b = ops.sigmoid(b)
        grad_a = ops.mul(grad_output, sigmoid_b)
        grad_b = ops.sub(ops.scalar_from_float(1.0), sigmoid_b)
        grad_b = ops.mul(grad_output, ops.mul(a, ops.mul(sigmoid_b, grad_b)))

        grad_x = torch.cat([grad_a, grad_b], dim=ctx.dim)
        return grad_x, None