import torch
from torchdt.autograd import DTFunction
from torchdt.ops import register_base_op

@register_base_op("relu")
def dt_relu(ops, x):
    return torch.where(
        ops.lt(x, ops.scalar_from_float(0.0, device=x.device)),
        ops.scalar_from_float(0.0, device=x.device), x
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
        grad_x = torch.where(
            ops.eq(output, ops.scalar_from_float(0.0, device=output.device)),
            ops.scalar_from_float(0.0, device=output.device), grad_output
        )
        return grad_x

@register_base_op("leaky_relu")
def dt_leaky_relu(ops, x, negative_slope):
    return torch.where(
        ops.lt(x, ops.scalar_from_float(0.0, device=x.device)),
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
        grad_x = torch.where(
            ops.lt(x, ops.scalar_from_float(0.0, device=x.device)),
            ops.mul(grad_output, negative_slope), grad_output
        )
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
        grad_x = torch.where(
            ops.gt(x, threshold), grad_output,
            ops.scalar_from_float(0.0, device=x.device)
        )
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
        grad_x = ops.mul(
            grad_output,
            ops.sub(ops.scalar_from_float(1.0, device=output.device), ops.square(output))
        )
        return grad_x

@register_base_op("sigmoid")
def dt_sigmoid(ops, x):
    exp_neg_x = ops.exp(ops.neg(x))
    one = ops.scalar_from_float(1.0, device=x.device)
    return ops.div(one, ops.add(one, exp_neg_x))

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
        grad_x = ops.mul(
            grad_output,
            ops.mul(output, ops.sub(ops.scalar_from_float(1.0, device=output.device), output))
        )
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
        grad_x = ops.mul(
            grad_output,
            ops.sub(ops.scalar_from_float(1.0, device=x.device), ops.sigmoid(x))
        )
        return grad_x

@register_base_op("softmin")
def dt_softmin(ops, x, dim=-1):
    neg_x = ops.neg(x)
    if x.numel() == 0:
        return x.clone()
    maximum = ops.max(neg_x) if dim is None else ops.max(neg_x, dim=dim, keepdim=True)[0]
    exp_neg_x = ops.exp(ops.sub(neg_x, maximum))
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
    if x.numel() == 0:
        return x.clone()
    maximum = ops.max(x) if dim is None else ops.max(x, dim=dim, keepdim=True)[0]
    exp_x = ops.exp(ops.sub(x, maximum))
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
            ops.scalar_from_float(0.0, device=grad_output.device), grad_output)
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
        grad_b = ops.sub(ops.scalar_from_float(1.0, device=sigmoid_b.device), sigmoid_b)
        grad_b = ops.mul(grad_output, ops.mul(a, ops.mul(sigmoid_b, grad_b)))

        grad_x = torch.cat([grad_a, grad_b], dim=ctx.dim)
        return grad_x, None


@register_base_op("erf")
def dt_erf(ops, x):
    return ops.from_float(torch.erf(ops.to_float(x)))


class DTErfFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.erf(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        coefficient = ops.scalar_from_float(2.0 / torch.pi ** 0.5, device=x.device)
        exponent = ops.neg(ops.square(x))
        return ops.mul(grad_output, ops.mul(coefficient, ops.exp(exponent)))


@register_base_op("gelu")
def dt_gelu(ops, x, approximate="none"):
    half = ops.scalar_from_float(0.5, device=x.device)
    one = ops.scalar_from_float(1.0, device=x.device)
    if approximate == "none":
        inv_sqrt_two = ops.scalar_from_float(2.0 ** -0.5, device=x.device)
        return ops.mul(ops.mul(half, x), ops.add(one, ops.erf(ops.mul(x, inv_sqrt_two))))
    if approximate != "tanh":
        raise ValueError("approximate must be 'none' or 'tanh'")
    coefficient = ops.scalar_from_float((2.0 / torch.pi) ** 0.5, device=x.device)
    cubic = ops.mul(ops.scalar_from_float(0.044715, device=x.device), ops.mul(ops.square(x), x))
    inner = ops.mul(coefficient, ops.add(x, cubic))
    return ops.mul(ops.mul(half, x), ops.add(one, ops.tanh(inner)))


class DTGeluFunction(DTFunction):

    @staticmethod
    def forward(ops, x, approximate="none"):
        return ops.gelu(x, approximate)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, approximate = inputs
        ctx.save_for_backward(x)
        ctx.approximate = approximate

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        half = ops.scalar_from_float(0.5, device=x.device)
        one = ops.scalar_from_float(1.0, device=x.device)
        if ctx.approximate == "none":
            inv_sqrt_two = ops.scalar_from_float(2.0 ** -0.5, device=x.device)
            cdf = ops.mul(half, ops.add(one, ops.erf(ops.mul(x, inv_sqrt_two))))
            density_scale = ops.scalar_from_float((2.0 * torch.pi) ** -0.5, device=x.device)
            density = ops.mul(density_scale, ops.exp(ops.mul(
                ops.scalar_from_float(-0.5, device=x.device), ops.square(x)
            )))
            derivative = ops.add(cdf, ops.mul(x, density))
        else:
            coefficient = ops.scalar_from_float((2.0 / torch.pi) ** 0.5, device=x.device)
            cubic_coefficient = ops.scalar_from_float(0.044715, device=x.device)
            inner = ops.mul(coefficient, ops.add(x, ops.mul(cubic_coefficient, ops.mul(ops.square(x), x))))
            tanh_inner = ops.tanh(inner)
            inner_derivative = ops.mul(coefficient, ops.add(
                one,
                ops.mul(ops.scalar_from_float(3.0 * 0.044715, device=x.device), ops.square(x)),
            ))
            derivative = ops.add(
                ops.mul(half, ops.add(one, tanh_inner)),
                ops.mul(ops.mul(half, x), ops.mul(
                    ops.sub(one, ops.square(tanh_inner)), inner_derivative
                )),
            )
        return ops.mul(grad_output, derivative), None


@register_base_op("silu")
def dt_silu(ops, x):
    return ops.mul(x, ops.sigmoid(x))


class DTSiluFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.silu(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        sigmoid = ops.sigmoid(x)
        one = ops.scalar_from_float(1.0, device=x.device)
        derivative = ops.mul(sigmoid, ops.add(one, ops.mul(x, ops.sub(one, sigmoid))))
        return ops.mul(grad_output, derivative)


@register_base_op("softplus")
def dt_softplus(ops, x, beta=1.0, threshold=20.0):
    beta_value = ops.scalar_from_float(beta, device=x.device)
    threshold_value = ops.scalar_from_float(threshold, device=x.device)
    one = ops.scalar_from_float(1.0, device=x.device)
    scaled = ops.mul(x, beta_value)
    unthresholded = ops.div(ops.log(ops.add(one, ops.exp(scaled))), beta_value)
    return torch.where(ops.gt(scaled, threshold_value), x, unthresholded)


class DTSoftplusFunction(DTFunction):

    @staticmethod
    def forward(ops, x, beta=1.0, threshold=20.0):
        return ops.softplus(x, beta, threshold)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, beta, threshold = inputs
        ctx.save_for_backward(x)
        ctx.beta = beta
        ctx.threshold = threshold

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        beta = ops.scalar_from_float(ctx.beta, device=x.device)
        threshold = ops.scalar_from_float(ctx.threshold, device=x.device)
        scaled = ops.mul(x, beta)
        one = ops.scalar_from_float(1.0, device=x.device)
        derivative = torch.where(ops.gt(scaled, threshold), one, ops.sigmoid(scaled))
        return ops.mul(grad_output, derivative), None, None


@register_base_op("mish")
def dt_mish(ops, x):
    return ops.mul(x, ops.tanh(ops.softplus(x)))


class DTMishFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.mish(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        one = ops.scalar_from_float(1.0, device=x.device)
        softplus = ops.softplus(x)
        tanh_softplus = ops.tanh(softplus)
        derivative = ops.add(
            tanh_softplus,
            ops.mul(x, ops.mul(
                ops.sub(one, ops.square(tanh_softplus)), ops.sigmoid(x)
            )),
        )
        return ops.mul(grad_output, derivative)
