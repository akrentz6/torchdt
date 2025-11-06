import torch
from torchdt.autograd import DTFunction, DTNonDifferentiableFunction
from torchdt.ops import register_base_op

@register_base_op("equal")
def dt_equal(ops, x, y):
    return torch.equal(x, y)

class DTEqualFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y):
        return ops.equal(x, y)

@register_base_op("eq")
def dt_eq(ops, x, y):
    return torch.eq(x, y)

class DTEqFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y):
        return ops.eq(x, y)

@register_base_op("ne")
def dt_ne(ops, x, y):
    return torch.ne(x, y)

class DTNeFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y):
        return ops.ne(x, y)

class DTGeFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y):
        return ops.ge(x, y)

class DTGtFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y):
        return ops.gt(x, y)

class DTLeFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y):
        return ops.le(x, y)

class DTLtFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y):
        return ops.lt(x, y)

@register_base_op("isclose")
def dt_isclose(ops, x, y, rtol, atol):
    abs_diff = ops.abs(ops.sub(x, y))
    eps = ops.add(atol, ops.mul(rtol, ops.abs(y)))
    return ops.le(abs_diff, eps)

class DTIsCloseFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y, rtol, atol):
        return ops.isclose(x, y, rtol, atol)

@register_base_op("allclose")
def dt_allclose(ops, x, y, rtol, atol):
    is_close = ops.isclose(x, y, rtol, atol)
    return torch.all(is_close)

class DTAllcloseFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y, rtol, atol):
        return ops.allclose(x, y, rtol, atol)

@register_base_op("any")
def dt_any(ops, x, dim=None, keepdim=False):
    return torch.any(ops.ne(x, ops.from_float(0.0)), dim=dim, keepdim=keepdim)

class DTAnyFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return ops.any(x, dim=dim, keepdim=keepdim)

@register_base_op("all")
def dt_all(ops, x, dim=None, keepdim=False):
    return torch.all(ops.ne(x, ops.from_float(0.0)), dim=dim, keepdim=keepdim)

class DTAllFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return ops.all(x, dim=dim, keepdim=keepdim)

@register_base_op("isin")
def dt_isin(ops, x, y, assume_unique=False, invert=False):
    return torch.isin(x, y, assume_unique=assume_unique, invert=invert)

class DTIsinFunction(DTNonDifferentiableFunction):

    output_indices = []

    @staticmethod
    def forward(ops, x, y, assume_unique=False, invert=False):
        return ops.isin(x, y, assume_unique=assume_unique, invert=invert)

@register_base_op("maximum")
def dt_maximum(ops, x, y):
    x_greater_y = ops.gt(x, y)
    return ops.where(x_greater_y, x, y)

class DTMaximumFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.maximum(x, y)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        x_y_equal = ops.eq(x, y)
        half_grad_output = ops.mul(grad_output, ops.from_float(0.5))

        grad_x = torch.where(x_y_equal, half_grad_output, torch.where(
            ops.gt(x, y), grad_output, ops.from_float(0.0)
        ))
        grad_y = torch.where(x_y_equal, half_grad_output, torch.where(
            ops.gt(y, x), grad_output, ops.from_float(0.0)
        ))

        grad_x = ops.sum_to_size(grad_x, x.shape)
        grad_y = ops.sum_to_size(grad_y, y.shape)

        return grad_x, grad_y

@register_base_op("minimum")
def dt_minimum(ops, x, y):
    x_less_y = ops.lt(x, y)
    return ops.where(x_less_y, x, y)

class DTMinimumFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.minimum(ops, x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        x_y_equal = ops.eq(x, y)
        half_grad_output = ops.mul(grad_output, ops.from_float(0.5))

        grad_x = torch.where(x_y_equal, half_grad_output, torch.where(
            ops.lt(x, y), grad_output, ops.from_float(0.0)
        ))
        grad_y = torch.where(x_y_equal, half_grad_output, torch.where(
            ops.lt(y, x), grad_output, ops.from_float(0.0)
        ))

        grad_x = ops.sum_to_size(grad_x, x.shape)
        grad_y = ops.sum_to_size(grad_y, y.shape)

        return grad_x, grad_y

@register_base_op("clamp")
def dt_clamp(ops, x, min=None, max=None):
    result = x.clone()

    if min is not None:
        lt_mask = ops.lt(result, min)
        result = torch.where(lt_mask, min, result)

    if max is not None:
        gt_mask = ops.gt(result, max)
        result = torch.where(gt_mask, max, result)

    return result

class DTClampFunction(DTFunction):

    @staticmethod
    def forward(ops, x, min=None, max=None):
        return ops.clamp(ops, x, min, max)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, min, max = inputs
        ctx.save_for_backward(x, min, max)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, min, max = ctx.saved_tensors

        grad_x = grad_output.clone()
        if min is not None:
            lt_mask = ops.lt(x, min)
            grad_x = torch.where(lt_mask, ops.from_float(0.0), grad_x)

        if max is not None:
            gt_mask = ops.gt(x, max)
            grad_x = torch.where(gt_mask, ops.from_float(0.0), grad_x)

        return grad_x, None, None