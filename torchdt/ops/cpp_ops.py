try:
    import torchdt._C as C
except ImportError:
    C = None
from torchdt.autograd import DTFunction
import torch

def register_cpp_ops(dtype_cls: type, backend: str) -> None:
    if C is None:
        raise ImportError("C++ extension is not built. Please build the C++ extension to use C++ backend.")

    bitwidth = dtype_cls.bitwidth

    dtype_cls.register_op("from_float")(lambda ops, x: from_float(backend, bitwidth, ops, x))
    dtype_cls.register_op("to_float")(lambda ops, x: to_float(backend, bitwidth, ops, x))
    dtype_cls.register_op("add")(lambda ops, x, y: add_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("sub")(lambda ops, x, y: sub_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("mul")(lambda ops, x, y: mul_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("div")(lambda ops, x, y: div_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("ge")(lambda ops, x, y: ge_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("gt")(lambda ops, x, y: gt_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("le")(lambda ops, x, y: le_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("lt")(lambda ops, x, y: lt_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("sum")(lambda ops, x, dim=None, keepdim=False: sum_op(backend, bitwidth, ops, x, dim, keepdim))
    dtype_cls.register_op("matmul")(
        lambda ops, A, B: matmul_op(backend, bitwidth, ops, A, B)
    )
    dtype_cls.register_op("matmul_backward")(
        lambda ops, grad_output, A, B: matmul_backward_op(backend, bitwidth, ops, grad_output, A, B)
    )
    dtype_cls.register_op("conv2d")(
        lambda ops, input, weight, bias, stride, padding, dilation, groups: conv2d_op(
            backend, bitwidth, ops, input, weight, bias, stride, padding, dilation, groups
        )
    )
    dtype_cls.register_op("conv2d_backward")(
        lambda ops, grad_output, input, weight, stride, padding, dilation, has_bias, groups: conv2d_backward_op(
            backend, bitwidth, ops, grad_output, input, weight, stride, padding, dilation, has_bias, groups
        )
    )

    # also register new torch. funcs to call ops that call into c++ for backward
    dtype_cls.register_func(torch.matmul, torch.Tensor.matmul, cast=("input", "other"))(matmul_func)
    dtype_cls.register_func(torch.nn.functional.conv2d, cast=("input", "weight", "bias"))(conv2d_func)

def from_float(backend, bitwidth, _, x):
    return C.from_float(backend, bitwidth, x)

def to_float(backend, bitwidth, _, x):
    return C.to_float(backend, bitwidth, x)

def add_op(backend, bitwidth, _, x, y):
    return C.add(backend, bitwidth, x, y)

def sub_op(backend, bitwidth, _, x, y):
    return C.sub(backend, bitwidth, x, y)

def mul_op(backend, bitwidth, _, x, y):
    return C.mul(backend, bitwidth, x, y)

def div_op(backend, bitwidth, _, x, y):
    return C.div(backend, bitwidth, x, y)

def ge_op(backend, bitwidth, _, x, y):
    return C.ge(backend, bitwidth, x, y)

def gt_op(backend, bitwidth, _, x, y):
    return C.gt(backend, bitwidth, x, y)

def le_op(backend, bitwidth, _, x, y):
    return C.le(backend, bitwidth, x, y)

def lt_op(backend, bitwidth, _, x, y):
    return C.lt(backend, bitwidth, x, y)

def sum_op(backend, bitwidth, _, x, dim=None, keepdim=False):
    if dim is None:
        dim = []
    elif isinstance(dim, int):
        dim = [dim]
    return C.sum(backend, bitwidth, x, dim, keepdim)

def matmul_op(backend, bitwidth, _, A, B):
    return C.matmul(backend, bitwidth, A, B)

def matmul_backward_op(backend, bitwidth, _, grad_output, A, B):
    return C.matmul_backward(backend, bitwidth, grad_output, A, B)

class DTMatmulFunction(DTFunction):

    @staticmethod
    def forward(ops, A, B):
        return ops.matmul(A, B)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        A, B = inputs
        ctx.save_for_backward(A, B)

    @staticmethod
    def backward(ops, ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A, grad_B = ops.matmul_backward(grad_output, A, B)
        return grad_A, grad_B

def matmul_func(input, other, *, out=None):
    result = DTMatmulFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

def conv2d_op(backend, bitwidth, _, input, weight, bias, stride, padding, dilation, groups):
    return C.conv2d(backend, bitwidth, input, weight, bias, stride, padding, dilation, groups)

def conv2d_backward_op(backend, bitwidth, _, grad_output, input, weight, stride, padding, dilation, has_bias, groups):
    return C.conv2d_backward(backend, bitwidth, grad_output, input, weight, stride, padding, dilation, has_bias, groups)

class DTConv2dFunction(DTFunction):

    @staticmethod
    def forward(ops, input, weight, bias, stride, padding, dilation, groups):
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)
        return ops.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        input, weight, bias, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(input, weight)
        ctx.stride = (stride, stride) if isinstance(stride, int) else stride
        ctx.padding = (padding, padding) if isinstance(padding, int) else padding
        ctx.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        ctx.has_bias = bias is not None
        ctx.groups = groups

    @staticmethod
    def backward(ctx, ops, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = ops.conv2d_backward(
            grad_output, input, weight,
            ctx.stride, ctx.padding, ctx.dilation,
            ctx.has_bias, ctx.groups
        )
        return grad_input, grad_weight, grad_bias if ctx.has_bias else None, None, None, None, None

def conv2d_func(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, out=None):
    result = DTConv2dFunction.apply(input, weight, bias, stride, padding, dilation, groups)

    if out is not None:
        return out.copy_(result)
    return result