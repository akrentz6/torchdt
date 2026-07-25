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
    handle = C.get_backend(backend, bitwidth)

    for method in ("from_float", "to_float", "add", "sub", "mul", "div",
                   "ge", "gt", "le", "lt", "matmul", "matmul_backward",
                   "conv2d", "conv2d_backward"):
        dtype_cls.register_op(method, backend="cpp", direct=True)(getattr(handle, method))

    def cpp_sum(x, dim=None, keepdim=False):
        if isinstance(dim, int):
            dim = [dim]
        return handle.sum(x, dim, keepdim)

    dtype_cls.register_op("sum", backend="cpp", direct=True)(cpp_sum)

    # also register new torch. funcs to call ops that call into c++ for backward
    dtype_cls.register_func(
        torch.matmul, torch.Tensor.matmul,
        cast=("input", "other"), backend="cpp"
    )(matmul_func)
    dtype_cls.register_func(
        torch.nn.functional.conv2d,
        cast=("input", "weight", "bias"), backend="cpp"
    )(conv2d_func)

    dtype_cls.ops.enable_backend("cpp", "cpu")

class DTMatmulFunction(DTFunction):

    @staticmethod
    def forward(ops, A, B):
        return ops.matmul(A, B)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        A, B = inputs
        ctx.save_for_backward(A, B)

    @staticmethod
    def backward(ctx, ops, grad_output):
        A, B = ctx.saved_tensors
        grad_A, grad_B = ops.matmul_backward(grad_output, A, B)
        return grad_A, grad_B

def matmul_func(input, other, *, out=None):
    result = DTMatmulFunction.apply(input, other)

    if out is not None:
        return out.copy_(result)
    return result

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
