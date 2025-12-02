import torchdt._C as C
from torchdt.autograd import DTFunction
import torch

def register_cpp_ops(dtype_cls: type, backend: str) -> None:
    bitwidth = dtype_cls.bitwidth

    dtype_cls.register_op("add")(lambda ops, x, y: add_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("sub")(lambda ops, x, y: sub_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("mul")(lambda ops, x, y: mul_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("div")(lambda ops, x, y: div_op(backend, bitwidth, ops, x, y))
    dtype_cls.register_op("sum")(lambda ops, x, dim=None, keepdim=False: sum_op(backend, bitwidth, ops, x, dim, keepdim))
    dtype_cls.register_op("matmul")(lambda ops, A, B: matmul_op(backend, bitwidth, ops, A, B))
    dtype_cls.register_op("matmul_backward")(lambda ops, grad_output, A, B: matmul_backward_op(backend, bitwidth, ops, grad_output, A, B))

    # also register new torch. funcs to call ops that call into c++ for backward
    dtype_cls.register_func(torch.matmul, torch.Tensor.matmul, cast=("input", "other"))(matmul_func)

def add_op(backend, bitwidth, _, x, y):
    return C.add(backend, bitwidth, x, y)

def sub_op(backend, bitwidth, _, x, y):
    return C.sub(backend, bitwidth, x, y)

def mul_op(backend, bitwidth, _, x, y):
    return C.mul(backend, bitwidth, x, y)

def div_op(backend, bitwidth, _, x, y):
    return C.div(backend, bitwidth, x, y)

def sum_op(backend, bitwidth, _, x, dim=None, keepdim=False):
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