from typing import Optional

from torchdt.ops.triton_ops import TritonAccumulatorOps, TritonScalarOps
from torchdt.ops._triton.context import create_registration_context
from torchdt.ops._triton.convolution import register_ops as register_convolution_ops
from torchdt.ops._triton.elementwise import register_ops as register_elementwise_ops
from torchdt.ops._triton.loss import register_ops as register_loss_ops
from torchdt.ops._triton.matmul import register_ops as register_matmul_ops
from torchdt.ops._triton.normalization import register_ops as register_normalization_ops
from torchdt.ops._triton.optimizers import register_ops as register_optimizer_ops
from torchdt.ops._triton.pooling import register_ops as register_pooling_ops
from torchdt.ops._triton.reductions import register_ops as register_reduction_ops


_REGISTRARS = (
    register_elementwise_ops,
    register_reduction_ops,
    register_matmul_ops,
    register_convolution_ops,
    register_pooling_ops,
    register_normalization_ops,
    register_loss_ops,
    register_optimizer_ops,
)


def register_triton_ops(
    dtype_cls: type,
    scalar_ops: TritonScalarOps,
    accumulator_ops: Optional[TritonAccumulatorOps] = None,
) -> None:
    """Register generic Triton kernels for one dtype configuration."""
    context = create_registration_context(dtype_cls, scalar_ops, accumulator_ops)
    for registrar in _REGISTRARS:
        registrar(context)
    dtype_cls.ops.enable_backend("triton", "cuda")
