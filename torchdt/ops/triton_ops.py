import hashlib
import inspect
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Callable, Optional

import torch


TRITON_IMPORT_ERROR = "Triton is not installed. Please install Triton to use Triton backend."


@dataclass(frozen=True)
class TritonScalarOps:
    """Scalar Triton functions required by the generic torchdt kernels."""

    from_float: Callable
    to_float: Callable
    add: Callable
    sub: Optional[Callable] = None
    mul: Optional[Callable] = None
    div: Optional[Callable] = None
    sqrt: Optional[Callable] = None
    gt: Optional[Callable] = None
    ge: Optional[Callable] = None
    lt: Optional[Callable] = None
    le: Optional[Callable] = None
    neg: Optional[Callable] = None
    exp: Optional[Callable] = None
    log: Optional[Callable] = None
    clamp: Optional[Callable] = None
    sign: Optional[Callable] = None


@dataclass(frozen=True)
class TritonAccumulatorOps:
    """Higher-precision scalar ops plus conversions to and from storage values."""

    int_dtype: torch.dtype
    scalar_ops: TritonScalarOps
    to_accumulator: Callable
    from_accumulator: Callable


def is_triton_available() -> bool:
    return find_spec("triton") is not None


HAS_TRITON = is_triton_available()


def require_triton():
    if not is_triton_available():
        raise ImportError(TRITON_IMPORT_ERROR)

    import triton
    import triton.language as tl

    return triton, tl


def _jit_with_captured_constants(fn):
    """
    JIT a scalar-op closure with its compile-time configuration in the cache key.
    Triton hashes source and called JIT functions, but not captured values.
    """
    triton, tl = require_triton()
    if triton.knobs.runtime.interpret:
        return triton.jit(fn)

    class CapturedConstantsJITFunction(triton.runtime.JITFunction):
        def __repr__(self):
            # Triton stringifies explicitly annotated constexpr arguments
            # for its in-memory launch cache, bypassing their cache_key.
            return f"{super().__repr__()}[{self.cache_key}]"

        @property
        def cache_key(self):
            native_key = super().cache_key
            captures = inspect.getclosurevars(self.fn).nonlocals
            constants = tuple(sorted(
                (name, repr(value)) for name, value in captures.items()
                if isinstance(value, (tl.constexpr, tl.dtype))
            ))
            return hashlib.sha256(repr((native_key, constants)).encode()).hexdigest()

    return CapturedConstantsJITFunction(fn)


def register_triton_ops(
    dtype_cls: type,
    scalar_ops: TritonScalarOps,
    accumulator_ops: Optional[TritonAccumulatorOps] = None,
) -> None:
    """Register generic Triton kernels for a dtype."""
    from torchdt.ops._triton import register_triton_ops as _register_triton_ops

    return _register_triton_ops(dtype_cls, scalar_ops, accumulator_ops)
