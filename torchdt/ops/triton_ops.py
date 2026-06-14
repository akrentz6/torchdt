from dataclasses import dataclass
from importlib.util import find_spec
from typing import Callable, Optional


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

    def to_kwargs(self) -> dict:
        return {
            "from_float": self.from_float,
            "to_float": self.to_float,
            "add": self.add,
            "sub": self.sub,
            "mul": self.mul,
            "div": self.div,
            "sqrt": self.sqrt,
            "gt": self.gt,
            "ge": self.ge,
            "lt": self.lt,
            "le": self.le,
            "neg": self.neg,
            "exp": self.exp,
            "log": self.log,
            "clamp": self.clamp,
            "sign": self.sign,
        }


def is_triton_available() -> bool:
    return find_spec("triton") is not None


HAS_TRITON = is_triton_available()


def require_triton():
    if not is_triton_available():
        raise ImportError(TRITON_IMPORT_ERROR)

    import triton
    import triton.language as tl

    return triton, tl


def register_triton_ops(dtype_cls: type, backend: Optional[TritonScalarOps] = None, *args, **kwargs) -> None:
    """Register generic Triton kernels for a dtype.

    User-defined dtypes can pass a TritonScalarOps object. The old positional
    argument style is still accepted for compatibility.
    """
    from ._triton_ops_impl import register_triton_ops as _register_triton_ops

    if isinstance(backend, TritonScalarOps):
        backend_kwargs = backend.to_kwargs()
        backend_kwargs.update(kwargs)
        return _register_triton_ops(dtype_cls, **backend_kwargs)

    if backend is not None:
        args = (backend, *args)

    return _register_triton_ops(dtype_cls, *args, **kwargs)
