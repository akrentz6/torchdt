import torch
from torch import Tensor, CharTensor, ShortTensor, IntTensor, LongTensor
from typing import Union, Callable

InternalTensor = Union[CharTensor, ShortTensor, IntTensor, LongTensor]

def register_op(dtype_cls: type, method: str) -> Callable:
    """Decorator to register an operation for a given DType subclass."""
    def decorator(func: Callable) -> Callable:
        ops_cls = dtype_cls.ops
        if not hasattr(ops_cls, method):
            raise ValueError(f"{ops_cls.__name__} has no method '{method}' to register.")
        setattr(ops_cls, method, classmethod(func))
        return func
    return decorator

def register_base_op(method: str) -> Callable:
    """Decorator to register a base operation."""
    def decorator(func: Callable) -> Callable:
        if not hasattr(OpsBase, method):
            raise ValueError(f"OpsBase has no method '{method}' to register.")
        setattr(OpsBase, method, classmethod(func))
        return func
    return decorator

class OpsBase:

    # ========== Useful helper functions ==========

    @classmethod
    def from_float(cls, x):
        return cls.dtype(x)._int

    @classmethod
    def to_float(cls, x):
        return cls.dtype.to_float(x)

    @classmethod
    def zeros(cls, size):
        return torch.full(size, cls.from_float(0.0), dtype=cls.dtype.int_dtype)

    @classmethod
    def zeros_like(cls, x):
        return torch.full_like(x, cls.from_float(0.0), dtype=cls.dtype.int_dtype)

    @classmethod
    def ones(cls, size):
        return torch.full(size, cls.from_float(1.0), dtype=cls.dtype.int_dtype)

    @classmethod
    def ones_like(cls, x):
        return torch.full_like(x, cls.from_float(1.0), dtype=cls.dtype.int_dtype)

    @classmethod
    def full(cls, size, fill_value):
        return torch.full(size, cls.from_float(fill_value), dtype=cls.dtype.int_dtype)

    @classmethod
    def full_like(cls, x, fill_value):
        return torch.full_like(x, cls.from_float(fill_value), dtype=cls.dtype.int_dtype)

    # ========== Operations to be implemented by subclasses ==========

    @classmethod
    def add(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def sub(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def mul(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def div(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def pow(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def sign(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def neg(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def ge(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def gt(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def le(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def lt(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    # ========== 'Base' operations with default implementations ==========

    @classmethod
    def square(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError