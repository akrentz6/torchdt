from torch import CharTensor, ShortTensor, IntTensor, LongTensor
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


    # ========== 'Base' operations with default implementations ==========

    @classmethod
    def square(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError