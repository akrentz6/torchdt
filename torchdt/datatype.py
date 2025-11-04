import torch
from torch import Tensor
from typing import Any, Optional, Union, Type, Dict, Callable
import sys

from torchdt.ops import OpsBase, register_op

_float_dtype = {
    8: torch.float8_e5m2, # we have several variants to pick from
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}

_int_dtype = {
    8: torch.uint8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64
}

# for functions that should not be overridden by __torch_function__
no_override_funcs = {
    Tensor.backward,
    Tensor.copy_,
    Tensor.numel,
    Tensor.requires_grad_,
    Tensor.register_hook,
    Tensor.register_post_accumulate_grad_hook,
    Tensor.size,
    torch.zeros_like,
}
# for functions that should not be overridden by __torch_function__
# where it is hard to reference them, so we do it by name
no_override_func_names = {
    "__get__",
}

class GradAccumHook:

    def __init__(self, tensor, dtype):
        self.value = dtype(torch.zeros_like(tensor), requires_grad=False)

        self.grad_hook_handle = tensor.register_hook(self.grad_hook)
        if tensor.is_leaf:
            self.grad_accum_hook_handle = tensor.register_post_accumulate_grad_hook(self.accumulate_hook)

    def grad_hook(self, grad):
        if grad is None:
            return None
        return self.value

    def accumulate_hook(self, tensor):
        tensor.grad.copy_(self.value)

    def register_edge_hook(self, edge, arg_index):

        def edge_hook(grad_inputs, grad_outputs):
            if grad_inputs[arg_index] is not None:
                self.value += type(self.value)(grad_inputs[arg_index], internal=True)

        edge.node.register_hook(edge_hook)

    # def __del__(self):
    #     if hasattr(self, "grad_hook_handle"):
    #         self.grad_hook_handle.remove()
    #     if hasattr(self, "grad_accum_hook_handle"):
    #         self.grad_accum_hook_handle.remove()

class DType(Tensor):
    """
    Parent class for custom dtypes (posit, LNS, etc) that live in a Tensor
    but expose their own semantics.
    """
    bit_width: int = 32 # subclasses override
    torch_funcs: Dict[Callable, Callable] = {} # mapping from 'torch.' function to custom implementation

    def __new__(
            cls,
            data: Any,
            *,
            internal: bool = False,
            device: Optional[Union[str, torch.device]] = None,
            requires_grad: Optional[bool] = None,
            memory_format: torch.memory_format = torch.preserve_format,
    ):
        cls.float_dtype = _float_dtype[cls.bit_width]
        cls.int_dtype = _int_dtype[cls.bit_width]

        if isinstance(data, DType):
            if type(data) == cls:
                payload = data
            else:
                payload = data.to_float(data._float)
                payload = ToDType.apply(payload, cls)
        elif isinstance(data, torch.Tensor):
            if internal:
                if data.dtype != cls.float_dtype:
                    payload = data.view(cls.float_dtype)
                else:
                    payload = data
            else:
                payload = data.to(dtype=cls.float_dtype, device=device, memory_format=memory_format)
                payload = ToDType.apply(payload, cls)
        else:
            if internal:
                payload = torch.tensor(data, dtype=cls.int_dtype, device=device).view(cls.float_dtype)
            else:
                payload = torch.tensor(data, dtype=cls.float_dtype, device=device)
            payload = ToDType.apply(payload, cls)
            payload = payload.to(memory_format=memory_format)

        obj = payload.as_subclass(cls)
        if requires_grad is None:
            if isinstance(data, torch.Tensor) and data.requires_grad:
                obj.requires_grad_(True)
        else:
            obj.requires_grad_(requires_grad)
        return obj

    def __init_subclass__(cls, bit_width: int = 32, **kwargs):
        super().__init_subclass__(**kwargs)

        if bit_width not in _float_dtype:
            raise ValueError(
                f"{cls.__name__} has invalid bit_width {bit_width}. "
                f"Must be one of {tuple(_float_dtype.keys())}."
            )
        cls.bit_width = bit_width

        if cls is DType:
            return # don't register base class

        # create a subclass of Ops for this DType
        ops_name = f"{cls.__name__}Ops"
        namespace = {
            '__module__': OpsBase.__module__,
            'dtype': cls,
        }
        ops_cls = type(ops_name, (OpsBase,), namespace)
        cls.ops = ops_cls

        # allow normal imports to see it
        # module = sys.modules[cls.__module__]
        # setattr(module, ops_name, ops_cls)

    def _track_operation(self, edge, arg_index):
        """
        Registers a hook to track the operation that produced this DType tensor.
        This is used to accumulate gradients from different paths in the computation
        graph since PyTorch will internally add these, but since they are DType
        tensors we must perform our custom DType addition.
        """
        self._grad_accum_hook.register_edge_hook(edge, arg_index)

    def requires_grad_(self, requires_grad: bool = True):
        """Sets the requires_grad flag for this DType tensor in-place."""
        super().requires_grad_(requires_grad)

        if requires_grad:
            self._grad_accum_hook = GradAccumHook(self, type(self))

        return self

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        """
        Computes the gradient of current DType tensor with respect to the graph leaves.
        This method is analogous to the standard PyTorch `Tensor.backward()` method, but
        works with DType tensors. See

        https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html

        for more details. Note that the `gradient` parameter, if provided, will be converted
        to the same DType as `self` before being used in the backward pass.
        """

        if gradient is None:

            if self.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")

            # create a tensor of ones in the same dtype as self
            gradient = type(self)(torch.ones(self.size()))

        elif type(gradient) != type(self):
            gradient = type(self)(gradient, requires_grad=False)

        # manually set the incoming gradients for the output
        # tensor since no hooks will be registered for it.
        self._grad_accum_hook.value.copy_(gradient)

        return super().backward(
            gradient=gradient,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs
        )

    @classmethod
    def register_func(cls, torch_func: Callable):
        """Decorator to register a custom implementation for a torch.* function."""
        def decorator(func: Callable) -> Callable:
            cls.torch_funcs[torch_func] = func
            return func
        return decorator

    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=None):
        """Override to handle torch.* functions for this DType subclass."""

        if kwargs is None:
            kwargs = {}

        if func not in cls.torch_funcs:
            if func in no_override_funcs or func.__name__ in no_override_func_names:
                return super().__torch_function__(func, types, args, kwargs)
            raise NotImplementedError(f"{cls.__name__} has no implementation for torch function '{func.__name__}'.")

        return cls.torch_funcs[func](*args, **kwargs)

    @classmethod
    def register_op(cls, method: str):
        """Decorator to register an operation for this DType subclass."""
        return register_op(cls, method)

    @staticmethod
    def from_float(t: Tensor) -> Tensor:
        "Convert a standard float tensor to this datatype."
        raise NotImplementedError

    @staticmethod
    def to_float(t: Tensor) -> Tensor:
        "Convert this datatype tensor to a standard float tensor."
        raise NotImplementedError

    @property
    def _float(self) -> Tensor:
        "Return the underlying storage as a plain *float* tensor."
        return self.as_subclass(Tensor)

    @property
    def _int(self) -> Tensor:
        "Integer bit-view of the same storage (no copy)."
        return self._float.view(_int_dtype[self.bit_width])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.to_float(self._float)}, bit_width={self.bit_width}, "
            f"shape={tuple(self.shape)}, device={self.device})"
        )

class ToDType(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: Tensor, dtype: Type[DType]) -> DType:
        return dtype.from_float(input).view(dtype.float_dtype)

    @staticmethod
    def backward(ctx, grad_output: DType) -> Tensor:
        return grad_output.to_float(grad_output), None