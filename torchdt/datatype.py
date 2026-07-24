import torch
from torch import Tensor
from typing import Any, Optional, Union, Type, Dict, Callable, Tuple
import functools
import inspect

from torchdt.transforms import register_collate_dtype_fn
from torchdt.ops import OpsBase, register_op, register_cpp_ops

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
    Tensor.detach,
    Tensor.dim,
    Tensor.numel,
    Tensor.requires_grad_,
    Tensor.register_hook,
    Tensor.register_post_accumulate_grad_hook,
    Tensor.size,
    Tensor.data_ptr,
    Tensor.__reduce_ex__
}
# for functions that should not be overridden by __torch_function__
# where it is hard to reference them, so we do it by name
no_override_func_names = {
    "__get__",
}

class GradAccumHook:

    def __init__(self, tensor, dtype):
        self.value = dtype(torch.zeros(tensor.size(), device=tensor.device), requires_grad=False)
        self.dtype = dtype

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
                # __torch_function__ doesn't work inside hooks, so we must
                # re-enable it manually with a context manager.
                with torch._C._EnableTorchFunction():
                    self.value = self.value + grad_inputs[arg_index].as_subclass(self.dtype)

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
    bitwidth: int = 32 # subclasses override
    # Python implementations shared by every dtype. Backend-specific and
    # dtype-specific implementations live on each concrete subclass.
    torch_funcs: Dict[Callable, Callable] = {}
    _torch_func_implementations: Dict[str, Dict[Callable, Callable]] = {}

    def __new__(
            cls,
            data: Any,
            *,
            internal: bool = False,
            device: Optional[Union[str, torch.device]] = None,
            requires_grad: Optional[bool] = None,
            memory_format: torch.memory_format = torch.preserve_format,
    ):
        if isinstance(data, DType):
            if data.__class__ == cls:
                payload = data
            else:
                payload = data.to_float()
                payload = ToDType.apply(payload, cls)
        elif isinstance(data, torch.Tensor):
            if internal:
                if data.dtype != cls.float_dtype:
                    payload = data.view(cls.float_dtype)
                else:
                    payload = data
            else:
                payload = data.to(dtype=torch.float32, device=device, memory_format=memory_format)
                payload = ToDType.apply(payload, cls)
        else:
            if internal:
                payload = torch.tensor(data, dtype=cls.int_dtype, device=device).view(cls.float_dtype)
            else:
                payload = torch.tensor(data, dtype=torch.float32, device=device)
            payload = ToDType.apply(payload, cls)
            payload = payload.to(memory_format=memory_format)

        obj = payload.as_subclass(cls)
        if requires_grad is None:
            if isinstance(data, torch.Tensor) and data.requires_grad:
                obj.requires_grad_(True)
        else:
            obj.requires_grad_(requires_grad)
        return obj

    def __init_subclass__(cls, bitwidth: int = 32, cpp_backend=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.float_dtype = _float_dtype[bitwidth]
        cls.int_dtype = _int_dtype[bitwidth]

        if bitwidth not in _float_dtype:
            raise ValueError(
                f"{cls.__name__} has invalid bitwidth {bitwidth}. "
                f"Must be one of {tuple(_float_dtype.keys())}."
            )
        cls.bitwidth = bitwidth
        cls.cpp_backend = cpp_backend

        if cls is DType:
            return # don't register base class

        # tell the collate function to handle this DType
        # this is used for DataLoader batching
        register_collate_dtype_fn(cls)

        # tell torch that this DType is safe to save/load
        torch.serialization.add_safe_globals([cls])

        # create a subclass of Ops for this DType
        ops_name = f"{cls.__name__}Ops"
        namespace = {
            '__module__': OpsBase.__module__,
            'dtype': cls,
            '_implementations': {},
            '_enabled_backends': {},
        }
        namespace.update({
            method: OpsBase.dispatch_method(method)
            for method in OpsBase._op_names
        })
        ops_cls = type(ops_name, (OpsBase,), namespace)
        cls.ops = ops_cls
        cls._torch_func_implementations = {}

        # allow normal imports to see it
        # module = sys.modules[cls.__module__]
        # setattr(module, ops_name, ops_cls)

    @classmethod
    def enable_cpp_backend(cls, backend=None):
        if cls.cpp_backend is None and backend is None:
            raise ValueError(f"{cls.__name__} has no C++ backend to enable.")
        register_cpp_ops(cls, backend or cls.cpp_backend)

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
            self._grad_accum_hook = GradAccumHook(self, self.__class__)

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
            gradient = self.__class__(torch.ones(self.size()), device=self.device)

        elif gradient.__class__ != self.__class__:
            gradient = self.__class__(gradient, device=self.device, requires_grad=False)

        # manually set the incoming gradients for the output
        # tensor since no hooks will be registered for it.
        self._grad_accum_hook.value.copy_(gradient)

        return super().backward(
            gradient=gradient,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs
        )

    @property
    def grad(self):
        """The gradient of this DType tensor."""
        if super().grad is None:
            return None
        return super().grad.as_subclass(self.__class__)

    @classmethod
    def register_func(
        cls,
        *torch_funcs: Callable,
        cast: Tuple[Union[str, int]] = (),
        backend: str = "python"):
        """Decorator to register a custom implementation for a torch.* function."""

        def decorator(func: Callable) -> Callable:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            def id_to_name(identifier: Union[str, int]):
                if isinstance(identifier, int):
                    try:
                        return param_names[identifier]
                    except IndexError:
                        raise IndexError(f"positional index {identifier} out of range for {func.__name__}")
                return identifier # assume string

            cast_names = [id_to_name(x) for x in cast]

            @functools.wraps(func)
            def wrapped_func(*args, _dtype_cls=None, **kwargs):
                if _dtype_cls is None:
                    raise ValueError("_dtype_cls must be provided when calling registered torch function.")

                bound = sig.bind_partial(*args, **kwargs)
                dtype_devices = {
                    _dtype_cls.ops.tensor_device(tensor)
                    for value in bound.arguments.values()
                    for tensor in _dtype_cls.ops._iter_tensors(value)
                    if isinstance(tensor, DType)
                }
                if len(dtype_devices) > 1:
                    device_list = ", ".join(str(device) for device in sorted(dtype_devices, key=str))
                    raise RuntimeError(
                        f"Expected all {DType.__name__} arguments to use one device, got {device_list}."
                    )
                cast_device = next(iter(dtype_devices), None)

                for pname in cast_names:
                    if pname in bound.arguments:
                        if bound.arguments[pname] is not None:
                            # convert argument to the correct DType subclass - can also handle lists, tuples, etc?
                            if isinstance(bound.arguments[pname], (list, tuple)):
                                bound.arguments[pname] = type(bound.arguments[pname])(
                                    _dtype_cls(x, device=cast_device) if type(x) != _dtype_cls else x
                                    for x in bound.arguments[pname]
                                )
                            elif type(bound.arguments[pname]) != _dtype_cls:
                                bound.arguments[pname] = _dtype_cls(
                                    bound.arguments[pname], device=cast_device
                                )

                return func(*bound.args, **bound.kwargs)

            for torch_func in torch_funcs:
                if cls is DType and backend == "python":
                    cls.torch_funcs[torch_func] = wrapped_func
                else:
                    cls._torch_func_implementations.setdefault(backend, {})[torch_func] = wrapped_func

            return wrapped_func

        return decorator

    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=None):
        """Override to handle torch.* functions for this DType subclass."""

        if kwargs is None:
            kwargs = {}

        dtype_devices = {
            cls.ops.tensor_device(tensor)
            for value in (*args, *kwargs.values())
            for tensor in cls.ops._iter_tensors(value)
            if isinstance(tensor, DType)
        }
        if len(dtype_devices) > 1:
            device_list = ", ".join(str(device) for device in sorted(dtype_devices, key=str))
            raise RuntimeError(
                f"Expected all DType arguments to use one device, got {device_list}."
            )
        device = next(iter(dtype_devices), None)
        backend = cls.ops.backend_for_device(device)
        implementation = cls._torch_func_implementations.get(backend, {}).get(func)
        if implementation is None:
            implementation = cls._torch_func_implementations.get("python", {}).get(func)
        if implementation is None:
            implementation = DType.torch_funcs.get(func)

        if implementation is None:
            if func in no_override_funcs or func.__name__ in no_override_func_names:
                return super().__torch_function__(func, types, args, kwargs)
            raise NotImplementedError(f"{cls.__name__} has no implementation for torch function '{func.__name__}'.")

        # pass cls to cast any floating point or number arguments to tensors of this DType
        return implementation(*args, _dtype_cls=cls, **kwargs)

    @classmethod
    def register_op(cls, method: str, backend: str = "python"):
        """Decorator to register an operation for this DType subclass."""
        return register_op(cls, method, backend=backend)

    @property
    def _float(self) -> Tensor:
        "Return the underlying storage as a plain *float* tensor."
        return self.as_subclass(Tensor)

    @property
    def _int(self) -> Tensor:
        "Integer bit-view of the same storage (no copy)."
        return self._float.view(_int_dtype[self.bitwidth])

    def to_float(self):
        return self.ops.to_float(self._int)

    def __float__(self):
        return self.ops.scalar_to_float(self._int)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.to_float()}, bitwidth={self.bitwidth}, "
            f"shape={tuple(self.shape)}, device={self.device})"
        )

class ToDType(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: Tensor, dtype: Type[DType]) -> DType:
        return dtype.ops.from_float(input).view(dtype.float_dtype)

    @staticmethod
    def backward(ctx, grad_output: DType) -> Tensor:
        return grad_output.to_float(), None
