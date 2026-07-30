import torch
from torch import Tensor
from typing import Any, Optional, Union, Type, Dict, Callable, Tuple
import functools
import inspect

from torchdt.transforms import register_collate_dtype_fn
from torchdt.ops import OpsBase, register_op, register_cpp_ops
from torchdt._dispatch import current_dispatch

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
    Tensor.element_size,
    Tensor.get_device,
    Tensor.is_contiguous,
    Tensor.is_pinned,
    Tensor.numel,
    Tensor.requires_grad_,
    Tensor.register_hook,
    Tensor.register_post_accumulate_grad_hook,
    Tensor.size,
    Tensor.storage_offset,
    Tensor.stride,
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
        self.value = None
        self.dtype = dtype

        self.grad_hook_handle = tensor.register_hook(self.grad_hook)
        if tensor.is_leaf:
            self.grad_accum_hook_handle = tensor.register_post_accumulate_grad_hook(self.accumulate_hook)

    def grad_hook(self, grad):
        if grad is None:
            return None
        return self.value if self.value is not None else grad

    def accumulate_hook(self, tensor):
        if self.value is not None:
            tensor.grad.copy_(self.value)
            # Keep the buffer PyTorch already owns and release the temporary
            # first-contribution view after leaf accumulation.
            self.value = tensor.grad.as_subclass(self.dtype)

    def register_edge_hook(self, edge, arg_index):

        def edge_hook(grad_inputs, grad_outputs):
            if grad_inputs[arg_index] is not None:
                # __torch_function__ doesn't work inside hooks, so we must
                # re-enable it manually with a context manager.
                with torch._C._EnableTorchFunction():
                    contribution = grad_inputs[arg_index].as_subclass(self.dtype)
                    if self.value is None:
                        self.value = contribution
                    else:
                        self.value = self.value + contribution

        edge.node.register_hook(edge_hook)

    def remove(self):
        self.grad_hook_handle.remove()
        if hasattr(self, "grad_accum_hook_handle"):
            self.grad_accum_hook_handle.remove()

    def reset(self, set_to_none=True):
        if set_to_none:
            self.value = None
            return False
        elif self.value is not None:
            self.value.zero_()
            return True
        return False

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
    _direct_torch_funcs: Dict[str, Dict[Callable, Callable]] = {}

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
                with torch._C.DisableTorchFunctionSubclass():
                    same_device = device is None or data.device == torch.device(device)
                    same_requires_grad = requires_grad is None or data.requires_grad == requires_grad
                if same_device and same_requires_grad and memory_format is torch.preserve_format:
                    return data
                payload = data._float.to(device=device, memory_format=memory_format)
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
            if isinstance(data, torch.Tensor):
                with torch._C.DisableTorchFunctionSubclass():
                    data_requires_grad = data.requires_grad
                if data_requires_grad:
                    obj.requires_grad_(True)
        else:
            obj.requires_grad_(requires_grad)
        return obj

    def __init_subclass__(cls, bitwidth: int = 32, cpp_backend=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if bitwidth not in _float_dtype:
            raise ValueError(
                f"{cls.__name__} has invalid bitwidth {bitwidth}. "
                f"Must be one of {tuple(_float_dtype.keys())}."
            )
        cls.float_dtype = _float_dtype[bitwidth]
        cls.int_dtype = _int_dtype[bitwidth]
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
            '_direct_ops': {},
            '_scalar_codes': {},
            '_direct_implementations': {},
        }
        namespace.update({
            method: OpsBase.dispatch_method(method)
            for method in OpsBase._op_names
        })
        ops_cls = type(ops_name, (OpsBase,), namespace)
        cls.ops = ops_cls
        cls._torch_func_implementations = {}
        cls._direct_torch_funcs = {}

        # allow normal imports to see it
        # module = sys.modules[cls.__module__]
        # setattr(module, ops_name, ops_cls)

    @classmethod
    def enable_cpp_backend(cls, backend=None):
        if cls.cpp_backend is None and backend is None:
            raise ValueError(f"{cls.__name__} has no C++ backend to enable.")
        backend_name = backend or cls.cpp_backend
        if getattr(cls.ops, "_cpp_backend_name", None) == backend_name:
            return
        register_cpp_ops(cls, backend_name)
        cls.ops._cpp_backend_name = backend_name

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

        hook = getattr(self, "_grad_accum_hook", None)
        if requires_grad and hook is None:
            self._grad_accum_hook = GradAccumHook(self, self.__class__)
        elif not requires_grad and hook is not None:
            hook.remove()
            del self._grad_accum_hook

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
            ops = self.ops.direct_for_device(self.device)
            gradient = self.__class__(ops.ones(self.size(), device=self.device), internal=True)

        elif gradient.__class__ != self.__class__:
            gradient = self.__class__(gradient, device=self.device, requires_grad=False)

        # manually set the incoming gradients for the output
        # tensor since no hooks will be registered for it.
        self._grad_accum_hook.value = gradient

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

    @grad.setter
    def grad(self, value):
        if isinstance(value, DType):
            value = value._float
        with torch._C.DisableTorchFunctionSubclass():
            Tensor.grad.__set__(self, value)

    @classmethod
    def register_func(
        cls,
        *torch_funcs: Callable,
        cast: Tuple[Union[str, int]] = (),
        backend: str = "python"):
        """Decorator to register a custom implementation for a torch.* function."""

        if isinstance(cast, (str, int)):
            cast = (cast,)

        def decorator(func: Callable) -> Callable:
            sig = inspect.signature(func)
            parameters = tuple(sig.parameters.values())
            param_names = [param.name for param in parameters]
            param_positions = {
                param.name: index
                for index, param in enumerate(parameters)
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            }

            def id_to_name(identifier: Union[str, int]):
                if isinstance(identifier, int):
                    try:
                        return param_names[identifier]
                    except IndexError:
                        raise IndexError(f"positional index {identifier} out of range for {func.__name__}")
                return identifier # assume string

            cast_names = [id_to_name(x) for x in cast]

            def cast_value(value, dtype_cls, device):
                if isinstance(value, tuple):
                    return tuple(cast_value(x, dtype_cls, device) for x in value)
                if isinstance(value, list):
                    return [cast_value(x, dtype_cls, device) for x in value]
                if isinstance(value, dict):
                    return {
                        key: cast_value(item, dtype_cls, device)
                        for key, item in value.items()
                    }
                if type(value) != dtype_cls:
                    return dtype_cls(value, device=device)
                return value

            @functools.wraps(func)
            def wrapped_func(*args, _dtype_cls=None, _device=None, **kwargs):
                if _dtype_cls is None:
                    raise ValueError("_dtype_cls must be provided when calling registered torch function.")

                mutable_args = None
                for pname in cast_names:
                    position = param_positions.get(pname)
                    if position is not None and position < len(args):
                        value = args[position]
                        if value is not None:
                            if mutable_args is None:
                                mutable_args = list(args)
                            mutable_args[position] = cast_value(value, _dtype_cls, _device)
                    elif pname in kwargs and kwargs[pname] is not None:
                        kwargs[pname] = cast_value(kwargs[pname], _dtype_cls, _device)

                return func(*(mutable_args if mutable_args is not None else args), **kwargs)

            for torch_func in torch_funcs:
                if cls is DType and backend == "python":
                    cls.torch_funcs[torch_func] = wrapped_func
                else:
                    cls._torch_func_implementations.setdefault(backend, {})[torch_func] = wrapped_func
            cls._direct_torch_funcs.clear()
            if cls is DType:
                pending = list(cls.__subclasses__())
                while pending:
                    subclass = pending.pop()
                    subclass._direct_torch_funcs.clear()
                    pending.extend(subclass.__subclasses__())

            return wrapped_func

        return decorator

    @classmethod
    def _torch_funcs_for_backend(cls, backend):
        implementations = cls._direct_torch_funcs.get(backend)
        if implementations is None:
            implementations = dict(DType.torch_funcs)
            implementations.update(cls._torch_func_implementations.get("python", {}))
            if backend != "python":
                implementations.update(cls._torch_func_implementations.get(backend, {}))
            cls._direct_torch_funcs[backend] = implementations
        return implementations

    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=None):
        """Override to handle torch.* functions for this DType subclass."""

        if kwargs is None:
            kwargs = {}

        device = cls.ops.resolve_device(args, kwargs)
        if device is None and kwargs.get("device") is not None:
            device = torch.device(kwargs["device"])
        backend = cls.ops.backend_for_device(device)
        implementation = cls._torch_funcs_for_backend(backend).get(func)

        if implementation is None:
            if func in no_override_funcs or func.__name__ in no_override_func_names:
                return super().__torch_function__(func, types, args, kwargs)
            raise NotImplementedError(f"{cls.__name__} has no implementation for torch function '{func.__name__}'.")

        # Share the already resolved table with nested operation/autograd calls.
        dispatch = (cls, device, cls.ops.direct_for_backend(backend))
        token = current_dispatch.set(dispatch)
        try:
            return implementation(*args, _dtype_cls=cls, _device=device, **kwargs)
        finally:
            current_dispatch.reset(token)

    @classmethod
    def register_op(cls, method: str, backend: str = "python", *, direct: bool = False):
        """Decorator to register an operation for this DType subclass."""
        return register_op(cls, method, backend=backend, direct=direct)

    @property
    def _float(self) -> Tensor:
        "Return the underlying storage as a plain *float* tensor."
        return self.as_subclass(Tensor)

    @property
    def _int(self) -> Tensor:
        "Integer bit-view of the same storage (no copy)."
        with torch._C.DisableTorchFunctionSubclass():
            return self.view(_int_dtype[self.bitwidth])

    def to_float(self):
        return self.ops.to_float(self._int)

    def copy_(self, src, non_blocking=False):
        if not isinstance(src, Tensor):
            raise TypeError("copy_(): argument 'src' must be Tensor")

        if type(src) is type(self):
            encoded = src._int
        else:
            values = src.to_float() if isinstance(src, DType) else src
            values = values.to(dtype=torch.float32, device=self.device)
            encoded = self.ops.direct_for_device(self.device).from_float(values)

        self._int.copy_(encoded, non_blocking=non_blocking)
        return self

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
        ops = dtype.ops.direct_for_device(input.device)
        return ops.from_float(input).view(dtype.float_dtype)

    @staticmethod
    def backward(ctx, grad_output: DType) -> Tensor:
        ops = grad_output.__class__.ops.direct_for_device(grad_output.device)
        return ops.to_float(grad_output._int), None
