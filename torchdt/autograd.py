import torch
from torch.utils._pytree import tree_flatten, tree_map
from typing import Optional, Tuple
import inspect
from torchdt._dispatch import current_dispatch

__all__ = [
    "DTFunction",
    "DTNonDifferentiableFunction",
]

_aten = torch.ops.aten
_gradient_view_ops = {
    _aten.alias.default,
    _aten.clone.default,
    _aten.detach.default,
    _aten.expand.default,
    _aten.permute.default,
    _aten.reshape.default,
    _aten.squeeze.default,
    _aten.squeeze.dim,
    _aten.transpose.int,
    _aten.unsqueeze.default,
    _aten.view.default,
}


class _Gradient(torch.Tensor):

    @staticmethod
    def __new__(cls, value, dtype):
        if type(value) is not torch.Tensor:
            raise TypeError("Expected a plain Tensor for gradient storage.")
        if value.layout != torch.strided:
            raise NotImplementedError("DType gradients must use a strided layout.")
        if value.dtype != dtype.float_dtype:
            raise TypeError(
                f"Expected {dtype.float_dtype} gradient storage, got {value.dtype}."
            )
        result = torch.Tensor._make_subclass(cls, value, value.requires_grad)
        result._dtype = dtype
        return result

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        values, _ = tree_flatten((args, kwargs))
        dtypes = {value._dtype for value in values if isinstance(value, cls)}
        if len(dtypes) != 1:
            names = ", ".join(sorted(dtype.__name__ for dtype in dtypes))
            raise TypeError(f"Cannot combine gradients of different DTypes: {names}.")
        dtype = next(iter(dtypes))

        with torch._C._DisableTorchDispatch():
            def unwrap(value):
                if isinstance(value, cls):
                    return value.as_subclass(torch.Tensor)
                return value

            base_args = tree_map(unwrap, args)
            base_kwargs = tree_map(unwrap, kwargs)

            if func in (_aten.add.Tensor, _aten.add_.Tensor):
                if base_kwargs.get("alpha", 1) != 1:
                    raise NotImplementedError("DType gradient addition requires alpha=1.")
                left, right = base_args[:2]
                if left.layout != torch.strided or right.layout != torch.strided:
                    raise NotImplementedError("Sparse DType gradients are not supported.")
                if left.dtype != dtype.float_dtype or right.dtype != dtype.float_dtype:
                    raise TypeError("DType gradient storage types do not match.")

                ops = dtype.ops.direct_for_device(left.device)
                result = ops.add(
                    left.view(dtype.int_dtype),
                    right.view(dtype.int_dtype),
                ).view(dtype.float_dtype)
                if func is _aten.add_.Tensor:
                    left.copy_(result)
                    return args[0]
                return cls(result, dtype)

            if func is _aten.zero_.default:
                target = base_args[0]
                integer = target.view(dtype.int_dtype)
                ops = dtype.ops.direct_for_device(target.device)
                integer.copy_(ops.zeros_like(integer))
                return args[0]

            if func is _aten.copy_.default:
                base_args[0].copy_(base_args[1], **base_kwargs)
                return args[0]

            if func in _gradient_view_ops:
                return cls(func(*base_args, **base_kwargs), dtype)

            # Backward implementations use a dtype-changing view to
            # recover the integer representation. Other operations
            # drop the engine-only gradient tag.
            return func(*base_args, **base_kwargs)


def _mark_gradient(value, dtype):
    if value is None:
        return value
    if isinstance(value, _Gradient):
        if value._dtype is not dtype:
            raise TypeError("Gradient has the wrong DType.")
        return value
    return _Gradient(value, dtype)


def _gradient_as_dtype(value):
    dtype = value._dtype
    with torch._C._DisableTorchDispatch():
        base = value.as_subclass(torch.Tensor)
        view = base.as_strided(base.size(), base.stride(), base.storage_offset())
    return view.as_subclass(dtype)


def _cast_int(x, dtype):
    return x.view(dtype.int_dtype) if isinstance(x, torch.Tensor) and x.dtype == dtype.float_dtype else x


def _cast_float(x, dtype):
    return x.view(dtype.float_dtype) if isinstance(x, torch.Tensor) and x.dtype == dtype.int_dtype else x


def _cast_values(values, indices, cast_fn, dtype):
    all_indices = indices is None

    if isinstance(values, tuple):
        return tuple(
            cast_fn(value, dtype) if all_indices or i in indices else value
            for i, value in enumerate(values)
        )
    if isinstance(values, list):
        return [
            cast_fn(value, dtype) if all_indices or i in indices else value
            for i, value in enumerate(values)
        ]
    return cast_fn(values, dtype) if all_indices or 0 in indices else values


def _wrap_outputs(result, dtype, output_indices):
    def wrap(value, index):
        if output_indices is None or index in output_indices:
            return dtype(value, internal=True)
        return value

    if isinstance(result, torch.Tensor):
        return wrap(result, 0)
    if isinstance(result, list):
        return [wrap(value, i) for i, value in enumerate(result)]
    if isinstance(result, tuple):
        return tuple(wrap(value, i) for i, value in enumerate(result))
    return result


def _combined_forward(ctx, call_spec, *inputs):
    dt_cls, ops, input_indices = call_spec
    dtype = ops.dtype

    ctx._dt_cls = dt_cls
    ctx._dtype = dtype
    ctx._ops = ops
    ctx._input_indices = input_indices

    cast_inputs = _cast_values(inputs, input_indices, _cast_int, dtype)
    if dt_cls._dt_forward_uses_ctx:
        output = dt_cls._dt_forward(ctx, ops, *cast_inputs)
    else:
        output = dt_cls._dt_forward(ops, *cast_inputs)
    if not dt_cls._dt_forward_uses_ctx and dt_cls._dt_setup_context is not None:
        dt_cls._dt_setup_context(ctx, ops, cast_inputs, output)
    return _cast_values(output, dt_cls.output_indices, _cast_float, dtype)


def _combined_backward(ctx, *grads):
    dt_cls = ctx._dt_cls
    dtype = ctx._dtype
    ops = ctx._ops

    cast_grads = _cast_values(grads, dt_cls.output_indices, _cast_int, dtype)
    output = dt_cls._dt_backward(ctx, ops, *cast_grads)
    cast_output = _cast_values(output, ctx._input_indices, _cast_float, dtype)
    cast_output = _cast_values(cast_output, ctx._input_indices, _mark_gradient, dtype)

    if isinstance(cast_output, tuple):
        return (None,) + cast_output
    return None, cast_output


def _common_dtype_and_device(args, dtype_base):
    dtype = None
    device = None
    tensor_devices = set()

    with torch._C.DisableTorchFunctionSubclass():
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                continue
            arg_device = arg.device
            tensor_devices.add(arg_device)
            if isinstance(arg, dtype_base):
                if dtype is None:
                    dtype = arg.__class__
                    device = arg_device
                elif arg.__class__ is not dtype:
                    raise ValueError("All DType arguments must be of the same type.")

    if dtype is None:
        raise ValueError("A DType tensor argument is required.")
    if len(tensor_devices) > 1:
        devices = ", ".join(str(value) for value in sorted(tensor_devices, key=str))
        raise RuntimeError(f"Expected all tensor arguments to use one device, got {devices}.")
    return dtype, device


class _NoGradContext:
    """Discard autograd-only state while running a ctx-style forward directly."""

    def save_for_backward(self, *tensors):
        pass

    def save_for_forward(self, *tensors):
        pass

    def mark_non_differentiable(self, *tensors):
        pass

    def set_materialize_grads(self, value):
        pass


class DTFunction(torch.autograd.Function):
    """Autograd Function base for operations on encoded DType tensors."""

    output_indices: Optional[Tuple[int, ...]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls._dt_forward = getattr(cls, "forward")
        cls._dt_backward = getattr(cls, "backward")
        forward_parameters = tuple(inspect.signature(cls._dt_forward).parameters)
        cls._dt_forward_uses_ctx = bool(forward_parameters and forward_parameters[0] == "ctx")
        setup_context = getattr(cls, "setup_context")
        inherited_setup = torch.autograd.function._SingleLevelFunction.setup_context
        cls._dt_setup_context = None if setup_context is inherited_setup else setup_context
        if cls._dt_forward_uses_ctx and cls._dt_setup_context is not None:
            raise TypeError(
                f"{cls.__name__} cannot define both forward(ctx, ...) and setup_context()."
            )

        cls.forward = staticmethod(_combined_forward)
        cls.backward = staticmethod(_combined_backward)
        if "setup_context" in cls.__dict__:
            delattr(cls, "setup_context")

    @classmethod
    def apply(cls, *args, **kwargs):
        from torchdt import DType  # avoid circular import

        if kwargs:
            raise ValueError(
                "torch.autograd.Function does not support keyword arguments. "
                "Please use positional arguments only."
            )

        dispatch = current_dispatch.get()
        if dispatch is None:
            dtype, device = _common_dtype_and_device(args, DType)
            ops = dtype.ops.direct_for_device(device)
        else:
            dtype, device, ops = dispatch
        input_indices = tuple(i for i, arg in enumerate(args) if isinstance(arg, DType))

        with torch._C.DisableTorchFunctionSubclass():
            needs_autograd = torch.is_grad_enabled() and any(
                isinstance(arg, DType) and arg.requires_grad for arg in args
            )
        if not needs_autograd:
            internal_inputs = tuple(arg._int if isinstance(arg, DType) else arg for arg in args)
            if cls._dt_forward_uses_ctx:
                result = cls._dt_forward(_NoGradContext(), ops, *internal_inputs)
            else:
                result = cls._dt_forward(ops, *internal_inputs)
            return _wrap_outputs(result, dtype, cls.output_indices)

        prepped_inputs = tuple(arg._float if isinstance(arg, DType) else arg for arg in args)
        call_spec = (cls, ops, input_indices)
        result = super().apply(call_spec, *prepped_inputs)
        return _wrap_outputs(result, dtype, cls.output_indices)


class DTNonDifferentiableFunction:
    output_indices: Optional[Tuple[int, ...]] = None

    @staticmethod
    def forward(ops, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        from torchdt import DType  # avoid circular import

        if kwargs:
            raise ValueError(
                "DTNonDifferentiableFunction does not support keyword arguments. "
                "Please use positional arguments only."
            )

        dispatch = current_dispatch.get()
        if dispatch is None:
            dtype, device = _common_dtype_and_device(args, DType)
            ops = dtype.ops.direct_for_device(device)
        else:
            dtype, device, ops = dispatch
        prepped_inputs = tuple(arg._int if isinstance(arg, DType) else arg for arg in args)
        result = cls.forward(ops, *prepped_inputs)
        return _wrap_outputs(result, dtype, cls.output_indices)
