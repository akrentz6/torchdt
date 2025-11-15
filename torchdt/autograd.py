import torch
from torch.autograd.graph import get_gradient_edge
import functools
from typing import Optional, Tuple

__all__ = [
    "DTFunction",
    "DTNonDifferentiableFunction",
]

def _find_first_grad_tensor(args):
    if not isinstance(args, (list, tuple)):
        args = (args,)
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            return arg
    return None

def _cast_int(x, dtype):
    return x.view(dtype.int_dtype) if isinstance(x, torch.Tensor) and x.dtype == dtype.float_dtype else x

def _cast_float(x, dtype):
    return x.view(dtype.float_dtype) if isinstance(x, torch.Tensor) and x.dtype == dtype.int_dtype else x

def _cast_values(values, indices, cast_fn, dtype):
    all_indices = indices is None

    if isinstance(values, tuple):
        return tuple(
            cast_fn(values[i], dtype) if all_indices or i in indices else values[i]
            for i in range(len(values))
        )

    elif isinstance(values, list):
        return [
            cast_fn(values[i], dtype) if all_indices or i in indices else values[i]
            for i in range(len(values))
        ]

    else:
        return cast_fn(values, dtype) if all_indices or 0 in indices else values

# added to forward only if setup_context is not defined
def forward_ctx_decorator(func, cls):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = args[0]
        ops = args[1]
        inputs = args[2:]

        ctx._dtype = cls.dtype
        ctx._input_indices = cls.input_indices

        cast_inputs = _cast_values(inputs, cls.input_indices, _cast_int, cls.dtype)
        output = func(ctx, ops, *cast_inputs, **kwargs)
        return _cast_values(output, cls.output_indices, _cast_float, cls.dtype)

    return wrapper

def forward_no_ctx_decorator(func, cls):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ops = args[0]
        inputs = args[1:]

        cast_inputs = _cast_values(inputs, cls.input_indices, _cast_int, cls.dtype)
        output = func(ops, *cast_inputs, **kwargs)
        return _cast_values(output, cls.output_indices, _cast_float, cls.dtype)

    return wrapper

def setup_context_decorator(func, cls):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = args[0]
        ops = args[1][0]
        inputs = args[1][1:]
        output = args[2]

        ctx._dtype = cls.dtype
        ctx._input_indices = cls.input_indices

        cast_inputs = _cast_values(inputs, cls.input_indices, _cast_int, cls.dtype)
        cast_output = _cast_values(output, cls.output_indices, _cast_int, cls.dtype)
        return func(ctx, ops, cast_inputs, cast_output, **kwargs)

    return wrapper

def backward_decorator(func, cls):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ctx = args[0]
        grads = args[1:]

        dtype = ctx._dtype
        ops = ctx._dtype.ops
        input_indices = ctx._input_indices

        cast_grads = _cast_values(grads, cls.output_indices, _cast_int, dtype)
        output = func(ctx, ops, *cast_grads, **kwargs)
        cast_output = _cast_values(output, input_indices, _cast_float, dtype)

        if isinstance(output, tuple):
            return (None,) + cast_output
        return None, cast_output

    return wrapper

class DTFunction(torch.autograd.Function):
    """
    Parent class for custom autograd Functions that work with DType tensors.
    Subclasses should implement static methods `forward` and `backward` (and
    optionally `setup_context`).
    """

    output_indices: Optional[Tuple[int, ...]] = None

    @classmethod
    def _register_ops_decorator(cls, func_name, decorator):
        func = getattr(cls, func_name)

        if not getattr(func, "_is_decorated", False):
            decorated = decorator(func, cls)
            decorated._is_lns_decorated = True
            setattr(cls, func_name, decorated)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # check if setup_context is defined in subclass or inherited
        # from torch.autograd.function._SingleLevelFunction. If it is
        # inherited, we only need to decorate forward and backward.
        # If it is defined, we need to decorate setup_context instead
        # of forward.
        if cls.setup_context is torch.autograd.function._SingleLevelFunction.setup_context:
            cls._register_ops_decorator("forward", forward_ctx_decorator)
        else:
            cls._register_ops_decorator("forward", forward_no_ctx_decorator)
            cls._register_ops_decorator("setup_context", setup_context_decorator)

        cls._register_ops_decorator("backward", backward_decorator)

    @classmethod
    def apply(cls, *args, **kwargs):
        from torchdt import DType # avoid circular import

        if kwargs:
            raise ValueError(
                "torch.autograd.Function does not support keyword arguments. "
                "Please use positional arguments only."
            )

        input_indices = []
        subtypes = []
        prepped_inputs = []

        for i, arg in enumerate(args):
            if isinstance(arg, DType):
                subtypes.append(arg.__class__)
                input_indices.append(i)
                prepped_inputs.append(arg._float)
            else:
                prepped_inputs.append(arg)

        if not subtypes:
            raise ValueError("DTFunction.apply() requires at least one DType tensor argument.")

        dtype = subtypes[0]
        if any(st != dtype for st in subtypes):
            raise ValueError("All DType arguments to DTFunction.apply() must be of the same type.")

        # perform the operation using the Ops class for this DType
        cls.dtype = dtype
        cls.input_indices = input_indices
        result = super().apply(dtype.ops, *prepped_inputs)
        del cls.input_indices
        del cls.dtype

        # get gradient edge to correctly handle grads for DType tensors
        first_tensor = _find_first_grad_tensor(result)
        if first_tensor is not None:
            edge = get_gradient_edge(first_tensor)

            j = 0
            for arg in args:

                # ignore non-tensor arguments
                if not isinstance(arg, torch.Tensor):
                    continue
                j += 1

                # only register hooks for DType inputs with gradients
                if not (isinstance(arg, dtype) and arg.requires_grad):
                    continue

                arg._track_operation(edge, j - 1)

        if isinstance(result, torch.Tensor):
            return dtype(
                result, internal=True
            ) if cls.output_indices is None or 0 in cls.output_indices else result

        elif isinstance(result, list):
            return [
                dtype(result[i], internal=True)
                if cls.output_indices is None or i in cls.output_indices else result[i]
                for i in range(len(result))
            ]

        elif isinstance(result, tuple):
            return tuple(
                dtype(result[i], internal=True)
                if cls.output_indices is None or i in cls.output_indices else result[i]
                for i in range(len(result))
            )

        return result

class DTNonDifferentiableFunction:

    output_indices: Optional[Tuple[int, ...]] = None

    @staticmethod
    def forward(ops, *args, **kwargs):
        """Forward pass for non-differentiable function."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        from torchdt import DType # avoid circular import

        if kwargs:
            raise ValueError(
                "DTNonDifferentiableFunction does not support keyword arguments. "
                "Please use positional arguments only."
            )

        subtypes = []
        prepped_inputs = []

        for i, arg in enumerate(args):
            if isinstance(arg, DType):
                subtypes.append(arg.__class__)
                prepped_inputs.append(arg._int)
            else:
                prepped_inputs.append(arg)

        if not subtypes:
            raise ValueError("DTFunction.apply() requires at least one DType tensor argument.")

        dtype = subtypes[0]
        if any(st != dtype for st in subtypes):
            raise ValueError("All DType arguments to DTFunction.apply() must be of the same type.")

        result = cls.forward(dtype.ops, *prepped_inputs)

        if isinstance(result, torch.Tensor):
            return dtype(
                result, internal=True
            ) if cls.output_indices is None or 0 in cls.output_indices else result

        elif isinstance(result, list):
            return [
                dtype(result[i], internal=True)
                if cls.output_indices is None or i in cls.output_indices else result[i]
                for i in range(len(result))
            ]

        elif isinstance(result, tuple):
            return tuple(
                dtype(result[i], internal=True)
                if cls.output_indices is None or i in cls.output_indices else result[i]
                for i in range(len(result))
            )

        return result