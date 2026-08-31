import torch
from torch import Tensor, CharTensor, ShortTensor, IntTensor, LongTensor
from typing import Union, Callable
import functools

InternalTensor = Union[CharTensor, ShortTensor, IntTensor, LongTensor]

def register_op(
    dtype_cls: type,
    method: str,
    backend: str = "python",
    *,
    direct: bool = False,
) -> Callable:
    """Decorator to register an operation for a given DType subclass."""
    def decorator(func: Callable) -> Callable:
        ops_cls = dtype_cls.ops
        if method not in OpsBase._op_names:
            raise ValueError(f"{ops_cls.__name__} has no method '{method}' to register.")
        registry = ops_cls._direct_implementations if direct else ops_cls._implementations
        registry.setdefault(backend, {})[method] = func
        ops_cls._direct_ops.clear()
        ops_cls._scalar_codes.clear()
        return func
    return decorator

def register_base_op(method: str) -> Callable:
    """Decorator to register a base operation."""
    def decorator(func: Callable) -> Callable:
        if method not in OpsBase._op_names:
            raise ValueError(f"OpsBase has no method '{method}' to register.")
        OpsBase._base_implementations[method] = func
        return func
    return decorator

class OpsBase:

    _base_implementations = {}
    _op_names = set()
    _direct_ops = {}
    _scalar_codes = {}
    _direct_implementations = {}

    @classmethod
    def _iter_tensors(cls, value):
        if isinstance(value, Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from cls._iter_tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from cls._iter_tensors(item)

    @staticmethod
    def tensor_device(tensor: Tensor):
        with torch._C.DisableTorchFunctionSubclass():
            return tensor.device

    @classmethod
    def resolve_device(cls, args, kwargs=None):
        """Return the common tensor device used by an operation."""
        kwargs = kwargs or {}
        device = None
        with torch._C.DisableTorchFunctionSubclass():
            for value in (*args, *kwargs.values()):
                tensors = (value,) if isinstance(value, Tensor) else cls._iter_tensors(value)
                for tensor in tensors:
                    tensor_device = tensor.device
                    if device is None:
                        device = tensor_device
                    elif tensor_device != device:
                        raise RuntimeError(
                            "Expected all tensor arguments to use one device, got "
                            f"{device} and {tensor_device}."
                        )
        return device

    @classmethod
    def backend_for_device(cls, device):
        if device is None:
            return "python"
        if not isinstance(device, torch.device):
            device = torch.device(device)
        return cls._enabled_backends.get(device.type, "python")

    @classmethod
    def enable_backend(cls, backend: str, device_type: str) -> None:
        current = cls._enabled_backends.get(device_type)
        if current is not None and current != backend:
            raise RuntimeError(
                f"{cls.__name__} already uses backend '{current}' for {device_type}."
            )
        cls._enabled_backends[device_type] = backend

    @classmethod
    def direct_for_backend(cls, backend: str):
        """Return an operation table with backend fallback resolved once."""
        direct = cls._direct_ops.get(backend)
        if direct is not None:
            return direct

        implementations = dict(OpsBase._base_implementations)
        implementations.update(cls._implementations.get("python", {}))
        if backend != "python":
            implementations.update(cls._implementations.get(backend, {}))
        direct_implementations = cls._direct_implementations.get(backend, {})

        namespace = {
            "__module__": cls.__module__,
            "dtype": cls.dtype,
            "backend": backend,
            "facade": cls,
        }
        namespace.update({
            method: classmethod(func)
            for method, func in implementations.items()
        })
        namespace.update({
            method: staticmethod(func)
            for method, func in direct_implementations.items()
        })
        direct = type(f"{cls.__name__}_{backend}", (OpsBase,), namespace)
        cls._direct_ops[backend] = direct
        return direct

    @classmethod
    def direct_for_device(cls, device):
        return cls.direct_for_backend(cls.backend_for_device(device))

    @classmethod
    def clear_scalar_cache(cls):
        owner = getattr(cls, "facade", cls)
        owner._scalar_codes.clear()

    @classmethod
    def encoded_scalar(cls, value):
        """Return a host integer encoding without a device scalar or sync."""
        owner = getattr(cls, "facade", cls)
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError("encoded_scalar expects a scalar value")
            value = value.detach().item()
        key = float(value)
        code = owner._scalar_codes.get(key)
        if code is None:
            python_ops = owner.direct_for_backend("python")
            tensor = torch.tensor(key, dtype=owner.dtype.conversion_dtype)
            code = int(python_ops.from_float(tensor).item())
            owner._scalar_codes[key] = code
        return code

    @classmethod
    def _get_implementation(cls, method: str, backend: str):
        implementation = cls._implementations.get(backend, {}).get(method)
        if implementation is None and backend == "python":
            implementation = OpsBase._base_implementations.get(method)
        return implementation

    @classmethod
    def _dispatch(cls, method: str, *args, **kwargs):
        device = cls.resolve_device(args, kwargs)
        backend = cls.backend_for_device(device)
        direct_ops = cls.direct_for_backend(backend)
        implementation = direct_ops.__dict__.get(method)

        if implementation is None:
            device_name = str(device) if device is not None else "no tensor device"
            raise NotImplementedError(
                f"{cls.dtype.__name__} has no implementation for operation '{method}' "
                f"on {device_name}."
            )
        return getattr(direct_ops, method)(*args, **kwargs)

    @staticmethod
    def dispatch_method(method: str):
        @functools.wraps(getattr(OpsBase, method))
        def dispatched(cls, *args, **kwargs):
            return cls._dispatch(method, *args, **kwargs)
        return classmethod(dispatched)

    # ========== Useful helper functions ==========

    @classmethod
    def scalar_from_float(cls, x: Union[float, int, Tensor], *, device=None) -> InternalTensor:
        if isinstance(x, Tensor):
            if x.numel() != 1:
                raise ValueError("scalar_from_float expects a scalar value")
            if device is None:
                device = x.device
            x_tensor = x.detach().to(dtype=cls.dtype.conversion_dtype, device=device).reshape(())
        else:
            x_tensor = torch.tensor(x, dtype=cls.dtype.conversion_dtype, device=device)
        return cls.from_float(x_tensor)

    @classmethod
    def scalar_to_float(cls, x: InternalTensor):
        return cls.to_float(x).item()

    @classmethod
    def zeros(cls, size, device=None):
        return torch.full(size, cls.encoded_scalar(0.0), dtype=cls.dtype.int_dtype, device=device)

    @classmethod
    def zeros_like(cls, x):
        return torch.full_like(x, cls.encoded_scalar(0.0), dtype=cls.dtype.int_dtype, device=x.device)

    @classmethod
    def ones(cls, size, device=None):
        return torch.full(size, cls.encoded_scalar(1.0), dtype=cls.dtype.int_dtype, device=device)

    @classmethod
    def ones_like(cls, x):
        return torch.full_like(x, cls.encoded_scalar(1.0), dtype=cls.dtype.int_dtype, device=x.device)

    @classmethod
    def full(cls, size, fill_value, device=None):
        return torch.full(size, cls.encoded_scalar(fill_value), dtype=cls.dtype.int_dtype, device=device)

    @classmethod
    def full_like(cls, x, fill_value):
        return torch.full_like(x, cls.encoded_scalar(fill_value), dtype=cls.dtype.int_dtype)

    @classmethod
    def sum_to_size(cls, x: InternalTensor, target_size: torch.Size) -> InternalTensor:
        if list(x.shape) == list(target_size):
            return x

        x_shape = list(x.shape)
        target_shape = list(target_size)
        if x.dim() > len(target_shape):
            target_shape = [1] * (x.dim() - len(target_shape)) + target_shape

        # reduce dimensions that were broadcasted
        leading = x.dim() - len(target_shape)
        if leading > 0:
            x = cls.sum(x, dim=tuple(range(leading)), keepdim=False)
            x_shape = x_shape[leading:]

        # reduce dimensions where target size is 1 but tensor has a larger size
        reduce_dims = [i for i, (ts, gs) in enumerate(zip(x_shape, target_shape)) if gs == 1 and ts != 1]
        if reduce_dims:
            x = cls.sum(x, dim=tuple(reduce_dims), keepdim=True)

        return x.reshape(target_size)

    # ========== Operations to be implemented by subclasses ==========

    @classmethod
    def from_float(cls, x: Tensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def to_float(cls, x: InternalTensor) -> Tensor:
        raise NotImplementedError

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
    def ge(cls, x: InternalTensor, y: InternalTensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def gt(cls, x: InternalTensor, y: InternalTensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def le(cls, x: InternalTensor, y: InternalTensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def lt(cls, x: InternalTensor, y: InternalTensor) -> Tensor:
        raise NotImplementedError

    # ========== Backend-specific backward operations ==========

    @classmethod
    def matmul_backward(cls, grad_output: InternalTensor, A: InternalTensor,
                        B: InternalTensor) -> tuple[InternalTensor, InternalTensor]:
        raise NotImplementedError

    @classmethod
    def conv2d_backward(cls, grad_output: InternalTensor, input: InternalTensor,
                        weight: InternalTensor, stride: tuple[int], padding: tuple[int],
                        dilation: tuple[int], has_bias: bool, groups: int
                        ) -> tuple[InternalTensor, InternalTensor, InternalTensor]:
        raise NotImplemented

    # ========== 'Base' operations with default implementations ==========

    @classmethod
    def sum(cls, x: InternalTensor, dim=None, keepdim=False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def square(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def sqrt(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def reciprocal(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def exp(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def log(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def prod(cls, x: InternalTensor, dim=None, keepdim=False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def mean(cls, x: InternalTensor, dim=None, keepdim=False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def var(cls, x: InternalTensor, correction: InternalTensor, dim=None, keepdim=False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def matmul(cls, A: InternalTensor, B: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def transpose(cls, A: InternalTensor, dim0: int, dim1: int) -> InternalTensor:
        raise NotImplementedError


    @classmethod
    def neg(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def abs(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError


    @classmethod
    def equal(cls, x: InternalTensor, y: InternalTensor) -> bool:
        raise NotImplementedError

    @classmethod
    def eq(cls, x: InternalTensor, y: InternalTensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def ne(cls, x: InternalTensor, y: InternalTensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def isclose(cls, x: InternalTensor, y: InternalTensor,
                rtol: InternalTensor, atol: InternalTensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def allclose(cls, x: InternalTensor, y: InternalTensor,
                 rtol: InternalTensor, atol: InternalTensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def any(cls, x: InternalTensor, dim=None, keepdim=False) -> Tensor:
        raise NotImplementedError

    @classmethod
    def all(cls, x: InternalTensor, dim=None, keepdim=False) -> Tensor:
        raise NotImplementedError

    @classmethod
    def isin(cls, x: InternalTensor, y: InternalTensor,
             assume_unique = False, invert = False) -> Tensor:
        raise NotImplementedError

    @classmethod
    def maximum(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def minimum(cls, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def max(cls, x: InternalTensor, dim=None, keepdim=False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def min(cls, x: InternalTensor, dim=None, keepdim=False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def argmax(cls, x: InternalTensor, dim=None, keepdim=False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def argmin(cls, x: InternalTensor, dim=None, keepdim=False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def clamp(cls, x: InternalTensor, min: InternalTensor = None, max: InternalTensor = None) -> InternalTensor:
        raise NotImplementedError


    @classmethod
    def relu(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def leaky_relu(cls, x: InternalTensor, negative_slope: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def threshold(cls, x: InternalTensor, threshold: InternalTensor, value: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def tanh(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def sigmoid(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def logsigmoid(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def softmin(cls, x: InternalTensor, dim: int = None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def softmax(cls, x: InternalTensor, dim: int = None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def log_softmax(cls, x: InternalTensor, dim: int = None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def hardtanh(cls, x: InternalTensor, min_val: InternalTensor, max_val: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def glu(cls, x: InternalTensor, dim: int = -1) -> InternalTensor:
        raise NotImplementedError


    @classmethod
    def broadcast_to(cls, x: InternalTensor, shape) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def clone(cls, x: InternalTensor, memory_format = torch.preserve_format) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def squeeze(cls, x: InternalTensor, dim=None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def unsqueeze(cls, x: InternalTensor, dim: int) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def stack(cls, tensors: list[InternalTensor], dim: int = 0) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def cat(cls, tensors: list[InternalTensor], dim: int = 0) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def chunk(cls, x: InternalTensor, chunks: int, dim: int = 0) -> list[InternalTensor]:
        raise NotImplementedError

    @classmethod
    def where(cls, condition: Tensor, x: InternalTensor, y: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def pad(cls, x: InternalTensor, pad: tuple[int], mode: str = 'constant', value: InternalTensor = None):
        raise NotImplementedError

    @classmethod
    def getitem(cls, x: InternalTensor, index) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def setitem(cls, x: InternalTensor, index, value: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def to(cls, x: InternalTensor, device) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def view(cls, x: InternalTensor, shape) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def contiguous(cls, x: InternalTensor) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def repeat(cls, x: InternalTensor, repeats: tuple[int]) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def flatten(cls, x: InternalTensor, start_dim: int = 0, end_dim: int = -1) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def reshape(cls, x: InternalTensor, shape: tuple[int]) -> InternalTensor:
        raise NotImplementedError


    @classmethod
    def mse_loss(cls, x: InternalTensor, y: InternalTensor,
                 reduction: str = 'mean', weight: InternalTensor = None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def l1_loss(cls, x: InternalTensor, y: InternalTensor,
                 reduction: str = 'mean', weight: InternalTensor = None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def binary_cross_entropy(cls, x: InternalTensor, y: InternalTensor,
                             weight: InternalTensor = None, reduction: str = 'mean') -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def binary_cross_entropy_with_logits(cls, x: InternalTensor, y: InternalTensor,
                                         weight: InternalTensor = None, reduction: str = 'mean',
                                         pos_weight: InternalTensor = None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def nll_loss(cls, x: InternalTensor, y: InternalTensor,
                 weight: InternalTensor = None, reduction: str = 'mean',
                 ignore_index: int = -100) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def poisson_nll_loss(cls, x: InternalTensor, y: InternalTensor,
                         eps: InternalTensor, log_input: bool = True,
                         full: bool = False, reduction: str = 'mean') -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def hinge_embedding_loss(cls, x: InternalTensor, y: InternalTensor,
                             margin: InternalTensor = None, reduction: str = 'mean') -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def kl_div(cls, x: InternalTensor, y: InternalTensor,
                reduction: str = 'mean', log_target: bool = False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def margin_ranking_loss(cls, x1: InternalTensor, x2: InternalTensor,
                            y: InternalTensor, margin: InternalTensor,
                            reduction: str = 'mean') -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def gaussian_nll_loss(cls, x: InternalTensor, y: InternalTensor,
                          var: InternalTensor, eps: InternalTensor,
                          full: bool = False, reduction: str = 'mean') -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def huber_loss(cls, x: InternalTensor, y: InternalTensor,
                   delta: InternalTensor, reduction: str = 'mean',
                   weight: InternalTensor = None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def smooth_l1_loss(cls, x: InternalTensor, y: InternalTensor,
                       beta: InternalTensor, reduction: str = 'mean') -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def cross_entropy(cls, x: InternalTensor, y: InternalTensor,
                      weight: InternalTensor = None, ignore_index: int = -100,
                      reduction: str = 'mean', label_smoothing: InternalTensor = None) -> InternalTensor:
        raise NotImplementedError


    @classmethod
    def linear(cls, x: InternalTensor, weight: InternalTensor, bias: InternalTensor = None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def dropout(cls, x: InternalTensor, p: float = 0.5) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def conv2d(cls, x: InternalTensor, weight: InternalTensor,
               bias: InternalTensor = None, stride: tuple[int] = 1,
               padding: tuple[int] = 0, dilation: tuple[int] = 1,
               groups: int = 1) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def avg_pool2d(cls, x: InternalTensor, kernel_size: tuple[int],
                  stride: tuple[int] = None, padding: tuple[int] = 0,
                  ceil_mode: bool = False, count_include_pad: bool = True,
                  divisor_override=None) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def adaptive_avg_pool2d(cls, x: InternalTensor, output_size: tuple[int]) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def max_pool2d(cls, x: InternalTensor, kernel_size: tuple[int],
                  stride: tuple[int] = None, padding: tuple[int] = 0,
                  dilation: tuple[int] = 1, ceil_mode: bool = False,
                  return_indices: bool = False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def batch_norm(cls, x: InternalTensor, running_mean: InternalTensor,
                   running_var: InternalTensor, momentum: InternalTensor,
                   eps: InternalTensor, weight: InternalTensor = None,
                   bias: InternalTensor = None, training: bool = False) -> InternalTensor:
        raise NotImplementedError

    @classmethod
    def layer_norm(cls, x: InternalTensor, eps: InternalTensor,
                   normalized_shape: tuple[int], weight: InternalTensor = None,
                   bias: InternalTensor = None) -> InternalTensor:
        raise NotImplementedError


    @classmethod
    def sgd_step(cls, params, grads, momentum_buffers,
                 lr: InternalTensor, momentum: InternalTensor,
                 dampening: InternalTensor, weight_decay: InternalTensor,
                 nesterov: bool, maximize: bool, use_momentum: bool):
        raise NotImplementedError

    @classmethod
    def madam_step(cls, params, grads, exp_avg_sqs, maxima,
                   lr: InternalTensor, beta: InternalTensor,
                   eps: InternalTensor, g_bound: InternalTensor,
                   bias_corr: InternalTensor, use_pow: bool,
                   maximize: bool):
        raise NotImplementedError

    @classmethod
    def adam_step(cls, params, grads, exp_avgs, exp_avg_sqs,
                  max_exp_avg_sqs, lr: InternalTensor,
                  beta1: InternalTensor, beta2: InternalTensor,
                  eps: InternalTensor, weight_decay: InternalTensor,
                  bias_corr1: InternalTensor, bias_corr2: InternalTensor,
                  amsgrad: bool, maximize: bool):
        raise NotImplementedError


_HELPER_METHODS = {
    "backend_for_device",
    "clear_scalar_cache",
    "direct_for_backend",
    "direct_for_device",
    "encoded_scalar",
    "enable_backend",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "resolve_device",
    "scalar_from_float",
    "scalar_to_float",
    "sum_to_size",
    "zeros",
    "zeros_like",
}
OpsBase._op_names = {
    name
    for name, value in OpsBase.__dict__.items()
    if isinstance(value, classmethod) and not name.startswith("_") and name not in _HELPER_METHODS
}
