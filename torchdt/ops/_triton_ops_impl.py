import torch
from torch.nn import _reduction as _Reduction
from torchdt.autograd import DTFunction
import math

from .triton_ops import TritonAccumulatorOps, TritonScalarOps

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def register_triton_ops(
    dtype_cls: type,
    scalar_ops: TritonScalarOps,
    accumulator_ops: TritonAccumulatorOps = None,
) -> None:
    if not HAS_TRITON:
        raise ImportError("Triton is not installed. Please install Triton to use Triton backend.")
    if not isinstance(scalar_ops, TritonScalarOps):
        raise TypeError("scalar_ops must be a TritonScalarOps instance.")
    if accumulator_ops is not None and not isinstance(accumulator_ops, TritonAccumulatorOps):
        raise TypeError("accumulator_ops must be a TritonAccumulatorOps instance.")

    from_float = scalar_ops.from_float
    to_float = scalar_ops.to_float
    add = scalar_ops.add
    sub = scalar_ops.sub
    mul = scalar_ops.mul
    div = scalar_ops.div
    sqrt = scalar_ops.sqrt
    gt = scalar_ops.gt
    ge = scalar_ops.ge
    lt = scalar_ops.lt
    le = scalar_ops.le
    neg = scalar_ops.neg
    exp = scalar_ops.exp
    log = scalar_ops.log
    clamp = scalar_ops.clamp
    sign = scalar_ops.sign

    cpu_from_float = dtype_cls.ops.from_float
    cpu_to_float = dtype_cls.ops.to_float

    def scalar_from_float_cpu(cls, x):
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                raise ValueError("scalar_from_float expects a scalar value")
            x_tensor = x.detach().to(device="cpu", dtype=torch.float32).reshape(())
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        return cpu_from_float(x_tensor)

    def scalar_to_float_cpu(cls, x):
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                raise ValueError("scalar_to_float expects a scalar tensor")
            x_tensor = x.detach().to(device="cpu", dtype=dtype_cls.int_dtype).reshape(())
        else:
            x_tensor = torch.tensor(x, dtype=dtype_cls.int_dtype)
        return cpu_to_float(x_tensor).item()

    dtype_cls.ops.scalar_from_float = classmethod(scalar_from_float_cpu)
    dtype_cls.ops.scalar_to_float = classmethod(scalar_to_float_cpu)

    if dtype_cls.bitwidth == 8:
        tl_int_dtype = tl.constexpr(tl.int8)
    elif dtype_cls.bitwidth == 16:
        tl_int_dtype = tl.constexpr(tl.int16)
    elif dtype_cls.bitwidth == 32:
        tl_int_dtype = tl.constexpr(tl.int32)
    elif dtype_cls.bitwidth == 64:
        tl_int_dtype = tl.constexpr(tl.int64)

    _ZERO = tl.constexpr(dtype_cls.ops.scalar_from_float(0.0).item())
    _NEG_INF = tl.constexpr(dtype_cls.ops.scalar_from_float(float("-inf")).item())
    _ONE = tl.constexpr(dtype_cls.ops.scalar_from_float(1.0).item())

    can_register_sign = sign is not None or (lt is not None and neg is not None)

    if exp is None:
        @triton.jit
        def exp(x):
            return from_float(tl.exp(to_float(x)))

    if log is None:
        @triton.jit
        def log(x):
            return from_float(tl.log(to_float(x)))

    if clamp is None:
        @triton.jit
        def clamp(x, min, max):
            return tl.where(lt(x, min), min, tl.where(gt(x, max), max, x))

    if sign is None:
        @triton.jit
        def sign(x):
            return tl.where(
                x == _ZERO, tl.cast(_ZERO, tl_int_dtype),
                tl.where(
                    lt(x, tl.cast(_ZERO, tl_int_dtype)), neg(tl.cast(_ONE, tl_int_dtype)), tl.cast(_ONE, tl_int_dtype)
                )
            )

    @triton.jit
    def max_combine_fn(a, b):
        return tl.where(gt(a, b), a, b)

    @triton.jit
    def atomic_add(x_ptrs, val, mask):
        active = mask

        while tl.max(active) != 0:
            old = tl.load(x_ptrs, mask=active, other=_ZERO)
            new = add(old, tl.where(active, val, tl.cast(_ZERO, tl_int_dtype)))
            prev = tl.atomic_cas(x_ptrs, old, new)

            success = prev == old
            active = active & (~success)

    @triton.jit
    def elementwise_unary_offsets(linear_offsets, shape_ptr, x_stride_ptr, NDIM: tl.constexpr):
        remaining = linear_offsets
        x_offsets = tl.cast(linear_offsets * 0, tl.int64)

        for rev_dim in tl.static_range(0, NDIM):
            dim = NDIM - 1 - rev_dim
            dim_size = tl.load(shape_ptr + dim)
            dim_index = remaining % dim_size
            remaining = remaining // dim_size

            x_stride = tl.load(x_stride_ptr + dim)
            x_offsets = x_offsets + dim_index * x_stride

        return x_offsets

    @triton.jit
    def elementwise_binary_offsets(linear_offsets, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM: tl.constexpr):
        remaining = linear_offsets
        x_offsets = tl.cast(linear_offsets * 0, tl.int64)
        y_offsets = tl.cast(linear_offsets * 0, tl.int64)

        for rev_dim in tl.static_range(0, NDIM):
            dim = NDIM - 1 - rev_dim
            dim_size = tl.load(shape_ptr + dim)
            dim_index = remaining % dim_size
            remaining = remaining // dim_size

            x_stride = tl.load(x_stride_ptr + dim)
            y_stride = tl.load(y_stride_ptr + dim)
            x_offsets = x_offsets + dim_index * x_stride
            y_offsets = y_offsets + dim_index * y_stride

        return x_offsets, y_offsets

    def _metadata_tensor(values, device):
        if len(values) == 0:
            values = (0,)
        return torch.tensor(values, dtype=torch.int64, device=device)

    def _prepare_unary(x, out_dtype):
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        if out.numel() == 0:
            return out, None, None, 0

        shape_meta = _metadata_tensor(tuple(x.shape), x.device)
        stride_meta = _metadata_tensor(tuple(x.stride()), x.device)
        return out, shape_meta, stride_meta, out.ndim

    @triton.jit
    def from_float_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets = elementwise_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

        out = from_float(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("from_float")
    def dt_from_float(ops, x):
        if x.device.type != "cuda":
            return cpu_from_float(x)

        out, shape_meta, stride_meta, ndim = _prepare_unary(x, dtype_cls.int_dtype)
        if out.numel() == 0:
            return out

        grid = (triton.cdiv(out.numel(), 1024),)
        from_float_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
        return out

    @triton.jit
    def to_float_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets = elementwise_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

        out = to_float(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("to_float")
    def dt_to_float(ops, x):
        if x.device.type != "cuda":
            return cpu_to_float(x)

        out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        if out.numel() == 0:
            return out

        shape_meta = _metadata_tensor(tuple(x.shape), x.device)
        stride_meta = _metadata_tensor(tuple(x.stride()), x.device)
        grid = (triton.cdiv(out.numel(), 1024),)
        to_float_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), out.ndim, BLOCK_SIZE=1024)
        return out

    def _normalize_binary_devices(x, y):
        if x.device == y.device:
            return x, y

        if x.dim() == 0 and x.device.type == "cpu":
            return x.to(device=y.device), y
        if y.dim() == 0 and y.device.type == "cpu":
            return x, y.to(device=x.device)

        raise RuntimeError(f"Expected tensors to be on the same device, got {x.device} and {y.device}.")

    def _prepare_binary(x, y, out_dtype):
        x, y = _normalize_binary_devices(x, y)
        out_shape = torch.broadcast_shapes(x.shape, y.shape)
        x = x.expand(out_shape)
        y = y.expand(out_shape)
        out = torch.empty(out_shape, dtype=out_dtype, device=x.device)

        if out.numel() == 0:
            return out, x, y, None, None, None, 0

        shape_meta = _metadata_tensor(tuple(out_shape), x.device)
        x_stride_meta = _metadata_tensor(tuple(x.stride()), x.device)
        y_stride_meta = _metadata_tensor(tuple(y.stride()), x.device)

        return out, x, y, shape_meta, x_stride_meta, y_stride_meta, out.ndim

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets, y_offsets = elementwise_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

        out = add(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("add")
    def dt_add(ops, x, y):
        out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim = _prepare_binary(x, y, dtype_cls.int_dtype)
        if out.numel() == 0:
            return out

        grid = (triton.cdiv(out.numel(), 1024),)
        add_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
        return out

    if sub is not None:
        @triton.jit
        def sub_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = elementwise_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = sub(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("sub")
        def dt_sub(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim = _prepare_binary(x, y, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            sub_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    if mul is not None:
        @triton.jit
        def mul_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = elementwise_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = mul(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("mul")
        def dt_mul(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim = _prepare_binary(x, y, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            mul_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    if div is not None:
        @triton.jit
        def div_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = elementwise_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = div(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("div")
        def dt_div(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim = _prepare_binary(x, y, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            div_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    if sqrt is not None:
        @triton.jit
        def sqrt_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets = elementwise_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

            out = sqrt(x)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("sqrt")
        def dt_sqrt(ops, x):
            out, shape_meta, stride_meta, ndim = _prepare_unary(x, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            sqrt_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    if neg is not None:
        @triton.jit
        def neg_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets = elementwise_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

            out = neg(x)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("neg")
        def dt_neg(ops, x):
            out, shape_meta, stride_meta, ndim = _prepare_unary(x, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            neg_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    @triton.jit
    def exp_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets = elementwise_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

        out = exp(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("exp")
    def dt_exp(ops, x):
        out, shape_meta, stride_meta, ndim = _prepare_unary(x, dtype_cls.int_dtype)
        if out.numel() == 0:
            return out

        grid = (triton.cdiv(out.numel(), 1024),)
        exp_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
        return out

    @triton.jit
    def log_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets = elementwise_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

        out = log(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("log")
    def dt_log(ops, x):
        out, shape_meta, stride_meta, ndim = _prepare_unary(x, dtype_cls.int_dtype)
        if out.numel() == 0:
            return out

        grid = (triton.cdiv(out.numel(), 1024),)
        log_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
        return out

    if can_register_sign:
        @triton.jit
        def sign_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets = elementwise_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

            out = sign(x)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("sign")
        def dt_sign(ops, x):
            out, shape_meta, stride_meta, ndim = _prepare_unary(x, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            sign_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    if lt is not None and neg is not None:
        @triton.jit
        def abs_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets = elementwise_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

            zero = tl.cast(_ZERO, tl_int_dtype)
            out = tl.where(lt(x, zero), neg(x), x)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("abs")
        def dt_abs(ops, x):
            out, shape_meta, stride_meta, ndim = _prepare_unary(x, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            abs_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    if gt is not None:
        @triton.jit
        def gt_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = elementwise_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = gt(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("gt")
        def dt_gt(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim = _prepare_binary(x, y, torch.bool)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            gt_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    if ge is not None:
        @triton.jit
        def ge_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = elementwise_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = ge(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("ge")
        def dt_ge(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim = _prepare_binary(x, y, torch.bool)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            ge_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    if lt is not None:
        @triton.jit
        def lt_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = elementwise_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = lt(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("lt")
        def dt_lt(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim = _prepare_binary(x, y, torch.bool)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            lt_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out

    if le is not None:
        @triton.jit
        def le_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = elementwise_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = le(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("le")
        def dt_le(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim = _prepare_binary(x, y, torch.bool)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            le_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, BLOCK_SIZE=1024)
            return out


    # @triton.jit
    # def relu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    #     pid = tl.program_id(0)
    #     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    #     mask = offs < N

    #     x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)

    #     out = tl.where(lt(x, _ZERO), _ZERO, x)
    #     tl.store(out_ptr + offs, out, mask=mask)

    # @dtype_cls.register_op("relu")
    # def dt_relu(ops, x):
    #     out = torch.empty(x.shape, dtype=dtype_cls.int_dtype, device=x.device)
    #     grid = (triton.cdiv(x.numel(), 1024),)
    #     relu_kernel[grid](x, out, x.numel(), BLOCK_SIZE=1024)
    #     return out

    # @triton.jit
    # def relu_backward_kernel(dy_ptr, y_ptr, dx_ptr, N, BLOCK_SIZE: tl.constexpr):
    #     pid = tl.program_id(0)
    #     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    #     mask = offs < N

    #     dy = tl.load(dy_ptr + offs, mask=mask, other=_ZERO)
    #     y = tl.load(y_ptr + offs, mask=mask, other=_ZERO)

    #     dx = tl.where(y == _ZERO, _ZERO, dy)
    #     tl.store(dx_ptr + offs, dx, mask=mask)

    # def relu_backward(grad_output, output):
    #     grad_input = torch.empty(grad_output.shape, dtype=dtype_cls.int_dtype, device=grad_output.device)
    #     grid = (triton.cdiv(grad_output.numel(), 1024),)
    #     relu_backward_kernel[grid](grad_output, grad_input, grad_input, grad_output.numel(), BLOCK_SIZE=1024)
    #     return grad_input

    # class DTReLUFunction(DTFunction):

    #     @staticmethod
    #     def forward(ops, x):
    #         return ops.relu(x)

    #     @staticmethod
    #     def setup_context(ctx, ops, inputs, output):
    #         ctx.save_for_backward(output)

    #     @staticmethod
    #     def backward(ctx, ops, grad_output):
    #         output, = ctx.saved_tensors
    #         return relu_backward(grad_output, output)

    # @dtype_cls.register_func(torch.nn.functional.relu, torch.Tensor.relu,
    #                  cast=("input",))
    # def dt_relu(input, inplace=False):
    #     result = DTReLUFunction.apply(input)
    #     return result


    # @triton.jit
    # def log_softmax_kernel(x_ptr, y_ptr, B, N, s_x_b, s_x_n, s_y_b, s_y_n, BLOCK_N: tl.constexpr):
    #     pid0 = tl.program_id(0)
    #     pid1 = tl.program_id(1)

    #     offs_n = tl.arange(0, BLOCK_N)
    #     n_idx = offs_n + pid1 * BLOCK_N

    #     x_ptrs = x_ptr + pid0 * s_x_b + n_idx * s_x_n
    #     y_ptrs = y_ptr + pid0 * s_y_b + n_idx * s_y_n

    #     mask = n_idx < N
    #     x = tl.load(x_ptrs, mask=mask, other=_NEG_INF)

    #     row_max = tl.reduce(x, 0, max_combine_fn)
    #     x_centered = sub(x, row_max)
    #     exps = exp(x_centered)
    #     sum_exp = tl.reduce(tl.where(mask, exps, _ZERO), 0, add)
    #     log_sum_exp = log(sum_exp)

    #     out = sub(x_centered, log_sum_exp)
    #     tl.store(y_ptrs, out, mask=mask)

    # @dtype_cls.register_op("log_softmax")
    # def log_softmax(ops, x, dim=None):
    #     assert x.dim() == 2
    #     BLOCK_N = 128

    #     B, N = x.shape
    #     y = torch.empty(x.shape, dtype=dtype_cls.int_dtype, device=x.device)

    #     num_tiles = (N + 128 - 1) // 128
    #     grid = (B, num_tiles)

    #     s_x_b, s_x_n = x.stride()
    #     s_y_b, s_y_n = y.stride()

    #     log_softmax_kernel[grid](x, y, B, N, s_x_b, s_x_n, s_y_b, s_y_n, BLOCK_N)
    #     return y

    # @triton.jit
    # def log_softmax_backward_kernel(dy_ptr, y_ptr, dx_ptr, B, N, s_dy_b, s_dy_n, s_y_b, s_y_n, s_dx_b, s_dx_n, BLOCK_N: tl.constexpr):
    #     pid0 = tl.program_id(0)
    #     pid1 = tl.program_id(1)

    #     offs_n = tl.arange(0, BLOCK_N)
    #     n_idx = pid1 * BLOCK_N + offs_n
    #     mask = n_idx < N

    #     dy_ptrs = dy_ptr + pid0 * s_dy_b + n_idx * s_dy_n
    #     y_ptrs  = y_ptr + pid0 * s_y_b + n_idx * s_y_n
    #     dx_ptrs = dx_ptr + pid0 * s_dx_b + n_idx * s_dx_n

    #     dy = tl.load(dy_ptrs, mask=mask, other=_ZERO)
    #     y = tl.load(y_ptrs, mask=mask, other=_NEG_INF)

    #     sum_dy = tl.reduce(dy, 0, combine_fn=add)
    #     softmax = exp(y)

    #     dx = sub(dy, mul(softmax, sum_dy))
    #     tl.store(dx_ptrs, dx, mask=mask)

    # def log_softmax_backward(grad_output, output):
    #     B, N = grad_output.shape
    #     BLOCK_N = 128

    #     grad_input = torch.empty(grad_output.shape, dtype=dtype_cls.int_dtype, device=grad_output.device)

    #     s_dy_b, s_dy_n = grad_output.stride()
    #     s_y_b, s_y_n = output.stride()
    #     s_dx_b, s_dx_n = grad_input.stride()

    #     grid = (B, (N + BLOCK_N - 1) // BLOCK_N)
    #     log_softmax_backward_kernel[grid](
    #         grad_output, output, grad_input,
    #         B, N,
    #         s_dy_b, s_dy_n,
    #         s_y_b, s_y_n,
    #         s_dx_b, s_dx_n,
    #         BLOCK_N
    #     )
    #     return grad_input

    # class DTLogSoftmaxFunction(DTFunction):

    #     @staticmethod
    #     def forward(ops, x, dim=None):
    #         return ops.log_softmax(x, dim=dim)

    #     @staticmethod
    #     def setup_context(ctx, ops, inputs, output):
    #         _, dim = inputs
    #         ctx.save_for_backward(output)
    #         ctx.dim = dim

    #     @staticmethod
    #     def backward(ctx, ops, grad_output):
    #         output, = ctx.saved_tensors
    #         return log_softmax_backward(grad_output, output), None

    # @dtype_cls.register_func(torch.nn.functional.log_softmax, torch.Tensor.log_softmax,
    #                  cast=("input",))
    # def dt_log_softmax(input, dim=None, _stacklevel=3, dtype=None, *, out=None):
    #     result = DTLogSoftmaxFunction.apply(input, dim)

    #     if out is not None:
    #         return out.copy_(result)
    #     return result


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128},  num_warps=2, num_stages=2),
            triton.Config({"BLOCK": 64},   num_warps=1, num_stages=2),
            triton.Config({"BLOCK": 128},  num_warps=1, num_stages=2),
            triton.Config({"BLOCK": 256},  num_warps=2, num_stages=2),
            triton.Config({"BLOCK": 256},  num_warps=4, num_stages=2),
        ],
        key=["N"],
    )
    @triton.jit
    def sum_kernel(x_ptr, y_ptr, M, N: tl.constexpr, s_x_r, s_x_c, s_y_r, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row_ptr = x_ptr + pid * s_x_r
        acc = tl.cast(_ZERO, tl_int_dtype)

        for tile_idx in range(0, tl.cdiv(N, BLOCK)):
            offs = tile_idx * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N

            vals = tl.load(row_ptr + offs * s_x_c, mask=mask, other=_ZERO)
            acc = add(acc, tl.reduce(vals, axis=0, combine_fn=add))

        tl.store(y_ptr + pid * s_y_r, acc)

    @dtype_cls.register_op("sum")
    def dt_sum(ops, x, dim=None, keepdim=False):
        orig_shape = x.shape
        ndim = x.dim()

        if dim is None:
            reduce_dims = tuple(range(ndim))
        elif isinstance(dim, int):
            reduce_dims = (dim,)
        else:
            reduce_dims = tuple(dim)

        reduce_dims = tuple(d + ndim if d < 0 else d for d in reduce_dims)
        reduce_dims = tuple(sorted(set(reduce_dims)))

        if len(reduce_dims) == 0:
            return x.clone()

        kept_dims = tuple(d for d in range(ndim) if d not in reduce_dims)
        perm = kept_dims + reduce_dims
        x_perm = x.permute(*perm)

        kept_shape = [orig_shape[d] for d in kept_dims]
        reduced_shape = [orig_shape[d] for d in reduce_dims]

        M = int(torch.prod(torch.tensor(kept_shape))) if kept_shape else 1
        N = int(torch.prod(torch.tensor(reduced_shape)))

        y = x_perm.reshape(M, N)
        stride_row, stride_col = y.stride()

        out = torch.empty((M,), device=x.device, dtype=dtype_cls.int_dtype)
        grid = (M,)

        sum_kernel[grid](
            y,
            out,
            M, N,
            stride_row, stride_col,
            out.stride(0),
        )

        if keepdim:
            out_shape = list(orig_shape)
            for d in reduce_dims:
                out_shape[d] = 1
            out = out.reshape(*out_shape)

        else:
            out_shape = [orig_shape[d] for d in kept_dims]
            out = out.reshape(*out_shape) if out_shape else out.view(())

        return out


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 8 }, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 8 }, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 8 }, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 8 }, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 4 }, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        BATCH, M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        tl.assume(BATCH >= 0)
        tl.assume(M >= 0)
        tl.assume(N >= 0)
        tl.assume(K >= 0)
        tl.assume(stride_am >= 0)
        tl.assume(stride_ak >= 0)
        tl.assume(stride_bn >= 0)
        tl.assume(stride_bk >= 0)
        tl.assume(stride_cm >= 0)
        tl.assume(stride_cn >= 0)

        base_a = a_ptr + pid_b * stride_ab
        base_b = b_ptr + pid_b * stride_bb
        base_c = c_ptr + pid_b * stride_cb

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.full((BLOCK_M, BLOCK_N), _ZERO, dtype=tl_int_dtype)

        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = k_offs < K

            for kk in tl.static_range(0, BLOCK_K):
                k = k0 * BLOCK_K + kk
                mask_k = k < K

                a_k_ptrs = base_a + offs_m * stride_am + k * stride_ak
                b_k_ptrs = base_b + k * stride_bk + offs_n * stride_bn

                a_k = tl.load(a_k_ptrs, mask=mask_m & mask_k, other=_ZERO)
                b_k = tl.load(b_k_ptrs, mask=mask_n & mask_k, other=_ZERO)

                acc = add(acc, mul(a_k[:, None], b_k[None, :]))

        c_ptrs = base_c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :] & (pid_b < BATCH))

    @dtype_cls.register_op("matmul")
    def dt_matmul(ops, A, B):
        a_was_1d = (A.ndim == 1)
        b_was_1d = (B.ndim == 1)

        if a_was_1d:
            A = A.unsqueeze(0)
        if b_was_1d:
            B = B.unsqueeze(1)

        if A.ndim < 2 or B.ndim < 2:
            raise ValueError("Inputs must be at least 1D")

        M, K_A = A.shape[-2:]
        K_B, N = B.shape[-2:]
        if K_A != K_B:
            raise ValueError(f"Incompatible dimensions: A(...,{M},{K_A}) and B(...,{K_B},{N})")

        A_batch = A.shape[:-2]
        B_batch = B.shape[:-2]
        try:
            batch_shape = torch.broadcast_shapes(A_batch, B_batch)
        except ValueError as e:
            raise ValueError("Incompatible batch dimensions for matmul") from e

        if A_batch != batch_shape:
            A = A.expand(*batch_shape, M, K_A)
        if B_batch != batch_shape:
            B = B.expand(*batch_shape, K_B, N)

        need_materialize = (A_batch != batch_shape) or (B_batch != batch_shape)
        if need_materialize or not A.is_contiguous():
            A = A.contiguous()
        if need_materialize or not B.is_contiguous():
            B = B.contiguous()

        if len(batch_shape) == 0:
            batch = 1
            A2 = A.reshape(1, M, K_A)
            B2 = B.reshape(1, K_B, N)
        else:
            batch = math.prod(batch_shape)
            A2 = A.reshape(batch, M, K_A)
            B2 = B.reshape(batch, K_B, N)

        C = torch.empty((batch, M, N), device=A.device, dtype=dtype_cls.int_dtype)

        stride_ab, stride_am, stride_ak = A2.stride()
        stride_bb, stride_bk, stride_bn = B2.stride()
        stride_cb, stride_cm, stride_cn = C.stride()

        grid = lambda META: (
            triton.cdiv(N, META["BLOCK_N"]),
            triton.cdiv(M, META["BLOCK_M"]),
            batch,
        )

        matmul_kernel[grid](
            A2, B2, C,
            batch, M, N, K_A,
            stride_ab, stride_am, stride_ak,
            stride_bb, stride_bk, stride_bn,
            stride_cb, stride_cm, stride_cn,
        )

        if len(batch_shape) == 0:
            C = C.reshape(M, N)
        else:
            C = C.reshape(*batch_shape, M, N)

        if a_was_1d and b_was_1d:
            C = C.squeeze(-1).squeeze(-2)
        elif a_was_1d:
            C = C.squeeze(-2)
        elif b_was_1d:
            C = C.squeeze(-1)

        return C

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_OC": 8,  "BLOCK_HW": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_OC": 4,  "BLOCK_HW": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_OC": 8,  "BLOCK_HW": 64},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_OC": 16, "BLOCK_HW": 64},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_OC": 4,  "BLOCK_HW": 256}, num_warps=4, num_stages=1),
        ],
        key=["Cin", "H", "W", "Cout", "Kh", "Kw", "Hout", "Wout", "sh", "sw", "ph", "pw", "dh", "dw", "groups"],
    )
    @triton.jit
    def conv2d_kernel(
        X_ptr, W_ptr, B_ptr, Y_ptr,
        N, Cin, H, W,
        Cout, Kh, Kw,
        Hout, Wout,
        sh, sw,
        ph, pw,
        dh, dw,
        groups,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_w_co, s_w_cinperg, s_w_kh, s_w_kw,
        s_y_n, s_y_c, s_y_h, s_y_w,
        BLOCK_OC: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        pid0 = tl.program_id(0) # spatial * oc
        pid1 = tl.program_id(1) # batch index
        pid2 = tl.program_id(2) # group index

        Cin_g = Cin // groups
        Cout_g = Cout // groups

        hw_tiles = tl.cdiv(Hout * Wout, BLOCK_HW)
        oc_tiles_per_group = tl.cdiv(Cout_g, BLOCK_OC)

        hw_block = pid0 % hw_tiles
        oc_block_in_group = pid0 // hw_tiles
        oc_base = pid2 * Cout_g + oc_block_in_group * BLOCK_OC

        oc_offsets = oc_base + tl.arange(0, BLOCK_OC)
        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)

        h = hw_offsets // Wout
        w = hw_offsets % Wout

        mask_oc = (oc_offsets < (pid2 + 1) * Cout_g) & (oc_offsets < Cout)
        mask_hw = hw_offsets < Hout * Wout
        mask_n = pid1 < N
        mask_group = pid2 < groups

        acc = tl.full((BLOCK_OC, BLOCK_HW), _ZERO, tl_int_dtype)

        Xb = X_ptr + pid1 * s_x_n
        Yb = Y_ptr + pid1 * s_y_n

        cin_group_start = pid2 * Cin_g
        Wb = W_ptr + oc_offsets[:, None] * s_w_co

        for icg in range(Cin_g):
            ic_base = Wb + icg * s_w_cinperg

            for ky in range(Kh):
                for kx in range(Kw):
                    in_h = h * sh + ky * dh - ph
                    in_w = w * sw + kx * dw - pw

                    in_bounds = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)

                    x_ptrs = Xb + (cin_group_start + icg) * s_x_c + in_h * s_x_h + in_w * s_x_w
                    x = tl.load(x_ptrs, mask=mask_hw & in_bounds & mask_n, other=_ZERO)

                    w_ptrs = ic_base + ky * s_w_kh + kx * s_w_kw
                    w_val = tl.load(w_ptrs, mask=mask_oc[:, None], other=_ZERO)

                    prod = mul(x[None, :], w_val)
                    acc = add(acc, prod)

        bias = tl.load(B_ptr + oc_offsets, mask=mask_oc, other=_ZERO)
        acc = add(acc, bias[:, None])

        out_ptrs = Yb + oc_offsets[:, None] * s_y_c + h[None, :] * s_y_h + w[None, :] * s_y_w
        tl.store(out_ptrs, acc, mask=mask_oc[:, None] & mask_hw & mask_n & mask_group)

    @dtype_cls.register_op("conv2d")
    def dt_conv2d(ops, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if bias is None:
            bias = torch.full((weight.shape[0],), _ZERO.value, device=x.device, dtype=x.dtype)

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        N, Cin, H, W = x.shape
        Cout, Cin_per_w, Kh, Kw = weight.shape
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation

        assert Cin % groups == 0, "Cin must be divisible by groups"
        assert Cout % groups == 0, "Cout must be divisible by groups"
        assert Cin_per_w == (Cin // groups), "w.shape[1] must equal Cin/groups"

        Kh_eff = dh * (Kh - 1) + 1
        Kw_eff = dw * (Kw - 1) + 1

        Hout = (H + 2 * ph - Kh_eff) // sh + 1
        Wout = (W + 2 * pw - Kw_eff) // sw + 1

        y = torch.empty((N, Cout, Hout, Wout), device=x.device, dtype=dtype_cls.int_dtype)

        s_x_n, s_x_c, s_x_h, s_x_w = x.stride()
        s_w_co, s_w_cinperg, s_w_kh, s_w_kw = weight.stride()
        s_y_n, s_y_c, s_y_h, s_y_w = y.stride()

        Cout_g = Cout // groups
        grid = lambda META: (
            triton.cdiv(Hout * Wout, META["BLOCK_HW"]) * triton.cdiv(Cout_g, META["BLOCK_OC"]),
            N,
            groups,
        )

        conv2d_kernel[grid](
            x, weight, bias, y,
            N, Cin, H, W,
            Cout, Kh, Kw,
            Hout, Wout,
            sh, sw,
            ph, pw,
            dh, dw,
            groups,
            s_x_n, s_x_c, s_x_h, s_x_w,
            s_w_co, s_w_cinperg, s_w_kh, s_w_kw,
            s_y_n, s_y_c, s_y_h, s_y_w,
        )

        return y

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_HW": 64},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_HW": 32},  num_warps=2, num_stages=1),
            triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=1),
        ],
        key=["Cin", "H", "W", "Cout", "Kh", "Kw", "Hout", "Wout", "sh", "sw", "ph", "pw", "dh", "dw", "groups"],
    )
    @triton.jit
    def conv2d_dinput_kernel(
        dX_ptr, dY_ptr, W_ptr,
        N, Cin, H, W,
        Cout, Kh, Kw,
        Hout, Wout,
        sh, sw,
        ph, pw,
        dh, dw,
        groups,
        s_dx_n, s_dx_c, s_dx_h, s_dx_w,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        s_w_co, s_w_cinperg, s_w_kh, s_w_kw,
        BLOCK_HW: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)

        n = pid0 // Cin
        cin = pid0 % Cin

        HW = H * W
        hw_start = pid1 * BLOCK_HW
        offs = tl.arange(0, BLOCK_HW)
        idx = hw_start + offs
        mask = idx < HW

        h = idx // W
        w = idx % W

        cin_per_g = Cin // groups
        cout_per_g = Cout // groups
        group_id = cin // cin_per_g
        base_w_cin = cin % cin_per_g
        cout_start = group_id * cout_per_g
        cout_end = cout_start + cout_per_g

        acc = tl.full((BLOCK_HW,), _ZERO, dtype=tl_int_dtype)

        for kh in range(Kh):
            numer_h = h + ph - kh * dh
            divisible_h = (numer_h % sh) == 0
            h_out = numer_h // sh

            for kw in range(Kw):
                numer_w = w + pw - kw * dw
                divisible_w = (numer_w % sw) == 0
                w_out = numer_w // sw

                valid_pos = mask & divisible_h & divisible_w & (h_out >= 0) & (h_out < Hout) & (w_out >= 0) & (w_out < Wout)

                for cout in range(cout_start, cout_end):
                    dy_idx = n * s_dy_n + cout * s_dy_c + h_out * s_dy_h + w_out * s_dy_w
                    dy_vals = tl.load(dY_ptr + dy_idx, mask=valid_pos, other=_ZERO)

                    w_idx = cout * s_w_co + base_w_cin * s_w_cinperg + kh * s_w_kh + kw * s_w_kw
                    w_val = tl.load(W_ptr + w_idx)

                    acc = add(acc, mul(dy_vals, w_val))

        dx_idx = n * s_dx_n + cin * s_dx_c + h * s_dx_h + w * s_dx_w
        tl.store(dX_ptr + dx_idx, acc, mask=mask)

    def conv2d_dinput(grad_output, weight, input_shape, stride, padding, dilation, groups):
        N, Cin, Hin, Win = input_shape
        N2, Cout, Hout, Wout = grad_output.shape
        Kh, Kw = weight.shape[2], weight.shape[3]

        grad_input = torch.empty((N, Cin, Hin, Win), device=grad_output.device, dtype=dtype_cls.int_dtype)

        sh, sw = stride[0], stride[1]
        ph, pw = padding[0], padding[1]
        dh, dw = dilation[0], dilation[1]

        s_dx_n, s_dx_c, s_dx_h, s_dx_w = grad_input.stride()
        s_dy_n, s_dy_c, s_dy_h, s_dy_w = grad_output.stride()
        s_w_co,  s_w_cinperg, s_w_kh, s_w_kw = weight.stride()

        grid = lambda META: (N * Cin, triton.cdiv(Hin * Win, META["BLOCK_HW"]))
        conv2d_dinput_kernel[grid](
            grad_input, grad_output, weight,
            N, Cin, Hin, Win,
            Cout, Kh, Kw,
            Hout, Wout,
            sh, sw,
            ph, pw,
            dh, dw,
            groups,
            s_dx_n, s_dx_c, s_dx_h, s_dx_w,
            s_dy_n, s_dy_c, s_dy_h, s_dy_w,
            s_w_co, s_w_cinperg, s_w_kh, s_w_kw,
        )
        return grad_input

    # @triton.jit
    # def conv2d_dweight_kernel(
    #     dW_ptr,
    #     X_ptr,
    #     dY_ptr,
    #     N, Cin, H, W,
    #     Cout, Kh, Kw,
    #     Hout, Wout,
    #     sh, sw,
    #     ph, pw,
    #     dh, dw,
    #     groups,
    #     s_dw_co, s_dw_cin, s_dw_kh, s_dw_kw,
    #     s_x_n, s_x_c, s_x_h, s_x_w,
    #     s_dy_n, s_dy_c, s_dy_h, s_dy_w,
    #     BLOCK_NHW: tl.constexpr,
    # ):
    #     pid0 = tl.program_id(0)
    #     pid1 = tl.program_id(1)

    #     Cin_per_group = Cin // groups
    #     Cout_per_group = Cout // groups

    #     cout = pid0 // (Cin_per_group * Kh * Kw)
    #     rem = pid0 - cout * (Cin_per_group * Kh * Kw)
    #     cin = rem // (Kh * Kw)
    #     rem2 = rem -  cin * (Kh * Kw)
    #     kh = rem2 // Kw
    #     kw = rem2 % Kw

    #     group_id = cout // Cout_per_group
    #     cin_abs = group_id * Cin_per_group + cin

    #     offs = pid1 * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    #     NHW = N * Hout * Wout
    #     mask = offs < NHW

    #     hw = Hout * Wout
    #     n = offs // hw
    #     r = offs - n * hw
    #     hout = r // Wout
    #     wout = r - hout * Wout

    #     h = hout * sh - ph + kh * dh
    #     w = wout * sw - pw + kw * dw
    #     inb = mask & (h >= 0) & (h < H) & (w >= 0) & (w < W)

    #     x_ptrs = X_ptr + n * s_x_n + cin_abs * s_x_c + h * s_x_h + w * s_x_w
    #     dy_ptrs = dY_ptr + n * s_dy_n + cout * s_dy_c + hout * s_dy_h + wout * s_dy_w

    #     x = tl.load(x_ptrs, mask=inb, other=_ZERO)
    #     dy = tl.load(dy_ptrs, mask=inb, other=_ZERO)

    #     partial = tl.reduce(mul(x, dy), axis=0, combine_fn=add)

    #     dw_idx = cout * s_dw_co + cin * s_dw_cin + kh * s_dw_kh + kw * s_dw_kw
    #     # atomic add with a scalar ptr
    #     atomic_add(dW_ptr + dw_idx + tl.zeros((1,), dtype=tl.int32), partial, tl.full((1,), True, tl.int1))

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_NHW": 64},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_NHW": 32},  num_warps=2, num_stages=1),
            triton.Config({"BLOCK_NHW": 128}, num_warps=4, num_stages=1),
        ],
        key=["N", "Cin", "H", "W", "Cout", "Kh", "Kw", "Hout", "Wout", "sh", "sw", "ph", "pw", "dh", "dw", "groups"],
    )
    @triton.jit
    def conv2d_dweight_kernel(
        dW_ptr,
        X_ptr,
        dY_ptr,
        N, Cin, H, W,
        Cout, Kh, Kw,
        Hout, Wout,
        sh, sw,
        ph, pw,
        dh, dw,
        groups,
        s_dw_co, s_dw_cin, s_dw_kh, s_dw_kw,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        BLOCK_NHW: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_NHW)

        Cin_per_group = Cin // groups
        Cout_per_group = Cout // groups

        cout = pid // (Cin_per_group * Kh * Kw)
        rem = pid - cout * (Cin_per_group * Kh * Kw)
        cin = rem // (Kh * Kw)
        rem2 = rem -  cin * (Kh * Kw)
        kh = rem2 // Kw
        kw = rem2 % Kw

        group_id = cout // Cout_per_group
        cin_abs = group_id * Cin_per_group + cin

        acc = tl.full((BLOCK_NHW,), _ZERO, dtype=tl_int_dtype)

        for nhw_start in range(0, N * Hout * Wout, BLOCK_NHW):
            idx = nhw_start + offs
            mask = idx < N * Hout * Wout

            n = idx // (Hout * Wout)
            rem3 = idx % (Hout * Wout)
            hout = rem3 // Wout
            wout = rem3 % Wout

            h = hout * sh - ph + kh * dh
            w = wout * sw - pw + kw * dw

            valid_mask = mask & (h >= 0) & (h < H) & (w >= 0) & (w < W)

            x_idx = n * s_x_n + cin_abs * s_x_c + h * s_x_h + w * s_x_w
            x_vals = tl.load(X_ptr + x_idx, mask=valid_mask, other=_ZERO)

            dy_idx = n * s_dy_n + cout * s_dy_c + hout * s_dy_h + wout * s_dy_w
            dy_vals = tl.load(dY_ptr + dy_idx, mask=valid_mask, other=_ZERO)

            prod = mul(dy_vals, x_vals)
            acc = add(prod, acc)

        dw_idx = cout * s_dw_co + cin * s_dw_cin + kh * s_dw_kh + kw * s_dw_kw
        tl.store(dW_ptr + dw_idx, tl.reduce(acc, axis=0, combine_fn=add))

    def conv2d_dweight(grad_output, input, weight_shape, stride, padding, dilation, groups):
        N, Cin, H, W = input.shape
        _, Cout, Hout, Wout = grad_output.shape
        Kh, Kw = weight_shape[2], weight_shape[3]

        Cin_per_group = Cin // groups
        grad_weight = torch.empty(weight_shape, device=grad_output.device, dtype=dtype_cls.int_dtype)

        sh, sw = stride[0], stride[1]
        ph, pw = padding[0], padding[1]
        dh, dw = dilation[0], dilation[1]

        s_dw_co, s_dw_cin, s_dw_kh, s_dw_kw = grad_weight.stride()
        s_x_n, s_x_c, s_x_h, s_x_w = input.stride()
        s_dy_n, s_dy_c, s_dy_h, s_dy_w = grad_output.stride()

        grid = (Cout * Cin_per_group * Kh * Kw,)
        conv2d_dweight_kernel[grid](
            grad_weight, input, grad_output,
            N, Cin, H, W,
            Cout, Kh, Kw,
            Hout, Wout,
            sh, sw,
            ph, pw,
            dh, dw,
            groups,
            s_dw_co, s_dw_cin, s_dw_kh, s_dw_kw,
            s_x_n, s_x_c, s_x_h, s_x_w,
            s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        )

        return grad_weight

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_NHW": 1024}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_NHW": 512},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_NHW": 2048}, num_warps=8, num_stages=1),
        ],
        key=["N", "Hout", "Wout"],
    )
    @triton.jit
    def conv2d_dbias_kernel(
        dB_ptr, dY_ptr,
        N, Cout, Hout, Wout,
        s_db_c,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        BLOCK_NHW: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_NHW)

        acc = tl.full((BLOCK_NHW,), _ZERO, dtype=tl_int_dtype)

        total = N * Hout * Wout
        for nhw_start in range(0, total, BLOCK_NHW):
            idx = nhw_start + offs
            mask = idx < total

            n = idx // (Hout * Wout)
            rem = idx % (Hout * Wout)
            hout = rem // Wout
            wout = rem % Wout

            dy_idx = n * s_dy_n + pid * s_dy_c + hout * s_dy_h + wout * s_dy_w
            dy_vals = tl.load(dY_ptr + dy_idx, mask=mask, other=_ZERO)

            acc = add(dy_vals, acc)

        db_idx = pid * s_db_c
        tl.store(dB_ptr + db_idx, acc.reduce(0, add))

    def conv2d_dbias(grad_output, bias_shape):
        N, Cout, Hout, Wout = grad_output.shape

        grad_bias = torch.empty(bias_shape, device=grad_output.device, dtype=dtype_cls.int_dtype)

        s_db_c, = grad_bias.stride()
        s_dy_n, s_dy_c, s_dy_h, s_dy_w = grad_output.stride()

        grid = (Cout,)
        conv2d_dbias_kernel[grid](
            grad_bias, grad_output,
            N, Cout, Hout, Wout,
            s_db_c,
            s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        )

        return grad_bias

    class DTConv2dFunction(DTFunction):

        @staticmethod
        def forward(ops, input, weight, bias, stride, padding, dilation, groups):
            return ops.conv2d(input, weight, bias, stride, padding, dilation, groups)

        @staticmethod
        def setup_context(ctx, ops, inputs, output):
            input, weight, bias, stride, padding, dilation, groups = inputs
            ctx.save_for_backward(input, weight, bias)
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.groups = groups

        @staticmethod
        def backward(ctx, ops, grad_output):
            input, weight, bias = ctx.saved_tensors
            stride = ctx.stride
            padding = ctx.padding
            dilation = ctx.dilation
            groups = ctx.groups

            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)

            # needs_input_grad pushes every index back by one since we pass ops...
            if ctx.needs_input_grad[1]:
                grad_input = conv2d_dinput(
                    grad_output, weight, input.shape,
                    stride, padding, dilation, groups
                )
            else:
                grad_input = None

            if weight is not None and ctx.needs_input_grad[2]:
                grad_weight = conv2d_dweight(
                    grad_output, input, weight.shape,
                    stride, padding, dilation, groups
                )
                # grad_weight = torch.nn.grad.conv2d_weight(
                #     ops.to_float(input), weight.shape,
                #     ops.to_float(grad_output),
                #     stride=stride, padding=padding,
                #     dilation=dilation, groups=groups
                # )
                # grad_weight = ops.from_float(grad_weight)
            else:
                grad_weight = None

            if bias is not None and ctx.needs_input_grad[3]:
                grad_bias = conv2d_dbias(
                    grad_output, bias.shape
                )
            else:
                grad_bias = None

            return grad_input, grad_weight, grad_bias, None, None, None, None

    @dtype_cls.register_func(torch.nn.functional.conv2d,
                             cast=("input", "weight", "bias"))
    def dt_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return DTConv2dFunction.apply(input, weight, bias, stride, padding, dilation, groups)


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_HW": 64},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_HW": 32},  num_warps=2, num_stages=1),
            triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=1),
        ],
        key=["H", "W", "Hout", "Wout", "Kh", "Kw", "sh", "sw", "ph", "pw", "dh", "dw"],
    )
    @triton.jit
    def max_pool2d_kernel(
        X_ptr, Y_ptr, idx_ptr,
        N, C, H, W,
        Hout, Wout,
        Kh, Kw,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_y_n, s_y_c, s_y_h, s_y_w,
        sh, sw,
        ph, pw,
        dh, dw,
        BLOCK_HW: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)
        pid2 = tl.program_id(2)

        offs = tl.arange(0, BLOCK_HW)

        n = pid0 // C
        c = pid0 - n * C

        ow = pid2 * BLOCK_HW + offs
        ow_mask = ow < Wout

        hstart = pid1 * sh - ph
        wstart = ow * sw - pw

        maxv = tl.full((BLOCK_HW,), _NEG_INF, tl_int_dtype)
        maxidx = tl.zeros([BLOCK_HW], tl.int64)

        base = X_ptr + n * s_x_n + c * s_x_c

        for kh in range(Kh):
            ih = hstart + kh * dh
            ih_ok = (ih >= 0) & (ih < H)

            for kw in range(Kw):
                iw = wstart + kw * dw
                iw_ok = (iw >= 0) & (iw < W)

                in_mask = ow_mask & ih_ok & iw_ok
                x_off = ih * s_x_h + iw * s_x_w
                xv = tl.load(base + x_off, mask=in_mask, other=_NEG_INF)

                better = gt(xv, maxv)
                maxv = tl.where(better, xv, maxv)

                cand_idx = ih * W + iw
                maxidx = tl.where(better, cand_idx, maxidx)

        y_off = n * s_y_n + c * s_y_c + pid1 * s_y_h + ow * s_y_w
        tl.store(Y_ptr + y_off, maxv, mask=ow_mask)
        tl.store(idx_ptr + y_off, maxidx, mask=ow_mask)

    def _pool_out_dim(in_size, kernel_size, padding, stride, dilation, ceil_mode):
        eff_k = dilation * (kernel_size - 1) + 1
        if ceil_mode:
            out = (in_size + 2 * padding - eff_k + stride - 1) // stride + 1
            if (out - 1) * stride >= in_size + padding:
                out -= 1
        else:
            out = (in_size + 2 * padding - eff_k) // stride + 1
        return max(out, 0)

    @dtype_cls.register_op("max_pool2d")
    def dt_max_pool2d(ops, x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)

        N, C, H, W = x.shape
        Kh, Kw = kernel_size[0], kernel_size[1]
        sh, sw = stride[0], stride[1]
        ph, pw = padding[0], padding[1]
        dh, dw = dilation[0], dilation[1]

        Hout = _pool_out_dim(H, Kh, ph, sh, dh, ceil_mode)
        Wout = _pool_out_dim(W, Kw, pw, sw, dw, ceil_mode)

        output = torch.empty((N, C, Hout, Wout), device=x.device, dtype=dtype_cls.int_dtype)
        indices = torch.empty((N, C, Hout, Wout), device=x.device, dtype=torch.int64)

        s_x_n, s_x_c, s_x_h, s_x_w = x.stride()
        s_y_n, s_y_c, s_y_h, s_y_w = output.stride()

        grid = lambda META: (N * C, Hout, triton.cdiv(Wout, META["BLOCK_HW"]))
        max_pool2d_kernel[grid](
            x, output, indices,
            N, C, H, W,
            Hout, Wout,
            Kh, Kw,
            s_x_n, s_x_c, s_x_h, s_x_w,
            s_y_n, s_y_c, s_y_h, s_y_w,
            sh, sw,
            ph, pw,
            dh, dw,
        )

        return (output, indices) if return_indices else output

    @triton.jit
    def max_pool2d_dinput_kernel(
        dY_ptr, dX_ptr, idx_ptr,
        N, C, H, W,
        Hout, Wout,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        s_dx_n, s_dx_c, s_dx_h, s_dx_w,
        s_idx_n, s_idx_c, s_idx_h, s_idx_w,
        BLOCK_W: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)
        pid2 = tl.program_id(2)

        offs_w = tl.arange(0, BLOCK_W)
        ow = pid2 * BLOCK_W + offs_w
        m_ow = ow < Wout
        oh = pid1

        n = pid0 // C
        c = pid0 - n * C

        dy_base = dY_ptr + n * s_dy_n + c * s_dy_c
        dx_base = dX_ptr + n * s_dx_n + c * s_dx_c
        idx_base = idx_ptr + n * s_idx_n + c * s_idx_c

        y_off_dy = oh * s_dy_h + ow * s_dy_w
        y_off_idx = oh * s_idx_h + ow * s_idx_w

        got_idx = tl.load(idx_base + y_off_idx, mask=m_ow, other=-1)
        dy = tl.load(dy_base + y_off_dy, mask=m_ow, other=_ZERO)

        valid = m_ow & (got_idx >= 0) & (got_idx < H * W)

        safe_idx = tl.where(valid, got_idx, 0)
        ih = safe_idx // W
        iw = safe_idx - ih * W

        ptr_x = dx_base + ih * s_dx_h + iw * s_dx_w
        atomic_add(ptr_x, dy, valid)

    def max_pool2d_dinput(grad_output, indices, input_shape):
        N, C, H, W = input_shape
        Hout = grad_output.shape[2]
        Wout = grad_output.shape[3]

        grad_input = torch.full((N, C, H, W), _ZERO.value, device=grad_output.device, dtype=dtype_cls.int_dtype)

        s_dy_n, s_dy_c, s_dy_h, s_dy_w = grad_output.stride()
        s_dx_n, s_dx_c, s_dx_h, s_dx_w = grad_input.stride()
        s_idx_n, s_idx_c, s_idx_h, s_idx_w = indices.stride()

        BLOCK_W = 128
        grid = (N * C, Hout, triton.cdiv(Wout, BLOCK_W))

        max_pool2d_dinput_kernel[grid](
            grad_output, grad_input, indices,
            N, C, H, W,
            Hout, Wout,
            s_dy_n, s_dy_c, s_dy_h, s_dy_w,
            s_dx_n, s_dx_c, s_dx_h, s_dx_w,
            s_idx_n, s_idx_c, s_idx_h, s_idx_w,
            BLOCK_W=BLOCK_W,
            num_warps=4,
        )

        return grad_input

    class DTMaxPool2dFunction(DTFunction):

        output_indices = [0]

        @staticmethod
        def forward(ctx, ops, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
            ctx.input_shape = input.shape
            output, indices = ops.max_pool2d(
                input, kernel_size,
                stride, padding, dilation,
                ceil_mode, return_indices=True
            )
            ctx.save_for_backward(indices)

            return (output, indices) if return_indices else output

        @staticmethod
        def backward(ctx, ops, grad_output):
            indices, = ctx.saved_tensors
            input_shape = ctx.input_shape

            return max_pool2d_dinput(grad_output, indices, input_shape), None, None, None, None, None, None

    @dtype_cls.register_func(torch.nn.functional.max_pool2d,
                             cast=("input",))
    def dt_max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        return DTMaxPool2dFunction.apply(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 8,  "BLOCK_HW": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_C": 4,  "BLOCK_HW": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_C": 8,  "BLOCK_HW": 64},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_C": 4,  "BLOCK_HW": 256}, num_warps=4, num_stages=1),
        ],
        key=["C", "H", "W", "Hout", "Wout", "Kh_max", "Kw_max"],
    )
    @triton.jit
    def adaptive_avg_pool2d_kernel(
        X_ptr, Y_ptr,
        N, C, H, W,
        Hout, Wout,
        Kh_max, Kw_max,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_y_n, s_y_c, s_y_h, s_y_w,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)

        hw_tiles = tl.cdiv(Hout * Wout, BLOCK_HW)

        hw_block = pid0 % hw_tiles
        c_block = pid0 // hw_tiles

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        oh = hw_offsets // Wout
        ow = hw_offsets % Wout

        mask_n = pid1 < N
        mask_c = c_offsets < C
        mask_hw = hw_offsets < (Hout * Wout)

        mask_c = mask_c[:, None]
        mask_hw = mask_hw[None, :]
        in_mask = mask_c & mask_n
        out_mask = in_mask & mask_hw

        Xb = X_ptr + pid1 * s_x_n
        Yb = Y_ptr + pid1 * s_y_n

        h_start = (oh * H) // Hout
        h_end = ((oh + 1) * H + (Hout - 1)) // Hout
        w_start = (ow * W) // Wout
        w_end = ((ow + 1) * W + (Wout - 1)) // Wout

        kh_len = h_end - h_start
        kw_len = w_end - w_start
        area = kh_len * kw_len

        acc = tl.full((BLOCK_C, BLOCK_HW), _ZERO, tl_int_dtype)

        for ky in range(Kh_max):
            ky_in = ky < kh_len
            in_h = h_start + ky

            for kx in range(Kw_max):
                kx_in = kx < kw_len
                in_w = w_start + kx

                valid_hw = ky_in & kx_in
                load_mask = in_mask & mask_hw & valid_hw[None, :]

                x_ptrs = Xb + c_offsets[:, None] * s_x_c + in_h[None, :] * s_x_h + in_w[None, :] * s_x_w
                x_vals = tl.load(x_ptrs, mask=load_mask, other=_ZERO)

                acc = add(x_vals, acc)

        area = tl.maximum(area, 1).to(tl.float32)
        acc = div(acc, from_float(area[None, :]))

        y_ptrs = Yb + c_offsets[:, None] * s_y_c + oh[None, :] * s_y_h + ow[None, :] * s_y_w
        tl.store(y_ptrs, acc, mask=out_mask)

    def _max_adaptive_window(in_size, out_size):
        m = 0
        for i in range(out_size):
            start = (i * in_size) // out_size
            end = ((i + 1) * in_size + (out_size - 1)) // out_size
            m = max(m, end - start)
        return m

    @dtype_cls.register_op("adaptive_avg_pool2d")
    def dt_adaptive_avg_pool2d(ops, x, output_size):
        N, C, H, W = x.shape
        Hout, Wout = output_size

        output = torch.empty((N, C, Hout, Wout), device=x.device, dtype=dtype_cls.int_dtype)

        s_x_n, s_x_c, s_x_h, s_x_w = x.stride()
        s_y_n, s_y_c, s_y_h, s_y_w = output.stride()

        Kh_max = _max_adaptive_window(H, Hout)
        Kw_max = _max_adaptive_window(W, Wout)

        grid = lambda META: (
            triton.cdiv(Hout * Wout, META["BLOCK_HW"]) * triton.cdiv(C, META["BLOCK_C"]),
            N,
        )
        adaptive_avg_pool2d_kernel[grid](
            x, output,
            N, C, H, W,
            Hout, Wout,
            Kh_max, Kw_max,
            s_x_n, s_x_c, s_x_h, s_x_w,
            s_y_n, s_y_c, s_y_h, s_y_w,
        )

        return output

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 8,  "BLOCK_HW": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_C": 4,  "BLOCK_HW": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_C": 8,  "BLOCK_HW": 64},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK_C": 4,  "BLOCK_HW": 256}, num_warps=4, num_stages=1),
        ],
        key=["C", "H", "W", "Hout", "Wout", "OVERLAP_H_MAX", "OVERLAP_W_MAX"],
    )
    @triton.jit
    def adaptive_avg_pool2d_dinput_kernel(
        dY_ptr,
        dX_ptr,
        N, C, H, W,
        Hout, Wout,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        s_dx_n, s_dx_c, s_dx_h, s_dx_w,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        OVERLAP_H_MAX: tl.constexpr,
        OVERLAP_W_MAX: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)

        hw_tiles_in = tl.cdiv(H * W, BLOCK_HW)

        hw_block = pid0 % hw_tiles_in
        c_block = pid0 // hw_tiles_in

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)

        ih = hw_offsets // W
        iw = hw_offsets % W

        mask_n = pid1 < N
        mask_c = c_offsets < C
        mask_hw = hw_offsets < (H * W)

        mask_c_2d = mask_c[:, None]
        mask_hw_2d = mask_hw[None, :]
        in_mask = mask_c_2d & mask_hw_2d & mask_n

        dYb = dY_ptr + pid1 * s_dy_n
        dXb = dX_ptr + pid1 * s_dx_n

        oh_low = tl.cdiv(ih * Hout + 1, H) - 1
        oh_high = tl.cdiv((ih + 1) * Hout, H) - 1
        oh_low = tl.maximum(oh_low, 0)
        oh_high = tl.minimum(oh_high, Hout - 1)
        oh_len = oh_high - oh_low + 1

        ow_low = tl.cdiv(iw * Wout + 1, W) - 1
        ow_high = tl.cdiv((iw + 1) * Wout, W) - 1
        ow_low = tl.maximum(ow_low, 0)
        ow_high = tl.minimum(ow_high, Wout - 1)
        ow_len = ow_high - ow_low + 1

        acc = tl.full((BLOCK_C, BLOCK_HW), _ZERO, tl_int_dtype)

        for dh in tl.static_range(OVERLAP_H_MAX):
            oh = oh_low + dh
            valid_oh = dh < oh_len

            h_start = (oh * H) // Hout
            h_end = ((oh + 1) * H + (Hout - 1)) // Hout
            kh_len = h_end - h_start

            for dw in tl.static_range(OVERLAP_W_MAX):
                ow = ow_low + dw
                valid_ow = dw < ow_len

                valid_hw = valid_oh & valid_ow
                load_mask = in_mask & valid_hw[None, :]

                w_start = (ow * W) // Wout
                w_end = ((ow + 1) * W + (Wout - 1)) // Wout
                kw_len = w_end - w_start

                area = kh_len * kw_len
                area = tl.maximum(area, 1).to(tl.float32)
                area_dt = from_float(area)[None, :]

                dy_ptrs = dYb + c_offsets[:, None] * s_dy_c + oh[None, :] * s_dy_h + ow[None, :] * s_dy_w
                dy_vals = tl.load(dy_ptrs, mask=load_mask, other=_ZERO)

                contrib = div(dy_vals, area_dt)
                acc = add(acc, contrib)

        dx_ptrs = dXb + c_offsets[:, None] * s_dx_c + ih[None, :] * s_dx_h + iw[None, :] * s_dx_w
        tl.store(dx_ptrs, acc, mask=in_mask)

    def _max_adaptive_overlap(in_size: int, out_size: int) -> int:
        m = 1
        for i in range(in_size):
            low = ((i * out_size + 1 + in_size - 1) // in_size) - 1
            high = (((i + 1) * out_size + in_size - 1) // in_size) - 1
            low = max(low, 0)
            high = min(high, out_size - 1)
            m = max(m, high - low + 1)
        return m

    def adaptive_avg_pool2d_dinput(grad_output, input_shape):
        N, C, H, W = input_shape
        Hout, Wout = grad_output.shape[2], grad_output.shape[3]

        grad_input = torch.empty((N, C, H, W), device=grad_output.device, dtype=dtype_cls.int_dtype)

        s_dy_n, s_dy_c, s_dy_h, s_dy_w = grad_output.stride()
        s_dx_n, s_dx_c, s_dx_h, s_dx_w = grad_input.stride()

        OVERLAP_H_MAX = _max_adaptive_overlap(H, Hout)
        OVERLAP_W_MAX = _max_adaptive_overlap(W, Wout)

        grid = lambda META: (
            triton.cdiv(H * W, META["BLOCK_HW"]) * triton.cdiv(C, META["BLOCK_C"]),
            N,
        )
        adaptive_avg_pool2d_dinput_kernel[grid](
            grad_output, grad_input,
            N, C, H, W,
            Hout, Wout,
            s_dy_n, s_dy_c, s_dy_h, s_dy_w,
            s_dx_n, s_dx_c, s_dx_h, s_dx_w,
            OVERLAP_H_MAX=OVERLAP_H_MAX,
            OVERLAP_W_MAX=OVERLAP_W_MAX,
        )

        return grad_input

    class DTAdaptiveAvgPool2dFunction(DTFunction):

        @staticmethod
        def forward(ctx, ops, input, output_size):
            ctx.input_shape = input.shape
            return ops.adaptive_avg_pool2d(input, output_size)

        @staticmethod
        def backward(ctx, ops, grad_output):
            input_shape = ctx.input_shape

            return adaptive_avg_pool2d_dinput(grad_output, input_shape), None

    @dtype_cls.register_func(torch.nn.functional.adaptive_avg_pool2d,
                             cast=("input",))
    def dt_adaptive_avg_pool2d(input, output_size):
        return DTAdaptiveAvgPool2dFunction.apply(input, output_size)


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK": 128}, num_warps=4, num_stages=1),
        ],
        key=["count", "HW", "W"],
    )
    @triton.jit
    def batch_norm2d_sum_kernel(
        X_ptr, partial_sum_ptr,
        HW, W, count,
        s_x_n, s_x_c, s_x_h, s_x_w,
        ps_s0, ps_s1,
        BLOCK: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)

        lane = tl.arange(0, BLOCK)
        idx = pid1 * BLOCK + lane
        mask = idx < count

        n = idx // HW
        rem = idx - n * HW
        h = rem // W
        w = rem - h * W

        x_ptrs = X_ptr + n * s_x_n + pid0 * s_x_c + h * s_x_h + w * s_x_w
        x = tl.load(x_ptrs, mask=mask, other=_ZERO)
        x = tl.where(mask, x, tl.cast(_ZERO, tl_int_dtype))

        block_sum = tl.reduce(x, axis=0, combine_fn=add)

        tl.store(partial_sum_ptr + pid0 * ps_s0 + pid1 * ps_s1, block_sum)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_T": 16},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK_T": 32},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK_T": 64},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK_T": 128}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_T": 128}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_T": 256}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_T": 512}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_T": 1024}, num_warps=4, num_stages=1),
        ],
        key=["ntiles"],
    )
    @triton.jit
    def batch_norm2d_mean_finalize_kernel(
        partial_sum_ptr,
        rm_ptr, sm_ptr,
        momentum, count_dt,
        ntiles,
        ps_s0, ps_s1,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK_T)

        momentum = tl.cast(momentum, tl_int_dtype)
        count_dt = tl.cast(count_dt, tl_int_dtype)

        acc_sum = tl.cast(_ZERO, tl_int_dtype)

        for t0 in range(0, ntiles, BLOCK_T):
            t = t0 + lane
            mask = t < ntiles

            s = tl.load(partial_sum_ptr + pid * ps_s0 + t * ps_s1, mask=mask, other=_ZERO)
            acc_sum = add(acc_sum, tl.reduce(s, axis=0, combine_fn=add))

        mean = div(acc_sum, count_dt)

        rm = tl.load(rm_ptr + pid)
        one_minus_m = sub(tl.cast(_ONE, tl_int_dtype), momentum)
        new_rm = add(mul(one_minus_m, rm), mul(momentum, mean))

        tl.store(rm_ptr + pid, new_rm)
        tl.store(sm_ptr + pid, mean)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK": 128}, num_warps=4, num_stages=1),
        ],
        key=["count", "HW", "W"],
    )
    @triton.jit
    def batch_norm2d_centered_var_kernel(
        X_ptr, sm_ptr, partial_var_ptr,
        HW, W, count,
        s_x_n, s_x_c, s_x_h, s_x_w,
        pv_s0, pv_s1,
        BLOCK: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)

        lane = tl.arange(0, BLOCK)
        idx = pid1 * BLOCK + lane
        mask = idx < count

        n = idx // HW
        rem = idx - n * HW
        h = rem // W
        w = rem - h * W

        mean = tl.load(sm_ptr + pid0)

        x_ptrs = X_ptr + n * s_x_n + pid0 * s_x_c + h * s_x_h + w * s_x_w
        x = tl.load(x_ptrs, mask=mask, other=_ZERO)

        centered = sub(x, mean)
        sq = mul(centered, centered)

        sq = tl.where(mask, sq, tl.cast(_ZERO, tl_int_dtype))
        block_var_sum = tl.reduce(sq, axis=0, combine_fn=add)

        tl.store(partial_var_ptr + pid0 * pv_s0 + pid1 * pv_s1, block_var_sum)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_T": 16},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK_T": 32},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK_T": 64},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK_T": 128}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_T": 128}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_T": 256}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_T": 512}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_T": 1024}, num_warps=4, num_stages=1),
        ],
        key=["ntiles"],
    )
    @triton.jit
    def batch_norm2d_var_finalize_kernel(
        partial_var_ptr,
        rv_ptr,
        sis_ptr,
        eps, momentum,
        count, count_dt,
        ntiles,
        pv_s0, pv_s1,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK_T)

        eps = tl.cast(eps, tl_int_dtype)
        momentum = tl.cast(momentum, tl_int_dtype)
        count_dt = tl.cast(count_dt, tl_int_dtype)

        acc_var_sum = tl.cast(_ZERO, tl_int_dtype)

        for t0 in range(0, ntiles, BLOCK_T):
            t = t0 + lane
            mask = t < ntiles

            v = tl.load(partial_var_ptr + pid * pv_s0 + t * pv_s1, mask=mask, other=_ZERO)
            acc_var_sum = add(acc_var_sum, tl.reduce(v, axis=0, combine_fn=add))

        var = div(acc_var_sum, count_dt)
        var = tl.where(lt(var, tl.cast(_ZERO, tl_int_dtype)), tl.cast(_ZERO, tl_int_dtype), var)

        if count > 1:
            sample_var = mul(var, div(count_dt, sub(count_dt, tl.cast(_ONE, tl_int_dtype))))
        else:
            sample_var = tl.cast(_ZERO, tl_int_dtype)

        rv = tl.load(rv_ptr + pid)

        one_minus_m = sub(tl.cast(_ONE, tl_int_dtype), momentum)
        new_rv = add(mul(one_minus_m, rv), mul(momentum, sample_var))

        tl.store(rv_ptr + pid, new_rv)

        invstd = div(tl.cast(_ONE, tl_int_dtype), sqrt(add(var, eps)))
        tl.store(sis_ptr + pid, invstd)

    @triton.jit
    def batch_norm2d_eval_stats_kernel(
        rm_ptr, rv_ptr,
        sm_ptr, sis_ptr,
        eps,
    ):
        pid = tl.program_id(0)

        eps = tl.cast(eps, tl_int_dtype)

        mean = tl.load(rm_ptr + pid)
        var = tl.load(rv_ptr + pid)
        var = tl.where(lt(var, tl.cast(_ZERO, tl_int_dtype)), tl.cast(_ZERO, tl_int_dtype), var)

        invstd = div(tl.cast(_ONE, tl_int_dtype), sqrt(add(var, eps)))
        tl.store(sm_ptr + pid, mean)
        tl.store(sis_ptr + pid, invstd)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 64},  num_warps=2, num_stages=1),
            triton.Config({"BLOCK": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        ],
        key=["count", "HW", "W"],
    )
    @triton.jit
    def batch_norm2d_apply_kernel(
        X_ptr, Y_ptr,
        w_ptr, b_ptr,
        sm_ptr, sis_ptr,
        HW, W, count,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_y_n, s_y_c, s_y_h, s_y_w,
        has_weight, has_bias,
        BLOCK: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)

        lane = tl.arange(0, BLOCK)
        idx = pid1 * BLOCK + lane
        mask = idx < count

        mean = tl.load(sm_ptr + pid0)
        invstd = tl.load(sis_ptr + pid0)

        if has_weight:
            weight = tl.load(w_ptr + pid0)
        else:
            weight = tl.cast(_ONE, tl_int_dtype)

        if has_bias:
            bias = tl.load(b_ptr + pid0)
        else:
            bias = tl.cast(_ZERO, tl_int_dtype)

        n = idx // HW
        rem = idx - n * HW
        h = rem // W
        w = rem - h * W

        x_ptrs = X_ptr + n * s_x_n + pid0 * s_x_c + h * s_x_h + w * s_x_w
        x = tl.load(x_ptrs, mask=mask, other=_ZERO)

        y = mul(sub(x, mean), invstd)
        y = add(mul(y, weight), bias)

        y_ptrs = Y_ptr + n * s_y_n + pid0 * s_y_c + h * s_y_h + w * s_y_w
        tl.store(y_ptrs, y, mask=mask)

    @dtype_cls.register_op("batch_norm")
    def dt_batch_norm(ops, x, running_mean, running_var, momentum, eps, weight=None, bias=None, training=False):
        PARTIAL_BLOCK = 128

        N, C, H, W = x.shape
        HW = H * W
        count = N * H * W
        partial_tiles = triton.cdiv(count, PARTIAL_BLOCK)

        has_weight = weight is not None
        has_bias = bias is not None

        if not has_weight:
            weight = torch.empty(0, device=x.device, dtype=dtype_cls.int_dtype)
        if not has_bias:
            bias = torch.empty(0, device=x.device, dtype=dtype_cls.int_dtype)

        count_dt = ops.scalar_from_float(count).item()
        eps_dt = ops.scalar_from_float(eps).item()
        momentum_dt = ops.scalar_from_float(momentum).item()

        output = torch.empty((N, C, H, W), device=x.device, dtype=dtype_cls.int_dtype)
        save_mean = torch.empty((C,), device=x.device, dtype=dtype_cls.int_dtype)
        save_invstd = torch.empty((C,), device=x.device, dtype=dtype_cls.int_dtype)

        s_x_n, s_x_c, s_x_h, s_x_w = x.stride()
        s_y_n, s_y_c, s_y_h, s_y_w = output.stride()

        if training:
            partial_sum = torch.empty((C, partial_tiles), device=x.device, dtype=dtype_cls.int_dtype)
            partial_var = torch.empty((C, partial_tiles), device=x.device, dtype=dtype_cls.int_dtype)

            ps_s0, ps_s1 = partial_sum.stride()
            pv_s0, pv_s1 = partial_var.stride()

            batch_norm2d_sum_kernel[(C, partial_tiles)](
                x, partial_sum,
                HW, W, count,
                s_x_n, s_x_c, s_x_h, s_x_w,
                ps_s0, ps_s1,
            )

            batch_norm2d_mean_finalize_kernel[(C,)](
                partial_sum,
                running_mean,
                save_mean,
                momentum_dt,
                count_dt,
                partial_tiles,
                ps_s0, ps_s1,
            )

            batch_norm2d_centered_var_kernel[(C, partial_tiles)](
                x, save_mean, partial_var,
                HW, W, count,
                s_x_n, s_x_c, s_x_h, s_x_w,
                pv_s0, pv_s1,
            )

            batch_norm2d_var_finalize_kernel[(C,)](
                partial_var,
                running_var,
                save_invstd,
                eps_dt, momentum_dt,
                count, count_dt,
                partial_tiles,
                pv_s0, pv_s1,
            )
        else:
            batch_norm2d_eval_stats_kernel[(C,)](
                running_mean,
                running_var,
                save_mean,
                save_invstd,
                eps_dt,
                num_warps=1,
            )

        grid_apply = lambda META: (C, triton.cdiv(count, META["BLOCK"]))
        batch_norm2d_apply_kernel[grid_apply](
            x, output,
            weight, bias,
            save_mean, save_invstd,
            HW, W, count,
            s_x_n, s_x_c, s_x_h, s_x_w,
            s_y_n, s_y_c, s_y_h, s_y_w,
            has_weight,
            has_bias,
        )

        return output, save_mean, save_invstd

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        ],
        key=["N", "HW", "W"],
    )
    @triton.jit
    def batch_norm2d_backward_partials_kernel(
        X_ptr, dY_ptr,
        p_dy_ptr, p_dy_xhat_ptr,
        sm_ptr, sis_ptr,
        N, HW, W,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        BLOCK: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)
        pid2 = tl.program_id(2)

        base = tl.arange(0, BLOCK)
        hw = pid2 * BLOCK + base
        mask = hw < HW

        mean = tl.load(sm_ptr + pid0)
        invstd = tl.load(sis_ptr + pid0)

        h = hw // W
        w = hw - h * W
        x_ptrs = X_ptr + pid1 * s_x_n + pid0 * s_x_c + h * s_x_h + w * s_x_w
        dy_ptrs = dY_ptr + pid1 * s_dy_n + pid0 * s_dy_c + h * s_dy_h + w * s_dy_w

        x = tl.load(x_ptrs, mask=mask, other=_ZERO)
        dy = tl.load(dy_ptrs, mask=mask, other=_ZERO)

        x = tl.where(mask, x, tl.cast(_ZERO, tl_int_dtype))
        dy = tl.where(mask, dy, tl.cast(_ZERO, tl_int_dtype))

        xhat = mul(sub(x, mean), invstd)
        xhat = tl.where(mask, xhat, tl.cast(_ZERO, tl_int_dtype))

        dy_xhat = mul(dy, xhat)
        dy_xhat = tl.where(mask, dy_xhat, tl.cast(_ZERO, tl_int_dtype))

        partial_dy = tl.reduce(dy, axis=0, combine_fn=add)
        partial_dy_xhat = tl.reduce(dy_xhat, axis=0, combine_fn=add)

        num_hw_blks = tl.cdiv(HW, BLOCK)
        tile_id = pid1 * num_hw_blks + pid2

        tl.store(p_dy_ptr + pid0 * (N * num_hw_blks) + tile_id, partial_dy)
        tl.store(p_dy_xhat_ptr + pid0 * (N * num_hw_blks) + tile_id, partial_dy_xhat)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_R": 64},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK_R": 128}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_R": 256}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_R": 256}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_R": 512}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_R": 512}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_R": 1024}, num_warps=4, num_stages=1),
        ],
        key=["K"],
    )
    @triton.jit
    def batch_norm2d_backward_reduce_kernel(
        p_dy_ptr, p_dy_xhat_ptr,
        dB_ptr, dW_ptr,
        m_dy_ptr, m_dy_xhat_ptr,
        count_dt, K,
        has_weight, has_bias,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base = tl.arange(0, BLOCK_R)

        sum_dy = tl.cast(_ZERO, tl_int_dtype)
        sum_dy_xhat = tl.cast(_ZERO, tl_int_dtype)

        for start in range(0, K, BLOCK_R):
            idx = start + base
            mask = idx < K

            pdy = tl.load(p_dy_ptr + pid * K + idx, mask=mask, other=_ZERO)
            pdyx = tl.load(p_dy_xhat_ptr + pid * K + idx, mask=mask, other=_ZERO)

            sum_dy = add(sum_dy, tl.reduce(pdy, axis=0, combine_fn=add))
            sum_dy_xhat = add(sum_dy_xhat, tl.reduce(pdyx, axis=0, combine_fn=add))

        if has_bias:
            tl.store(dB_ptr + pid, sum_dy)
        if has_weight:
            tl.store(dW_ptr + pid, sum_dy_xhat)

        tl.store(m_dy_ptr + pid, div(sum_dy, tl.cast(count_dt, tl_int_dtype)))
        tl.store(m_dy_xhat_ptr + pid, div(sum_dy_xhat, tl.cast(count_dt, tl_int_dtype)))

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 32},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 64},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 64},  num_warps=2, num_stages=1),
            triton.Config({"BLOCK": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        ],
        key=["HW", "W"],
    )
    @triton.jit
    def batch_norm2d_backward_dx_kernel(
        X_ptr, dY_ptr, dX_ptr,
        w_ptr, sm_ptr, sis_ptr,
        m_dy_ptr, m_dy_xhat_ptr,
        HW, W,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        s_dx_n, s_dx_c, s_dx_h, s_dx_w,
        has_weight,
        BLOCK: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)
        pid2 = tl.program_id(2)

        base = tl.arange(0, BLOCK)
        hw = pid2 * BLOCK + base
        mask = hw < HW

        mean = tl.load(sm_ptr + pid0)
        invstd = tl.load(sis_ptr + pid0)

        if has_weight:
            weight = tl.load(w_ptr + pid0)
        else:
            weight = tl.cast(_ONE, tl_int_dtype)
        w_over_std = mul(weight, invstd)

        mean_dy = tl.load(m_dy_ptr + pid0)
        mean_dy_xhat = tl.load(m_dy_xhat_ptr + pid0)

        h = hw // W
        w = hw - h * W

        x_ptrs = X_ptr + pid1 * s_x_n + pid0 * s_x_c + h * s_x_h + w * s_x_w
        dy_ptrs = dY_ptr + pid1 * s_dy_n + pid0 * s_dy_c + h * s_dy_h + w * s_dy_w
        dx_ptrs = dX_ptr + pid1 * s_dx_n + pid0 * s_dx_c + h * s_dx_h + w * s_dx_w

        x = tl.load(x_ptrs, mask=mask, other=_ZERO)
        dy = tl.load(dy_ptrs, mask=mask, other=_ZERO)

        xhat = mul(sub(x, mean), invstd)

        inner = sub(dy, mean_dy)
        inner = sub(inner, mul(xhat, mean_dy_xhat))
        dx = mul(w_over_std, inner)

        tl.store(dx_ptrs, dx, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 32},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 64},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 64},  num_warps=2, num_stages=1),
            triton.Config({"BLOCK": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        ],
        key=["HW", "W"],
    )
    @triton.jit
    def batch_norm2d_backward_dx_eval_kernel(
        dY_ptr, dX_ptr,
        w_ptr, sis_ptr,
        HW, W,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        s_dx_n, s_dx_c, s_dx_h, s_dx_w,
        has_weight,
        BLOCK: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)
        pid2 = tl.program_id(2)

        base = tl.arange(0, BLOCK)
        hw = pid2 * BLOCK + base
        mask = hw < HW

        invstd = tl.load(sis_ptr + pid0)

        if has_weight:
            weight = tl.load(w_ptr + pid0)
        else:
            weight = tl.cast(_ONE, tl_int_dtype)
        w_over_std = mul(weight, invstd)

        h = hw // W
        w = hw - h * W

        dy_ptrs = dY_ptr + pid1 * s_dy_n + pid0 * s_dy_c + h * s_dy_h + w * s_dy_w
        dx_ptrs = dX_ptr + pid1 * s_dx_n + pid0 * s_dx_c + h * s_dx_h + w * s_dx_w

        dy = tl.load(dy_ptrs, mask=mask, other=_ZERO)
        dx = mul(w_over_std, dy)

        tl.store(dx_ptrs, dx, mask=mask)

    def batch_norm2d_backward(
        input, grad_output,
        save_mean, save_invstd,
        weight=None, bias=None,
        training=False,
    ):
        PARTIAL_BLOCK = 256

        N, C, H, W = input.shape
        HW = H * W
        count = N * H * W

        has_weight = weight is not None
        has_bias = bias is not None

        count_dt = dtype_cls.ops.scalar_from_float(count).item()

        if not has_weight:
            weight = torch.empty(0, device=input.device, dtype=dtype_cls.int_dtype)
        if not has_bias:
            bias = torch.empty(0, device=input.device, dtype=dtype_cls.int_dtype)

        grad_input = torch.empty((N, C, H, W), device=input.device, dtype=dtype_cls.int_dtype)

        if has_weight:
            grad_weight = torch.empty((C,), device=input.device, dtype=dtype_cls.int_dtype)
        else:
            grad_weight = torch.empty(0, device=input.device, dtype=dtype_cls.int_dtype)

        if has_bias:
            grad_bias = torch.empty((C,), device=input.device, dtype=dtype_cls.int_dtype)
        else:
            grad_bias = torch.empty(0, device=input.device, dtype=dtype_cls.int_dtype)

        s_x_n, s_x_c, s_x_h, s_x_w = input.stride()
        s_dy_n, s_dy_c, s_dy_h, s_dy_w = grad_output.stride()
        s_dx_n, s_dx_c, s_dx_h, s_dx_w = grad_input.stride()

        num_hw_blks = triton.cdiv(HW, PARTIAL_BLOCK)
        K = N * num_hw_blks

        if training or has_weight or has_bias:
            partial_dy = torch.empty((C, K), device=input.device, dtype=dtype_cls.int_dtype)
            partial_dy_xhat = torch.empty((C, K), device=input.device, dtype=dtype_cls.int_dtype)

            mean_dy = torch.empty((C,), device=input.device, dtype=dtype_cls.int_dtype)
            mean_dy_xhat = torch.empty((C,), device=input.device, dtype=dtype_cls.int_dtype)

            grid_partials = (C, N, num_hw_blks)
            batch_norm2d_backward_partials_kernel[grid_partials](
                input, grad_output,
                partial_dy, partial_dy_xhat,
                save_mean, save_invstd,
                N, HW, W,
                s_x_n, s_x_c, s_x_h, s_x_w,
                s_dy_n, s_dy_c, s_dy_h, s_dy_w,
            )

            grid_reduce = (C,)
            batch_norm2d_backward_reduce_kernel[grid_reduce](
                partial_dy, partial_dy_xhat,
                grad_bias, grad_weight,
                mean_dy, mean_dy_xhat,
                count_dt, K,
                has_weight, has_bias,
            )

        else:
            mean_dy = None
            mean_dy_xhat = None

        grid_dx = lambda META: (C, N, triton.cdiv(HW, META["BLOCK"]))

        if training:
            batch_norm2d_backward_dx_kernel[grid_dx](
                input, grad_output, grad_input,
                weight, save_mean, save_invstd,
                mean_dy, mean_dy_xhat,
                HW, W,
                s_x_n, s_x_c, s_x_h, s_x_w,
                s_dy_n, s_dy_c, s_dy_h, s_dy_w,
                s_dx_n, s_dx_c, s_dx_h, s_dx_w,
                has_weight,
            )
        else:
            batch_norm2d_backward_dx_eval_kernel[grid_dx](
                grad_output, grad_input,
                weight, save_invstd,
                HW, W,
                s_dy_n, s_dy_c, s_dy_h, s_dy_w,
                s_dx_n, s_dx_c, s_dx_h, s_dx_w,
                has_weight,
            )

        if not has_weight:
            grad_weight = None
        if not has_bias:
            grad_bias = None

        return grad_input, grad_weight, grad_bias

    class DTBatchNormFunction(DTFunction):

        @staticmethod
        def forward(ctx, ops, x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
            output, save_mean, save_invstd = ops.batch_norm(
                x, running_mean, running_var,
                momentum, eps,
                weight, bias,
                training
            )

            ctx.save_for_backward(x, weight, bias, save_mean, save_invstd)
            ctx.training = training

            return output

        @staticmethod
        def backward(ctx, ops, grad_output):
            training = ctx.training
            x, weight, bias, save_mean, save_invstd = ctx.saved_tensors

            grad_input, grad_weight, grad_bias = batch_norm2d_backward(
                x, grad_output,
                save_mean, save_invstd,
                weight, bias,
                training
            )

            return grad_input, None, None, grad_weight, grad_bias, None, None, None

    @dtype_cls.register_func(torch.nn.functional.batch_norm,
                             cast=("input", "running_mean", "running_var", "weight", "bias"))
    def dt_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
        assert input.dim() == 4, "torchdt only supports 2D batch norm for now"
        return DTBatchNormFunction.apply(input, running_mean, running_var, weight, bias, training, momentum, eps)


    @triton.jit
    def nll_target_offsets(linear_offsets, target_shape_ptr, x_nonclass_stride_ptr, t_stride_ptr, TARGET_NDIM: tl.constexpr):
        remaining = linear_offsets
        x_offsets = tl.cast(linear_offsets * 0, tl.int64)
        t_offsets = tl.cast(linear_offsets * 0, tl.int64)

        for rev_dim in tl.static_range(0, TARGET_NDIM):
            dim = TARGET_NDIM - 1 - rev_dim
            dim_size = tl.load(target_shape_ptr + dim)
            dim_index = remaining % dim_size
            remaining = remaining // dim_size

            x_stride = tl.load(x_nonclass_stride_ptr + dim)
            t_stride = tl.load(t_stride_ptr + dim)
            x_offsets = x_offsets + dim_index * x_stride
            t_offsets = t_offsets + dim_index * t_stride

        return x_offsets, t_offsets

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 256},  num_warps=2, num_stages=1),
            triton.Config({"BLOCK": 512},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        ],
        key=["total", "HAS_WEIGHT", "HAS_DENOM", "TARGET_NDIM"],
    )
    @triton.jit
    def nll_loss_kernel(
        x_ptr, t_ptr, w_ptr, loss_ptr, denom_ptr,
        target_shape_ptr, x_nonclass_stride_ptr, t_stride_ptr,
        total, x_class_stride, w_stride,
        HAS_WEIGHT: tl.constexpr,
        HAS_DENOM: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        TARGET_NDIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total

        x_base_offsets, t_offsets = nll_target_offsets(
            offs, target_shape_ptr, x_nonclass_stride_ptr, t_stride_ptr, TARGET_NDIM
        )
        target = tl.load(t_ptr + t_offsets, mask=mask, other=IGNORE_INDEX)
        valid = mask & (target != IGNORE_INDEX)
        safe_target = tl.where(valid, target, 0)

        x_vals = tl.load(x_ptr + x_base_offsets + safe_target * x_class_stride, mask=valid, other=_ZERO)

        if HAS_WEIGHT:
            weights = tl.load(w_ptr + safe_target * w_stride, mask=valid, other=_ZERO)
            weighted = mul(x_vals, weights)
            denom = weights
        else:
            weighted = x_vals
            denom = tl.full((BLOCK,), _ONE, dtype=tl_int_dtype)

        loss = tl.where(valid, neg(weighted), tl.cast(_ZERO, tl_int_dtype))
        tl.store(loss_ptr + offs, loss, mask=mask)

        if HAS_DENOM:
            denom = tl.where(valid, denom, tl.cast(_ZERO, tl_int_dtype))
            tl.store(denom_ptr + offs, denom, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 256},  num_warps=2, num_stages=1),
            triton.Config({"BLOCK": 512},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        ],
        key=["total", "HAS_WEIGHT", "TARGET_NDIM"],
    )
    @triton.jit
    def nll_denominator_kernel(
        t_ptr, w_ptr, denom_ptr,
        target_shape_ptr, x_nonclass_stride_ptr, t_stride_ptr,
        total, w_stride,
        HAS_WEIGHT: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        TARGET_NDIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total

        _, t_offsets = nll_target_offsets(
            offs, target_shape_ptr, x_nonclass_stride_ptr, t_stride_ptr, TARGET_NDIM
        )
        target = tl.load(t_ptr + t_offsets, mask=mask, other=IGNORE_INDEX)
        valid = mask & (target != IGNORE_INDEX)
        safe_target = tl.where(valid, target, 0)

        if HAS_WEIGHT:
            denom = tl.load(w_ptr + safe_target * w_stride, mask=valid, other=_ZERO)
        else:
            denom = tl.full((BLOCK,), _ONE, dtype=tl_int_dtype)

        denom = tl.where(valid, denom, tl.cast(_ZERO, tl_int_dtype))
        tl.store(denom_ptr + offs, denom, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128},  num_warps=1, num_stages=1),
            triton.Config({"BLOCK": 256},  num_warps=2, num_stages=1),
            triton.Config({"BLOCK": 512},  num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        ],
        key=["total", "HAS_WEIGHT", "REDUCTION_NONE", "REDUCTION_MEAN", "TARGET_NDIM"],
    )
    @triton.jit
    def nll_loss_backward_kernel(
        dy_ptr, t_ptr, w_ptr, dx_ptr, denom_ptr,
        target_shape_ptr, dx_nonclass_stride_ptr, t_stride_ptr, dy_stride_ptr,
        total, dx_class_stride, w_stride,
        HAS_WEIGHT: tl.constexpr,
        REDUCTION_NONE: tl.constexpr,
        REDUCTION_MEAN: tl.constexpr,
        IGNORE_INDEX: tl.constexpr,
        TARGET_NDIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total

        dx_base_offsets, t_offsets = nll_target_offsets(
            offs, target_shape_ptr, dx_nonclass_stride_ptr, t_stride_ptr, TARGET_NDIM
        )
        target = tl.load(t_ptr + t_offsets, mask=mask, other=IGNORE_INDEX)
        valid = mask & (target != IGNORE_INDEX)
        safe_target = tl.where(valid, target, 0)

        if HAS_WEIGHT:
            weights = tl.load(w_ptr + safe_target * w_stride, mask=valid, other=_ZERO)
        else:
            weights = tl.full((BLOCK,), _ONE, dtype=tl_int_dtype)

        coeff = neg(weights)

        if REDUCTION_MEAN:
            denom = tl.load(denom_ptr)
            coeff = div(coeff, denom)

        if REDUCTION_NONE:
            _, dy_offsets = nll_target_offsets(
                offs, target_shape_ptr, dx_nonclass_stride_ptr, dy_stride_ptr, TARGET_NDIM
            )
            upstream = tl.load(dy_ptr + dy_offsets, mask=valid, other=_ZERO)
        else:
            upstream = tl.load(dy_ptr)

        coeff = mul(coeff, upstream)
        tl.store(dx_ptr + dx_base_offsets + safe_target * dx_class_stride, coeff, mask=valid)

    def _nll_target_shape_and_strides(x, target):
        if x.dim() < 1:
            raise ValueError("nll_loss input must have at least 1 dimension.")

        if x.dim() == 1:
            target_shape = tuple(target.shape)
            x_nonclass_strides = (0,) * target.dim()
            x_class_stride = x.stride(0)
        else:
            expected_target_shape = (x.shape[0], *x.shape[2:])
            if tuple(target.shape) != expected_target_shape:
                raise ValueError(
                    f"Expected target shape {expected_target_shape} for input shape {tuple(x.shape)}, "
                    f"got {tuple(target.shape)}."
                )
            target_shape = tuple(target.shape)
            x_nonclass_strides = (x.stride(0), *x.stride()[2:])
            x_class_stride = x.stride(1)

        return target_shape, x_nonclass_strides, x_class_stride

    def _nll_metadata(x, target):
        target_shape, x_nonclass_strides, x_class_stride = _nll_target_shape_and_strides(x, target)
        target_shape_meta = _metadata_tensor(target_shape, target.device)
        x_nonclass_stride_meta = _metadata_tensor(x_nonclass_strides, target.device)
        target_stride_meta = _metadata_tensor(tuple(target.stride()), target.device)
        return target_shape, target_shape_meta, x_nonclass_stride_meta, target_stride_meta, x_class_stride

    def _nll_empty_weight(device):
        return torch.empty(0, device=device, dtype=dtype_cls.int_dtype)

    def _nll_weight_and_stride(weight, device):
        has_weight = weight is not None
        if has_weight:
            return weight, weight.stride(0), True
        return _nll_empty_weight(device), 0, False

    def _nll_denominator(ops, target, weight, x_shape, ignore_index):
        if len(x_shape) == 1:
            target_shape = tuple(target.shape)
            x_nonclass_strides = (0,) * target.dim()
        else:
            target_shape = tuple(target.shape)
            x_nonclass_strides = (0,) * len(target_shape)

        target_shape_meta = _metadata_tensor(target_shape, target.device)
        x_nonclass_stride_meta = _metadata_tensor(x_nonclass_strides, target.device)
        target_stride_meta = _metadata_tensor(tuple(target.stride()), target.device)
        weight, weight_stride, has_weight = _nll_weight_and_stride(weight, target.device)

        denom_values = torch.empty(target_shape, device=target.device, dtype=dtype_cls.int_dtype)
        if denom_values.numel() == 0:
            return ops.sum(denom_values)

        grid = lambda META: (triton.cdiv(denom_values.numel(), META["BLOCK"]),)
        nll_denominator_kernel[grid](
            target, weight, denom_values,
            target_shape_meta, x_nonclass_stride_meta, target_stride_meta,
            denom_values.numel(), weight_stride,
            has_weight,
            ignore_index,
            len(target_shape),
        )
        return ops.sum(denom_values)

    @dtype_cls.register_op("nll_loss")
    def dt_nll_loss(ops, x, target, weight=None, reduction='mean', ignore_index=-100):
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"Invalid reduction: {reduction}")

        target_shape, target_shape_meta, x_nonclass_stride_meta, target_stride_meta, x_class_stride = _nll_metadata(x, target)
        loss = torch.empty(target_shape, device=x.device, dtype=dtype_cls.int_dtype)
        denom_values = torch.empty(target_shape if reduction == "mean" else (0,), device=x.device, dtype=dtype_cls.int_dtype)
        weight, weight_stride, has_weight = _nll_weight_and_stride(weight, x.device)

        if loss.numel() != 0:
            grid = lambda META: (triton.cdiv(loss.numel(), META["BLOCK"]),)
            nll_loss_kernel[grid](
                x, target, weight, loss, denom_values,
                target_shape_meta, x_nonclass_stride_meta, target_stride_meta,
                loss.numel(), x_class_stride, weight_stride,
                has_weight,
                reduction == "mean",
                ignore_index,
                len(target_shape),
            )

        if reduction == "none":
            return loss

        loss_sum = ops.sum(loss)
        if reduction == "sum":
            return loss_sum

        return ops.div(loss_sum, ops.sum(denom_values))

    def nll_loss_backward(ops, grad_output, target, weight, input_shape, reduction, ignore_index):
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"Invalid reduction: {reduction}")

        grad_input = torch.full(input_shape, _ZERO.value, device=grad_output.device, dtype=dtype_cls.int_dtype)
        target_shape, target_shape_meta, dx_nonclass_stride_meta, target_stride_meta, dx_class_stride = _nll_metadata(grad_input, target)
        weight, weight_stride, has_weight = _nll_weight_and_stride(weight, grad_output.device)

        if reduction == "none":
            dy_stride_meta = _metadata_tensor(tuple(grad_output.stride()), grad_output.device)
            denom = torch.empty(1, device=grad_output.device, dtype=dtype_cls.int_dtype)
        elif reduction == "mean":
            dy_stride_meta = _metadata_tensor((0,) * len(target_shape), grad_output.device)
            denom = _nll_denominator(ops, target, None if not has_weight else weight, input_shape, ignore_index)
        else:
            dy_stride_meta = _metadata_tensor((0,) * len(target_shape), grad_output.device)
            denom = torch.empty(1, device=grad_output.device, dtype=dtype_cls.int_dtype)

        if target.numel() != 0:
            grid = lambda META: (triton.cdiv(target.numel(), META["BLOCK"]),)
            nll_loss_backward_kernel[grid](
                grad_output, target, weight, grad_input, denom,
                target_shape_meta, dx_nonclass_stride_meta, target_stride_meta, dy_stride_meta,
                target.numel(), dx_class_stride, weight_stride,
                has_weight,
                reduction == "none",
                reduction == "mean",
                ignore_index,
                len(target_shape),
            )

        return grad_input

    class DTNLLLossFunction(DTFunction):

        @staticmethod
        def forward(ops, x, y, weight=None, reduction='mean', ignore_index=-100):
            return ops.nll_loss(x, y, weight, reduction, ignore_index)

        @staticmethod
        def setup_context(ctx, ops, inputs, output):
            x, y, weight, reduction, ignore_index = inputs
            ctx.save_for_backward(x, y, weight)
            ctx.reduction = reduction
            ctx.ignore_index = ignore_index

        @staticmethod
        def backward(ctx, ops, grad_output):
            x, y, weight = ctx.saved_tensors
            grad_input = nll_loss_backward(
                ops, grad_output, y, weight, x.shape, ctx.reduction, ctx.ignore_index
            )
            return grad_input, None, None, None, None

    @dtype_cls.register_func(torch.nn.functional.nll_loss,
                             cast=("input", "weight"))
    def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        if size_average is not None or reduce is not None:
            reduction = _Reduction.legacy_get_string(size_average, reduce)
        return DTNLLLossFunction.apply(input, target, weight, reduction, ignore_index)


    @triton.jit
    def sgd_step_kernel(
        p_ptr, g_ptr, buf_ptr,
        lr_ptr, momentum_ptr, dampening_ptr, weight_decay_ptr,
        N,
        MAXIMIZE: tl.constexpr, NESTEROV: tl.constexpr,
        FIRST_MOMENTUM: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        p = tl.load(p_ptr + offs, mask=mask, other=_ZERO)
        g = tl.load(g_ptr + offs, mask=mask, other=_ZERO)
        lr = tl.load(lr_ptr)
        momentum = tl.load(momentum_ptr)
        dampening = tl.load(dampening_ptr)
        weight_decay = tl.load(weight_decay_ptr)

        if MAXIMIZE:
            g = neg(g)

        if weight_decay != _ZERO:
            g = add(g, mul(p, tl.cast(weight_decay, tl_int_dtype)))

        if momentum != _ZERO:
            if FIRST_MOMENTUM:
                buf_new = g
            else:
                buf = tl.load(buf_ptr + offs, mask=mask, other=_ZERO)
                buf_new = add(mul(buf, tl.cast(momentum, tl_int_dtype)), mul(g, sub(tl.cast(_ONE, tl_int_dtype), tl.cast(dampening, tl_int_dtype))))

            tl.store(buf_ptr + offs, buf_new, mask=mask)

            if NESTEROV:
                g = add(g, mul(buf_new, tl.cast(momentum, tl_int_dtype)))
            else:
                g = buf_new

        p_new = sub(p, mul(g, tl.cast(lr, tl_int_dtype)))
        tl.store(p_ptr + offs, p_new, mask=mask)

    @dtype_cls.register_op("triton_sgd_step")
    def triton_sgd_step(ops, p, grad, buf, lr, momentum, dampening, weight_decay, nesterov, maximize):
        N = p.numel()
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        first_mom = False
        if buf is None:
            buf = torch.empty(p.shape, dtype=p.dtype, device=p.device)
            first_mom = True

        sgd_step_kernel[grid](
            p, grad, buf,
            lr, momentum, dampening, weight_decay,
            N,
            maximize, nesterov,
            first_mom, BLOCK
        )

        return buf


    @triton.jit
    def madam_step_kernel(
        p_ptr, g_ptr, exp_avg_sq_ptr,
        lr_ptr, beta_ptr, eps_ptr,
        g_bound_ptr, max_ptr, bias_corr_ptr,
        N,
        MAXIMIZE: tl.constexpr, USE_POW: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        p = tl.load(p_ptr + offs, mask=mask, other=_ZERO)
        g = tl.load(g_ptr + offs, mask=mask, other=_ZERO)
        v = tl.load(exp_avg_sq_ptr + offs, mask=mask, other=_ZERO)
        lr = tl.load(lr_ptr)
        beta = tl.load(beta_ptr)
        eps = tl.load(eps_ptr)
        g_bound = tl.load(g_bound_ptr)
        max = tl.load(max_ptr)
        bias_corr = tl.load(bias_corr_ptr)

        g2 = mul(g, g)
        v_new = add(
            mul(tl.cast(beta, tl_int_dtype), v),
            mul(sub(tl.cast(_ONE, tl_int_dtype), tl.cast(beta, tl_int_dtype)), g2)
        )
        tl.store(exp_avg_sq_ptr + offs, v_new, mask=mask)

        corr = add(div(v_new, tl.cast(bias_corr, tl_int_dtype)), tl.cast(eps, tl_int_dtype))
        denom = sqrt(corr)
        g_normed = div(g, denom)

        g_clipped = clamp(g_normed, neg(tl.cast(g_bound, tl_int_dtype)), tl.cast(g_bound, tl_int_dtype))
        delta = mul(mul(tl.cast(lr, tl_int_dtype), g_clipped), sign(p))

        if not MAXIMIZE:
            delta = neg(delta)

        if USE_POW:
            mul_update = mul(p, exp(delta))
        else:
            mul_update = mul(p, add(tl.cast(_ONE, tl_int_dtype), delta))

        p_new = clamp(mul_update, neg(tl.cast(max, tl_int_dtype)), tl.cast(max, tl_int_dtype))
        tl.store(p_ptr + offs, p_new, mask=mask)

    @dtype_cls.register_op("triton_madam_step")
    def triton_madam_step(ops, p, grad, exp_avg_sq, lr, beta, eps, g_bound, max, bias_corr, use_pow, maximize):
        N = p.numel()
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        madam_step_kernel[grid](
            p, grad, exp_avg_sq,
            lr, beta, eps,
            g_bound, max, bias_corr,
            N,
            maximize, use_pow,
            BLOCK=BLOCK,
        )
        return exp_avg_sq


    @triton.jit
    def adam_step_kernel(
        p_ptr, g_ptr,
        m_ptr, v_ptr, vhat_ptr,
        lr_ptr, beta1_ptr, beta2_ptr, eps_ptr, weight_decay_ptr,
        bias_corr1_ptr, bias_corr2_ptr,
        N,
        MAXIMIZE: tl.constexpr, AMSGRAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        p = tl.load(p_ptr + offs, mask=mask, other=_ZERO)
        g = tl.load(g_ptr + offs, mask=mask, other=_ZERO)
        lr = tl.load(lr_ptr)
        beta1 = tl.load(beta1_ptr)
        beta2 = tl.load(beta2_ptr)
        eps = tl.load(eps_ptr)
        weight_decay = tl.load(weight_decay_ptr)
        bias_corr1 = tl.load(bias_corr1_ptr)
        bias_corr2 = tl.load(bias_corr2_ptr)

        if MAXIMIZE:
            g = neg(g)

        if weight_decay != _ZERO:
            g = add(g, mul(p, tl.cast(weight_decay, tl_int_dtype)))

        m = tl.load(m_ptr + offs, mask=mask, other=_ZERO)
        v = tl.load(v_ptr + offs, mask=mask, other=_ZERO)

        m_new = add(
            mul(m, tl.cast(beta1, tl_int_dtype)),
            mul(g, sub(tl.cast(_ONE, tl_int_dtype), tl.cast(beta1, tl_int_dtype)))
        )

        g2 = mul(g, g)
        v_new = add(
            mul(v, tl.cast(beta2, tl_int_dtype)),
            mul(g2, sub(tl.cast(_ONE, tl_int_dtype), tl.cast(beta2, tl_int_dtype)))
        )

        if AMSGRAD:
            vhat = tl.load(vhat_ptr + offs, mask=mask, other=_ZERO)
            vhat_new = tl.where(gt(vhat, v_new), vhat, v_new)
            tl.store(vhat_ptr + offs, vhat_new, mask=mask)
            v_denom = vhat_new

        else:
            v_denom = v_new

        step_size = div(
            mul(tl.cast(lr, tl_int_dtype), sqrt(tl.cast(bias_corr2, tl_int_dtype))),
            tl.cast(bias_corr1, tl_int_dtype)
        )

        denom = add(sqrt(v_denom), tl.cast(eps, tl_int_dtype))
        step_update = mul(step_size, div(m_new, denom))
        p_new = sub(p, step_update)

        tl.store(p_ptr + offs, p_new, mask=mask)
        tl.store(m_ptr + offs, m_new, mask=mask)
        tl.store(v_ptr + offs, v_new, mask=mask)

    @dtype_cls.register_op("triton_adam_step")
    def triton_adam_step(ops,
                        p, grad, exp_avg, exp_avg_sq, max_exp_avg_sq,
                        lr, beta1, beta2, eps, weight_decay,
                        bias_corr1, bias_corr2, amsgrad, maximize):
        N = p.numel()
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        adam_step_kernel[grid](
            p, grad,
            exp_avg, exp_avg_sq, max_exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay,
            bias_corr1, bias_corr2,
            N,
            maximize, amsgrad,
            BLOCK=BLOCK,
        )

        return exp_avg, exp_avg_sq, max_exp_avg_sq
