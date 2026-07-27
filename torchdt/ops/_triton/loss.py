import torch
from torch.nn import _reduction as _Reduction

from torchdt.autograd import DTFunction

def register_ops(context):
    triton = context.triton
    tl = context.tl
    dtype_cls = context.dtype_cls

    from_float = context.from_float
    to_float = context.to_float
    add = context.add
    sub = context.sub
    mul = context.mul
    div = context.div
    sqrt = context.sqrt
    gt = context.gt
    ge = context.ge
    lt = context.lt
    le = context.le
    neg = context.neg
    exp = context.exp
    log = context.log
    clamp = context.clamp
    sign = context.sign

    acc_int_dtype = context.acc_int_dtype
    acc_from_float = context.acc_from_float
    acc_add = context.acc_add
    acc_div = context.acc_div
    to_accumulator = context.to_accumulator
    from_accumulator = context.from_accumulator

    tl_int_dtype = context.tl_int_dtype
    _ZERO = context.zero
    _NEG_INF = context.neg_inf
    _ONE = context.one
    _metadata_tensor = context.metadata_tensor
    can_register_sign = context.can_register_sign

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

    @triton.jit
    def nll_loss_reduce_kernel(
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
        x_vals = tl.load(
            x_ptr + x_base_offsets + safe_target * x_class_stride,
            mask=valid, other=_ZERO,
        )

        if HAS_WEIGHT:
            weights = tl.load(w_ptr + safe_target * w_stride, mask=valid, other=_ZERO)
            weighted = mul(x_vals, weights)
            denominator = weights
        else:
            weighted = x_vals
            denominator = tl.full((BLOCK,), _ONE, dtype=tl_int_dtype)

        loss = tl.where(valid, neg(weighted), tl.cast(_ZERO, tl_int_dtype))
        loss_partial = tl.reduce(to_accumulator(loss), axis=0, combine_fn=acc_add)
        tl.store(loss_ptr + pid, from_accumulator(loss_partial))

        if HAS_DENOM:
            denominator = tl.where(valid, denominator, tl.cast(_ZERO, tl_int_dtype))
            denom_partial = tl.reduce(to_accumulator(denominator), axis=0, combine_fn=acc_add)
            tl.store(denom_ptr + pid, from_accumulator(denom_partial))

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

    @dtype_cls.register_op("nll_loss", backend="triton")
    def dt_nll_loss(ops, x, target, weight=None, reduction='mean', ignore_index=-100, return_denominator=False):
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"Invalid reduction: {reduction}")

        target_shape, target_shape_meta, x_nonclass_stride_meta, target_stride_meta, x_class_stride = _nll_metadata(x, target)
        weight, weight_stride, has_weight = _nll_weight_and_stride(weight, x.device)
        total = target.numel()

        if reduction == "none":
            loss = torch.empty(target_shape, device=x.device, dtype=dtype_cls.int_dtype)
            denom_values = torch.empty((0,), device=x.device, dtype=dtype_cls.int_dtype)
            if total != 0:
                grid = lambda META: (triton.cdiv(total, META["BLOCK"]),)
                nll_loss_kernel[grid](
                    x, target, weight, loss, denom_values,
                    target_shape_meta, x_nonclass_stride_meta, target_stride_meta,
                    total, x_class_stride, weight_stride,
                    has_weight, False, ignore_index, len(target_shape),
                )
            return loss

        block = 512
        partial_count = triton.cdiv(total, block)
        loss_partials = torch.empty((partial_count,), device=x.device, dtype=dtype_cls.int_dtype)
        denom_partials = torch.empty(
            (partial_count if reduction == "mean" else 0,),
            device=x.device, dtype=dtype_cls.int_dtype,
        )
        if total != 0:
            nll_loss_reduce_kernel[(partial_count,)](
                x, target, weight, loss_partials, denom_partials,
                target_shape_meta, x_nonclass_stride_meta, target_stride_meta,
                total, x_class_stride, weight_stride,
                has_weight, reduction == "mean", ignore_index, len(target_shape),
                BLOCK=block, num_warps=4,
            )

        loss_sum = ops.sum(loss_partials)
        if reduction == "sum":
            denominator = torch.empty((0,), device=x.device, dtype=dtype_cls.int_dtype)
            result = loss_sum
        else:
            denominator = ops.sum(denom_partials)
            result = ops.div(loss_sum, denominator)

        return (result, denominator) if return_denominator else result

    def nll_loss_backward(
        ops, grad_output, target, weight, input_shape, reduction, ignore_index,
        saved_denominator=None,
    ):
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
            if saved_denominator is not None and saved_denominator.numel() != 0:
                denom = saved_denominator
            else:
                denom = _nll_denominator(
                    ops, target, None if not has_weight else weight, input_shape, ignore_index
                )
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
        def forward(ctx, ops, x, y, weight=None, reduction='mean', ignore_index=-100):
            if reduction == "none":
                result = ops.nll_loss(x, y, weight, reduction, ignore_index)
                denominator = torch.empty((0,), device=x.device, dtype=dtype_cls.int_dtype)
            else:
                result, denominator = ops.nll_loss(
                    x, y, weight, reduction, ignore_index, True
                )
            ctx.save_for_backward(x, y, weight, denominator)
            ctx.reduction = reduction
            ctx.ignore_index = ignore_index
            return result

        @staticmethod
        def backward(ctx, ops, grad_output):
            x, y, weight, denominator = ctx.saved_tensors
            grad_input = nll_loss_backward(
                ops, grad_output, y, weight, x.shape, ctx.reduction, ctx.ignore_index,
                denominator,
            )
            return grad_input, None, None, None, None

    @dtype_cls.register_func(torch.nn.functional.nll_loss,
                             cast=("input", "weight"), backend="triton")
    def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        if size_average is not None or reduce is not None:
            reduction = _Reduction.legacy_get_string(size_average, reduce)
        return DTNLLLossFunction.apply(input, target, weight, reduction, ignore_index)


