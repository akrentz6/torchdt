import torch

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
    def elementwise_unary_offsets(linear_offsets, shape_ptr, x_stride_ptr, NDIM: tl.constexpr):
        remaining = linear_offsets
        x_offsets = tl.cast(linear_offsets * 0, tl.int64)

        for rev_dim in tl.static_range(0, NDIM):
            dim = NDIM - 1 - rev_dim
            dim_size = tl.cast(tl.load(shape_ptr + dim), tl.int64)
            dim_index = remaining % dim_size
            remaining = remaining // dim_size

            x_stride = tl.cast(tl.load(x_stride_ptr + dim), tl.int64)
            x_offsets = x_offsets + dim_index * x_stride

        return x_offsets

    @triton.jit
    def elementwise_binary_offsets(linear_offsets, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM: tl.constexpr):
        remaining = linear_offsets
        x_offsets = tl.cast(linear_offsets * 0, tl.int64)
        y_offsets = tl.cast(linear_offsets * 0, tl.int64)

        for rev_dim in tl.static_range(0, NDIM):
            dim = NDIM - 1 - rev_dim
            dim_size = tl.cast(tl.load(shape_ptr + dim), tl.int64)
            dim_index = remaining % dim_size
            remaining = remaining // dim_size

            x_stride = tl.cast(tl.load(x_stride_ptr + dim), tl.int64)
            y_stride = tl.cast(tl.load(y_stride_ptr + dim), tl.int64)
            x_offsets = x_offsets + dim_index * x_stride
            y_offsets = y_offsets + dim_index * y_stride

        return x_offsets, y_offsets

    def _prepare_unary(x, out_dtype):
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        if out.numel() == 0:
            return out, None, None, 0, True

        contiguous = x.is_contiguous()
        if contiguous:
            # Triton may type-check the unreachable strided branch. Give it an
            # integer pointer even though no metadata is loaded at runtime.
            metadata_dummy = x if x.dtype == dtype_cls.int_dtype else out
            return out, metadata_dummy, metadata_dummy, out.ndim, True

        shape_meta = _metadata_tensor(tuple(x.shape), x.device)
        stride_meta = _metadata_tensor(tuple(x.stride()), x.device)
        return out, shape_meta, stride_meta, out.ndim, False

    @triton.jit
    def prepared_unary_offsets(offs, shape_ptr, stride_ptr, NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr):
        offs = tl.cast(offs, tl.int64)
        if CONTIGUOUS:
            return offs
        return elementwise_unary_offsets(offs, shape_ptr, stride_ptr, NDIM)

    @triton.jit
    def from_float_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets = prepared_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM, CONTIGUOUS)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

        out = from_float(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("from_float", backend="triton")
    def dt_from_float(ops, x):
        out, shape_meta, stride_meta, ndim, contiguous = _prepare_unary(x, dtype_cls.int_dtype)
        if out.numel() == 0:
            return out

        grid = (triton.cdiv(out.numel(), 1024),)
        from_float_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, contiguous, BLOCK_SIZE=1024, num_warps=2)
        return out

    @triton.jit
    def to_float_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets = prepared_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM, CONTIGUOUS)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

        out = to_float(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("to_float", backend="triton")
    def dt_to_float(ops, x):
        out, shape_meta, stride_meta, ndim, contiguous = _prepare_unary(x, torch.float32)
        if out.numel() == 0:
            return out

        grid = (triton.cdiv(out.numel(), 1024),)
        to_float_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, contiguous, BLOCK_SIZE=1024, num_warps=2)
        return out

    def _prepare_binary(x, y, out_dtype):
        out_shape = torch.broadcast_shapes(x.shape, y.shape)
        x_scalar = x.numel() == 1
        y_scalar = y.numel() == 1
        x = x.expand(out_shape)
        y = y.expand(out_shape)
        out = torch.empty(out_shape, dtype=out_dtype, device=x.device)

        if out.numel() == 0:
            return out, x, y, None, None, None, 0, 1

        if x_scalar and y_scalar:
            mode = 4
        elif x_scalar and y.is_contiguous():
            mode = 2
        elif y_scalar and x.is_contiguous():
            mode = 3
        elif x.is_contiguous() and y.is_contiguous():
            mode = 1
        else:
            mode = 0

        if mode:
            return out, x, y, x, x, y, out.ndim, mode

        shape_meta = _metadata_tensor(tuple(out_shape), x.device)
        x_stride_meta = _metadata_tensor(tuple(x.stride()), x.device)
        y_stride_meta = _metadata_tensor(tuple(y.stride()), x.device)

        return out, x, y, shape_meta, x_stride_meta, y_stride_meta, out.ndim, mode

    @triton.jit
    def prepared_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM: tl.constexpr, MODE: tl.constexpr):
        offs = tl.cast(offs, tl.int64)
        if MODE == 1:
            return offs, offs
        if MODE == 2:
            return offs * 0, offs
        if MODE == 3:
            return offs, offs * 0
        if MODE == 4:
            return offs * 0, offs * 0
        return elementwise_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM)

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets, y_offsets = prepared_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM, MODE)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

        out = add(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("add", backend="triton")
    def dt_add(ops, x, y):
        out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim, mode = _prepare_binary(x, y, dtype_cls.int_dtype)
        if out.numel() == 0:
            return out

        grid = (triton.cdiv(out.numel(), 1024),)
        add_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, mode, BLOCK_SIZE=1024, num_warps=2)
        return out

    if sub is not None:
        @triton.jit
        def sub_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = prepared_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM, MODE)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = sub(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("sub", backend="triton")
        def dt_sub(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim, mode = _prepare_binary(x, y, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            sub_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, mode, BLOCK_SIZE=1024, num_warps=2)
            return out

    if mul is not None:
        @triton.jit
        def mul_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = prepared_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM, MODE)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = mul(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("mul", backend="triton")
        def dt_mul(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim, mode = _prepare_binary(x, y, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            mul_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, mode, BLOCK_SIZE=1024, num_warps=2)
            return out

    if div is not None:
        @triton.jit
        def div_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = prepared_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM, MODE)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = div(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("div", backend="triton")
        def dt_div(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim, mode = _prepare_binary(x, y, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            div_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, mode, BLOCK_SIZE=1024, num_warps=2)
            return out

    if sqrt is not None:
        @triton.jit
        def sqrt_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets = prepared_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM, CONTIGUOUS)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

            out = sqrt(x)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("sqrt", backend="triton")
        def dt_sqrt(ops, x):
            out, shape_meta, stride_meta, ndim, contiguous = _prepare_unary(x, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            sqrt_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, contiguous, BLOCK_SIZE=1024, num_warps=2)
            return out

    if neg is not None:
        @triton.jit
        def neg_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets = prepared_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM, CONTIGUOUS)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

            out = neg(x)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("neg", backend="triton")
        def dt_neg(ops, x):
            out, shape_meta, stride_meta, ndim, contiguous = _prepare_unary(x, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            neg_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, contiguous, BLOCK_SIZE=1024, num_warps=2)
            return out

    @triton.jit
    def exp_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets = prepared_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM, CONTIGUOUS)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

        out = exp(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("exp", backend="triton")
    def dt_exp(ops, x):
        out, shape_meta, stride_meta, ndim, contiguous = _prepare_unary(x, dtype_cls.int_dtype)
        if out.numel() == 0:
            return out

        grid = (triton.cdiv(out.numel(), 1024),)
        exp_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, contiguous, BLOCK_SIZE=1024, num_warps=2)
        return out

    @triton.jit
    def log_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x_offsets = prepared_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM, CONTIGUOUS)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

        out = log(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("log", backend="triton")
    def dt_log(ops, x):
        out, shape_meta, stride_meta, ndim, contiguous = _prepare_unary(x, dtype_cls.int_dtype)
        if out.numel() == 0:
            return out

        grid = (triton.cdiv(out.numel(), 1024),)
        log_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, contiguous, BLOCK_SIZE=1024, num_warps=2)
        return out

    if can_register_sign:
        @triton.jit
        def sign_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets = prepared_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM, CONTIGUOUS)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

            out = sign(x)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("sign", backend="triton")
        def dt_sign(ops, x):
            out, shape_meta, stride_meta, ndim, contiguous = _prepare_unary(x, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            sign_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, contiguous, BLOCK_SIZE=1024, num_warps=2)
            return out

    if lt is not None and neg is not None:
        @triton.jit
        def abs_kernel(x_ptr, out_ptr, shape_ptr, x_stride_ptr, N, NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets = prepared_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM, CONTIGUOUS)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

            zero = tl.cast(_ZERO, tl_int_dtype)
            out = tl.where(lt(x, zero), neg(x), x)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("abs", backend="triton")
        def dt_abs(ops, x):
            out, shape_meta, stride_meta, ndim, contiguous = _prepare_unary(x, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            abs_kernel[grid](x, out, shape_meta, stride_meta, out.numel(), ndim, contiguous, BLOCK_SIZE=1024, num_warps=2)
            return out

    if gt is not None:
        @triton.jit
        def gt_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = prepared_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM, MODE)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = gt(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("gt", backend="triton")
        def dt_gt(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim, mode = _prepare_binary(x, y, torch.bool)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            gt_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, mode, BLOCK_SIZE=1024, num_warps=2)
            return out

    if ge is not None:
        @triton.jit
        def ge_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = prepared_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM, MODE)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = ge(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("ge", backend="triton")
        def dt_ge(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim, mode = _prepare_binary(x, y, torch.bool)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            ge_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, mode, BLOCK_SIZE=1024, num_warps=2)
            return out

    if lt is not None:
        @triton.jit
        def lt_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = prepared_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM, MODE)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = lt(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("lt", backend="triton")
        def dt_lt(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim, mode = _prepare_binary(x, y, torch.bool)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            lt_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, mode, BLOCK_SIZE=1024, num_warps=2)
            return out

    if le is not None:
        @triton.jit
        def le_kernel(x_ptr, y_ptr, out_ptr, shape_ptr, x_stride_ptr, y_stride_ptr, N, NDIM: tl.constexpr, MODE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets, y_offsets = prepared_binary_offsets(offs, shape_ptr, x_stride_ptr, y_stride_ptr, NDIM, MODE)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            out = le(x, y)
            tl.store(out_ptr + offs, out, mask=mask)

        @dtype_cls.register_op("le", backend="triton")
        def dt_le(ops, x, y):
            out, x, y, shape_meta, x_stride_meta, y_stride_meta, ndim, mode = _prepare_binary(x, y, torch.bool)
            if out.numel() == 0:
                return out

            grid = (triton.cdiv(out.numel(), 1024),)
            le_kernel[grid](x, y, out, shape_meta, x_stride_meta, y_stride_meta, out.numel(), ndim, mode, BLOCK_SIZE=1024, num_warps=2)
            return out

    if lt is not None:
        @triton.jit
        def relu_kernel(
            x_ptr, out_ptr, shape_ptr, x_stride_ptr, N,
            NDIM: tl.constexpr, CONTIGUOUS: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            x_offsets = prepared_unary_offsets(offs, shape_ptr, x_stride_ptr, NDIM, CONTIGUOUS)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=_ZERO)

            zero = tl.cast(_ZERO, tl_int_dtype)
            tl.store(out_ptr + offs, tl.where(lt(x, zero), zero, x), mask=mask)

        @dtype_cls.register_op("relu", backend="triton")
        def dt_relu(ops, x):
            out, shape_meta, stride_meta, ndim, contiguous = _prepare_unary(x, dtype_cls.int_dtype)
            if out.numel() == 0:
                return out

            block = 512
            grid = (triton.cdiv(out.numel(), block),)
            relu_kernel[grid](
                x, out, shape_meta, stride_meta, out.numel(), ndim, contiguous,
                BLOCK_SIZE=block, num_warps=4,
            )
            return out

        @triton.jit
        def relu_backward_kernel(
            dy_ptr, y_ptr, dx_ptr,
            shape_ptr, dy_stride_ptr, y_stride_ptr,
            N, NDIM: tl.constexpr, MODE: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N

            dy_offsets, y_offsets = prepared_binary_offsets(
                offs, shape_ptr, dy_stride_ptr, y_stride_ptr, NDIM, MODE
            )
            dy = tl.load(dy_ptr + dy_offsets, mask=mask, other=_ZERO)
            y = tl.load(y_ptr + y_offsets, mask=mask, other=_ZERO)

            tl.store(dx_ptr + offs, tl.where(y == _ZERO, _ZERO, dy), mask=mask)

        def relu_backward(grad_output, output):
            dx, grad_output, output, shape_meta, dy_stride_meta, y_stride_meta, ndim, mode = _prepare_binary(
                grad_output, output, dtype_cls.int_dtype
            )
            if dx.numel() == 0:
                return dx
            block = 512
            grid = (triton.cdiv(dx.numel(), block),)
            relu_backward_kernel[grid](
                grad_output, output, dx,
                shape_meta, dy_stride_meta, y_stride_meta,
                dx.numel(), ndim, mode,
                BLOCK_SIZE=block, num_warps=4,
            )
            return dx

        class DTReLUFunction(DTFunction):
            @staticmethod
            def forward(ops, x):
                return ops.relu(x)

            @staticmethod
            def setup_context(ctx, ops, inputs, output):
                ctx.save_for_backward(output)

            @staticmethod
            def backward(ctx, ops, grad_output):
                output, = ctx.saved_tensors
                return relu_backward(grad_output, output)

        @dtype_cls.register_func(
            torch.nn.functional.relu, torch.Tensor.relu,
            cast=("input",), backend="triton",
        )
        def dt_relu(input, inplace=False):
            result = DTReLUFunction.apply(input)
            return result

