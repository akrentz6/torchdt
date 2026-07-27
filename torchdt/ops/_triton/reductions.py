import math

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
    def max_combine_fn(a, b):
        return tl.where(gt(a, b), a, b)

    @triton.jit
    def reduction_row_base_offset(
        row, kept_shape_ptr, kept_stride_ptr,
        KEPT_NDIM: tl.constexpr,
    ):
        remaining = tl.cast(row, tl.int64)
        base = remaining * 0
        for rev_dim in tl.static_range(0, KEPT_NDIM):
            dim = KEPT_NDIM - 1 - rev_dim
            dim_size = tl.load(kept_shape_ptr + dim).to(tl.int64)
            dim_index = remaining % dim_size
            remaining = remaining // dim_size
            stride = tl.load(kept_stride_ptr + dim).to(tl.int64)
            base += dim_index * stride
        return base

    @triton.jit
    def max_reduce_kernel(
        x_ptr, value_ptr, index_ptr,
        kept_shape_ptr, kept_stride_ptr,
        M, N, reduce_stride,
        KEPT_NDIM: tl.constexpr,
        CONTIGUOUS_LAST: tl.constexpr,
        WRITE_INDICES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if CONTIGUOUS_LAST:
            x_base = row * N
        else:
            x_base = reduction_row_base_offset(row, kept_shape_ptr, kept_stride_ptr, KEPT_NDIM)

        lane = tl.arange(0, BLOCK)
        best = tl.cast(_NEG_INF, tl_int_dtype)
        best_idx = 0
        for start in range(0, N, BLOCK):
            idx = start + lane
            mask = idx < N
            values = tl.load(x_ptr + x_base + idx * reduce_stride, mask=mask, other=_NEG_INF)
            tile_max = tl.reduce(values, axis=0, combine_fn=max_combine_fn)
            candidate_idx = tl.min(tl.where(mask & (values == tile_max), idx, 0x7fffffff), axis=0)
            better = gt(tile_max, best)
            tie_earlier = (tile_max == best) & (candidate_idx < best_idx)
            best = tl.where(better, tile_max, best)
            best_idx = tl.where(better | tie_earlier, candidate_idx, best_idx)

        tl.store(value_ptr + row, best)
        if WRITE_INDICES:
            tl.store(index_ptr + row, best_idx)

    def _reduction_block_size(n):
        return max(1, triton.next_power_of_2(min(int(n), 1024)))

    if gt is not None:
        @dtype_cls.register_op("max", backend="triton")
        def dt_max(ops, x, dim=None, keepdim=False):
            if x.numel() == 0 and dim is None:
                raise RuntimeError("max(): Expected reduction dim to be specified for input.numel() == 0")

            original_shape = tuple(x.shape)
            write_indices = dim is not None
            if dim is None:
                # Arbitrary flattened strides need offset decoding across every
                # dimension, so use a contiguous copy only for this rare form.
                x_work = x.contiguous().view(-1)
                reduce_dim = 0
            else:
                reduce_dim = int(dim)
                if x.dim() == 0 and reduce_dim in (0, -1):
                    reduce_dim = 0
                    x_work = x.reshape(1)
                else:
                    reduce_dim %= x.dim()
                    x_work = x

            N = x_work.shape[reduce_dim]
            if N == 0:
                raise IndexError("max(): Expected reduction dim to have non-zero size.")
            kept_dims = tuple(d for d in range(x_work.dim()) if d != reduce_dim)
            kept_shape = tuple(x_work.shape[d] for d in kept_dims)
            M = math.prod(kept_shape) if kept_shape else 1
            kept_strides = tuple(x_work.stride(d) for d in kept_dims)
            contiguous_last = x_work.is_contiguous() and reduce_dim == x_work.dim() - 1

            meta_shape = _metadata_tensor(kept_shape, x.device)
            meta_stride = _metadata_tensor(kept_strides, x.device)
            values = torch.empty((M,), dtype=dtype_cls.int_dtype, device=x.device)
            indices = torch.empty((M,), dtype=torch.int64, device=x.device)
            block = _reduction_block_size(N)
            num_warps = 1 if block <= 128 else (2 if block <= 256 else 4)
            if M:
                max_reduce_kernel[(M,)](
                    x_work, values, indices,
                    meta_shape, meta_stride,
                    M, N, x_work.stride(reduce_dim),
                    len(kept_dims), contiguous_last, write_indices,
                    BLOCK=block, num_warps=num_warps,
                )

            if dim is None:
                return values.view(())

            out_shape = list(original_shape)
            del out_shape[reduce_dim]
            values = values.reshape(out_shape)
            indices = indices.reshape(out_shape)
            if keepdim:
                values = values.unsqueeze(reduce_dim)
                indices = indices.unsqueeze(reduce_dim)
            return torch.return_types.max((values, indices))

    def _prepare_rowwise(x, dim):
        original_shape = tuple(x.shape)
        flattened = dim is None
        if flattened:
            x_work = x.contiguous().reshape(-1)
            reduce_dim = 0
        elif x.dim() == 0:
            x_work = x.reshape(1)
            reduce_dim = 0
        else:
            x_work = x
            reduce_dim = int(dim) % x.dim()

        out = torch.empty(original_shape, dtype=dtype_cls.int_dtype, device=x.device)
        out_work = out.reshape(-1) if flattened or x.dim() == 0 else out
        kept_dims = tuple(d for d in range(x_work.dim()) if d != reduce_dim)
        kept_shape = tuple(x_work.shape[d] for d in kept_dims)
        M = math.prod(kept_shape) if kept_shape else 1
        N = x_work.shape[reduce_dim]
        x_kept_strides = tuple(x_work.stride(d) for d in kept_dims)
        out_kept_strides = tuple(out_work.stride(d) for d in kept_dims)
        contiguous_last = (
            x_work.is_contiguous() and out_work.is_contiguous()
            and reduce_dim == x_work.dim() - 1
        )
        return (
            x_work, out_work, out,
            _metadata_tensor(kept_shape, x.device),
            _metadata_tensor(x_kept_strides, x.device),
            _metadata_tensor(out_kept_strides, x.device),
            M, N, reduce_dim, len(kept_dims), contiguous_last,
        )

    @triton.jit
    def softmax_forward_kernel(
        x_ptr, y_ptr,
        kept_shape_ptr, x_kept_stride_ptr, y_kept_stride_ptr,
        M, N, x_reduce_stride, y_reduce_stride,
        KEPT_NDIM: tl.constexpr,
        CONTIGUOUS_LAST: tl.constexpr,
        LOG_SOFTMAX: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if CONTIGUOUS_LAST:
            x_base = row * N
            y_base = row * N
        else:
            x_base = reduction_row_base_offset(row, kept_shape_ptr, x_kept_stride_ptr, KEPT_NDIM)
            y_base = reduction_row_base_offset(row, kept_shape_ptr, y_kept_stride_ptr, KEPT_NDIM)

        lane = tl.arange(0, BLOCK)
        row_max = tl.cast(_NEG_INF, tl_int_dtype)
        for start in range(0, N, BLOCK):
            idx = start + lane
            mask = idx < N
            x = tl.load(x_ptr + x_base + idx * x_reduce_stride, mask=mask, other=_NEG_INF)
            tile_max = tl.reduce(x, axis=0, combine_fn=max_combine_fn)
            row_max = tl.where(gt(tile_max, row_max), tile_max, row_max)

        sum_acc = to_accumulator(tl.cast(_ZERO, tl_int_dtype))
        for start in range(0, N, BLOCK):
            idx = start + lane
            mask = idx < N
            x = tl.load(x_ptr + x_base + idx * x_reduce_stride, mask=mask, other=row_max)
            centered = sub(x, row_max)
            exp_value = tl.where(mask, exp(centered), tl.cast(_ZERO, tl_int_dtype))
            sum_acc = acc_add(sum_acc, tl.reduce(to_accumulator(exp_value), axis=0, combine_fn=acc_add))
            temporary = centered if LOG_SOFTMAX else exp_value
            tl.store(y_ptr + y_base + idx * y_reduce_stride, temporary, mask=mask)

        sum_value = from_accumulator(sum_acc)
        log_sum = log(sum_value) if LOG_SOFTMAX else tl.cast(_ZERO, tl_int_dtype)
        for start in range(0, N, BLOCK):
            idx = start + lane
            mask = idx < N
            temporary = tl.load(y_ptr + y_base + idx * y_reduce_stride, mask=mask, other=_ZERO)
            result = sub(temporary, log_sum) if LOG_SOFTMAX else div(temporary, sum_value)
            tl.store(y_ptr + y_base + idx * y_reduce_stride, result, mask=mask)

    def softmax_forward(x, dim, log_softmax):
        prepared = _prepare_rowwise(x, dim)
        x_work, out_work, out, shape_meta, x_stride_meta, out_stride_meta, M, N, reduce_dim, kept_ndim, fast = prepared
        if out.numel() == 0:
            return out
        block = _reduction_block_size(N)
        num_warps = 1 if block <= 128 else (2 if block <= 256 else 4)
        softmax_forward_kernel[(M,)](
            x_work, out_work,
            shape_meta, x_stride_meta, out_stride_meta,
            M, N, x_work.stride(reduce_dim), out_work.stride(reduce_dim),
            kept_ndim, fast, log_softmax,
            BLOCK=block, num_warps=num_warps,
        )
        return out

    if exp is not None and div is not None and sub is not None and mul is not None:
        @dtype_cls.register_op("softmax", backend="triton")
        def dt_softmax(ops, x, dim=None):
            return softmax_forward(x, dim, False)

        @dtype_cls.register_op("log_softmax", backend="triton")
        def dt_log_softmax(ops, x, dim=None):
            return softmax_forward(x, dim, True)

        @triton.jit
        def softmax_backward_kernel(
            dy_ptr, y_ptr, dx_ptr,
            kept_shape_ptr,
            dy_kept_stride_ptr, y_kept_stride_ptr, dx_kept_stride_ptr,
            M, N, dy_reduce_stride, y_reduce_stride, dx_reduce_stride,
            KEPT_NDIM: tl.constexpr,
            CONTIGUOUS_LAST: tl.constexpr,
            LOG_SOFTMAX: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0)
            if CONTIGUOUS_LAST:
                dy_base = row * N
                y_base = row * N
                dx_base = row * N
            else:
                dy_base = reduction_row_base_offset(row, kept_shape_ptr, dy_kept_stride_ptr, KEPT_NDIM)
                y_base = reduction_row_base_offset(row, kept_shape_ptr, y_kept_stride_ptr, KEPT_NDIM)
                dx_base = reduction_row_base_offset(row, kept_shape_ptr, dx_kept_stride_ptr, KEPT_NDIM)

            lane = tl.arange(0, BLOCK)
            sum_acc = to_accumulator(tl.cast(_ZERO, tl_int_dtype))
            for start in range(0, N, BLOCK):
                idx = start + lane
                mask = idx < N
                dy = tl.load(dy_ptr + dy_base + idx * dy_reduce_stride, mask=mask, other=_ZERO)
                if LOG_SOFTMAX:
                    term = dy
                else:
                    y = tl.load(y_ptr + y_base + idx * y_reduce_stride, mask=mask, other=_ZERO)
                    term = mul(dy, y)
                term = tl.where(mask, term, tl.cast(_ZERO, tl_int_dtype))
                sum_acc = acc_add(sum_acc, tl.reduce(to_accumulator(term), axis=0, combine_fn=acc_add))

            row_sum = from_accumulator(sum_acc)
            for start in range(0, N, BLOCK):
                idx = start + lane
                mask = idx < N
                dy = tl.load(dy_ptr + dy_base + idx * dy_reduce_stride, mask=mask, other=_ZERO)
                y = tl.load(y_ptr + y_base + idx * y_reduce_stride, mask=mask, other=_ZERO)
                if LOG_SOFTMAX:
                    dx = sub(dy, mul(exp(y), row_sum))
                else:
                    dx = mul(y, sub(dy, row_sum))
                tl.store(dx_ptr + dx_base + idx * dx_reduce_stride, dx, mask=mask)

        def softmax_backward(grad_output, output, dim, log_softmax):
            original_shape = tuple(output.shape)
            flattened = dim is None
            if flattened:
                dy_work = grad_output.contiguous().reshape(-1)
                y_work = output.contiguous().reshape(-1)
                reduce_dim = 0
            elif output.dim() == 0:
                dy_work = grad_output.reshape(1)
                y_work = output.reshape(1)
                reduce_dim = 0
            else:
                dy_work = grad_output
                y_work = output
                reduce_dim = int(dim) % output.dim()

            dx = torch.empty(original_shape, dtype=dtype_cls.int_dtype, device=output.device)
            dx_work = dx.reshape(-1) if flattened or output.dim() == 0 else dx
            if dx.numel() == 0:
                return dx

            kept_dims = tuple(d for d in range(y_work.dim()) if d != reduce_dim)
            kept_shape = tuple(y_work.shape[d] for d in kept_dims)
            M = math.prod(kept_shape) if kept_shape else 1
            N = y_work.shape[reduce_dim]
            shape_meta = _metadata_tensor(kept_shape, output.device)
            dy_stride_meta = _metadata_tensor(tuple(dy_work.stride(d) for d in kept_dims), output.device)
            y_stride_meta = _metadata_tensor(tuple(y_work.stride(d) for d in kept_dims), output.device)
            dx_stride_meta = _metadata_tensor(tuple(dx_work.stride(d) for d in kept_dims), output.device)
            fast = (
                dy_work.is_contiguous() and y_work.is_contiguous() and dx_work.is_contiguous()
                and reduce_dim == y_work.dim() - 1
            )
            block = _reduction_block_size(N)
            num_warps = 1 if block <= 128 else (2 if block <= 256 else 4)
            softmax_backward_kernel[(M,)](
                dy_work, y_work, dx_work,
                shape_meta, dy_stride_meta, y_stride_meta, dx_stride_meta,
                M, N,
                dy_work.stride(reduce_dim), y_work.stride(reduce_dim), dx_work.stride(reduce_dim),
                len(kept_dims), fast, log_softmax,
                BLOCK=block, num_warps=num_warps,
            )
            return dx

        class DTSoftmaxFunction(DTFunction):
            @staticmethod
            def forward(ops, x, dim=None):
                return ops.softmax(x, dim=dim)

            @staticmethod
            def setup_context(ctx, ops, inputs, output):
                _, dim = inputs
                ctx.save_for_backward(output)
                ctx.dim = dim

            @staticmethod
            def backward(ctx, ops, grad_output):
                output, = ctx.saved_tensors
                return softmax_backward(grad_output, output, ctx.dim, False), None

        class DTLogSoftmaxFunction(DTFunction):
            @staticmethod
            def forward(ops, x, dim=None):
                return ops.log_softmax(x, dim=dim)

            @staticmethod
            def setup_context(ctx, ops, inputs, output):
                _, dim = inputs
                ctx.save_for_backward(output)
                ctx.dim = dim

            @staticmethod
            def backward(ctx, ops, grad_output):
                output, = ctx.saved_tensors
                return softmax_backward(grad_output, output, ctx.dim, True), None

        @dtype_cls.register_func(
            torch.nn.functional.softmax, torch.Tensor.softmax,
            cast=("input",), backend="triton",
        )
        def dt_softmax(input, dim=None, _stacklevel=3, dtype=None, *, out=None):
            result = DTSoftmaxFunction.apply(input, dim)

            if out is not None:
                return out.copy_(result)
            return result

        @dtype_cls.register_func(
            torch.nn.functional.log_softmax, torch.Tensor.log_softmax,
            cast=("input",), backend="triton",
        )
        def dt_log_softmax(input, dim=None, _stacklevel=3, dtype=None, *, out=None):
            result = DTLogSoftmaxFunction.apply(input, dim)

            if out is not None:
                return out.copy_(result)
            return result


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128},  num_warps=2, num_stages=2),
            triton.Config({"BLOCK": 64},   num_warps=1, num_stages=2),
            triton.Config({"BLOCK": 128},  num_warps=1, num_stages=2),
            triton.Config({"BLOCK": 256},  num_warps=2, num_stages=2),
            triton.Config({"BLOCK": 256},  num_warps=4, num_stages=2),
        ],
        key=["N", "DO_MEAN"],
    )
    @triton.jit
    def sum_kernel(
        x_ptr, y_ptr, M, N: tl.constexpr, divisor,
        s_x_r, s_x_c, s_y_r,
        DO_MEAN: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_ptr = x_ptr + pid * s_x_r
        acc = to_accumulator(tl.cast(_ZERO, tl_int_dtype))

        for tile_idx in range(0, tl.cdiv(N, BLOCK)):
            offs = tile_idx * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N

            vals = tl.load(row_ptr + offs * s_x_c, mask=mask, other=_ZERO)
            vals = to_accumulator(vals)
            acc = acc_add(acc, tl.reduce(vals, axis=0, combine_fn=acc_add))

        result = from_accumulator(acc)
        if DO_MEAN:
            result = div(result, tl.cast(divisor, tl_int_dtype))
        tl.store(y_ptr + pid * s_y_r, result)

    def sum_or_mean(ops, x, dim=None, keepdim=False, do_mean=False):
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

        M = math.prod(kept_shape) if kept_shape else 1
        N = math.prod(reduced_shape)

        y = x_perm.reshape(M, N)
        stride_row, stride_col = y.stride()

        out = torch.empty((M,), device=x.device, dtype=dtype_cls.int_dtype)
        grid = (M,)

        sum_kernel[grid](
            y,
            out,
            M, N, ops.encoded_scalar(N),
            stride_row, stride_col,
            out.stride(0),
            do_mean,
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

    @dtype_cls.register_op("sum", backend="triton")
    def dt_sum(ops, x, dim=None, keepdim=False):
        return sum_or_mean(ops, x, dim, keepdim, False)

    if div is not None:
        @dtype_cls.register_op("mean", backend="triton")
        def dt_mean(ops, x, dim=None, keepdim=False):
            return sum_or_mean(ops, x, dim, keepdim, True)


