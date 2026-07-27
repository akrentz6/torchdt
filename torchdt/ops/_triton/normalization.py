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
        x = to_accumulator(tl.load(x_ptrs, mask=mask, other=_ZERO))
        x = tl.where(mask, x, to_accumulator(tl.cast(_ZERO, tl_int_dtype)))

        block_sum = tl.reduce(x, axis=0, combine_fn=acc_add)

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
        count_acc = to_accumulator(tl.cast(count_dt, tl_int_dtype))

        acc_sum = to_accumulator(tl.cast(_ZERO, tl_int_dtype))

        for t0 in range(0, ntiles, BLOCK_T):
            t = t0 + lane
            mask = t < ntiles

            s = tl.load(partial_sum_ptr + pid * ps_s0 + t * ps_s1, mask=mask, other=to_accumulator(tl.cast(_ZERO, tl_int_dtype)))
            acc_sum = acc_add(acc_sum, tl.reduce(s, axis=0, combine_fn=acc_add))

        mean = from_accumulator(acc_div(acc_sum, count_acc))

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
        block_var_sum = tl.reduce(to_accumulator(sq), axis=0, combine_fn=acc_add)

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
        count_acc = to_accumulator(tl.cast(count_dt, tl_int_dtype))

        acc_var_sum = to_accumulator(tl.cast(_ZERO, tl_int_dtype))

        for t0 in range(0, ntiles, BLOCK_T):
            t = t0 + lane
            mask = t < ntiles

            v = tl.load(partial_var_ptr + pid * pv_s0 + t * pv_s1, mask=mask, other=to_accumulator(tl.cast(_ZERO, tl_int_dtype)))
            acc_var_sum = acc_add(acc_var_sum, tl.reduce(v, axis=0, combine_fn=acc_add))

        var = from_accumulator(acc_div(acc_var_sum, count_acc))
        var = tl.where(lt(var, tl.cast(_ZERO, tl_int_dtype)), tl.cast(_ZERO, tl_int_dtype), var)

        if count > 1:
            sample_var = mul(var, div(tl.cast(count_dt, tl_int_dtype), sub(tl.cast(count_dt, tl_int_dtype), tl.cast(_ONE, tl_int_dtype))))
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
        key=["count", "HW", "W", "EVAL_FUSED"],
    )
    @triton.jit
    def batch_norm2d_apply_kernel(
        X_ptr, Y_ptr,
        w_ptr, b_ptr,
        rm_ptr, rv_ptr, sm_ptr, sis_ptr,
        eps,
        HW, W, count,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_y_n, s_y_c, s_y_h, s_y_w,
        has_weight, has_bias,
        EVAL_FUSED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)

        lane = tl.arange(0, BLOCK)
        idx = pid1 * BLOCK + lane
        mask = idx < count

        if EVAL_FUSED:
            mean = tl.load(rm_ptr + pid0)
            var = tl.load(rv_ptr + pid0)
            var = tl.where(lt(var, tl.cast(_ZERO, tl_int_dtype)), tl.cast(_ZERO, tl_int_dtype), var)
            invstd = div(tl.cast(_ONE, tl_int_dtype), sqrt(add(var, tl.cast(eps, tl_int_dtype))))
            if pid1 == 0:
                tl.store(sm_ptr + pid0, mean)
                tl.store(sis_ptr + pid0, invstd)
        else:
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

    @dtype_cls.register_op("batch_norm", backend="triton")
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

        count_dt = ops.encoded_scalar(count)
        eps_dt = ops.encoded_scalar(eps)
        momentum_dt = ops.encoded_scalar(momentum)

        output = torch.empty((N, C, H, W), device=x.device, dtype=dtype_cls.int_dtype)
        save_mean = torch.empty((C,), device=x.device, dtype=dtype_cls.int_dtype)
        save_invstd = torch.empty((C,), device=x.device, dtype=dtype_cls.int_dtype)

        s_x_n, s_x_c, s_x_h, s_x_w = x.stride()
        s_y_n, s_y_c, s_y_h, s_y_w = output.stride()
        fused_eval = (not training) and count <= 256

        if training:
            partial_sum = torch.empty((C, partial_tiles), device=x.device, dtype=acc_int_dtype)
            partial_var = torch.empty((C, partial_tiles), device=x.device, dtype=acc_int_dtype)

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
        elif not fused_eval:
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
            running_mean, running_var, save_mean, save_invstd,
            eps_dt,
            HW, W, count,
            s_x_n, s_x_c, s_x_h, s_x_w,
            s_y_n, s_y_c, s_y_h, s_y_w,
            has_weight,
            has_bias,
            fused_eval,
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

        partial_dy = tl.reduce(to_accumulator(dy), axis=0, combine_fn=acc_add)
        partial_dy_xhat = tl.reduce(to_accumulator(dy_xhat), axis=0, combine_fn=acc_add)

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

        count_dt = to_accumulator(tl.cast(count_dt, tl_int_dtype))
        sum_dy = to_accumulator(tl.cast(_ZERO, tl_int_dtype))
        sum_dy_xhat = to_accumulator(tl.cast(_ZERO, tl_int_dtype))

        for start in range(0, K, BLOCK_R):
            idx = start + base
            mask = idx < K

            pdy = tl.load(p_dy_ptr + pid * K + idx, mask=mask, other=to_accumulator(tl.cast(_ZERO, tl_int_dtype)))
            pdyx = tl.load(p_dy_xhat_ptr + pid * K + idx, mask=mask, other=to_accumulator(tl.cast(_ZERO, tl_int_dtype)))

            sum_dy = acc_add(sum_dy, tl.reduce(pdy, axis=0, combine_fn=acc_add))
            sum_dy_xhat = acc_add(sum_dy_xhat, tl.reduce(pdyx, axis=0, combine_fn=acc_add))

        if has_bias:
            tl.store(dB_ptr + pid, from_accumulator(sum_dy))
        if has_weight:
            tl.store(dW_ptr + pid, from_accumulator(sum_dy_xhat))

        tl.store(m_dy_ptr + pid, from_accumulator(acc_div(sum_dy, count_dt)))
        tl.store(m_dy_xhat_ptr + pid, from_accumulator(acc_div(sum_dy_xhat, count_dt)))

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

        count_dt = dtype_cls.ops.encoded_scalar(count)

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
            partial_dy = torch.empty((C, K), device=input.device, dtype=acc_int_dtype)
            partial_dy_xhat = torch.empty((C, K), device=input.device, dtype=acc_int_dtype)

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
                             cast=("input", "running_mean", "running_var", "weight", "bias"),
                             backend="triton")
    def dt_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
        assert input.dim() == 4, "torchdt only supports 2D batch norm for now"
        return DTBatchNormFunction.apply(input, running_mean, running_var, weight, bias, training, momentum, eps)


