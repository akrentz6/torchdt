import torch

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

    @dtype_cls.register_op("triton_sgd_step", backend="triton")
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
    def sgd_step_group_kernel(
        meta_ptr,
        lr_ptr, momentum_ptr, dampening_ptr, weight_decay_ptr,
        MAXIMIZE: tl.constexpr, NESTEROV: tl.constexpr,
        USE_MOMENTUM: tl.constexpr, FIRST_MOMENTUM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        tensor_id = tl.program_id(1)
        meta_base = tensor_id * 4
        p_ptr = tl.load(meta_ptr + meta_base).to(tl.pointer_type(tl_int_dtype))
        g_ptr = tl.load(meta_ptr + meta_base + 1).to(tl.pointer_type(tl_int_dtype))
        buf_ptr = tl.load(meta_ptr + meta_base + 2).to(tl.pointer_type(tl_int_dtype))
        N = tl.load(meta_ptr + meta_base + 3)
        offs = block_id * BLOCK + tl.arange(0, BLOCK)
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
        if USE_MOMENTUM:
            if FIRST_MOMENTUM:
                buf_new = g
            else:
                buf = tl.load(buf_ptr + offs, mask=mask, other=_ZERO)
                buf_new = add(
                    mul(buf, tl.cast(momentum, tl_int_dtype)),
                    mul(g, sub(tl.cast(_ONE, tl_int_dtype), tl.cast(dampening, tl_int_dtype))),
                )
            tl.store(buf_ptr + offs, buf_new, mask=mask)
            if NESTEROV:
                g = add(g, mul(buf_new, tl.cast(momentum, tl_int_dtype)))
            else:
                g = buf_new

        tl.store(p_ptr + offs, sub(p, mul(g, tl.cast(lr, tl_int_dtype))), mask=mask)

    def _size_bucket_entries(entries):
        buckets = {}
        for entry in entries:
            size = entry[0].numel()
            bucket = max(0, (max(1, size) - 1).bit_length())
            buckets.setdefault(bucket, []).append(entry)
        return buckets.values()

    @dtype_cls.register_op("triton_sgd_step_group", backend="triton")
    def triton_sgd_step_group(
        ops, params, grads, bufs,
        lr, momentum, dampening, weight_decay, nesterov, maximize, use_momentum,
    ):
        if not params:
            return []
        outputs = list(bufs)
        grouped = {}
        for index, (param, grad, buf) in enumerate(zip(params, grads, bufs)):
            first = use_momentum and buf is None
            if first:
                buf = torch.empty_like(param)
                outputs[index] = buf
            elif buf is None:
                buf = param
            size_bucket = max(0, (max(1, param.numel()) - 1).bit_length())
            grouped.setdefault((first, size_bucket), []).append((index, param, grad, buf))

        block = 512
        for (first, _), entries in grouped.items():
            metadata = torch.tensor(
                [
                    [param.data_ptr(), grad.data_ptr(), buf.data_ptr(), param.numel()]
                    for _, param, grad, buf in entries
                ],
                dtype=torch.int64, device=params[0].device,
            )
            max_size = max(param.numel() for _, param, _, _ in entries)
            grid = (triton.cdiv(max_size, block), len(entries))
            sgd_step_group_kernel[grid](
                metadata, lr, momentum, dampening, weight_decay,
                maximize, nesterov, use_momentum, first,
                BLOCK=block, num_warps=4,
            )
        return outputs


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

    @dtype_cls.register_op("triton_madam_step", backend="triton")
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
    def madam_step_group_kernel(
        meta_ptr,
        lr_ptr, beta_ptr, eps_ptr, g_bound_ptr, bias_corr_ptr,
        MAXIMIZE: tl.constexpr, USE_POW: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        tensor_id = tl.program_id(1)
        meta_base = tensor_id * 5
        p_ptr = tl.load(meta_ptr + meta_base).to(tl.pointer_type(tl_int_dtype))
        g_ptr = tl.load(meta_ptr + meta_base + 1).to(tl.pointer_type(tl_int_dtype))
        v_ptr = tl.load(meta_ptr + meta_base + 2).to(tl.pointer_type(tl_int_dtype))
        max_ptr = tl.load(meta_ptr + meta_base + 3).to(tl.pointer_type(tl_int_dtype))
        N = tl.load(meta_ptr + meta_base + 4)
        offs = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        p = tl.load(p_ptr + offs, mask=mask, other=_ZERO)
        g = tl.load(g_ptr + offs, mask=mask, other=_ZERO)
        v = tl.load(v_ptr + offs, mask=mask, other=_ZERO)
        lr = tl.load(lr_ptr)
        beta = tl.load(beta_ptr)
        eps = tl.load(eps_ptr)
        g_bound = tl.load(g_bound_ptr)
        maximum = tl.load(max_ptr)
        bias_corr = tl.load(bias_corr_ptr)

        g2 = mul(g, g)
        v_new = add(
            mul(tl.cast(beta, tl_int_dtype), v),
            mul(sub(tl.cast(_ONE, tl_int_dtype), tl.cast(beta, tl_int_dtype)), g2),
        )
        tl.store(v_ptr + offs, v_new, mask=mask)
        corr = add(div(v_new, tl.cast(bias_corr, tl_int_dtype)), tl.cast(eps, tl_int_dtype))
        g_normed = div(g, sqrt(corr))
        g_clipped = clamp(
            g_normed, neg(tl.cast(g_bound, tl_int_dtype)), tl.cast(g_bound, tl_int_dtype)
        )
        delta = mul(mul(tl.cast(lr, tl_int_dtype), g_clipped), sign(p))
        if not MAXIMIZE:
            delta = neg(delta)
        if USE_POW:
            updated = mul(p, exp(delta))
        else:
            updated = mul(p, add(tl.cast(_ONE, tl_int_dtype), delta))
        updated = clamp(
            updated, neg(tl.cast(maximum, tl_int_dtype)), tl.cast(maximum, tl_int_dtype)
        )
        tl.store(p_ptr + offs, updated, mask=mask)

    @dtype_cls.register_op("triton_madam_step_group", backend="triton")
    def triton_madam_step_group(
        ops, params, grads, exp_avg_sqs, maxima,
        lr, beta, eps, g_bound, bias_corr, use_pow, maximize,
    ):
        if not params:
            return exp_avg_sqs
        entries = list(zip(params, grads, exp_avg_sqs, maxima))
        block = 512
        for bucket_entries in _size_bucket_entries(entries):
            metadata = torch.tensor(
                [
                    [
                        param.data_ptr(), grad.data_ptr(), exp_avg_sq.data_ptr(),
                        maximum.data_ptr(), param.numel(),
                    ]
                    for param, grad, exp_avg_sq, maximum in bucket_entries
                ],
                dtype=torch.int64, device=params[0].device,
            )
            max_size = max(param.numel() for param, _, _, _ in bucket_entries)
            grid = (triton.cdiv(max_size, block), len(bucket_entries))
            madam_step_group_kernel[grid](
                metadata,
                lr, beta, eps, g_bound, bias_corr,
                maximize, use_pow,
                BLOCK=block, num_warps=4,
            )
        return exp_avg_sqs


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

    @dtype_cls.register_op("triton_adam_step", backend="triton")
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


    @triton.jit
    def adam_step_group_kernel(
        meta_ptr,
        lr_ptr, beta1_ptr, beta2_ptr, eps_ptr, weight_decay_ptr,
        bias_corr1_ptr, bias_corr2_ptr,
        MAXIMIZE: tl.constexpr, AMSGRAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        tensor_id = tl.program_id(1)
        meta_base = tensor_id * 6
        p_ptr = tl.load(meta_ptr + meta_base).to(tl.pointer_type(tl_int_dtype))
        g_ptr = tl.load(meta_ptr + meta_base + 1).to(tl.pointer_type(tl_int_dtype))
        m_ptr = tl.load(meta_ptr + meta_base + 2).to(tl.pointer_type(tl_int_dtype))
        v_ptr = tl.load(meta_ptr + meta_base + 3).to(tl.pointer_type(tl_int_dtype))
        vhat_ptr = tl.load(meta_ptr + meta_base + 4).to(tl.pointer_type(tl_int_dtype))
        N = tl.load(meta_ptr + meta_base + 5)
        offs = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        p = tl.load(p_ptr + offs, mask=mask, other=_ZERO)
        g = tl.load(g_ptr + offs, mask=mask, other=_ZERO)
        if MAXIMIZE:
            g = neg(g)
        weight_decay = tl.load(weight_decay_ptr)
        if weight_decay != _ZERO:
            g = add(g, mul(p, tl.cast(weight_decay, tl_int_dtype)))

        beta1 = tl.load(beta1_ptr)
        beta2 = tl.load(beta2_ptr)
        m = tl.load(m_ptr + offs, mask=mask, other=_ZERO)
        v = tl.load(v_ptr + offs, mask=mask, other=_ZERO)
        m_new = add(
            mul(m, tl.cast(beta1, tl_int_dtype)),
            mul(g, sub(tl.cast(_ONE, tl_int_dtype), tl.cast(beta1, tl_int_dtype))),
        )
        g2 = mul(g, g)
        v_new = add(
            mul(v, tl.cast(beta2, tl_int_dtype)),
            mul(g2, sub(tl.cast(_ONE, tl_int_dtype), tl.cast(beta2, tl_int_dtype))),
        )
        if AMSGRAD:
            vhat = tl.load(vhat_ptr + offs, mask=mask, other=_ZERO)
            vhat_new = tl.where(gt(vhat, v_new), vhat, v_new)
            tl.store(vhat_ptr + offs, vhat_new, mask=mask)
            v_denom = vhat_new
        else:
            v_denom = v_new

        step_size = div(
            mul(tl.cast(tl.load(lr_ptr), tl_int_dtype), sqrt(tl.cast(tl.load(bias_corr2_ptr), tl_int_dtype))),
            tl.cast(tl.load(bias_corr1_ptr), tl_int_dtype),
        )
        denom = add(sqrt(v_denom), tl.cast(tl.load(eps_ptr), tl_int_dtype))
        p_new = sub(p, mul(step_size, div(m_new, denom)))
        tl.store(p_ptr + offs, p_new, mask=mask)
        tl.store(m_ptr + offs, m_new, mask=mask)
        tl.store(v_ptr + offs, v_new, mask=mask)

    @dtype_cls.register_op("triton_adam_step_group", backend="triton")
    def triton_adam_step_group(
        ops, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
        lr, beta1, beta2, eps, weight_decay,
        bias_corr1, bias_corr2, amsgrad, maximize,
    ):
        if not params:
            return exp_avgs, exp_avg_sqs, max_exp_avg_sqs
        entries = list(zip(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs))
        block = 512
        for bucket_entries in _size_bucket_entries(entries):
            metadata = torch.tensor(
                [
                    [
                        param.data_ptr(), grad.data_ptr(), exp_avg.data_ptr(),
                        exp_avg_sq.data_ptr(),
                        (max_exp_avg_sq if max_exp_avg_sq is not None else param).data_ptr(),
                        param.numel(),
                    ]
                    for param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq in bucket_entries
                ],
                dtype=torch.int64, device=params[0].device,
            )
            max_size = max(param.numel() for param, _, _, _, _ in bucket_entries)
            grid = (triton.cdiv(max_size, block), len(bucket_entries))
            adam_step_group_kernel[grid](
                metadata,
                lr, beta1, beta2, eps, weight_decay,
                bias_corr1, bias_corr2,
                maximize, amsgrad,
                BLOCK=block, num_warps=4,
            )
        return exp_avgs, exp_avg_sqs, max_exp_avg_sqs

