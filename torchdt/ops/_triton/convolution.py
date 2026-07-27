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
        Cout, Kh: tl.constexpr, Kw: tl.constexpr,
        Hout, Wout,
        sh: tl.constexpr, sw: tl.constexpr,
        ph: tl.constexpr, pw: tl.constexpr,
        dh: tl.constexpr, dw: tl.constexpr,
        groups: tl.constexpr,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_w_co, s_w_cinperg, s_w_kh, s_w_kw,
        s_y_n, s_y_c, s_y_h, s_y_w,
        HAS_BIAS: tl.constexpr,
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

        acc = to_accumulator(tl.full((BLOCK_OC, BLOCK_HW), _ZERO, tl_int_dtype))

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
                    acc = acc_add(acc, to_accumulator(prod))

        if HAS_BIAS:
            bias = to_accumulator(tl.load(B_ptr + oc_offsets, mask=mask_oc, other=_ZERO))
            acc = acc_add(acc, bias[:, None])

        out_ptrs = Yb + oc_offsets[:, None] * s_y_c + h[None, :] * s_y_h + w[None, :] * s_y_w
        tl.store(out_ptrs, from_accumulator(acc), mask=mask_oc[:, None] & mask_hw & mask_n & mask_group)

    @dtype_cls.register_op("conv2d", backend="triton")
    def dt_conv2d(ops, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        has_bias = bias is not None
        if bias is None:
            bias = weight

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
            HAS_BIAS=has_bias,
        )

        return y

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_IC": 4, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_IC": 8, "BLOCK_HW": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_IC": 2, "BLOCK_HW": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_IC": 4, "BLOCK_HW": 32}, num_warps=2, num_stages=1),
        ],
        key=["Cin", "H", "W", "Cout", "Kh", "Kw", "Hout", "Wout", "sh", "sw", "ph", "pw", "dh", "dw", "groups"],
    )
    @triton.jit
    def conv2d_dinput_kernel(
        dX_ptr, dY_ptr, W_ptr,
        N, Cin, H, W,
        Cout, Kh: tl.constexpr, Kw: tl.constexpr,
        Hout, Wout,
        sh: tl.constexpr, sw: tl.constexpr,
        ph: tl.constexpr, pw: tl.constexpr,
        dh: tl.constexpr, dw: tl.constexpr,
        groups: tl.constexpr,
        s_dx_n, s_dx_c, s_dx_h, s_dx_w,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        s_w_co, s_w_cinperg, s_w_kh, s_w_kw,
        BLOCK_IC: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)
        cin_per_g = Cin // groups
        cout_per_g = Cout // groups
        ic_tiles = tl.cdiv(cin_per_g, BLOCK_IC)

        n = pid0 // (groups * ic_tiles)
        group_tile = pid0 % (groups * ic_tiles)
        group_id = group_tile // ic_tiles
        ic_tile = group_tile % ic_tiles
        ic_in_group = ic_tile * BLOCK_IC + tl.arange(0, BLOCK_IC)
        cin = group_id * cin_per_g + ic_in_group
        mask_ic = ic_in_group < cin_per_g

        HW = H * W
        idx = pid1 * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask_hw = idx < HW
        h = idx // W
        w = idx % W
        acc = to_accumulator(tl.full((BLOCK_IC, BLOCK_HW), _ZERO, dtype=tl_int_dtype))

        cout_start = group_id * cout_per_g
        cout_end = cout_start + cout_per_g
        for kh in tl.static_range(0, Kh):
            numer_h = h + ph - kh * dh
            divisible_h = (numer_h % sh) == 0
            h_out = numer_h // sh
            for kw in tl.static_range(0, Kw):
                numer_w = w + pw - kw * dw
                divisible_w = (numer_w % sw) == 0
                w_out = numer_w // sw
                valid_pos = mask_hw & divisible_h & divisible_w & (h_out >= 0) & (h_out < Hout) & (w_out >= 0) & (w_out < Wout)

                for cout in range(cout_start, cout_end):
                    dy_idx = n * s_dy_n + cout * s_dy_c + h_out * s_dy_h + w_out * s_dy_w
                    dy_vals = tl.load(dY_ptr + dy_idx, mask=valid_pos, other=_ZERO)
                    w_idx = (
                        cout * s_w_co + ic_in_group * s_w_cinperg
                        + kh * s_w_kh + kw * s_w_kw
                    )
                    w_vals = tl.load(W_ptr + w_idx, mask=mask_ic, other=_ZERO)
                    product = mul(w_vals[:, None], dy_vals[None, :])
                    acc = acc_add(acc, to_accumulator(product))

        dx_idx = (
            n * s_dx_n + cin[:, None] * s_dx_c
            + h[None, :] * s_dx_h + w[None, :] * s_dx_w
        )
        tl.store(dX_ptr + dx_idx, from_accumulator(acc), mask=mask_ic[:, None] & mask_hw[None, :])

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

        cin_per_group = Cin // groups
        grid = lambda META: (
            N * groups * triton.cdiv(cin_per_group, META["BLOCK_IC"]),
            triton.cdiv(Hin * Win, META["BLOCK_HW"]),
        )
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_OC": 4, "BLOCK_IC": 4, "BLOCK_NHW": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_OC": 8, "BLOCK_IC": 4, "BLOCK_NHW": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_OC": 4, "BLOCK_IC": 8, "BLOCK_NHW": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_OC": 2, "BLOCK_IC": 4, "BLOCK_NHW": 64}, num_warps=2, num_stages=1),
        ],
        key=["N", "Cin", "H", "W", "Cout", "Kh", "Kw", "Hout", "Wout", "sh", "sw", "ph", "pw", "dh", "dw", "groups", "SPLIT_K"],
    )
    @triton.jit
    def conv2d_dweight_kernel(
        partial_ptr, X_ptr, dY_ptr,
        N, Cin, H, W,
        Cout, Kh: tl.constexpr, Kw: tl.constexpr,
        Hout, Wout,
        sh: tl.constexpr, sw: tl.constexpr,
        ph: tl.constexpr, pw: tl.constexpr,
        dh: tl.constexpr, dw: tl.constexpr,
        groups: tl.constexpr,
        s_x_n, s_x_c, s_x_h, s_x_w,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        SPLIT_K: tl.constexpr,
        BLOCK_OC: tl.constexpr,
        BLOCK_IC: tl.constexpr,
        BLOCK_NHW: tl.constexpr,
    ):
        tile = tl.program_id(0)
        split_id = tl.program_id(1)
        cin_per_group = Cin // groups
        cout_per_group = Cout // groups
        ic_tiles = tl.cdiv(cin_per_group, BLOCK_IC)
        oc_tiles = tl.cdiv(cout_per_group, BLOCK_OC)

        kw = tile % Kw
        tile = tile // Kw
        kh = tile % Kh
        tile = tile // Kh
        ic_tile = tile % ic_tiles
        tile = tile // ic_tiles
        oc_tile = tile % oc_tiles
        group_id = tile // oc_tiles

        ic = ic_tile * BLOCK_IC + tl.arange(0, BLOCK_IC)
        oc_in_group = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
        oc = group_id * cout_per_group + oc_in_group
        ic_abs = group_id * cin_per_group + ic
        mask_ic = ic < cin_per_group
        mask_oc = oc_in_group < cout_per_group

        acc = to_accumulator(tl.full((BLOCK_OC, BLOCK_IC), _ZERO, dtype=tl_int_dtype))
        lane = tl.arange(0, BLOCK_NHW)
        total = N * Hout * Wout
        for start in range(split_id * BLOCK_NHW, total, SPLIT_K * BLOCK_NHW):
            idx = start + lane
            mask_nhw = idx < total
            n = idx // (Hout * Wout)
            spatial = idx % (Hout * Wout)
            hout = spatial // Wout
            wout = spatial % Wout
            h = hout * sh - ph + kh * dh
            w = wout * sw - pw + kw * dw
            in_bounds = mask_nhw & (h >= 0) & (h < H) & (w >= 0) & (w < W)

            x_idx = (
                n[None, :] * s_x_n + ic_abs[:, None] * s_x_c
                + h[None, :] * s_x_h + w[None, :] * s_x_w
            )
            x_vals = tl.load(
                X_ptr + x_idx,
                mask=mask_ic[:, None] & in_bounds[None, :], other=_ZERO,
            )
            dy_idx = (
                n[None, :] * s_dy_n + oc[:, None] * s_dy_c
                + hout[None, :] * s_dy_h + wout[None, :] * s_dy_w
            )
            dy_vals = tl.load(
                dY_ptr + dy_idx,
                mask=mask_oc[:, None] & mask_nhw[None, :], other=_ZERO,
            )
            product = mul(dy_vals[:, None, :], x_vals[None, :, :])
            tile_sum = tl.reduce(to_accumulator(product), axis=2, combine_fn=acc_add)
            acc = acc_add(acc, tile_sum)

        weight_count = Cout * cin_per_group * Kh * Kw
        weight_idx = (
            ((oc[:, None] * cin_per_group + ic[None, :]) * Kh + kh) * Kw + kw
        )
        weight_mask = mask_oc[:, None] & mask_ic[None, :]
        if SPLIT_K == 1:
            tl.store(partial_ptr + weight_idx, from_accumulator(acc), mask=weight_mask)
        else:
            tl.store(
                partial_ptr + split_id * weight_count + weight_idx,
                acc, mask=weight_mask,
            )

    @triton.jit
    def conv2d_dweight_finalize_kernel(
        partial_ptr, dW_ptr, weight_count,
        SPLIT_K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < weight_count
        acc = to_accumulator(tl.full((BLOCK,), _ZERO, dtype=tl_int_dtype))
        for split in tl.static_range(0, SPLIT_K):
            value = tl.load(
                partial_ptr + split * weight_count + idx,
                mask=mask, other=to_accumulator(tl.cast(_ZERO, tl_int_dtype)),
            )
            acc = acc_add(acc, value)
        tl.store(dW_ptr + idx, from_accumulator(acc), mask=mask)

    def conv2d_dweight(grad_output, input, weight_shape, stride, padding, dilation, groups):
        N, Cin, H, W = input.shape
        _, Cout, Hout, Wout = grad_output.shape
        Kh, Kw = weight_shape[2], weight_shape[3]
        cin_per_group = Cin // groups
        cout_per_group = Cout // groups
        weight_count = Cout * cin_per_group * Kh * Kw
        grad_weight = torch.empty(weight_shape, device=grad_output.device, dtype=dtype_cls.int_dtype)

        sh, sw = stride[0], stride[1]
        ph, pw = padding[0], padding[1]
        dh, dw = dilation[0], dilation[1]
        s_x_n, s_x_c, s_x_h, s_x_w = input.stride()
        s_dy_n, s_dy_c, s_dy_h, s_dy_w = grad_output.stride()

        total = N * Hout * Wout
        split_k = min(8, max(1, triton.next_power_of_2(triton.cdiv(total, 4096))))
        accumulator_bytes = torch.empty((), dtype=acc_int_dtype).element_size()
        while split_k > 1 and split_k * weight_count * accumulator_bytes > 64 * 1024 * 1024:
            split_k //= 2

        if split_k == 1:
            partials = grad_weight
        else:
            partials = torch.empty(
                (split_k, weight_count), device=grad_output.device, dtype=acc_int_dtype
            )

        grid = lambda META: (
            groups * Kh * Kw
            * triton.cdiv(cin_per_group, META["BLOCK_IC"])
            * triton.cdiv(cout_per_group, META["BLOCK_OC"]),
            split_k,
        )
        conv2d_dweight_kernel[grid](
            partials, input, grad_output,
            N, Cin, H, W,
            Cout, Kh, Kw,
            Hout, Wout,
            sh, sw, ph, pw, dh, dw, groups,
            s_x_n, s_x_c, s_x_h, s_x_w,
            s_dy_n, s_dy_c, s_dy_h, s_dy_w,
            SPLIT_K=split_k,
        )

        if split_k > 1:
            block = 256
            conv2d_dweight_finalize_kernel[(triton.cdiv(weight_count, block),)](
                partials, grad_weight, weight_count,
                SPLIT_K=split_k, BLOCK=block, num_warps=4,
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

        acc = to_accumulator(tl.full((BLOCK_NHW,), _ZERO, dtype=tl_int_dtype))

        total = N * Hout * Wout
        for nhw_start in range(0, total, BLOCK_NHW):
            idx = nhw_start + offs
            mask = idx < total

            n = idx // (Hout * Wout)
            rem = idx % (Hout * Wout)
            hout = rem // Wout
            wout = rem % Wout

            dy_idx = n * s_dy_n + pid * s_dy_c + hout * s_dy_h + wout * s_dy_w
            dy_vals = to_accumulator(tl.load(dY_ptr + dy_idx, mask=mask, other=_ZERO))

            acc = acc_add(dy_vals, acc)

        db_idx = pid * s_db_c
        tl.store(dB_ptr + db_idx, from_accumulator(acc.reduce(0, acc_add)))

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
                             cast=("input", "weight", "bias"), backend="triton")
    def dt_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return DTConv2dFunction.apply(input, weight, bias, stride, padding, dilation, groups)


