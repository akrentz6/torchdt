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

    @dtype_cls.register_op("max_pool2d", backend="triton")
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
    def max_pool2d_dinput_scatter_kernel(
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
        # Non-overlapping windows guarantee one writer per selected input.
        tl.store(ptr_x, dy, mask=valid)


    @triton.jit
    def max_pool2d_dinput_gather_kernel(
        dY_ptr, dX_ptr, idx_ptr,
        N, C, H, W, Hout, Wout,
        s_dy_n, s_dy_c, s_dy_h, s_dy_w,
        s_dx_n, s_dx_c, s_dx_h, s_dx_w,
        s_idx_n, s_idx_c, s_idx_h, s_idx_w,
        Kh: tl.constexpr, Kw: tl.constexpr,
        sh: tl.constexpr, sw: tl.constexpr,
        ph: tl.constexpr, pw: tl.constexpr,
        dh: tl.constexpr, dw: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        ih = tl.program_id(1)
        iw = tl.program_id(2) * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_w = iw < W
        n = pid0 // C
        c = pid0 % C
        input_index = ih * W + iw
        acc = to_accumulator(tl.full((BLOCK_W,), _ZERO, dtype=tl_int_dtype))

        dy_base = dY_ptr + n * s_dy_n + c * s_dy_c
        idx_base = idx_ptr + n * s_idx_n + c * s_idx_c
        for kh in tl.static_range(0, Kh):
            numer_h = ih + ph - kh * dh
            valid_h = (numer_h % sh) == 0
            oh = numer_h // sh
            valid_h = valid_h & (oh >= 0) & (oh < Hout)
            for kw in tl.static_range(0, Kw):
                numer_w = iw + pw - kw * dw
                valid_w = (numer_w % sw) == 0
                ow = numer_w // sw
                valid = mask_w & valid_h & valid_w & (ow >= 0) & (ow < Wout)
                idx_offset = oh * s_idx_h + ow * s_idx_w
                selected = tl.load(idx_base + idx_offset, mask=valid, other=-1)
                contributes = valid & (selected == input_index)
                dy_offset = oh * s_dy_h + ow * s_dy_w
                dy = tl.load(dy_base + dy_offset, mask=contributes, other=_ZERO)
                acc = acc_add(acc, to_accumulator(dy))

        dx_base = dX_ptr + n * s_dx_n + c * s_dx_c
        dx_offset = ih * s_dx_h + iw * s_dx_w
        tl.store(dx_base + dx_offset, from_accumulator(acc), mask=mask_w)

    def max_pool2d_dinput(grad_output, indices, input_shape, kernel_size, stride, padding, dilation):
        N, C, H, W = input_shape
        Hout = grad_output.shape[2]
        Wout = grad_output.shape[3]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        Kh, Kw = kernel_size
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation
        non_overlapping = sh >= dh * (Kh - 1) + 1 and sw >= dw * (Kw - 1) + 1

        if non_overlapping:
            grad_input = torch.full((N, C, H, W), _ZERO.value, device=grad_output.device, dtype=dtype_cls.int_dtype)
        else:
            grad_input = torch.empty((N, C, H, W), device=grad_output.device, dtype=dtype_cls.int_dtype)

        s_dy_n, s_dy_c, s_dy_h, s_dy_w = grad_output.stride()
        s_dx_n, s_dx_c, s_dx_h, s_dx_w = grad_input.stride()
        s_idx_n, s_idx_c, s_idx_h, s_idx_w = indices.stride()

        BLOCK_W = 128
        common_args = (
            grad_output, grad_input, indices, N, C, H, W, Hout, Wout,
            s_dy_n, s_dy_c, s_dy_h, s_dy_w,
            s_dx_n, s_dx_c, s_dx_h, s_dx_w,
            s_idx_n, s_idx_c, s_idx_h, s_idx_w,
        )
        if non_overlapping:
            grid = (N * C, Hout, triton.cdiv(Wout, BLOCK_W))
            max_pool2d_dinput_scatter_kernel[grid](
                *common_args, BLOCK_W=BLOCK_W, num_warps=4,
            )
        else:
            grid = (N * C, H, triton.cdiv(W, BLOCK_W))
            max_pool2d_dinput_gather_kernel[grid](
                *common_args,
                Kh, Kw, sh, sw, ph, pw, dh, dw,
                BLOCK_W=BLOCK_W, num_warps=4,
            )

        return grad_input

    class DTMaxPool2dFunction(DTFunction):

        output_indices = [0]

        @staticmethod
        def forward(ctx, ops, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
            ctx.input_shape = input.shape
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
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

            grad_input = max_pool2d_dinput(
                grad_output, indices, input_shape,
                ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation,
            )
            return grad_input, None, None, None, None, None, None

    @dtype_cls.register_func(torch.nn.functional.max_pool2d,
                             cast=("input",), backend="triton")
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

        acc = to_accumulator(tl.full((BLOCK_C, BLOCK_HW), _ZERO, tl_int_dtype))

        for ky in range(Kh_max):
            ky_in = ky < kh_len
            in_h = h_start + ky

            for kx in range(Kw_max):
                kx_in = kx < kw_len
                in_w = w_start + kx

                valid_hw = ky_in & kx_in
                load_mask = in_mask & mask_hw & valid_hw[None, :]

                x_ptrs = Xb + c_offsets[:, None] * s_x_c + in_h[None, :] * s_x_h + in_w[None, :] * s_x_w
                x_vals = to_accumulator(tl.load(x_ptrs, mask=load_mask, other=_ZERO))

                acc = acc_add(x_vals, acc)

        area = tl.maximum(area, 1).to(tl.float32)
        acc = acc_div(acc, acc_from_float(area[None, :]))

        y_ptrs = Yb + c_offsets[:, None] * s_y_c + oh[None, :] * s_y_h + ow[None, :] * s_y_w
        tl.store(y_ptrs, from_accumulator(acc), mask=out_mask)

    def _max_adaptive_window(in_size, out_size):
        m = 0
        for i in range(out_size):
            start = (i * in_size) // out_size
            end = ((i + 1) * in_size + (out_size - 1)) // out_size
            m = max(m, end - start)
        return m

    @dtype_cls.register_op("adaptive_avg_pool2d", backend="triton")
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

        acc = to_accumulator(tl.full((BLOCK_C, BLOCK_HW), _ZERO, tl_int_dtype))

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
                acc = acc_add(acc, to_accumulator(contrib))

        dx_ptrs = dXb + c_offsets[:, None] * s_dx_c + ih[None, :] * s_dx_h + iw[None, :] * s_dx_w
        tl.store(dx_ptrs, from_accumulator(acc), mask=in_mask)

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
                             cast=("input",), backend="triton")
    def dt_adaptive_avg_pool2d(input, output_size):
        return DTAdaptiveAvgPool2dFunction.apply(input, output_size)


