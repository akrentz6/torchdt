import torch
from torchdt.autograd import DTFunction
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def register_triton_ops(
    dtype_cls: type,
    from_float=None,
    to_float=None,
    add=None,
    sub=None,
    mul=None,
    div=None,
    sqrt=None,
    gt=None,
    ge=None,
    lt=None,
    le=None,
    _ZERO=None,
) -> None:
    if not HAS_TRITON:
        raise ImportError("Triton is not installed. Please install Triton to use Triton backend.")

    if dtype_cls.bitwidth == 8:
        tl_int_dtype = tl.constexpr(tl.int8)
    elif dtype_cls.bitwidth == 16:
        tl_int_dtype = tl.constexpr(tl.int16)
    elif dtype_cls.bitwidth == 32:
        tl_int_dtype = tl.constexpr(tl.int32)
    elif dtype_cls.bitwidth == 64:
        tl_int_dtype = tl.constexpr(tl.int64)

    @triton.jit
    def atomic_add(x_ptrs, val, mask):
        active = mask

        while tl.max(active) != 0:
            old = tl.load(x_ptrs, mask=active, other=_ZERO)
            new = add(old, tl.where(active, val, _ZERO))
            prev = tl.atomic_cas(x_ptrs, old, new)

            success = prev == old
            active = active & (~success)

    @triton.jit
    def from_float_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask)

        out = from_float(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("from_float")
    def dt_from_float(ops, x):
        out = torch.empty(x.shape, dtype=dtype_cls.int_dtype, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        from_float_kernel[grid](x, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def to_float_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)

        out = to_float(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("to_float")
    def dt_to_float(ops, x):
        out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        to_float_kernel[grid](x, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + offs, mask=mask, other=_ZERO)

        out = add(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("add")
    def dt_add(ops, x, y):
        out = torch.empty(x.shape, dtype=dtype_cls.int_dtype, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        add_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def sub_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + offs, mask=mask, other=_ZERO)

        out = sub(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("sub")
    def dt_sub(ops, x, y):
        out = torch.empty(x.shape, dtype=dtype_cls.int_dtype, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        sub_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def mul_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + offs, mask=mask, other=_ZERO)

        out = mul(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("mul")
    def dt_mul(ops, x, y):
        out = torch.empty(x.shape, dtype=dtype_cls.int_dtype, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        mul_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def div_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + offs, mask=mask, other=_ZERO)

        out = div(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("div")
    def dt_div(ops, x, y):
        out = torch.empty(x.shape, dtype=dtype_cls.int_dtype, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        div_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def sqrt_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)

        out = sqrt(x)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("sqrt")
    def dt_sqrt(ops, x):
        out = torch.empty(x.shape, dtype=dtype_cls.int_dtype, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        sqrt_kernel[grid](x, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def gt_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + offs, mask=mask, other=_ZERO)

        out = gt(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("gt")
    def dt_gt(ops, x, y):
        out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        gt_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def ge_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + offs, mask=mask, other=_ZERO)

        out = ge(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("ge")
    def dt_ge(ops, x, y):
        out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        ge_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def lt_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + offs, mask=mask, other=_ZERO)

        out = lt(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("lt")
    def dt_lt(ops, x, y):
        out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        lt_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.jit
    def le_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_ptr + offs, mask=mask, other=_ZERO)
        y = tl.load(y_ptr + offs, mask=mask, other=_ZERO)

        out = le(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("le")
    def dt_le(ops, x, y):
        out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        le_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
        return out

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 128},  num_warps=2, num_stages=2),
            triton.Config({"BLOCK": 256},  num_warps=2, num_stages=2),
            triton.Config({"BLOCK": 256},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 512},  num_warps=8, num_stages=2),
            triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
        ],
        key=["N"],
    )
    @triton.jit
    def sum_kernel(x_ptr, y_ptr, M, N, s_x_r, s_x_c, s_y_r, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        in_bounds = pid < M

        row_ptr = x_ptr + pid * s_x_r
        acc = _ZERO

        num_tiles = (N + BLOCK - 1) // BLOCK
        for tile_idx in range(0, num_tiles):
            offs = tile_idx * BLOCK + tl.arange(0, BLOCK)
            mask = in_bounds & (offs < N)

            vals = tl.load(row_ptr + offs * s_x_c, mask=mask, other=_ZERO)
            acc = add(acc, tl.reduce(vals, axis=0, combine_fn=add))

        if in_bounds:
            tl.store(y_ptr + pid * s_y_r, acc)

    @dtype_cls.register_op("sum")
    def dt_sum(x, dim=None, keepdim=False):
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
        x_perm = x.permute(*perm).contiguous()

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
            out = out.view(*out_shape)

        else:
            out_shape = [orig_shape[d] for d in kept_dims]
            out = out.view(*out_shape) if out_shape else out.view(())

        return out


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 8 }, num_warps=4, num_stages=2),
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

            a_ptrs = base_a + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
            b_ptrs = base_b + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn

            # prod = lns_mul(a_blk[:, :, None], b_blk[None, :, :])
            # acc = lns_add(acc, tl.reduce(prod, axis=1, combine_fn=lns_add))

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
            A2 = A.view(1, M, K_A)
            B2 = B.view(1, K_B, N)
        else:
            batch = math.prod(batch_shape)
            A2 = A.view(batch, M, K_A)
            B2 = B.view(batch, K_B, N)

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
            C = C.view(M, N)
        else:
            C = C.view(*batch_shape, M, N)

        if a_was_1d and b_was_1d:
            C = C.squeeze(-1).squeeze(-2)
        elif a_was_1d:
            C = C.squeeze(-2)
        elif b_was_1d:
            C = C.squeeze(-1)

        return C

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
        oc_block = pid2 * oc_tiles_per_group + oc_block_in_group

        oc_offsets = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)

        h = hw_offsets // Wout
        w = hw_offsets % Wout

        mask_oc = oc_offsets < Cout
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
    def dt_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
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

        BLOCK_OC = 8
        BLOCK_HW = 128
        hw_tiles = triton.cdiv(Hout * Wout, BLOCK_HW)
        Cout_g = Cout // groups
        oc_tiles_per_group = triton.cdiv(Cout_g, BLOCK_OC)

        grid = (hw_tiles * oc_tiles_per_group, N, groups)
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
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=1,
        )

        return y

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
                    w_val = tl.load(W_ptr + w_idx, mask=valid_pos, other=_ZERO)

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

        BLOCK_HW = 64
        grid = (N * Cin, triton.cdiv(Hin * Win, BLOCK_HW))
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
            BLOCK_HW
        )
        return grad_input

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
        rem = pid % (Cin_per_group * Kh * Kw)
        cin = rem // (Kh * Kw)
        rem2 = rem % (Kh * Kw)
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

        BLOCK_NHW = 64
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
            BLOCK_NHW,
        )

        return grad_weight

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

        BLOCK_NHW = 1024
        grid = (Cout,)
        conv2d_dbias_kernel[grid](
            grad_bias, grad_output,
            N, Cout, Hout, Wout,
            s_db_c,
            s_dy_n, s_dy_c, s_dy_h, s_dy_w,
            BLOCK_NHW,
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
        def backward(ctx, grad_output):
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

            if ctx.needs_input_grad[0]:
                grad_input = conv2d_dinput(
                    grad_output, weight, input.shape,
                    stride, padding, dilation, groups
                )
            else:
                grad_input = None

            if ctx.needs_input_grad[1]:
                grad_weight = conv2d_dweight(
                    grad_output, input, weight.shape,
                    stride, padding, dilation, groups
                )
            else:
                grad_weight = None

            if bias is not None and ctx.needs_input_grad[2]:
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