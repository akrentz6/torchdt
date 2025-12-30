import torch
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def register_triton_ops(
    dtype_cls: type,
    _ZERO,
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

) -> None:
    if not HAS_TRITON:
        raise ImportError("Triton is not installed. Please install Triton to use Triton backend.")

    if dtype_cls.bitwidth == 8:
        int_dtype = tl.constexpr(tl.int8)
    elif dtype_cls.bitwidth == 16:
        int_dtype = tl.constexpr(tl.int16)
    elif dtype_cls.bitwidth == 32:
        int_dtype = tl.constexpr(tl.int32)
    elif dtype_cls.bitwidth == 64:
        int_dtype = tl.constexpr(tl.int64)

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

        acc = tl.full((BLOCK_M, BLOCK_N), _ZERO, dtype=int_dtype)

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