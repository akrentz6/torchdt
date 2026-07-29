import math

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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 8 }, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 8 }, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 8 }, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 8 }, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 4 }, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 4 }, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        ],
        key=["M_BUCKET", "N_BUCKET", "K_BUCKET"],
    )
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        BATCH, M, N, K,
        M_BUCKET, N_BUCKET, K_BUCKET,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_b = tl.program_id(1)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

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

        acc = to_accumulator(tl.full((BLOCK_M, BLOCK_N), _ZERO, dtype=tl_int_dtype))

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

                acc = acc_add(acc, to_accumulator(mul(a_k[:, None], b_k[None, :])))

        c_ptrs = base_c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, from_accumulator(acc), mask=mask_m[:, None] & mask_n[None, :] & (pid_b < BATCH))

    @dtype_cls.register_op("matmul", backend="triton")
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

        if len(batch_shape) == 0:
            batch = 1
            A2 = A.unsqueeze(0)
            B2 = B.unsqueeze(0)
        elif len(batch_shape) == 1:
            # Preserve transposes and stride-zero broadcasting.
            batch = batch_shape[0]
            A2 = A
            B2 = B
        else:
            batch = math.prod(batch_shape)
            try:
                A2 = A.view(batch, M, K_A)
            except RuntimeError:
                A2 = A.contiguous().view(batch, M, K_A)
            try:
                B2 = B.view(batch, K_B, N)
            except RuntimeError:
                B2 = B.contiguous().view(batch, K_B, N)

        C = torch.empty((batch, M, N), device=A.device, dtype=dtype_cls.int_dtype)

        stride_ab, stride_am, stride_ak = A2.stride()
        stride_bb, stride_bk, stride_bn = B2.stride()
        stride_cb, stride_cm, stride_cn = C.stride()

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            batch,
        )

        def shape_bucket(value):
            return min(4096, triton.next_power_of_2(max(1, int(value))))

        matmul_kernel[grid](
            A2, B2, C,
            batch, M, N, K_A,
            shape_bucket(M), shape_bucket(N), shape_bucket(K_A),
            stride_ab, stride_am, stride_ak,
            stride_bb, stride_bk, stride_bn,
            stride_cb, stride_cm, stride_cn,
            GROUP_SIZE_M=8,
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

