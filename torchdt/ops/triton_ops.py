import torch
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
) -> None:
    if not HAS_TRITON:
        raise ImportError("Triton is not installed. Please install Triton to use Triton backend.")

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

        x = tl.load(x_ptr + offs, mask=mask)

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

        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)

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

        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)

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

        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)

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

        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)

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

        x = tl.load(x_ptr + offs, mask=mask)

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

        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)

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

        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)

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

        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)

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

        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)

        out = le(x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @dtype_cls.register_op("le")
    def dt_le(ops, x, y):
        out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
        grid = (triton.cdiv(x.numel(), 1024),)
        le_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
        return out