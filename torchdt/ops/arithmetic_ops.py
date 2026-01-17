import torch
from torchdt.autograd import DTFunction
from torchdt.ops import register_base_op

class DTAddFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.add(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.sum_to_size(grad_output, x.shape)
        grad_y = ops.sum_to_size(grad_output, y.shape)

        return grad_x, grad_y

class DTSubFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.sub(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.sum_to_size(grad_output, x.shape)
        grad_y = ops.sum_to_size(ops.neg(grad_output), y.shape)

        return grad_x, grad_y

@register_base_op("sum")
def dt_sum(ops, x, dim=None, keepdim=False):
    if dim is None:
        flat = x.reshape(-1)
        out = flat[0]

        for i in range(1, flat.numel()):
            out = ops.add(out, flat[i])

        if keepdim:
            out = out.reshape([1] * x.dim())

        return out

    red_dims = (dim,) if isinstance(dim, int) else tuple(dim)
    red_dims = tuple(sorted(d % x.dim() for d in red_dims))

    permute_order = [d for d in range(x.dim()) if d not in red_dims] + list(red_dims)
    transposed = x.permute(*permute_order)

    outer_shape = transposed.shape[:-len(red_dims)]
    transposed = transposed.reshape(*outer_shape, -1)

    out = transposed[..., 0]
    for i in range(1, transposed.shape[-1]):
        out = ops.add(out, transposed[..., i])

    if keepdim:
        for d in red_dims:
            out = out.unsqueeze(d)

    return out

class DTSumFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return ops.sum(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        grad_x = grad_output
        if ctx.dim is None:
            grad_x = grad_x.expand(x.shape)

        else:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(d % x.dim() for d in red_dims)

            if not ctx.keepdim:
                for d in sorted(red_dims):
                    grad_x = grad_x.unsqueeze(d)

            grad_x = grad_x.expand(x.shape)

        return grad_x, None, None

class DTMulFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.mul(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.sum_to_size(ops.mul(grad_output, y), x.shape)
        grad_y = ops.sum_to_size(ops.mul(grad_output, x), y.shape)

        return grad_x, grad_y

class DTDivFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.div(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.sum_to_size(ops.div(grad_output, y), x.shape)
        grad_y = ops.sum_to_size(ops.neg(ops.div(ops.mul(grad_output, x), ops.mul(y, y))), y.shape)

        return grad_x, grad_y

class DTPowFunction(DTFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.pow(x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y, output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y, output = ctx.saved_tensors

        grad_x = ops.mul(grad_output, ops.mul(ops.div(output, x), y))
        grad_y = ops.mul(grad_output, ops.mul(output, ops.log(x)))

        return grad_x, grad_y

@register_base_op("square")
def dt_square(ops, x):
    return ops.mul(x, x)

class DTSquareFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.square(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        grad_x = ops.mul(grad_output, ops.mul(ops.scalar_from_float(2), x))
        return grad_x

@register_base_op("sqrt")
def dt_sqrt(ops, x):
    return ops.pow(x, ops.scalar_from_float(0.5))

class DTSqrtFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.sqrt(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        grad_x = ops.div(grad_output, ops.mul(ops.scalar_from_float(2), output))
        return grad_x

@register_base_op("reciprocal")
def dt_reciprocal(ops, x):
    return ops.div(ops.scalar_from_float(1), x)

class DTReciprocalFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.reciprocal(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        grad_x = ops.neg(ops.mul(grad_output, ops.mul(output, output)))
        return grad_x

@register_base_op("exp")
def dt_exp(ops, x):
    return ops.pow(ops.scalar_from_float(torch.e), x)

class DTExpFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.exp(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        grad_x = ops.mul(grad_output, output)
        return grad_x

@register_base_op("log")
def dt_log(ops, x):
    return ops.from_float(torch.log(ops.to_float(x)))

class DTLogFunction(DTFunction):

    @staticmethod
    def forward(ops, x):
        return ops.log(x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors
        grad_x = ops.div(grad_output, x)
        return grad_x

@register_base_op("prod")
def dt_prod(ops, x, dim=None, keepdim=False):

    if dim is None:
        flat = x.reshape(-1)

        out = flat[0]
        for i in range(1, flat.numel()):
            out = ops.mul(out, flat[i])

        if keepdim:
            out = out.reshape([1] * x.dim())

        return out

    # Reduction over a subset of the dimensions
    red_dims = (dim,) if isinstance(dim, int) else tuple(dim)
    red_dims = tuple(sorted(d % x.dim() for d in red_dims))

    # transpose so that the reduction dimensions are at the end, then flatten.
    permute_order = [d for d in range(x.dim()) if d not in red_dims] + list(red_dims)
    transposed = x.permute(*permute_order)
    outer_shape = transposed.shape[:-len(red_dims)]
    transposed = transposed.reshape(*outer_shape, -1)

    out = transposed[..., 0]
    for i in range(1, transposed.shape[-1]):
        out = ops.mul(out, transposed[..., i])

    # re-insert the reduced axes
    if keepdim:
        for d in red_dims:
            out = out.unsqueeze(d)

    return out

class DTProdFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return ops.prod(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        ctx.save_for_backward(x, output)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, output = ctx.saved_tensors

        # 1. Broadcast the forward result so it matches x's shape
        if ctx.dim is not None and not ctx.keepdim:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(sorted(d % x.dim() for d in red_dims))

            for d in red_dims:
                output = output.unsqueeze(d)

        output_broadcast = output.expand_as(x)
        ratio = ops.div(output_broadcast, x)

        # broadcast grad_output to match x's shape
        if ctx.dim is not None and not ctx.keepdim:
            for d in red_dims:
                grad_output = grad_output.unsqueeze(d)

        grad_output = grad_output.expand_as(x)
        grad_x = ops.mul(grad_output, ratio)

        return grad_x, None, None

@register_base_op("mean")
def dt_mean(ops, x, dim=None, keepdim=False):
    if dim is None:
        dims = None
    else:
        if isinstance(dim, int):
            dims = (dim,)
        else:
            dims = tuple(dim)
        # canonicalise negative indices
        dims = tuple(d % x.dim() for d in dims)

    if dims is None:
        n_elem = x.numel()
    else:
        n_elem = 1
        for d in dims:
            n_elem *= x.shape[d]

    total = ops.sum(x, dims, keepdim)
    return ops.div(total, ops.scalar_from_float(n_elem))

class DTMeanFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return ops.mean(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        if ctx.dim is None:
            dims = None
        else:
            if isinstance(ctx.dim, int):
                dims = (ctx.dim,)
            else:
                dims = tuple(ctx.dim)
            # canonicalise negative indices
            dims = tuple(d % x.dim() for d in dims)

        if dims is None:
            n_elem = x.numel()
        else:
            n_elem = 1
            for d in dims:
                n_elem *= x.shape[d]

        grad_x = ops.div(grad_output, ops.scalar_from_float(n_elem))
        if dims is None:
            grad_x = grad_x.expand(x.shape)

        else:
            if not ctx.keepdim:
                for d in sorted(dims):
                    grad_x = grad_x.unsqueeze(d)
            grad_x = grad_x.expand(x.shape)

        return grad_x, None, None

@register_base_op("var")
def dt_var(ops, x, correction, dim=None, keepdim=False):
    if dim is None:
        red_dims = None
        N = x.numel()

    else:
        red_dims = (dim,) if isinstance(dim, int) else tuple(dim)
        red_dims = tuple(d % x.dim() for d in red_dims)
        N = 1
        for d in red_dims:
            N *= x.shape[d]

    n_elems = ops.scalar_from_float(N)

    denom = ops.sub(n_elems, correction)
    if denom <= 0:
        raise ValueError("Degrees of freedom <= 0 for slice")

    total_x = ops.sum(x, dim=red_dims, keepdim=True)
    mean = ops.div(total_x, n_elems)

    diff = ops.sub(x, mean)
    sq_diff = ops.mul(diff, diff)
    total_sq = ops.sum(sq_diff, dim=red_dims, keepdim=keepdim)
    var = ops.div(total_sq, denom)

    return var

class DTVarFunction(DTFunction):

    @staticmethod
    def forward(ops, x, correction, dim=None, keepdim=False):
        return ops.var(ops, x, correction, dim, keepdim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, correction, dim, keepdim = inputs
        ctx.save_for_backward(x, correction)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, correction = ctx.saved_tensors

        if ctx.dim is None:
            red_dims = None
            N = x.numel()

        else:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(d % x.dim() for d in red_dims)

            N = 1
            for d in red_dims:
                N *= x.shape[d]

        total_x = ops.sum(x, dim=red_dims, keepdim=True)
        n_elems = ops.scalar_from_float(N)
        denom = ops.sub(n_elems, correction)
        mean = ops.div(total_x, n_elems)

        diff = ops.sub(x, mean)
        scale = ops.div(ops.scalar_from_float(2.0), denom)

        grad_x = grad_output
        if red_dims is None:
            grad_x = grad_x.expand(x.shape)

        else:
            if not ctx.keepdim:
                for d in sorted(red_dims):
                    grad_x = grad_x.unsqueeze(d)
            grad_x = grad_x.expand(x.shape)

        grad_x = ops.mul(grad_x, ops.mul(diff, scale))

        return grad_x, None, None, None

@register_base_op("matmul")
def dt_matmul(ops, A, B):
    # 1. (..., M, K)  @  (..., K, N)  -> (..., M, N)          (regular case)
    # 2. (..., M, K)  @  (..., K)     -> (..., M)             (rhs vector)
    # 3. (..., K)     @  (..., K, N)  -> (..., N)             (lhs vector)
    # 4. (..., K)     @  (..., K)     -> (..., K)             (dot product)
    orig_A_dim = A.dim()
    orig_B_dim = B.dim()

    prepended_A = False
    appended_B = False

    if orig_A_dim == 1:
        A = A.unsqueeze(0) # (K,) -> (1, K)
        prepended_A = True

    if orig_B_dim == 1:
        B = B.unsqueeze(-1) # (K,) -> (K, 1)
        appended_B = True

    # Now perform the actual matrix multiplication
    # A has shape (..., M, K) and B has shape (..., K, N)
    # For broadcasting, align batch dimensions
    M, K_A = A.shape[-2:]
    K_B, N = B.shape[-2:]

    assert K_A == K_B, "Inner dimensions of A and B must match for matrix multiplication: {K_A} vs {K_B}"

    # Handle broadcasting of batch dimensions - get batch shapes (everything except last 2 dims)
    A_batch_shape = A.shape[:-2]
    B_batch_shape = B.shape[:-2]

    try:
        output_batch_shape = torch.broadcast_shapes(A_batch_shape, B_batch_shape)
    except RuntimeError as e:
        raise RuntimeError(f"Batch dimensions are not broadcastable: {A_batch_shape} vs {B_batch_shape}") from e

    # Expand A and B to have the same batch dimensions
    A = A.expand(*output_batch_shape, M, K_A)
    B = B.expand(*output_batch_shape, K_B, N)

    result = ops.full((*output_batch_shape, M, N), 0.0, device=A.device)

    # Perform matrix multiplication in log space
    for k in range(K_A):
        term = ops.mul(
            A[..., :, k].unsqueeze(-1), # (..., M, 1)
            B[..., k, :].unsqueeze(-2)  # (..., 1, N)
        )
        result = ops.add(result, term)

    if prepended_A:
        result = result.squeeze(-2) # Remove extra M dimension
    if appended_B:
        result = result.squeeze(-1) # Remove extra N dimension

    return result

class DTMatmulFunction(DTFunction):

    @staticmethod
    def forward(ops, A, B):
        return ops.matmul(A, B)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        # Here we must repeat the unsqueezing logic from forward
        # to ensure that we can correctly compute the gradients.
        # todo: This is a bit of a hack, we should ideally handle
        # the unsqueezing in a more elegant way. 
        A, B = inputs

        ctx.prepended_A = False
        ctx.appended_B = False

        if A.dim() == 1:
            A = A.unsqueeze(0)
            ctx.prepended_A = True

        if B.dim() == 1:
            B = B.unsqueeze(-1)
            ctx.appended_B = True

        ctx.A_shape_before_broadcast = A.shape
        ctx.B_shape_before_broadcast = B.shape

        A_batch_shape = A.shape[:-2]
        B_batch_shape = B.shape[:-2]
        output_batch_shape = torch.broadcast_shapes(A_batch_shape, B_batch_shape)

        A = A.expand(*output_batch_shape, *A.shape[-2:])
        B = B.expand(*output_batch_shape, *B.shape[-2:])

        ctx.save_for_backward(A, B)

    @staticmethod
    def backward(ctx, ops, grad_output):
        A, B = ctx.saved_tensors

        #  Re-introduce squeezed dimensions
        if ctx.prepended_A and not ctx.appended_B:
            grad_output = grad_output.unsqueeze(-2)
        elif ctx.appended_B and not ctx.prepended_A:
            grad_output = grad_output.unsqueeze(-1)
        elif ctx.prepended_A and ctx.appended_B:
            grad_output = grad_output.unsqueeze(-1).unsqueeze(-1)

        # Compute gradients w.r.t A and B after broadcasting
        grad_A = ops.matmul(grad_output, B.transpose(-1, -2))
        grad_B = ops.matmul(A.transpose(-1, -2), grad_output)

        # Reduce gradients to match original shapes before broadcasting
        # We need to sum over dimensions that were broadcasted

        # For grad_A: reduce to shape before broadcasting
        A_shape_before_broadcast = ctx.A_shape_before_broadcast
        while grad_A.dim() > len(A_shape_before_broadcast):
            grad_A = ops.sum(grad_A, dim=0)

        # Sum over any dimensions that were size 1 and got broadcasted
        for i in range(len(A_shape_before_broadcast) - 2):  # Don't touch matrix dims
            if A_shape_before_broadcast[i] == 1 and grad_A.shape[i] > 1:
                grad_A = ops.sum(grad_A, dim=i, keepdim=True)

        # For grad_B: reduce to shape before broadcasting
        B_shape_before_broadcast = ctx.B_shape_before_broadcast
        while grad_B.dim() > len(B_shape_before_broadcast):
            grad_B = ops.sum(grad_B, dim=0)

        # Sum over any dimensions that were size 1 and got broadcasted
        for i in range(len(B_shape_before_broadcast) - 2):  # Don't touch matrix dims
            if B_shape_before_broadcast[i] == 1 and grad_B.shape[i] > 1:
                grad_B = ops.sum(grad_B, dim=i, keepdim=True)

        if ctx.prepended_A:
            grad_A = grad_A.squeeze(0) # Remove extra M dimension
        if ctx.appended_B:
            grad_B = grad_B.squeeze(-1) # Remove extra N dimension

        return grad_A, grad_B

@register_base_op("transpose")
def dt_transpose(ops, x, dim0, dim1):
    return x.transpose(dim0, dim1).clone()

class DTTransposeFunction(DTFunction):

    @staticmethod
    def forward(ops, x, dim0, dim1):
        return ops.transpose(x, dim0, dim1)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim0, dim1 = inputs
        ctx.dim0 = dim0
        ctx.dim1 = dim1

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = ops.transpose(grad_output, ctx.dim0, ctx.dim1)
        return grad_x, None, None
