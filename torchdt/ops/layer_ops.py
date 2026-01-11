import math
import torch
from torchdt.autograd import DTFunction
from torchdt.ops import register_base_op

@register_base_op("linear")
def dt_linear(ops, x, weight, bias=None):
    output = ops.matmul(x, weight.transpose(-2, -1))
    if bias is not None:
        output = ops.add(output, bias)
    return output

class DTLinearFunction(DTFunction):

    @staticmethod
    def forward(ops, x, weight, bias=None):
        return ops.linear(x, weight, bias)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, A, bias = inputs
        ctx.biased = bias is not None
        ctx.save_for_backward(x, A)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, A = ctx.saved_tensors
        grad_x = ops.matmul(grad_output, A)
        out_features = A.shape[0]
        in_features = A.shape[1]

        grad_output_2d = grad_output.reshape(-1, out_features)
        x_2d = x.reshape(-1, in_features)
        grad_output_T = grad_output_2d.transpose(0, 1)
        grad_A = ops.matmul(grad_output_T, x_2d)

        if ctx.biased:
            if grad_output.dim() == 1:
                grad_bias = grad_output
            else:
                grad_bias = ops.sum(grad_output, dim=tuple(range(grad_output.dim() - 1)))
        else:
            grad_bias = None

        return grad_x, grad_A, grad_bias

@register_base_op("dropout")
def dt_dropout(ops, x, p=0.5):
    mask = ops.from_float(torch.bernoulli(torch.full(x.shape, 1 - p)))
    return ops.mul(x, mask)

class DTDropoutFunction(DTFunction):

    @staticmethod
    def forward(ops, x, p=0.5):
        return ops.dropout(x, p)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, p = inputs
        ctx.save_for_backward(output)
        ctx.p = p

    @staticmethod
    def backward(ctx, ops, grad_output):
        output, = ctx.saved_tensors
        grad_x = torch.where(
            output == ops.scalar_from_float(0.0),
            ops.scalar_from_float(0.0),
            ops.mul(grad_output, ops.scalar_from_float(1 / (1 - ctx.p))))
        return grad_x, None

@register_base_op("conv2d")
def dt_conv2d(ops, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Handle stride, padding, dilation as tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # Add batch dimension if needed
    squeeze_batch = False
    if x.dim() == 3:
        squeeze_batch = True
        x = x.unsqueeze(0)

    N, C_in, H_in, W_in = x.shape # [batch, in_channels, height, width]
    C_out, C_w, K_H, K_W = weight.shape  # [out_channels, in_per_group, kh, kw]
    assert C_in % groups == 0
    assert C_out % groups == 0
    assert C_w == C_in // groups
    g_Cin = C_in // groups
    g_Cout = C_out // groups

    # Pad input
    if pad_h > 0 or pad_w > 0:
        x_padded = torch.nn.functional.pad(
            x, (pad_w, pad_w, pad_h, pad_h), value=ops.scalar_from_float(0.0).item()
        )
    else:
        x_padded = x

    H_pad, W_pad = x_padded.shape[2:]
    # Output shape calculation as per PyTorch
    H_out = (H_in + 2 * pad_h - dil_h * (K_H - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_W - 1) - 1) // stride_w + 1
    out = ops.zeros(N, C_out, H_out, W_out)

    for n in range(N):
        for g in range(groups):
            inp_group = x_padded[n, g * g_Cin : (g + 1) * g_Cin] # [g_Cin, H_pad, W_pad]
            for c_out in range(g * g_Cout, (g + 1) * g_Cout):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride_h
                        w_start = w * stride_w
                        h_end = h_start + K_H * dil_h
                        w_end = w_start + K_W * dil_w
                        # Extract appropriate input window
                        inp_slice = inp_group[:, h_start:h_end:dil_h, w_start:w_end:dil_w] # shape [g_Cin, K_H, K_W]
                        out[n, c_out, h, w] = ops.sum(ops.mul(inp_slice, weight[c_out]))
                        if bias is not None:
                            out[n, c_out, h, w] = ops.add(out[n, c_out, h, w], bias[c_out])

    if squeeze_batch:
        out = out.squeeze(0)
    return out

class DTConv2dFunction(DTFunction):

    @staticmethod
    def forward(ops, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return ops.conv2d(x, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, weight, bias, stride, padding, dilation, groups = inputs
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, weight, bias = ctx.saved_tensors

        # Canonicalize params
        stride = ctx.stride if isinstance(ctx.stride, tuple) else (ctx.stride, ctx.stride)
        padding = ctx.padding if isinstance(ctx.padding, tuple) else (ctx.padding, ctx.padding)
        dilation = ctx.dilation if isinstance(ctx.dilation, tuple) else (ctx.dilation, ctx.dilation)
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation
        groups = ctx.groups

        squeeze_batch = (x.dim() == 3)
        if grad_output.dim() == 3:
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True

        N, C_out, H_out, W_out = grad_output.shape
        C_in = weight.shape[1] * groups
        K_H, K_W = weight.shape[2:]
        g_Cin = C_in // groups
        g_Cout = C_out // groups

        # Pad input for correct alignment
        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), value=ops.scalar_from_float(0.0).item())
        else:
            x_padded = x

        H_pad = x_padded.shape[2]
        W_pad = x_padded.shape[3]

        grad_x_padded = ops.zeros(N, C_in, H_pad, W_pad)
        grad_weight = ops.zeros_like(weight)

        # Input gradient: for each pixel in input, sum all contributions from grad_output via the receptive field
        for n in range(N):
            for g in range(groups):
                in_start = g * g_Cin
                in_end = (g + 1) * g_Cin
                out_start = g * g_Cout
                out_end = (g + 1) * g_Cout
                for c_in in range(g_Cin):
                    for h_in in range(H_pad):
                        for w_in in range(W_pad):
                            grad = ops.scalar_from_float(0.0)
                            for c_out in range(out_start, out_end):
                                w = weight[c_out, c_in, :, :]
                                for k_h in range(K_H):
                                    for k_w in range(K_W):
                                        # Figure out which output position this input contributes to
                                        h_out_nom = h_in - k_h * dil_h
                                        w_out_nom = w_in - k_w * dil_w
                                        if h_out_nom % stride_h == 0 and w_out_nom % stride_w == 0:
                                            h_out = h_out_nom // stride_h
                                            w_out = w_out_nom // stride_w
                                            if (0 <= h_out < H_out) and (0 <= w_out < W_out):
                                                grad = ops.add(
                                                    grad,
                                                    ops.mul(grad_output[n, c_out, h_out, w_out], w[k_h, k_w])
                                                )
                            grad_x_padded[n, in_start + c_in, h_in, w_in] = grad

        # Remove padding to match original input shape
        if pad_h > 0 or pad_w > 0:
            grad_x = grad_x_padded[:, :, pad_h: H_pad - pad_h, pad_w: W_pad - pad_w]
        else:
            grad_x = grad_x_padded

        # Weight gradient: correlate grad_output windows with input
        for g in range(groups):
            in_start = g * g_Cin
            in_end = (g + 1) * g_Cin
            out_start = g * g_Cout
            out_end = (g + 1) * g_Cout
            for c_out in range(out_start, out_end):
                for c_in in range(g_Cin):
                    for k_h in range(K_H):
                        for k_w in range(K_W):
                            grad = ops.scalar_from_float(0.0)
                            for n in range(N):
                                for h_out in range(H_out):
                                    for w_out in range(W_out):
                                        h_in = h_out * stride_h + k_h * dil_h
                                        w_in = w_out * stride_w + k_w * dil_w
                                        inp_padded = x_padded[n, in_start + c_in, :, :]
                                        if (0 <= h_in < inp_padded.size(0)) and (0 <= w_in < inp_padded.size(1)):
                                            grad = ops.add(
                                                grad,
                                                ops.mul(grad_output[n, c_out, h_out, w_out], inp_padded[h_in, w_in])
                                            )
                            grad_weight[c_out, c_in, k_h, k_w] = grad

        # Bias gradient: sum grad_output over batch, spatial dims
        if bias is not None:
            grad_bias = ops.sum(grad_output, dim=[0, 2, 3])
        else:
            grad_bias = None

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)
        return grad_x, grad_weight, grad_bias, None, None, None, None

@register_base_op("avg_pool2d")
def dt_avg_pool2d(ops, x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    kernel_w, kernel_h = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    N, C, H_in, W_in = x.shape
    if pad_h > 0 or pad_w > 0:
        x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), value=ops.scalar_from_float(0.0).item())
    else:
        x_padded = x

    if ceil_mode:
        H_out = int(math.ceil((H_in + 2 * pad_h - kernel_h) / stride_h)) + 1
        W_out = int(math.ceil((W_in + 2 * pad_w - kernel_w) / stride_w)) + 1
    else:
        H_out = (H_in + 2 * pad_h - kernel_h) // stride_h + 1
        W_out = (W_in + 2 * pad_w - kernel_w) // stride_w + 1

    out = ops.zeros(N, C, H_out, W_out)
    kernel_area_dt = ops.scalar_from_float(kernel_h * kernel_w)

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride_h
                    h_end = h_start + kernel_h
                    w_start = w_out * stride_w
                    w_end = w_start + kernel_w
                    if h_end > x_padded.size(-2) or w_end > x_padded.size(-1):
                        continue

                    window = x_padded[n, c, h_start:h_end, w_start:w_end]
                    sm = ops.sum(window)

                    if divisor_override is not None:
                        divisor = divisor_override
                    elif count_include_pad:
                        divisor = kernel_area_dt
                    else:
                        left_pad = max(0, pad_w - w_start)
                        right_pad = max(0, w_end - (W_in + pad_w))
                        top_pad = max(0, pad_h - h_start)
                        bot_pad = max(0, h_end - (H_in + pad_h))
                        valid_h = kernel_h - (top_pad + bot_pad)
                        valid_w = kernel_w - (left_pad + right_pad)
                        valid_count = max(valid_h, 0) * max(valid_w, 0)
                        divisor = ops.scalar_from_float(max(valid_count, 1))

                    out[n, c, h_out, w_out] = ops.div(sm, divisor)

    if squeeze_batch:
        out = out.squeeze(0)
    return out

class DTAvgPool2dFunction(DTFunction):

    @staticmethod
    def forward(ops, x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        return ops.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override = inputs
        if stride is None:
            stride = kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if divisor_override is None:
            ctx.divisor_override = False
            ctx.save_for_backward(x)
        else:
            ctx.divisor_override = True
            ctx.save_for_backward(x, divisor_override)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad

    @staticmethod
    def backward(ctx, ops, grad_output):
        if ctx.divisor_override:
            x, divisor_override = ctx.saved_tensors
        else:
            x, = ctx.saved_tensors

        kernel_h, kernel_w = ctx.kernel_size
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding

        if x.dim() == 3:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, H_in, W_in = x.shape
        H_out, W_out = grad_output.shape[-2:]

        grad_x = ops.zeros_like(x)
        kernel_area_dt = ops.scalar_from_float(kernel_h * kernel_w)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride_h
                        h_end = h_start + kernel_h
                        w_start = w_out * stride_w
                        w_end = w_start + kernel_w

                        if ctx.divisor_override:
                            divisor = divisor_override
                        elif ctx.count_include_pad:
                            divisor = kernel_area_dt
                        else:
                            left_pad = max(0, pad_w - w_start)
                            right_pad = max(0, w_end - (W_in + pad_w))
                            top_pad = max(0, pad_h - h_start)
                            bot_pad = max(0, h_end - (H_in + pad_h))
                            valid_h = kernel_h - (top_pad + bot_pad)
                            valid_w = kernel_w - (left_pad + right_pad)
                            valid_count = max(valid_h, 0) * max(valid_w, 0)
                            divisor = ops.scalar_from_float(max(valid_count, 1))

                        grad = grad_output[n, c, h_out, w_out]
                        for i in range(kernel_h):
                            for j in range(kernel_w):
                                h_idx = h_start + i - pad_h
                                w_idx = w_start + j - pad_w
                                if 0 <= h_idx < H_in and 0 <= w_idx < W_in:
                                    grad_x[n, c, h_idx, w_idx] = ops.add(
                                        grad_x[n, c, h_idx, w_idx],
                                        ops.div(grad, divisor)
                                    )

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)
        return grad_x, None, None, None, None, None, None

@register_base_op("adaptive_avg_pool2d")
def dt_adaptive_avg_pool2d(ops, x, output_size):
    if isinstance(output_size, int):
        H_out, W_out = output_size, output_size
    else:
        assert len(output_size) == 2, "output_size must be int or tuple of length 2"
        H_out, W_out = output_size

    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    N, C, H_in, W_in = x.shape
    out = ops.zeros(N, C, H_out, W_out)

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                h_start = int(math.floor(h_out * H_in / H_out))
                h_end = int(math.ceil((h_out + 1) * H_in / H_out))
                for w_out in range(W_out):
                    w_start = int(math.floor(w_out * W_in / W_out))
                    w_end = int(math.ceil((w_out + 1) * W_in / W_out))

                    window = x[n, c, h_start:h_end, w_start:w_end]
                    sm = ops.sum(window)
                    divisor = ops.scalar_from_float(max((h_end - h_start) * (w_end - w_start), 1))
                    out[n, c, h_out, w_out] = ops.div(sm, divisor)

    if squeeze_batch:
        out = out.squeeze(0)
    return out

class DTAdaptiveAvgPool2dFunction(DTFunction):

    @staticmethod
    def forward(ops, x, output_size):
        return ops.adaptive_avg_pool2d(x, output_size)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, output_size = inputs
        ctx.save_for_backward(x)
        ctx.output_size = output_size

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        if isinstance(ctx.output_size, int):
            H_out, W_out = ctx.output_size, ctx.output_size
        else:
            H_out, W_out = ctx.output_size

        if x.dim() == 3:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, H_in, W_in = x.shape
        grad_x = ops.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    h_start = int(math.floor(h_out * H_in / H_out))
                    h_end = int(math.ceil((h_out + 1) * H_in / H_out))
                    for w_out in range(W_out):
                        w_start = int(math.floor(w_out * W_in / W_out))
                        w_end = int(math.ceil((w_out + 1) * W_in / W_out))

                        divisor = ops.scalar_from_float(max((h_end - h_start) * (w_end - w_start), 1))
                        grad = ops.div(grad_output[n, c, h_out, w_out], divisor)

                        for i in range(h_start, h_end):
                            for j in range(w_start, w_end):
                                grad_x[n, c, i, j] = ops.add(grad_x[n, c, i, j], grad)

        if squeeze_batch:
            grad_x = grad_x.squeeze(0)
        return grad_x, None

@register_base_op("max_pool2d")
def dt_max_pool2d(ops, x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    N, C, H_in, W_in = x.shape
    if pad_h > 0 or pad_w > 0:
        x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h),
                                           value=ops.scalar_from_float(float("-inf")).item())
    else:
        x_padded = x

    eff_kh = (kernel_h - 1) * dil_h + 1
    eff_kw = (kernel_w - 1) * dil_w + 1
    if ceil_mode:
        H_out = int(math.ceil((H_in + 2 * pad_h - eff_kh) / stride_h)) + 1
        W_out = int(math.ceil((W_in + 2 * pad_w - eff_kw) / stride_w)) + 1
    else:
        H_out = (H_in + 2 * pad_h - eff_kh) // stride_h + 1
        W_out = (W_in + 2 * pad_w - eff_kw) // stride_w + 1

    out_vals = ops.zeros(N, C, H_out, W_out)
    if return_indices:
        out_idx = torch.empty((N, C, H_out, W_out), dtype=torch.int64, device=x.device)

    padded_H = x_padded.size(-2)
    padded_W = x_padded.size(-1)

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                h_start = h_out * stride_h
                h_end = h_start + eff_kh

                if h_start >= padded_H:
                    continue

                for w_out in range(W_out):
                    w_start = w_out * stride_w
                    w_end = w_start + eff_kw

                    if w_start >= padded_W:
                        continue

                    h_end_eff = min(h_end, padded_H)
                    w_end_eff = min(w_end, padded_W)

                    window = x_padded[n, c, h_start:h_end_eff:dil_h, w_start:w_end_eff:dil_w]
                    window_flat = window.reshape(-1)

                    # pass dim to get indices
                    mx_val, mx_idx = ops.max(window_flat, dim=0)
                    out_vals[n, c, h_out, w_out] = mx_val

                    len_w = (w_end_eff - w_start + dil_w - 1) // dil_w

                    if return_indices:
                        mx_idx_int = int(mx_idx)
                        i = mx_idx_int // len_w
                        j = mx_idx_int % len_w
                        h_in_idx = (h_start + i * dil_h) - pad_h
                        w_in_idx = (w_start + j * dil_w) - pad_w
                        out_idx[n, c, h_out, w_out] = h_in_idx * W_in + w_in_idx

    if squeeze_batch:
        out_vals = out_vals.squeeze(0)
        if return_indices:
            out_idx = out_idx.squeeze(0)

    if return_indices:
        return out_vals, out_idx
    else:
        return out_vals

class DTMaxPool2dFunction(DTFunction):

    output_indices = [0]

    @staticmethod
    def forward(ops, x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        return ops.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, kernel_size, stride, padding, dilation, ceil_mode, return_indices = inputs

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        ctx.return_indices = return_indices

    @staticmethod
    def backward(ctx, ops, grad_output, grad_out_indices=None):
        x, = ctx.saved_tensors

        kernel_h, kernel_w = ctx.kernel_size
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding
        dil_h, dil_w = ctx.dilation

        if x.dim() == 3:
            x = x.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        N, C, H_in, W_in = x.shape

        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h),
                                               value=ops.scalar_from_float(float("-inf")).item())
        else:
            x_padded = x

        padded_H, padded_W = x_padded.shape[-2:]
        eff_kh = (kernel_h - 1) * dil_h + 1
        eff_kw = (kernel_w - 1) * dil_w + 1

        if ctx.ceil_mode:
            H_out = int(math.ceil((H_in + 2 * pad_h - eff_kh) / stride_h)) + 1
            W_out = int(math.ceil((W_in + 2 * pad_w - eff_kw) / stride_w)) + 1
        else:
            H_out = (H_in + 2 * pad_h - eff_kh) // stride_h + 1
            W_out = (W_in + 2 * pad_w - eff_kw) // stride_w + 1

        grad_padded = ops.zeros_like(x_padded)
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    h_start = h_out * stride_h
                    h_end = h_start + eff_kh

                    if h_start >= padded_H:
                        continue

                    for w_out in range(W_out):
                        w_start = w_out * stride_w
                        w_end = w_start + eff_kw

                        if w_start >= padded_W:
                            continue

                        h_end_eff = min(h_end, padded_H)
                        w_end_eff = min(w_end, padded_W)

                        len_h = (h_end_eff - h_start + dil_h - 1) // dil_h
                        len_w = (w_end_eff - w_start + dil_w - 1) // dil_w
                        if len_h <= 0 or len_w <= 0:
                            continue

                        window = x_padded[n, c, h_start:h_end_eff:dil_h, w_start:w_end_eff:dil_w]
                        window_flat = window.reshape(-1)

                        _, mx_idx = ops.max(window_flat, dim=0)

                        mx_idx = int(mx_idx)
                        i = mx_idx // len_w
                        j = mx_idx %  len_w

                        h_in_idx = h_start + i * dil_h
                        w_in_idx = w_start + j * dil_w

                        grad_padded[n, c, h_in_idx, w_in_idx] = ops.add(grad_padded[n, c, h_in_idx, w_in_idx],
                                                                        grad_output[n, c, h_out, w_out])

        if pad_h > 0 or pad_w > 0:
            grad_padded = grad_padded[:, :, pad_h:pad_h + H_in, pad_w:pad_w + W_in]
        grad_x = grad_padded.squeeze(0) if squeeze_batch else grad_padded
        return grad_x, None, None, None, None, None, None

@register_base_op("batch_norm")
def dt_batch_norm(ops, x, running_mean, running_var, momentum, eps, weight=None, bias=None, training=True):
    red_dims = tuple(i for i in range(x.dim()) if i != 1)

    if training:
        batch_mean = ops.mean(x, dim=red_dims, keepdim=True)
        batch_var = ops.var(x, ops.scalar_from_float(0.0), dim=red_dims, keepdim=True)
        batch_var_corrected = ops.var(x, ops.scalar_from_float(1.0), dim=red_dims, keepdim=True)

        with torch.no_grad():
            one_minus_momentum = ops.sub(ops.scalar_from_float(1.0), momentum)
            new_running_mean = ops.add(
                ops.mul(one_minus_momentum, running_mean),
                ops.mul(momentum, batch_mean.squeeze()))
            running_mean.copy_(new_running_mean)
            new_running_var = ops.add(
                ops.mul(one_minus_momentum, running_var),
                ops.mul(momentum, batch_var_corrected.squeeze()))
            running_var.copy_(new_running_var)

        mean = batch_mean
        var = batch_var

    else:
        mean = running_mean.view(1, -1, *([1] * (x.dim() - 2)))
        var = running_var.view(1, -1, *([1] * (x.dim() - 2)))

    var_eps = ops.add(var, eps)
    inv_std = ops.div(ops.scalar_from_float(1.0), ops.sqrt(var_eps))
    x_centered = ops.sub(x, mean)
    x_hat = ops.mul(x_centered, inv_std)

    if weight is not None:
        y = ops.mul(x_hat, weight.view(1, -1, *([1] * (x.dim() - 2))))
    else:
        y = x_hat

    if bias is not None:
        y = ops.add(y, bias.view(1, -1, *([1] * (x.dim() - 2))))
    return y

class DTBatchNormFunction(DTFunction):

    @staticmethod
    def forward(ops, x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
        return ops.batch_norm(x, running_mean, running_var, momentum, eps, weight, bias, training)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, running_mean, running_var, weight, bias, training, _, eps = inputs
        ctx.red_dims = tuple(i for i in range(x.dim()) if i != 1)
        ctx.training = training

        if training:
            mean = ops.mean(x, dim=ctx.red_dims, keepdim=True)
            var = ops.var(x, ops.scalar_from_float(1.0), dim=ctx.red_dims, keepdim=True)
        else:
            mean = running_mean.reshape(1, -1, *([1] * (x.dim() - 2)))
            var = running_var.reshape(1, -1, *([1] * (x.dim() - 2)))
        ctx.save_for_backward(x, weight, bias, mean, var, eps)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, weight, bias, mean, var, eps = ctx.saved_tensors

        var_eps = ops.add(var, eps)
        inv_std = ops.div(ops.scalar_from_float(1.0), ops.sqrt(var_eps))
        x_centered = ops.sub(x, mean)
        x_hat = ops.mul(x_centered, inv_std)

        grad_x = grad_weight = grad_bias = None
        if bias is not None:
            grad_bias = ops.sum(grad_output, dim=ctx.red_dims, keepdim=False)

        if weight is not None:
            grad_y_wrt_x_hat = ops.mul(grad_output, weight.reshape(1, -1, *([1] * (x.dim() - 2))))
            grad_weight = ops.sum(ops.mul(grad_output, x_hat), dim=ctx.red_dims, keepdim=False)
        else:
            grad_y_wrt_x_hat = grad_output

        if ctx.training:
            N = 1
            for dim in ctx.red_dims:
                N *= x.shape[dim]
            n_elems = ops.scalar_from_float(N)

            var_eps = ops.add(var, eps)
            inv_std_cubed = ops.mul(inv_std, ops.mul(inv_std, inv_std))
            neg_half = ops.scalar_from_float(-0.5)

            grad_var = ops.sum(ops.mul(
                ops.mul(grad_y_wrt_x_hat, x_centered),
                ops.mul(neg_half, inv_std_cubed)),
                dim=ctx.red_dims, keepdim=True)

            neg_inv_std = ops.neg(inv_std)
            grad_mean_term1 = ops.sum(ops.mul(grad_y_wrt_x_hat, neg_inv_std),
                                      dim=ctx.red_dims, keepdim=True)

            neg_two = ops.scalar_from_float(-2.0)
            neg_two_over_N = ops.div(neg_two, n_elems)
            sum_x_centered = ops.sum(x_centered, dim=ctx.red_dims, keepdim=True)
            grad_mean_term2 = ops.mul(ops.mul(grad_var, neg_two_over_N), sum_x_centered)

            grad_mean = ops.add(grad_mean_term1, grad_mean_term2)

            two = ops.scalar_from_float(2.0)
            two_over_N = ops.div(two, n_elems)
            one_over_N = ops.div(ops.scalar_from_float(1.0), n_elems)

            grad_x_term1 = ops.mul(grad_y_wrt_x_hat, inv_std)
            grad_x_term2 = ops.mul(ops.mul(grad_var, two_over_N), x_centered)
            grad_x_term3 = ops.mul(grad_mean, one_over_N)
            grad_x = ops.add(grad_x_term1, ops.add(grad_x_term2, grad_x_term3))

        else:
            grad_x = ops.mul(grad_y_wrt_x_hat, inv_std)

        return grad_x, None, None, grad_weight, grad_bias, None, None, None

@register_base_op("layer_norm")
def dt_layer_norm(ops, x, eps, normalized_shape, weight=None, bias=None):
    reduce_dims = tuple(range(x.dim() - len(normalized_shape), x.dim()))
    mean = ops.mean(x, dim=reduce_dims, keepdim=True)
    var = ops.var(x, ops.scalar_from_float(0.0), dim=reduce_dims, keepdim=True)

    var_eps = ops.add(var, eps)
    inv_std = ops.div(ops.scalar_from_float(1.0), ops.sqrt(var_eps))
    x_hat = ops.mul(ops.sub(x, mean), inv_std)

    if weight is not None:
        shape = [1] * (x.dim() - len(normalized_shape)) + list(normalized_shape)
        x_hat = ops.mul(x_hat, weight.view(*shape))
    if bias is not None:
        shape = [1] * (x.dim() - len(normalized_shape)) + list(normalized_shape)
        x_hat = ops.add(x_hat, bias.view(*shape))

    return x_hat

class DTLayerNormFunction(DTFunction):

    @staticmethod
    def forward(ops, x, eps, normalized_shape, weight=None, bias=None):
        return ops.layer_norm(x, eps, normalized_shape, weight, bias)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, eps, normalized_shape, weight, bias = inputs
        ctx.red_dims = tuple(range(x.dim() - len(normalized_shape), x.dim()))
        mean = ops.mean(x, dim=ctx.red_dims, keepdim=True)
        var = ops.var(x, ops.scalar_from_float(0.0), dim=ctx.red_dims, keepdim=True)
        ctx.save_for_backward(x, weight, bias, mean, var, eps)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, weight, bias, mean, var, eps = ctx.saved_tensors
        red_dims = ctx.red_dims
        var_eps = ops.add(var, eps)
        inv_std = ops.div(ops.scalar_from_float(1.0), ops.sqrt(var_eps))
        x_centered = ops.sub(x, mean)
        x_hat = ops.mul(x_centered, inv_std)

        param_red_dims = tuple(i for i in range(x.dim()) if i not in red_dims)
        grad_bias = None
        if bias is not None:
            grad_bias = ops.sum(grad_output, dim=param_red_dims, keepdim=False)

        grad_weight = None
        if weight is not None:
            w_view = [1] * x.dim()
            for d in red_dims:
                w_view[d] = x.size(d)
            weight_b = weight.view(*w_view)
            grad_y_wrt_x_hat = ops.mul(grad_output, weight_b)
            grad_weight = ops.sum(
                ops.mul(grad_output, x_hat),
                dim=param_red_dims, keepdim=False
            )
        else:
            grad_y_wrt_x_hat = grad_output

        N = 1
        for d in red_dims:
            N *= x.shape[d]
        n_elems = ops.scalar_from_float(N)

        sum_grad = ops.sum(grad_y_wrt_x_hat, dim=red_dims, keepdim=True)
        sum_grad_xhat = ops.sum(
            ops.mul(grad_y_wrt_x_hat, x_hat),
            dim=red_dims, keepdim=True
        )

        Ng = ops.mul(n_elems, grad_y_wrt_x_hat)
        term_inner = ops.sub(
            ops.sub(Ng, sum_grad),
            ops.mul(x_hat, sum_grad_xhat)
        )
        inv_N = ops.div(ops.scalar_from_float(1.0), n_elems)
        grad_x = ops.mul(ops.mul(inv_std, inv_N), term_inner)

        return grad_x, None, None, grad_weight, grad_bias
