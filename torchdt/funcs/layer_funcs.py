import torch
from torchdt import DType
from torchdt.ops.layer_ops import (
    DTLinearFunction,
    DTDropoutFunction,
    DTConv2dFunction,
    DTAvgPool2dFunction,
    DTAdaptiveAvgPool2dFunction,
    DTMaxPool2dFunction,
    DTBatchNormFunction,
    DTLayerNormFunction,
)

@DType.register_func(torch.nn.functional.linear,
                     cast=("input", "weight", "bias"))
def dt_linear(input, weight, bias=None):
    return DTLinearFunction.apply(input, weight, bias)

@DType.register_func(torch.nn.functional.dropout,
                     cast=("input",))
def dt_dropout(input, p=0.5, training=True, inplace=False):
    if not training or p == 0.0:
        return input
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability p must be in the range [0, 1], but got {p}.")
    return DTDropoutFunction.apply(input, p)

@DType.register_func(torch.nn.functional.conv2d,
                     cast=("input", "weight", "bias"))
def dt_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return DTConv2dFunction.apply(input, weight, bias, stride, padding, dilation, groups)

@DType.register_func(torch.nn.functional.avg_pool2d,
                     cast=("input", "divisor_override"))
def dt_avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return DTAvgPool2dFunction.apply(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

@DType.register_func(torch.nn.functional.adaptive_avg_pool2d,
                     cast=("input",))
def dt_adaptive_avg_pool2d(input, output_size):
    return DTAdaptiveAvgPool2dFunction.apply(input, output_size)

@DType.register_func(torch.nn.functional.max_pool2d,
                     cast=("input",))
def dt_max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return DTMaxPool2dFunction.apply(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

@DType.register_func(torch.nn.functional.batch_norm,
                     cast=("input", "running_mean", "running_var", "weight", "bias", "momentum", "eps"))
def dt_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    return DTBatchNormFunction.apply(input, running_mean, running_var, weight, bias, training, momentum, eps)

@DType.register_func(torch.nn.functional.layer_norm,
                     cast=("input", "weight", "bias", "eps"))
def dt_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return DTLayerNormFunction.apply(input, normalized_shape, weight, bias, eps)