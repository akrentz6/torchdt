import torch
from torchdt import DType
from torchdt.ops.activation_ops import (
    DTReLUFunction,
    DTLeakyReLUFunction,
    DTThresholdFunction,
    DTTanhFunction,
    DTSigmoidFunction,
    DTLogSigmoidFunction,
    DTSoftminFunction,
    DTSoftmaxFunction,
    DTLogSoftmaxFunction,
    DTHardtanhFunction,
    DTGluFunction,
)

@DType.register_func(torch.nn.functional.relu, torch.Tensor.relu,
                     cast=("input",))
def dt_relu(input, inplace=False):
    result = DTReLUFunction.apply(input)
    return result

@DType.register_func(torch.nn.functional.leaky_relu,
                     cast=("input", "negative_slope"))
def dt_leaky_relu(input, negative_slope=0.01, inplace=False):
    result = DTLeakyReLUFunction.apply(input, negative_slope)
    return result

@DType.register_func(torch.nn.functional.threshold,
                     cast=("input", "threshold", "value"))
def dt_threshold(input, threshold, value, inplace=False):
    result = DTThresholdFunction.apply(input, threshold, value)
    return result

@DType.register_func(torch.nn.functional.tanh, torch.Tensor.tanh,
                     cast=("input",))
def dt_tanh(input, *, out=None):
    result = DTTanhFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.nn.functional.sigmoid, torch.Tensor.sigmoid,
                     cast=("input",))
def dt_sigmoid(input, *, out=None):
    result = DTSigmoidFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.nn.functional.logsigmoid,
                     cast=("input",))
def dt_logsigmoid(input, *, out=None):
    result = DTLogSigmoidFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.nn.functional.softmin,
                     cast=("input",))
def dt_softmin(input, dim=None, *, out=None):
    result = DTSoftminFunction.apply(input, dim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.nn.functional.softmax, torch.Tensor.softmax,
                     cast=("input",))
def dt_softmax(input, dim=None, *, out=None):
    result = DTSoftmaxFunction.apply(input, dim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.nn.functional.log_softmax, torch.Tensor.log_softmax,
                     cast=("input",))
def dt_log_softmax(input, dim=None, _stacklevel=3, dtype=None, *, out=None):
    result = DTLogSoftmaxFunction.apply(input, dim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.nn.functional.hardtanh,
                     cast=("input", "min_val", "max_val"))
def dt_hardtanh(input, min_val=-1.0, max_val=1.0, *, out=None):
    result = DTHardtanhFunction.apply(input, min_val, max_val)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.nn.functional.glu,
                     cast=("input",))
def dt_glu(input, dim=-1, *, out=None):
    result = DTGluFunction.apply(input, dim)

    if out is not None:
        return out.copy_(result)
    return result