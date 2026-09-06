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
    DTErfFunction,
    DTGeluFunction,
    DTSiluFunction,
    DTSoftplusFunction,
    DTMishFunction,
)

@DType.register_func(torch.nn.functional.relu, torch.Tensor.relu,
                     cast=("input",))
def dt_relu(input, inplace=False):
    if inplace:
        raise NotImplementedError("in-place ReLU is not supported for DType tensors")
    result = DTReLUFunction.apply(input)
    return result

@DType.register_func(torch.nn.functional.leaky_relu,
                     cast=("input", "negative_slope"))
def dt_leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        raise NotImplementedError("in-place leaky ReLU is not supported for DType tensors")
    result = DTLeakyReLUFunction.apply(input, negative_slope)
    return result

@DType.register_func(torch.nn.functional.threshold,
                     cast=("input", "threshold", "value"))
def dt_threshold(input, threshold, value, inplace=False):
    if inplace:
        raise NotImplementedError("in-place threshold is not supported for DType tensors")
    result = DTThresholdFunction.apply(input, threshold, value)
    return result

@DType.register_func(torch.tanh, torch.nn.functional.tanh, torch.Tensor.tanh,
                     cast=("input",))
def dt_tanh(input, *, out=None):
    result = DTTanhFunction.apply(input)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.sigmoid, torch.nn.functional.sigmoid, torch.Tensor.sigmoid,
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

@DType.register_func(torch.softmax, torch.nn.functional.softmax, torch.Tensor.softmax,
                     cast=("input",))
def dt_softmax(input, dim=None, _stacklevel=3, dtype=None, *, out=None):
    if dtype is not None and dtype is not input.__class__:
        raise NotImplementedError("softmax dtype conversion is not supported for DType tensors")
    result = DTSoftmaxFunction.apply(input, dim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.log_softmax, torch.nn.functional.log_softmax, torch.Tensor.log_softmax,
                     cast=("input",))
def dt_log_softmax(input, dim=None, _stacklevel=3, dtype=None, *, out=None):
    if dtype is not None and dtype is not input.__class__:
        raise NotImplementedError("log_softmax dtype conversion is not supported for DType tensors")
    result = DTLogSoftmaxFunction.apply(input, dim)

    if out is not None:
        return out.copy_(result)
    return result

@DType.register_func(torch.nn.functional.hardtanh,
                     cast=("input", "min_val", "max_val"))
def dt_hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False, *, out=None):
    if inplace:
        raise NotImplementedError("in-place hardtanh is not supported for DType tensors")
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


@DType.register_func(torch.erf, torch.Tensor.erf, cast=("input",))
def dt_erf(input, *, out=None):
    result = DTErfFunction.apply(input)
    return out.copy_(result) if out is not None else result


@DType.register_func(torch.nn.functional.gelu, cast=("input",))
def dt_gelu(input, approximate="none"):
    return DTGeluFunction.apply(input, approximate)


@DType.register_func(torch.nn.functional.silu, cast=("input",))
def dt_silu(input, inplace=False):
    if inplace:
        raise NotImplementedError("in-place SiLU is not supported for DType tensors")
    return DTSiluFunction.apply(input)


@DType.register_func(torch.nn.functional.softplus, cast=("input",))
def dt_softplus(input, beta=1.0, threshold=20.0):
    if beta <= 0:
        raise ValueError("softplus beta must be positive")
    return DTSoftplusFunction.apply(input, beta, threshold)


@DType.register_func(torch.nn.functional.mish, cast=("input",))
def dt_mish(input, inplace=False):
    if inplace:
        raise NotImplementedError("in-place Mish is not supported for DType tensors")
    return DTMishFunction.apply(input)
