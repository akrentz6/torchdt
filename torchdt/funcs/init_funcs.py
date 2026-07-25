import torch
from torchdt import DType


def _fill_encoded(tensor, value):
    ops = tensor.__class__.ops.direct_for_device(tensor.device)
    tensor._int.fill_(ops.encoded_scalar(value))
    return tensor

@DType.register_func(torch.nn.init.uniform_, torch.Tensor.uniform_)
def dt_uniform_(tensor, a=0.0, b=1.0, *, generator=None):
    result = torch.nn.init.uniform_(torch.empty(tensor.size(), device=tensor.device), a=a, b=b, generator=generator)
    tensor._int.copy_(tensor.__class__.ops.direct_for_device(tensor.device).from_float(result))
    return tensor

@DType.register_func(torch.nn.init.normal_, torch.Tensor.normal_)
def dt_normal_(tensor, mean=0.0, std=1.0, *, generator=None):
    result = torch.nn.init.normal_(torch.empty(tensor.size(), device=tensor.device), mean=mean, std=std, generator=generator)
    tensor._int.copy_(tensor.__class__.ops.direct_for_device(tensor.device).from_float(result))
    return tensor

@DType.register_func(torch.nn.init.constant_, torch.Tensor.fill_)
def dt_constant_(tensor, val):
    return _fill_encoded(tensor, val)

@DType.register_func(torch.nn.init.ones_)
def dt_ones_(tensor):
    return _fill_encoded(tensor, 1.0)

@DType.register_func(torch.nn.init.zeros_, torch.Tensor.zero_)
def dt_zeros_(tensor):
    return _fill_encoded(tensor, 0.0)

@DType.register_func(torch.nn.init.xavier_uniform_)
def dt_xavier_uniform_(tensor, gain=1.0, generator=None):
    result = torch.nn.init.xavier_uniform_(torch.empty(tensor.size(), device=tensor.device), gain=gain, generator=generator)
    tensor._int.copy_(tensor.__class__.ops.direct_for_device(tensor.device).from_float(result))
    return tensor

@DType.register_func(torch.nn.init.xavier_normal_)
def dt_xavier_normal_(tensor, gain=1.0, generator=None):
    result = torch.nn.init.xavier_normal_(torch.empty(tensor.size(), device=tensor.device), gain=gain, generator=generator)
    tensor._int.copy_(tensor.__class__.ops.direct_for_device(tensor.device).from_float(result))
    return tensor

@DType.register_func(torch.nn.init.kaiming_uniform_)
def dt_kaiming_uniform_(tensor, a=0.0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
    result = torch.nn.init.kaiming_uniform_(torch.empty(tensor.size(), device=tensor.device), a=a, mode=mode, nonlinearity=nonlinearity, generator=generator)
    tensor._int.copy_(tensor.__class__.ops.direct_for_device(tensor.device).from_float(result))
    return tensor

@DType.register_func(torch.nn.init.kaiming_normal_)
def dt_kaiming_normal_(tensor, a=0.0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
    result = torch.nn.init.kaiming_normal_(torch.empty(tensor.size(), device=tensor.device), a=a, mode=mode, nonlinearity=nonlinearity, generator=generator)
    tensor._int.copy_(tensor.__class__.ops.direct_for_device(tensor.device).from_float(result))
    return tensor
