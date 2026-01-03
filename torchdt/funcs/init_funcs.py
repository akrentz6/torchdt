import torch
from torchdt import DType

@DType.register_func(torch.nn.init.uniform_, torch.Tensor.uniform_)
def dt_uniform_(tensor, a=0.0, b=1.0, *, generator=None):
    result = torch.nn.init.uniform_(torch.empty(tensor.size(), device=tensor.device), a=a, b=b, generator=generator)
    return tensor.data.copy_(tensor.__class__(result))

@DType.register_func(torch.nn.init.normal_, torch.Tensor.normal_)
def dt_normal_(tensor, mean=0.0, std=1.0, *, generator=None):
    result = torch.nn.init.normal_(torch.empty(tensor.size(), device=tensor.device), mean=mean, std=std, generator=generator)
    return tensor.data.copy_(tensor.__class__(result))

@DType.register_func(torch.nn.init.constant_, torch.Tensor.fill_)
def dt_constant_(tensor, val):
    # here we can precompute the result since it's constant
    val_dt = tensor.__class__(val, device=tensor.device)
    return tensor.data.copy_(val_dt)

@DType.register_func(torch.nn.init.ones_)
def dt_ones_(tensor):
    # precompute the one value in the target dtype
    one_dt = tensor.__class__(1.0, device=tensor.device)
    return tensor.data.copy_(one_dt)

@DType.register_func(torch.nn.init.zeros_, torch.Tensor.zero_)
def dt_zeros_(tensor):
    # precompute the zero value in the target dtype
    zero_dt = tensor.__class__(0.0, device=tensor.device)
    return tensor.data.copy_(zero_dt)

@DType.register_func(torch.nn.init.xavier_uniform_)
def dt_xavier_uniform_(tensor, gain=1.0, generator=None):
    result = torch.nn.init.xavier_uniform_(torch.empty(tensor.size(), device=tensor.device), gain=gain, generator=generator)
    return tensor.data.copy_(tensor.__class__(result))

@DType.register_func(torch.nn.init.xavier_normal_)
def dt_xavier_normal_(tensor, gain=1.0, generator=None):
    result = torch.nn.init.xavier_normal_(torch.empty(tensor.size(), device=tensor.device), gain=gain, generator=generator)
    return tensor.data.copy_(tensor.__class__(result))

@DType.register_func(torch.nn.init.kaiming_uniform_)
def dt_kaiming_uniform_(tensor, a=0.0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
    result = torch.nn.init.kaiming_uniform_(torch.empty(tensor.size(), device=tensor.device), a=a, mode=mode, nonlinearity=nonlinearity, generator=generator)
    return tensor.data.copy_(tensor.__class__(result))

@DType.register_func(torch.nn.init.kaiming_normal_)
def dt_kaiming_normal_(tensor, a=0.0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
    result = torch.nn.init.kaiming_normal_(torch.empty(tensor.size(), device=tensor.device), a=a, mode=mode, nonlinearity=nonlinearity, generator=generator)
    return tensor.data.copy_(tensor.__class__(result))