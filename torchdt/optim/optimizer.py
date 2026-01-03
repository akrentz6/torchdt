import torch
from torchdt import DType

class DTOptimizer(torch.optim.Optimizer):

    def __init__(self, dtype, device, params, defaults):
        if not issubclass(dtype, DType):
            raise ValueError("dtype must be a subclass of DType.")
        self.dtype = dtype
        self.device = device
        super().__init__(params, defaults)

    def step(self, closure=None):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def zero_grad(self, set_to_none: bool = True): # support set_to_none later
        for group in self.param_groups:
            for param in group['params']:
                param._grad_accum_hook.value.fill_(0.0)

    def convert_params(self, *param_names):
        for group in self.param_groups:
            for name in param_names:
                if name in group and not isinstance(group[name], self.dtype):
                    group[name] = self.dtype(group[name], device=self.device)

    def validate_param(self, param_name, condition):
        for group in self.param_groups:
            if param_name not in group:
                continue
            if not condition(group[param_name]):
                str_val = group[param_name].item() if group[param_name].numel() == 1 else group[param_name]
                raise ValueError(f"Invalid {param_name}: {str_val}")