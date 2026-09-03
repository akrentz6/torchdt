import torch
from torchdt import DType

class DTOptimizer(torch.optim.Optimizer):

    def __init__(self, dtype, params, defaults):
        if not issubclass(dtype, DType):
            raise ValueError("dtype must be a subclass of DType.")
        self.dtype = dtype
        super().__init__(params, defaults)

        devices = {
            param.device
            for group in self.param_groups
            for param in group["params"]
        }
        if len(devices) != 1:
            device_names = ", ".join(sorted(map(str, devices))) or "none"
            raise ValueError(
                "DTOptimizer parameters must use exactly one device, got "
                f"{device_names}."
            )
        self.device = next(iter(devices))

        # These scalars are shared by validation and every parameter update.
        # Constructing native scalar tensors in the inner loops would force the
        # dtype interception path to encode them again for every parameter.
        self._zero = dtype(0, device=self.device)
        self._one = dtype(1, device=self.device)

    def step(self, closure=None):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for param in group['params']:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.zero_()

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

    def encoded_step(self, step, cache):
        """Return one shared encoded step scalar for this optimizer step."""
        value = cache.get(step)
        if value is None:
            value = cache[step] = self.dtype(step, device=self.device)
        return value
