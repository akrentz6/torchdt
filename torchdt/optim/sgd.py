import torch
from torchdt.optim import DTOptimizer

class SGD(DTOptimizer):

    def __init__(
            self,
            dtype,
            device,
            params,
            lr=0.001,
            momentum=0.0,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            *,
            maximize=False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        super().__init__(dtype, device, params, defaults)
        self.convert_params("lr", "momentum", "dampening", "weight_decay")

        self.validate_param("lr", lambda lr: lr >= torch.tensor(0.0, device=device))
        self.validate_param("momentum", lambda momentum: momentum >= torch.tensor(0.0, device=device))
        self.validate_param("dampening", lambda dampening: dampening >= torch.tensor(0.0, device=device))
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= torch.tensor(0.0, device=device))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if maximize:
                    grad = -grad

                if weight_decay != torch.tensor(0.0, device=self.device):
                    grad = grad + p * weight_decay

                if momentum != torch.tensor(0.0, device=self.device):
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = state["momentum_buffer"] = grad.clone()
                    else:
                        buf = (buf * momentum) + (grad * (torch.tensor(1.0, device=self.device) - dampening))

                    if nesterov:
                        grad = grad + buf * momentum
                    else:
                        grad = buf

                p.data.copy_(p - grad * lr)

        return loss

class TritonSGD(DTOptimizer):

    def __init__(
            self,
            dtype,
            device,
            params,
            lr=0.001,
            momentum=0.0,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            *,
            maximize=False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        super().__init__(dtype, device, params, defaults)
        self.convert_params("lr", "momentum", "dampening", "weight_decay")

        self.validate_param("lr", lambda lr: lr >= torch.tensor(0.0, device=device))
        self.validate_param("momentum", lambda momentum: momentum >= torch.tensor(0.0, device=device))
        self.validate_param("dampening", lambda dampening: dampening >= torch.tensor(0.0, device=device))
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= torch.tensor(0.0, device=device))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                buf = state.get("momentum_buffer", None)

                new_buf = self.dtype.ops.triton_sgd_step(
                    p._int, p.grad._int, buf,
                    lr._int, momentum._int,
                    dampening._int, weight_decay._int,
                    nesterov, maximize
                )
                state["momentum_buffer"] = new_buf

        return loss