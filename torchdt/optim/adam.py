import torch
from torchdt.optim import DTOptimizer

class Adam(DTOptimizer):

    def __init__(
            self,
            dtype,
            device,
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            *,
            maximize=False,
    ):
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super().__init__(dtype, device, params, defaults)
        self.convert_params("lr", "beta1", "beta2", "eps", "weight_decay")

        self.validate_param("lr", lambda lr: lr >= torch.tensor(0.0, device=device))
        self.validate_param("eps", lambda eps: eps > torch.tensor(0.0, device=device))
        self.validate_param("beta1",lambda beta1: torch.tensor(0.0, device=device) <= beta1 < torch.tensor(1.0, device=device))
        self.validate_param("beta2",lambda beta2: torch.tensor(0.0, device=device) <= beta2 < torch.tensor(1.0, device=device))
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= torch.tensor(0.0, device=device))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
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

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = self.dtype(0, device=self.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"] + torch.tensor(1.0, device=self.device)

                exp_avg = (exp_avg * beta1) + (grad * (torch.tensor(1.0, device=self.device) - beta1))
                exp_avg_sq = (exp_avg_sq * beta2) + (grad * grad * (torch.tensor(1.0, device=self.device) - beta2))

                bias_corr1 = torch.tensor(1.0, device=self.device) - beta1 ** step
                bias_corr2 = torch.tensor(1.0, device=self.device) - beta2 ** step

                if amsgrad:
                    max_exp_avg_sq = torch.maximum(state["max_exp_avg_sq"], exp_avg_sq)
                    state["max_exp_avg_sq"] = max_exp_avg_sq
                    v_denom = max_exp_avg_sq
                else:
                    v_denom = exp_avg_sq

                step_size = lr * torch.sqrt(bias_corr2) / bias_corr1
                p.data.copy_(p - step_size * exp_avg / (torch.sqrt(v_denom) + eps))

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss


class TritonAdam(DTOptimizer):

    def __init__(
            self,
            dtype,
            device,
            params,
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            *,
            maximize=False,
    ):
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super().__init__(dtype, device, params, defaults)
        self.convert_params("lr", "beta1", "beta2", "eps", "weight_decay")

        self.validate_param("lr", lambda lr: lr >= torch.tensor(0.0, device=device))
        self.validate_param("eps", lambda eps: eps > torch.tensor(0.0, device=device))
        self.validate_param("beta1",lambda beta1: torch.tensor(0.0, device=device) <= beta1 < torch.tensor(1.0, device=device))
        self.validate_param("beta2",lambda beta2: torch.tensor(0.0, device=device) <= beta2 < torch.tensor(1.0, device=device))
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= torch.tensor(0.0, device=device))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = self.dtype(0, device=self.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)._int
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)._int
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)._int

                step = state["step"] + torch.tensor(1.0, device=self.device)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                max_exp_avg_sq = state.get("max_exp_avg_sq", None)

                bias_corr1 = torch.tensor(1.0, device=self.device) - (beta1 ** step)
                bias_corr2 = torch.tensor(1.0, device=self.device) - (beta2 ** step)

                exp_avg, exp_avg_sq, max_exp_avg_sq = self.dtype.ops.triton_adam_step(
                    p._int, grad._int,
                    exp_avg, exp_avg_sq, max_exp_avg_sq,
                    lr._int, beta1._int, beta2._int,
                    eps._int, weight_decay._int,
                    bias_corr1._int, bias_corr2._int,
                    amsgrad, maximize,
                )

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                if amsgrad:
                    state["max_exp_avg_sq"] = max_exp_avg_sq

        return loss