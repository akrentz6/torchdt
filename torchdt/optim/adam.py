import torch
from torchdt.optim import DTOptimizer

class Adam(DTOptimizer):

    def __init__(
            self,
            dtype,
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
        super().__init__(dtype, params, defaults)
        self.convert_params("lr", "beta1", "beta2", "eps", "weight_decay")

        self.validate_param("lr", lambda lr: lr >= 0.0)
        self.validate_param("eps", lambda eps: eps > 0.0)
        self.validate_param("beta1",lambda beta1: 0.0 <= beta1 < 1.0)
        self.validate_param("beta2",lambda beta2: 0.0 <= beta2 < 1.0)
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= 0.0)

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

                if weight_decay != 0.0:
                    grad = grad + p * weight_decay

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = self.dtype(0)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"] + 1

                exp_avg = (exp_avg * beta1) + (grad * (1.0 - beta1))
                exp_avg_sq = (exp_avg_sq * beta2) + (grad * grad * (1.0 - beta2))

                beta1 = beta1 ** step
                beta2 = beta2 ** step

                exp_avg_hat = exp_avg / (1.0 - beta1)
                if amsgrad:
                    max_exp_avg_sq = torch.maximum(state["max_exp_avg_sq"], exp_avg_sq)
                    state["max_exp_avg_sq"] = max_exp_avg_sq
                    denom_sq = (max_exp_avg_sq / (1.0 - beta2))
                else:
                    denom_sq = (exp_avg_sq / (1.0 - beta2))

                p.data = p - lr * exp_avg_hat / (torch.sqrt(denom_sq) + eps)

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                state["step"] = step

        return loss