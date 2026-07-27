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

        self.validate_param("lr", lambda lr: lr >= self._zero)
        self.validate_param("eps", lambda eps: eps > self._zero)
        self.validate_param("beta1", lambda beta1: self._zero <= beta1 < self._one)
        self.validate_param("beta2", lambda beta2: self._zero <= beta2 < self._one)
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= self._zero)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            step_cache = {}
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

                if weight_decay != self._zero:
                    grad = grad + p * weight_decay

                if len(state) == 0:
                    # First time we see this parameter
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"] + 1
                step_value = self.encoded_step(step, step_cache)

                exp_avg = (exp_avg * beta1) + (grad * (self._one - beta1))
                exp_avg_sq = (exp_avg_sq * beta2) + (grad * grad * (self._one - beta2))

                bias_corr1 = self._one - beta1 ** step_value
                bias_corr2 = self._one - beta2 ** step_value

                if amsgrad:
                    max_exp_avg_sq = torch.maximum(state["max_exp_avg_sq"], exp_avg_sq)
                    state["max_exp_avg_sq"] = max_exp_avg_sq
                    v_denom = max_exp_avg_sq
                else:
                    v_denom = exp_avg_sq

                step_size = lr * torch.sqrt(bias_corr2) / bias_corr1
                p.data.copy_(p - step_size * exp_avg / (torch.sqrt(v_denom) + eps))

                state["step"] = step
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

        self.validate_param("lr", lambda lr: lr >= self._zero)
        self.validate_param("eps", lambda eps: eps > self._zero)
        self.validate_param("beta1", lambda beta1: self._zero <= beta1 < self._one)
        self.validate_param("beta2", lambda beta2: self._zero <= beta2 < self._one)
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= self._zero)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            step_cache = {}
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]

            active = [p for p in group["params"] if p.grad is not None]
            step_groups = {}
            for p in active:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)
                step = state["step"] + 1
                step_groups.setdefault(step, []).append(p)

            for step, params in step_groups.items():
                step_value = self.encoded_step(step, step_cache)
                bias_corr1 = self._one - (beta1 ** step_value)
                bias_corr2 = self._one - (beta2 ** step_value)
                states = [self.state[p] for p in params]
                exp_avgs = [state["exp_avg"]._int for state in states]
                exp_avg_sqs = [state["exp_avg_sq"]._int for state in states]
                max_exp_avg_sqs = [state["max_exp_avg_sq"]._int if amsgrad else None for state in states]
                self.dtype.ops.triton_adam_step_group(
                    [p._int for p in params],
                    [p.grad._int for p in params],
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    lr._int,
                    beta1._int,
                    beta2._int,
                    eps._int,
                    weight_decay._int,
                    bias_corr1._int,
                    bias_corr2._int,
                    amsgrad,
                    maximize,
                )
                for state in states:
                    state["step"] = step

        return loss
