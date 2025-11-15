import torch
from torchdt.optim import DTOptimizer

class Madam(DTOptimizer):

    def __init__(
            self,
            dtype,
            params,
            lr=0.01,
            beta=0.999,
            eps=1e-8,
            p_scale=3.0,
            g_bound=10.0,
            use_pow=False,
            *,
            maximize=False,
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            p_scale=p_scale,
            g_bound=g_bound,
            use_pow=use_pow,
            maximize=maximize,
        )
        super().__init__(dtype, params, defaults)
        self.convert_params("lr", "beta", "eps", "p_scale", "g_bound")

        self.validate_param("lr", lambda lr: lr >= 0.0)
        self.validate_param("eps", lambda eps: eps > 0.0)
        self.validate_param("beta",lambda beta: 0.0 < beta < 1.0)
        self.validate_param("p_scale", lambda p_scale: p_scale > 0.0)
        self.validate_param("g_bound", lambda g_bound: g_bound > 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            p_scale = group["p_scale"]
            g_bound = group["g_bound"]
            use_pow = group["use_pow"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    # First time we see this parameter
                    rms = torch.sqrt(torch.mean(p * p))
                    state["max"] = p_scale * rms
                    state["step"] = self.dtype(0.0)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

                max = state["max"]
                step = state["step"] + 1
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg_sq = (beta * exp_avg_sq) + (grad * grad * (1.0 - beta))
                corr_exp_avg_sq = exp_avg_sq / (1.0 - beta ** step) + eps

                g_normed = grad / torch.sqrt(corr_exp_avg_sq)
                g_clipped = torch.clamp(g_normed, -g_bound, g_bound)
                delta = lr * g_clipped * torch.sign(p)

                if not maximize:
                    delta = -delta

                if use_pow:
                    mul_update = p * torch.exp(delta)
                else:
                    mul_update = p * (1.0 + delta)

                p.data.copy_(torch.clamp(mul_update, -max, max))

                state["step"] = step
                state["exp_avg_sq"] = exp_avg_sq

        return loss