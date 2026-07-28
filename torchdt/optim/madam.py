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

        self.validate_param("lr", lambda lr: lr >= self._zero)
        self.validate_param("eps", lambda eps: eps > self._zero)
        self.validate_param("beta", lambda beta: self._zero < beta < self._one)
        self.validate_param("p_scale", lambda p_scale: p_scale > self._zero)
        self.validate_param("g_bound", lambda g_bound: g_bound > self._zero)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        ops = self.dtype.ops.direct_for_device(self.device)
        for group in self.param_groups:
            step_cache = {}
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            p_scale = group["p_scale"]
            g_bound = group["g_bound"]
            use_pow = group["use_pow"]
            maximize = group["maximize"]

            active = [p for p in group["params"] if p.grad is not None]
            step_groups = {}
            for p in active:
                state = self.state[p]
                if len(state) == 0:
                    rms = torch.sqrt(torch.mean(p * p))
                    state["max"] = p_scale * rms
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype, device=self.device)
                step = state["step"] + 1
                step_groups.setdefault(step, []).append(p)

            for step, params in step_groups.items():
                step_value = self.encoded_step(step, step_cache)
                bias_corr = self._one - beta ** step_value
                states = [self.state[p] for p in params]
                ops.madam_step(
                    [p._int for p in params],
                    [p.grad._int for p in params],
                    [state["exp_avg_sq"]._int for state in states],
                    [state["max"]._int for state in states],
                    lr._int,
                    beta._int,
                    eps._int,
                    g_bound._int,
                    bias_corr._int,
                    use_pow,
                    maximize,
                )
                for state in states:
                    state["step"] = step

        return loss
