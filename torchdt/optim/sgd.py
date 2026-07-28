import torch
from torchdt.optim import DTOptimizer

class SGD(DTOptimizer):

    def __init__(
            self,
            dtype,
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
        super().__init__(dtype, params, defaults)
        self.convert_params("lr", "momentum", "dampening", "weight_decay")

        self.validate_param("lr", lambda lr: lr >= self._zero)
        self.validate_param("momentum", lambda momentum: momentum >= self._zero)
        self.validate_param("dampening", lambda dampening: dampening >= self._zero)
        self.validate_param("weight_decay", lambda weight_decay: weight_decay >= self._zero)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        ops = self.dtype.ops.direct_for_device(self.device)
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]

            active = [p for p in group["params"] if p.grad is not None]
            if not active:
                continue
            states = [self.state[p] for p in active]
            buffers = [state.get("momentum_buffer", None) for state in states]
            use_momentum = bool(momentum != self._zero)
            new_buffers = ops.sgd_step(
                [p._int for p in active],
                [p.grad._int for p in active],
                [buffer._int if buffer is not None else None for buffer in buffers],
                lr._int,
                momentum._int,
                dampening._int,
                weight_decay._int,
                nesterov,
                maximize,
                use_momentum,
            )
            for state, new_buffer in zip(states, new_buffers):
                if new_buffer is not None:
                    state["momentum_buffer"] = self.dtype(new_buffer, internal=True)

        return loss
