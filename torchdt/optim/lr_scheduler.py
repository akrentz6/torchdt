import torch
import torch.optim.lr_scheduler as lr_sched
from torchdt.optim import DTOptimizer
from collections import Counter
from bisect import bisect_right
import math
import warnings

__all__ = [
    "StepLR",
    "MultiStepLR",
    "LinearLR",
    "CosineAnnealingLR",
    "SequentialLR",
    "ReduceLROnPlateau",
]

def _param_groups_val_list(optimizer, key):
    return [group[key] for group in optimizer.param_groups]

def _update_param_group_val(param_group, key, val):
    if isinstance(param_group[key], torch.Tensor):
        param_group[key].copy_(val)
    else:
        param_group[key] = val

def _warn_get_lr_called_within_step(lr_scheduler):
    if not lr_scheduler._get_lr_called_within_step:
        warnings.warn(
            "To get the last learning rate computed by the scheduler, "
            "please use `get_last_lr()`.",
            UserWarning,
            stacklevel=2,
        )


class StepLR(lr_sched.LRScheduler):

    def __init__(
        self,
        optimizer,
        step_size,
        gamma = 0.1,
        last_epoch = -1,
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return _param_groups_val_list(self.optimizer, "lr")
        return [
            group["lr"] * self.optimizer.dtype(self.gamma, device=group["lr"].device)
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr * self.optimizer.dtype(
                self.gamma ** (self.last_epoch // self.step_size),
                device=base_lr.device
            )
            for base_lr in self.base_lrs
        ]

class MultiStepLR(lr_sched.LRScheduler):

    def __init__(
        self,
        optimizer,
        milestones,
        gamma = 0.1,
        last_epoch = -1,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        if self.last_epoch not in self.milestones:
            return _param_groups_val_list(self.optimizer, "lr")
        return [
            group["lr"] * self.optimizer.dtype(
                self.gamma ** self.milestones[self.last_epoch],
                device=group["lr"].device,
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        milestones = sorted(self.milestones.elements())
        return [
            base_lr * self.optimizer.dtype(
                self.gamma ** bisect_right(milestones, self.last_epoch),
                device=base_lr.device,
            )
            for base_lr in self.base_lrs
        ]

class LinearLR(lr_sched.LRScheduler):

    def __init__(
        self,
        optimizer,
        start_factor = 1.0 / 3,
        end_factor = 1.0,
        total_iters = 5,
        last_epoch = -1,
    ):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(
                "Starting multiplicative factor expected to be greater than 0 "
                "and less or equal to 1."
            )
        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [
                group["lr"] * self.optimizer.dtype(
                    self.start_factor,
                    device=group["lr"].device,
                )
                for group in self.optimizer.param_groups
            ]

        if getattr(self, "_is_initial", False) or self.last_epoch > self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        factor = 1.0 + (
            (self.end_factor - self.start_factor)
            / (
                self.total_iters * self.start_factor
                + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
            )
        )
        return [
            group["lr"] * self.optimizer.dtype(factor, device=group["lr"].device)
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        factor = (
            self.start_factor
            + (self.end_factor - self.start_factor)
            * min(self.total_iters, self.last_epoch)
            / self.total_iters
        )
        return [
            base_lr * self.optimizer.dtype(factor, device=base_lr.device)
            for base_lr in self.base_lrs
        ]

class CosineAnnealingLR(lr_sched.LRScheduler):

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min = 0,
        last_epoch = -1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return _param_groups_val_list(self.optimizer, "lr")

        if self._step_count == 1 and self.last_epoch > 0:
            factor = (1 + math.cos(self.last_epoch * math.pi / self.T_max)) / 2
            return [
                self.optimizer.dtype(self.eta_min, device=group["lr"].device)
                + (
                    base_lr
                    - self.optimizer.dtype(self.eta_min, device=group["lr"].device)
                ) * self.optimizer.dtype(factor, device=group["lr"].device)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            factor = (1 - math.cos(math.pi / self.T_max)) / 2
            return [
                group["lr"]
                + (
                    base_lr
                    - self.optimizer.dtype(self.eta_min, device=group["lr"].device)
                ) * self.optimizer.dtype(factor, device=group["lr"].device)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        factor = (
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
        )
        return [
            (
                group["lr"]
                - self.optimizer.dtype(self.eta_min, device=group["lr"].device)
            ) * self.optimizer.dtype(factor, device=group["lr"].device)
            + self.optimizer.dtype(self.eta_min, device=group["lr"].device)
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        factor = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        return [
            self.optimizer.dtype(self.eta_min, device=base_lr.device)
            + (
                base_lr
                - self.optimizer.dtype(self.eta_min, device=base_lr.device)
            ) * self.optimizer.dtype(factor, device=base_lr.device)
            for base_lr in self.base_lrs
        ]

class SequentialLR(lr_sched.LRScheduler):

    def __init__(
        self,
        optimizer,
        schedulers,
        milestones,
        last_epoch = -1,
    ):
        if len(schedulers) < 1:
            raise ValueError(
                f"{self.__class__.__name__} expects at least one scheduler, "
                "but got no scheduler."
            )

        for scheduler_idx, scheduler in enumerate(schedulers):
            if not hasattr(scheduler, "optimizer"):
                raise TypeError(
                    f"{self.__class__.__name__} at index {scheduler_idx} should "
                    "have `optimizer` as its attribute."
                )
            if isinstance(
                scheduler,
                (ReduceLROnPlateau, lr_sched.ReduceLROnPlateau),
            ):
                raise ValueError(
                    f"{self.__class__.__name__} does not support "
                    "`ReduceLROnPlateau` because it requires metrics when stepping."
                )
            if optimizer != scheduler.optimizer:
                raise ValueError(
                    f"{self.__class__.__name__} expects every scheduler to use "
                    "the same optimizer."
                )

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                "SequentialLR expects one fewer milestone than schedulers, but "
                f"got {len(milestones)} milestones and {len(schedulers)} schedulers."
            )

        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

        for group in self.optimizer.param_groups:
            initial_lr = group["initial_lr"]
            group["lr"] = (
                initial_lr.clone()
                if isinstance(initial_lr, torch.Tensor)
                else initial_lr
            )

        self.recursive_undo()
        self._schedulers[0]._initial_step()
        self._last_lr = self._schedulers[0].get_last_lr()

    def recursive_undo(self, scheduler=None):
        scheduler = self if scheduler is None else scheduler

        if hasattr(scheduler, "_schedulers"):
            for child_scheduler in scheduler._schedulers:
                self.recursive_undo(child_scheduler)
        elif hasattr(scheduler, "last_epoch"):
            scheduler.last_epoch -= 1

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]

        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler.step(0)
        else:
            scheduler.step()

        self._last_lr = scheduler.get_last_lr()

    def state_dict(self):
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        state_dict["_schedulers"] = [
            scheduler.state_dict() for scheduler in self._schedulers
        ]
        return state_dict

    def load_state_dict(self, state_dict):
        scheduler_states = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        state_dict["_schedulers"] = scheduler_states

        for scheduler, scheduler_state in zip(
            self._schedulers,
            scheduler_states,
        ):
            scheduler.load_state_dict(scheduler_state)

class ReduceLROnPlateau(lr_sched.LRScheduler):

    def __init__(
        self,
        optimizer,
        mode = "min",
        factor = 0.1,
        patience = 10,
        threshold = 1e-4,
        threshold_mode = "rel",
        cooldown = 0,
        min_lr = 0,
        eps = 1e-8,
    ):
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        if not isinstance(optimizer, DTOptimizer):
            raise TypeError(f"{type(optimizer).__name__} is not a DTOptimizer")
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}"
                )
            self.default_min_lr = None
            self.min_lrs = list(min_lr)

        else:
            self.default_min_lr = min_lr
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.cooldown = cooldown
        self.eps = eps
        self.last_epoch = 0
        self._last_lr = _param_groups_val_list(self.optimizer, "lr")
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        self.best = None
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        current = float(metrics)
        self.last_epoch += 1

        if self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(self.last_epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = _param_groups_val_list(self.optimizer, "lr")

    def _reduce_lr(self, epoch):
        if len(self.optimizer.param_groups) != len(self.min_lrs):
            if self.default_min_lr is None:
                raise RuntimeError(
                    "The number of param groups in the `optimizer` "
                    f"({len(self.optimizer.param_groups)}) differs "
                    f"from when `ReduceLROnPlateau` was initialized "
                    f"({len(self.min_lrs)}), usually due to a new "
                    "param group being added to the optimizer. Please "
                    "modify the `min_lrs` field to match the length "
                    "of the `optimizer` param groups."
                )
            else:
                self.min_lrs = [self.default_min_lr] * len(self.optimizer.param_groups)

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group["lr"]
            factor = self.optimizer.dtype(self.factor, device=old_lr.device)
            min_lr = self.optimizer.dtype(self.min_lrs[i], device=old_lr.device)
            eps = self.optimizer.dtype(self.eps, device=old_lr.device)
            new_lr = torch.maximum(old_lr * factor, min_lr)
            if old_lr - new_lr > eps:
                _update_param_group_val(param_group, "lr", new_lr)

    def _is_better(self, a, best):
        if best is None:
            return True

        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
