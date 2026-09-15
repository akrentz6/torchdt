from __future__ import annotations

import math

from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "AUTOTUNE_KERNELS",
    "AutotuneConfig",
    "exclude_autotune_configs",
    "get_autotune_configs",
    "reset_autotune_configs",
    "select_autotune_config",
    "set_autotune_configs",
    "set_autotune_numerical_check",
]


AUTOTUNE_KERNELS = (
    "adaptive_avg_pool2d",
    "adaptive_avg_pool2d_dinput",
    "batch_norm2d_backward_finalize",
    "batch_norm2d_backward_partials",
    "batch_norm2d_centered_var",
    "batch_norm2d_dinput_eval",
    "batch_norm2d_dinput_train",
    "batch_norm2d_forward",
    "batch_norm2d_mean_finalize",
    "batch_norm2d_sum",
    "batch_norm2d_var_finalize",
    "conv2d",
    "conv2d_dbias",
    "conv2d_dinput",
    "conv2d_dweight",
    "matmul",
    "max_pool2d",
    "nll_denominator",
    "nll_loss",
    "nll_loss_backward",
    "sum",
)


@dataclass(frozen=True)
class AutotuneConfig:
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 3
    num_ctas: int = 1
    maxnreg: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))
        if self.num_warps <= 0 or self.num_stages <= 0 or self.num_ctas <= 0:
            raise ValueError("num_warps, num_stages, and num_ctas must be positive")


_overrides: dict[str, tuple[AutotuneConfig, ...]] = {}
_exclusions: dict[str, tuple[object, ...]] = {}
_numerical_policies: dict[str, tuple[float, float]] = {}
_numerical_revisions: dict[str, int] = {}
_revision = 0
_lock = RLock()


def _check_kernel(kernel: str) -> None:
    if kernel not in AUTOTUNE_KERNELS:
        names = ", ".join(AUTOTUNE_KERNELS)
        raise ValueError(
            f"Unknown torchdt Triton kernel {kernel!r}. Choose from: {names}"
        )


def _coerce_config(config: AutotuneConfig | Mapping[str, Any] | Any) -> AutotuneConfig:
    if isinstance(config, AutotuneConfig):
        return config
    if isinstance(config, Mapping):
        return AutotuneConfig(config)
    kwargs = getattr(config, "kwargs", None)
    if kwargs is None:
        raise TypeError(
            "configs must contain AutotuneConfig, mapping, or triton.Config values"
        )
    return AutotuneConfig(
        kwargs,
        num_warps=getattr(config, "num_warps", 4),
        num_stages=getattr(config, "num_stages", 3),
        num_ctas=getattr(config, "num_ctas", 1),
        maxnreg=getattr(config, "maxnreg", None),
    )


def set_autotune_configs(
    kernel: str,
    configs: Iterable[AutotuneConfig | Mapping[str, Any] | Any],
) -> None:
    global _revision
    _check_kernel(kernel)
    values = tuple(_coerce_config(config) for config in configs)
    if not values:
        raise ValueError("At least one autotune config is required")
    with _lock:
        _overrides[kernel] = values
        _revision += 1


def select_autotune_config(
    kernel: str,
    config: AutotuneConfig | Mapping[str, Any] | Any,
) -> None:
    global _revision
    _check_kernel(kernel)
    selected = _coerce_config(config)
    with _lock:
        _overrides[kernel] = (selected,)
        _exclusions.pop(kernel, None)
        _revision += 1


def exclude_autotune_configs(
    kernel: str,
    configs: Iterable[AutotuneConfig | Mapping[str, Any] | Any],
) -> None:
    global _revision
    _check_kernel(kernel)
    selectors = []
    for config in configs:
        selectors.append(
            dict(config) if isinstance(config, Mapping) else _coerce_config(config)
        )
    with _lock:
        _exclusions[kernel] = _exclusions.get(kernel, ()) + tuple(selectors)
        _revision += 1


def reset_autotune_configs(kernel: Optional[str] = None) -> None:
    global _revision
    if kernel is not None:
        _check_kernel(kernel)
    with _lock:
        if kernel is None:
            _overrides.clear()
            _exclusions.clear()
        else:
            _overrides.pop(kernel, None)
            _exclusions.pop(kernel, None)
        _revision += 1


def _config_matches(config: AutotuneConfig, selector: object) -> bool:
    if isinstance(selector, Mapping):
        return all(config.kwargs.get(key) == value for key, value in selector.items())
    return config == selector


def _resolve_autotune_specs(
    kernel: str, defaults: Iterable[Any]
) -> tuple[AutotuneConfig, ...]:
    _check_kernel(kernel)
    default_values = tuple(defaults)
    with _lock:
        source = _overrides.get(kernel)
        selectors = _exclusions.get(kernel, ())
    specs = (
        tuple(_coerce_config(value) for value in default_values)
        if source is None
        else source
    )
    specs = tuple(
        config
        for config in specs
        if not any(_config_matches(config, selector) for selector in selectors)
    )
    if not specs:
        raise ValueError(f"Autotune policy removed every config for kernel {kernel!r}")
    return specs


def get_autotune_configs(kernel: str) -> tuple[AutotuneConfig, ...]:
    _check_kernel(kernel)
    from torchdt.ops._triton.autotune import DEFAULT_AUTOTUNE_CONFIGS

    return _resolve_autotune_specs(kernel, DEFAULT_AUTOTUNE_CONFIGS[kernel])


def _resolve_autotune_configs(
    kernel: str, triton: Any, defaults: Iterable[Any]
) -> list[Any]:
    _check_kernel(kernel)
    default_values = tuple(defaults)
    specs = _resolve_autotune_specs(kernel, default_values)

    result = []
    for config in specs:
        kwargs = {
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "num_ctas": config.num_ctas,
        }
        if config.maxnreg is not None:
            kwargs["maxnreg"] = config.maxnreg
        try:
            result.append(triton.Config(dict(config.kwargs), **kwargs))
        except TypeError:
            # Triton 2.x does not accept all newer launch options.
            kwargs.pop("num_ctas", None)
            kwargs.pop("maxnreg", None)
            result.append(triton.Config(dict(config.kwargs), **kwargs))
    return result


def _autotune_revision() -> int:
    return _revision


def set_autotune_numerical_check(
    enabled: bool = False, *, atol: float = 0.0, rtol: float = 0.0,
    kernels: Iterable[str] = AUTOTUNE_KERNELS,
) -> None:
    """
    Opt into numerical pruning when autotuning a new key.

    Compare decoded outputs against float64 arithmetic on decoded inputs using
    abs(actual - expected) <= atol + rtol * abs(expected), requiring finite
    values. All configs (including a single forced config) are validated before
    timing. If none pass, raise rather than silently use an inaccurate config.
    Checked configs are cached per autotune key and device. Later calls reuse
    the result without checking new values. Updating this policy invalidates
    checked caches for the selected kernels. Disabled by default; applies to
    all torchdt datatypes.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    atol, rtol = float(atol), float(rtol)
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError("atol and rtol must be finite and non-negative")
    names = (kernels,) if isinstance(kernels, str) else tuple(kernels)
    for name in names:
        _check_kernel(name)
    with _lock:
        for name in names:
            _numerical_revisions[name] = _numerical_revisions.get(name, 0) + 1
            if enabled:
                _numerical_policies[name] = (atol, rtol)
            else:
                _numerical_policies.pop(name, None)


def _get_numerical_policy(kernel: str):
    with _lock:
        return _numerical_policies.get(kernel)


def _get_numerical_policy_state(kernel: str):
    with _lock:
        return _numerical_policies.get(kernel), _numerical_revisions.get(kernel, 0)
