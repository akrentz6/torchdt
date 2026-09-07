from torchdt.triton import AutotuneConfig, _resolve_autotune_configs

__all__ = ["DEFAULT_AUTOTUNE_CONFIGS", "autotune_configs"]


def _config(kwargs, warps, stages):
    return AutotuneConfig(kwargs, num_warps=warps, num_stages=stages)


DEFAULT_AUTOTUNE_CONFIGS = {
    "matmul": (
        _config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 8}, 4, 2),
        _config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 8}, 2, 2),
        _config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 8}, 2, 2),
        _config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 8}, 1, 2),
        _config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 4}, 2, 2),
        _config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 4}, 4, 2),
        _config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, 4, 2),
    ),
    "sum": (
        _config({"BLOCK": 128}, 2, 2),
        _config({"BLOCK": 64}, 1, 2),
        _config({"BLOCK": 128}, 1, 2),
        _config({"BLOCK": 256}, 2, 2),
        _config({"BLOCK": 256}, 4, 2),
    ),
    "conv2d": (
        _config({"BLOCK_OC": 8, "BLOCK_HW": 128}, 4, 1),
        _config({"BLOCK_OC": 4, "BLOCK_HW": 128}, 4, 1),
        _config({"BLOCK_OC": 8, "BLOCK_HW": 64}, 4, 1),
        _config({"BLOCK_OC": 16, "BLOCK_HW": 64}, 4, 1),
        _config({"BLOCK_OC": 4, "BLOCK_HW": 256}, 4, 1),
    ),
    "conv2d_dinput": (
        _config({"BLOCK_IC": 4, "BLOCK_HW": 64}, 4, 1),
        _config({"BLOCK_IC": 8, "BLOCK_HW": 32}, 4, 1),
        _config({"BLOCK_IC": 2, "BLOCK_HW": 128}, 4, 1),
        _config({"BLOCK_IC": 4, "BLOCK_HW": 32}, 2, 1),
    ),
    "conv2d_dweight": (
        _config({"BLOCK_OC": 4, "BLOCK_IC": 4, "BLOCK_NHW": 64}, 4, 1),
        _config({"BLOCK_OC": 8, "BLOCK_IC": 4, "BLOCK_NHW": 32}, 4, 1),
        _config({"BLOCK_OC": 4, "BLOCK_IC": 8, "BLOCK_NHW": 32}, 4, 1),
        _config({"BLOCK_OC": 2, "BLOCK_IC": 4, "BLOCK_NHW": 64}, 2, 1),
    ),
    "conv2d_dbias": (
        _config({"BLOCK_NHW": 1024}, 4, 1),
        _config({"BLOCK_NHW": 512}, 4, 1),
        _config({"BLOCK_NHW": 2048}, 8, 1),
    ),
    "max_pool2d": (
        _config({"BLOCK_HW": 64}, 4, 1),
        _config({"BLOCK_HW": 32}, 2, 1),
        _config({"BLOCK_HW": 128}, 4, 1),
    ),
    "adaptive_avg_pool2d": (
        _config({"BLOCK_C": 8, "BLOCK_HW": 128}, 4, 1),
        _config({"BLOCK_C": 4, "BLOCK_HW": 128}, 4, 1),
        _config({"BLOCK_C": 8, "BLOCK_HW": 64}, 4, 1),
        _config({"BLOCK_C": 16, "BLOCK_HW": 64}, 4, 1),
        _config({"BLOCK_C": 4, "BLOCK_HW": 256}, 4, 1),
    ),
    "adaptive_avg_pool2d_dinput": (
        _config({"BLOCK_C": 8, "BLOCK_HW": 128}, 4, 1),
        _config({"BLOCK_C": 4, "BLOCK_HW": 128}, 4, 1),
        _config({"BLOCK_C": 8, "BLOCK_HW": 64}, 4, 1),
        _config({"BLOCK_C": 16, "BLOCK_HW": 64}, 4, 1),
        _config({"BLOCK_C": 4, "BLOCK_HW": 256}, 4, 1),
    ),
    "nll_loss": tuple(
        _config({"BLOCK": block}, warps, 1)
        for block, warps in ((128, 1), (256, 2), (512, 4), (1024, 4), (1024, 8))
    ),
    "nll_denominator": tuple(
        _config({"BLOCK": block}, warps, 1)
        for block, warps in ((128, 1), (256, 2), (512, 4), (1024, 4), (1024, 8))
    ),
    "nll_loss_backward": tuple(
        _config({"BLOCK": block}, warps, 1)
        for block, warps in ((128, 1), (256, 2), (512, 4), (1024, 4), (1024, 8))
    ),
    "batch_norm2d_sum": tuple(
        _config({"BLOCK": 128}, warps, 1) for warps in (1, 2, 4)
    ),
    "batch_norm2d_mean_finalize": tuple(
        _config({"BLOCK_T": block}, warps, 1)
        for block, warps in (
            (16, 1), (32, 1), (64, 1), (128, 1),
            (128, 2), (256, 2), (512, 4), (1024, 4),
        )
    ),
    "batch_norm2d_centered_var": tuple(
        _config({"BLOCK": 128}, warps, 1) for warps in (1, 2, 4)
    ),
    "batch_norm2d_var_finalize": tuple(
        _config({"BLOCK_T": block}, warps, 1)
        for block, warps in (
            (16, 1), (32, 1), (64, 1), (128, 1),
            (128, 2), (256, 2), (512, 4), (1024, 4),
        )
    ),
    "batch_norm2d_forward": tuple(
        _config({"BLOCK": block}, warps, 1)
        for block, warps in (
            (64, 2), (128, 4), (256, 4), (512, 4),
            (512, 8), (1024, 4), (1024, 8),
        )
    ),
    "batch_norm2d_backward_partials": tuple(
        _config({"BLOCK": 256}, warps, 1) for warps in (1, 2, 4)
    ),
    "batch_norm2d_backward_finalize": tuple(
        _config({"BLOCK_R": block}, warps, 1)
        for block, warps in (
            (64, 1), (128, 1), (256, 1), (256, 2),
            (512, 2), (512, 4), (1024, 4),
        )
    ),
    "batch_norm2d_dinput_train": tuple(
        _config({"BLOCK": block}, warps, 1)
        for block, warps in (
            (32, 1), (64, 1), (64, 2), (128, 4),
            (256, 4), (512, 4), (512, 8), (1024, 8),
        )
    ),
    "batch_norm2d_dinput_eval": tuple(
        _config({"BLOCK": block}, warps, 1)
        for block, warps in (
            (32, 1), (64, 1), (64, 2), (128, 4),
            (256, 4), (512, 4), (512, 8), (1024, 8),
        )
    ),
}


def autotune_configs(kernel: str, triton):
    try:
        defaults = DEFAULT_AUTOTUNE_CONFIGS[kernel]
    except KeyError as error:
        raise ValueError(f"No default autotune configs for kernel {kernel!r}") from error
    return _resolve_autotune_configs(kernel, triton, defaults)
