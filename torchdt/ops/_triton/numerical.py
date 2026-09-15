from __future__ import annotations

import torch

from torchdt.triton import _get_numerical_policy, _get_numerical_policy_state


def launch_checked(kernel_name, tuner, grid, args, kwargs, *, reference, outputs, decode,
                   initialize=(), exact=(), reference_for_config=None, config_filter=None,
                   return_tuner=False):
    """
    Validate candidates and tune a cache miss for checked_autotune.

    ``reference()`` returns a tuple of ordinary torch tensors. ``outputs`` lists
    positional output argument indices, including all buffers written by the
    kernel. Outputs must not alias other arguments. ``initialize`` outputs are
    copied to scratch before each trial (in-place state or partially written
    outputs). Other outputs are write-only. ``exact`` outputs use exact equality;
    ``decode`` is a callable or a mapping of output indices to decoders.
    ``reference_for_config`` supports config-dependent partial layouts.
    Other arguments must be read-only. This interface is datatype-independent.
    """
    policy = _get_numerical_policy(kernel_name)
    if policy is None:
        return tuner[grid](*args, **kwargs)
    import triton

    with torch.no_grad():
        expected = reference() if reference_for_config is None else None
        if not outputs or (expected is not None and len(expected) != len(outputs)):
            raise ValueError("reference must return one tensor per output")
        survivors = []
        failures = []
        for config in tuner.configs:
            if config_filter is not None and not config_filter(config):
                failures.append(f"{config}: incompatible intermediate buffer layout")
                continue
            targets = expected if reference_for_config is None else reference_for_config(config)
            if len(targets) != len(outputs):
                raise ValueError("reference must return one tensor per output")
            trial = list(args)
            for index in outputs:
                output = args[index]
                trial[index] = torch.empty_strided(
                    output.shape, output.stride(), dtype=output.dtype, device=output.device
                )
                if index in initialize:
                    trial[index].copy_(output)
            # Compilation/runtime failures are errors, not numerical mismatches.
            tuner.fn[grid](*trial, **kwargs, **config.all_kwargs())
            passed = True
            max_error = 0.0
            for index, target in zip(outputs, targets):
                decoder = decode[index] if isinstance(decode, dict) else decode
                actual = decoder(trial[index]).detach()
                target = target.detach()
                if actual.shape != target.shape:
                    raise ValueError("numerical reference output shape mismatch")
                target = target.to(actual.device)
                if index in exact:
                    if not torch.equal(actual, target):
                        passed = False
                        max_error = float("inf")
                    continue
                actual, target = actual.double(), target.double()
                finite = torch.isfinite(actual) & torch.isfinite(target)
                error = (actual - target).abs()
                close = finite & (error <= policy[0] + policy[1] * target.abs())
                passed = passed and bool(close.all())
                if error.numel():
                    max_error = max(max_error, float(error.nan_to_num(nan=float('inf')).max()))
            if passed:
                survivors.append(config)
            else:
                failures.append(f"{config}: max_abs_error={max_error:g}")
        if not survivors:
            raise RuntimeError(
                f"Numerical pruning removed every config for {kernel_name!r} "
                f"(atol={policy[0]}, rtol={policy[1]}, "
                f"tensor_shapes={[tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]}). "
                + "; ".join(failures)
            )
    # Keep checked timing decisions separate from unchecked ones. Filtering
    # explicitly also covers Triton's single-config pruning bypass.
    checked = triton.autotune(
        configs=survivors, key=tuner.keys,
        restore_value=getattr(tuner, "restore_value", None),
        reset_to_zero=getattr(tuner, "reset_to_zero", None),
    )(tuner.fn)
    result = checked[grid](*args, **kwargs)
    return (result, checked) if return_tuner else result


def checked_autotune(kernel_name, context, **options):
    """
    Attach a reference adapter to the normal Triton autotuner.

    The disabled path uses the original tuner and hooks unchanged. Reference
    adapters describe every written buffer, including in-place state and sparse
    writes. Accumulator buffers use the registered accumulator decoder.
    """
    triton, tl = context.triton, context.tl
    acc_to_float = context.acc_to_float

    @triton.jit
    def decode_accumulator_kernel(src, dst, size, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(src + i, i < size, other=0)
        tl.store(dst + i, acc_to_float(value), i < size)

    def decode(value):
        return context.dtype_cls(value, internal=True).to_float().double()

    def decode_accumulator(value):
        result = torch.empty(value.shape, dtype=torch.float64, device=value.device)
        if value.numel():
            decode_accumulator_kernel[(triton.cdiv(value.numel(), 256),)](
                value.contiguous(), result, value.numel(), BLOCK=256,
            )
        return result

    def decorate(fn):
        tuner = triton.autotune(**options)(fn)
        checked_cache = {}
        cache_revision = None

        class CheckedAutotuner:
            def __getattr__(self, name):
                return getattr(tuner, name)

            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    nonlocal cache_revision
                    policy, revision = _get_numerical_policy_state(kernel_name)
                    if revision != cache_revision:
                        checked_cache.clear()
                        cache_revision = revision
                    if policy is None:
                        return tuner[grid](*args, **kwargs)

                    named = {**dict(zip(fn.arg_names, args)), **kwargs}
                    device = next(x.device for x in args if isinstance(x, torch.Tensor))
                    # Match Triton's key fields and dtype specialization, with
                    # device isolation. Shape buckets intentionally share a key.
                    key_args = {k: v for k, v in named.items() if k in fn.arg_names}
                    key = (device, tuple(key_args[k] for k in tuner.keys if k in key_args),
                           tuple(str(v.dtype) for v in key_args.values() if hasattr(v, "dtype")))
                    if key in checked_cache:
                        return checked_cache[key][grid](*args, **kwargs)

                    from torchdt.ops._triton.references import make_reference

                    def scalar(value):
                        return decode(torch.tensor(value, device=device,
                                                   dtype=context.dtype_cls.int_dtype))

                    with torch.no_grad():
                        plan = make_reference(kernel_name, named, decode,
                                              decode_accumulator, scalar)
                    output_names = tuple(plan.formats)
                    outputs = tuple(fn.arg_names.index(name) for name in output_names)
                    decoders = {"dtype": decode, "acc": decode_accumulator, "raw": lambda x: x}
                    def targets(config):
                        values = plan.values(config)
                        return tuple(values[n] for n in output_names)

                    result, checked = launch_checked(
                        kernel_name, tuner, grid, args, kwargs,
                        reference=None, outputs=outputs,
                        decode={i: decoders[plan.formats[n]] for n, i in zip(output_names, outputs)},
                        initialize=tuple(fn.arg_names.index(n) for n in plan.initialize),
                        exact=tuple(fn.arg_names.index(n) for n in output_names
                                    if plan.formats[n] == "raw"),
                        reference_for_config=targets,
                        config_filter=plan.valid_config,
                        return_tuner=True,
                    )
                    checked_cache[key] = checked
                    return result
                return run
        return CheckedAutotuner()
    return decorate
