from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from torchdt.ops.triton_ops import (
    TritonAccumulatorOps,
    TritonScalarOps,
    require_triton,
)


@dataclass(frozen=True)
class TritonRegistrationContext:
    """Per-dtype state captured by one Triton backend registration."""

    dtype_cls: type
    triton: Any
    tl: Any
    from_float: Callable
    to_float: Callable
    add: Callable
    sub: Optional[Callable]
    mul: Optional[Callable]
    div: Optional[Callable]
    sqrt: Optional[Callable]
    gt: Optional[Callable]
    ge: Optional[Callable]
    lt: Optional[Callable]
    le: Optional[Callable]
    neg: Optional[Callable]
    exp: Callable
    log: Callable
    clamp: Callable
    sign: Callable
    acc_int_dtype: torch.dtype
    acc_from_float: Callable
    acc_add: Callable
    acc_div: Optional[Callable]
    to_accumulator: Callable
    from_accumulator: Callable
    tl_int_dtype: Any
    zero: Any
    neg_inf: Any
    one: Any
    metadata_tensor: Callable
    can_register_sign: bool


def create_registration_context(
    dtype_cls: type,
    scalar_ops: TritonScalarOps,
    accumulator_ops: Optional[TritonAccumulatorOps] = None,
) -> TritonRegistrationContext:
    if not isinstance(scalar_ops, TritonScalarOps):
        raise TypeError("scalar_ops must be a TritonScalarOps instance.")
    if accumulator_ops is not None and not isinstance(
        accumulator_ops, TritonAccumulatorOps
    ):
        raise TypeError("accumulator_ops must be a TritonAccumulatorOps instance.")

    triton, tl = require_triton()

    from_float = scalar_ops.from_float
    to_float = scalar_ops.to_float
    add = scalar_ops.add
    sub = scalar_ops.sub
    mul = scalar_ops.mul
    div = scalar_ops.div
    sqrt = scalar_ops.sqrt
    gt = scalar_ops.gt
    ge = scalar_ops.ge
    lt = scalar_ops.lt
    le = scalar_ops.le
    neg = scalar_ops.neg
    exp = scalar_ops.exp
    log = scalar_ops.log
    clamp = scalar_ops.clamp
    sign = scalar_ops.sign

    if accumulator_ops is None:
        accumulator_scalar_ops = scalar_ops
        acc_int_dtype = dtype_cls.int_dtype

        @triton.jit
        def to_accumulator(x):
            return x

        @triton.jit
        def from_accumulator(x):
            return x
    else:
        acc_int_dtype = accumulator_ops.int_dtype
        accumulator_scalar_ops = accumulator_ops.scalar_ops
        to_accumulator = accumulator_ops.to_accumulator
        from_accumulator = accumulator_ops.from_accumulator

    acc_from_float = accumulator_scalar_ops.from_float
    acc_add = accumulator_scalar_ops.add
    acc_div = accumulator_scalar_ops.div

    int_types = {
        8: tl.int8,
        16: tl.int16,
        32: tl.int32,
        64: tl.int64,
    }
    try:
        tl_int_dtype = tl.constexpr(int_types[dtype_cls.bitwidth])
    except KeyError as error:
        raise ValueError(
            f"Triton backend does not support bitwidth {dtype_cls.bitwidth}."
        ) from error

    zero = tl.constexpr(dtype_cls.ops.encoded_scalar(0.0))
    neg_inf = tl.constexpr(dtype_cls.ops.encoded_scalar(float("-inf")))
    one = tl.constexpr(dtype_cls.ops.encoded_scalar(1.0))
    can_register_sign = sign is not None or (lt is not None and neg is not None)

    if exp is None:
        @triton.jit
        def exp(x):
            return from_float(tl.exp(to_float(x)))

    if log is None:
        @triton.jit
        def log(x):
            return from_float(tl.log(to_float(x)))

    if clamp is None:
        @triton.jit
        def clamp(x, min, max):
            return tl.where(lt(x, min), min, tl.where(gt(x, max), max, x))

    if sign is None:
        @triton.jit
        def sign(x):
            return tl.where(
                x == zero,
                tl.cast(zero, tl_int_dtype),
                tl.where(
                    lt(x, tl.cast(zero, tl_int_dtype)),
                    neg(tl.cast(one, tl_int_dtype)),
                    tl.cast(one, tl_int_dtype),
                ),
            )

    # Metadata is immutable and repeated layouts are common in eager workloads.
    metadata_cache = OrderedDict()
    metadata_cache_limit = 256

    def metadata_tensor(values, device):
        if len(values) == 0:
            values = (0,)
        values = tuple(int(value) for value in values)
        device = torch.device(device)
        key = (device.type, device.index, values)
        cached = metadata_cache.get(key)
        if cached is not None:
            metadata_cache.move_to_end(key)
            return cached

        result = torch.tensor(values, dtype=torch.int64, device=device)
        metadata_cache[key] = result
        if len(metadata_cache) > metadata_cache_limit:
            metadata_cache.popitem(last=False)
        return result

    return TritonRegistrationContext(
        dtype_cls=dtype_cls,
        triton=triton,
        tl=tl,
        from_float=from_float,
        to_float=to_float,
        add=add,
        sub=sub,
        mul=mul,
        div=div,
        sqrt=sqrt,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        neg=neg,
        exp=exp,
        log=log,
        clamp=clamp,
        sign=sign,
        acc_int_dtype=acc_int_dtype,
        acc_from_float=acc_from_float,
        acc_add=acc_add,
        acc_div=acc_div,
        to_accumulator=to_accumulator,
        from_accumulator=from_accumulator,
        tl_int_dtype=tl_int_dtype,
        zero=zero,
        neg_inf=neg_inf,
        one=one,
        metadata_tensor=metadata_tensor,
        can_register_sign=can_register_sign,
    )
