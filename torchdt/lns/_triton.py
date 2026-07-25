import hashlib

import torch
from torchdt.ops import TritonAccumulatorOps, TritonScalarOps, register_triton_ops, require_triton


def _bump_triton_jit_hash(fn, **values):
    items = tuple(sorted((key, repr(getattr(value, "value", value))) for key, value in values.items()))
    fn.hash = hashlib.sha256(repr(items).encode()).hexdigest()
    return fn


def _lns_triton_bit_config(bitwidth: int, tl):
    if bitwidth == 16:
        return (
            tl.int16,
            tl.constexpr("=h, l"),
            tl.constexpr("ld.global.b16 $0, [$1];"),
            tl.constexpr(2),
        )
    if bitwidth == 32:
        return (
            tl.int32,
            tl.constexpr("=r, l"),
            tl.constexpr("ld.global.b32 $0, [$1];"),
            tl.constexpr(4),
        )
    if bitwidth == 64:
        return (
            tl.int64,
            tl.constexpr("=l, l"),
            tl.constexpr("ld.global.b64 $0, [$1];"),
            tl.constexpr(8),
        )
    raise ValueError(f"LNS Triton backend does not support bitwidth {bitwidth}.")


def enable_lns_triton_backend(
    dtype_cls: type,
    *,
    base: torch.Tensor,
    zero_value: int,
    pos_inf_value: int,
    neg_inf_value: int,
    tab_sbdb=None,
    tab_ez=None,
    accumulator_ops: TritonAccumulatorOps = None,
) -> None:
    fingerprint = (
        dtype_cls.bitwidth,
        float(base),
        zero_value,
        pos_inf_value,
        neg_inf_value,
        tab_sbdb.data_ptr() if tab_sbdb is not None else None,
        tab_ez.data_ptr() if tab_ez is not None else None,
        id(accumulator_ops) if accumulator_ops is not None else None,
    )
    if getattr(dtype_cls.ops, "_triton_fingerprint", None) == fingerprint:
        return

    scalar_ops = make_lns_triton_scalar_ops(
        bitwidth=dtype_cls.bitwidth,
        base=base,
        zero_value=zero_value,
        pos_inf_value=pos_inf_value,
        neg_inf_value=neg_inf_value,
        tab_sbdb=tab_sbdb,
        tab_ez=tab_ez,
    )
    register_triton_ops(dtype_cls, scalar_ops, accumulator_ops)
    dtype_cls.ops._triton_fingerprint = fingerprint


def make_lns_triton_scalar_ops(
    *,
    bitwidth: int,
    base: torch.Tensor,
    zero_value: int,
    pos_inf_value: int,
    neg_inf_value: int,
    tab_sbdb=None,
    tab_ez=None,
) -> TritonScalarOps:
    triton, tl = require_triton()

    tl_int_dtype, asm_output_constraint, asm_load, bytes_per_value = _lns_triton_bit_config(bitwidth, tl)

    LOG_BASE = tl.constexpr(torch.log(base).item())
    ZERO = tl.constexpr(zero_value)
    POS_INF = tl.constexpr(pos_inf_value)
    NEG_INF = tl.constexpr(neg_inf_value)
    MIN_LOG = tl.constexpr(zero_value >> 1)
    MAX_LOG = tl.constexpr(pos_inf_value >> 1)
    MIN_FINITE_LOG = tl.constexpr((zero_value >> 1) + 1)
    MAX_FINITE_LOG = tl.constexpr((pos_inf_value >> 1) - 1)

    @triton.jit
    def from_float(x):
        abs_x = tl.abs(tl.cast(x, tl.float64))
        log_x = tl.log(abs_x) / tl.cast(LOG_BASE, tl.float64)

        rounded = tl.where(log_x >= 0, tl.floor(log_x + 0.5), tl.ceil(log_x - 0.5))
        sign_bit = tl.cast(x < 0, tl_int_dtype)
        finite_rounded = tl.minimum(
            tl.maximum(rounded, tl.cast(MIN_FINITE_LOG, tl.float64)),
            tl.cast(MAX_FINITE_LOG, tl.float64),
        )
        packed = (tl.cast(finite_rounded, tl_int_dtype) << 1) | sign_bit
        overflow = rounded >= tl.cast(MAX_LOG, tl.float64)
        underflow = rounded <= tl.cast(MIN_LOG, tl.float64)
        inf = tl.where(sign_bit == 0, tl.cast(POS_INF, tl_int_dtype), tl.cast(NEG_INF, tl_int_dtype))

        return tl.where(
            x == 0.0,
            tl.cast(ZERO, tl_int_dtype),
            tl.where(overflow, inf, tl.where(underflow, tl.cast(ZERO, tl_int_dtype), packed)),
        )

    @triton.jit
    def to_float(x):
        log_x = x >> 1
        sign = tl.where((x & 1) == 1, -1.0, 1.0)

        abs_x = tl.exp(tl.cast(LOG_BASE, tl.float64) * tl.cast(log_x, tl.float64))
        float_x = sign * abs_x

        return tl.where(
            x == ZERO,
            0.0,
            tl.where(x == POS_INF, float("inf"), tl.where(x == NEG_INF, float("-inf"), float_x.to(tl.float32))),
        )

    @triton.jit
    def sub(x, y):
        return add(x, neg(y))

    @triton.jit
    def checked_add(a, b, overflow_sign):
        result = tl.cast(a + b, tl_int_dtype)

        underflow = (a < 0) & (b < 0) & (result >= 0)
        overflow = (a >= 0) & (b >= 0) & (result < 0)
        inf_signed = tl.where(overflow_sign == 0, tl.cast(POS_INF, tl_int_dtype), tl.cast(NEG_INF, tl_int_dtype))

        return tl.where(underflow, tl.cast(ZERO, tl_int_dtype), tl.where(overflow, inf_signed, result))

    @triton.jit
    def mul(x, y):
        y_magnitude = y - (y & 1)
        prod_unsigned = checked_add(x, y_magnitude, x & 1)
        prod = tl.where(prod_unsigned == ZERO, tl.cast(ZERO, tl_int_dtype), prod_unsigned ^ (y & 1))
        return tl.where(x == ZERO, tl.cast(ZERO, tl_int_dtype), tl.where(y == tl.cast(ZERO, tl_int_dtype), tl.cast(ZERO, tl_int_dtype), prod))

    @triton.jit
    def div(x, y):
        safe_y = tl.where(y == ZERO, tl.cast(0, tl_int_dtype), y)
        divisor_delta = -safe_y + (safe_y & 1)
        quotient_unsigned = checked_add(x, divisor_delta, x & 1)
        quotient = tl.where(quotient_unsigned == ZERO, tl.cast(ZERO, tl_int_dtype), quotient_unsigned ^ (safe_y & 1))
        div_by_zero = tl.where((x & 1) == 0, tl.cast(POS_INF, tl_int_dtype), tl.cast(NEG_INF, tl_int_dtype))
        return tl.where(x == ZERO, tl.cast(ZERO, tl_int_dtype), tl.where(y == ZERO, div_by_zero, quotient))

    @triton.jit
    def sqrt(x):
        result = ((x & (-2)) // 2) & (-2)
        return tl.where(x == ZERO, tl.cast(ZERO, tl_int_dtype), tl.where(x == POS_INF, tl.cast(POS_INF, tl_int_dtype), result))

    @triton.jit
    def neg(x):
        return tl.where(x == ZERO, tl.cast(ZERO, tl_int_dtype), x ^ 1)

    @triton.jit
    def gt(x, y):
        x_log = x >> 1
        y_log = y >> 1
        x_sign = x & 1
        y_sign = y & 1

        both_pos = (x_sign == 0) & (y_sign == 0)
        x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
        both_neg = (x_sign == 1) & (y_sign == 1)

        return x_pos_y_neg | (both_pos & (x_log > y_log)) | (both_neg & (y_log > x_log))

    @triton.jit
    def ge(x, y):
        x_log = x >> 1
        y_log = y >> 1
        x_sign = x & 1
        y_sign = y & 1

        both_pos = (x_sign == 0) & (y_sign == 0)
        x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
        both_neg = (x_sign == 1) & (y_sign == 1)

        return x_pos_y_neg | (both_pos & (x_log >= y_log)) | (both_neg & (y_log >= x_log))

    @triton.jit
    def lt(x, y):
        x_log = x >> 1
        y_log = y >> 1
        x_sign = x & 1
        y_sign = y & 1

        both_pos = (x_sign == 0) & (y_sign == 0)
        x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
        both_neg = (x_sign == 1) & (y_sign == 1)

        return x_neg_y_pos | (both_pos & (x_log < y_log)) | (both_neg & (y_log < x_log))

    @triton.jit
    def le(x, y):
        x_log = x >> 1
        y_log = y >> 1
        x_sign = x & 1
        y_sign = y & 1

        both_pos = (x_sign == 0) & (y_sign == 0)
        x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
        both_neg = (x_sign == 1) & (y_sign == 1)

        return x_neg_y_pos | (both_pos & (x_log <= y_log)) | (both_neg & (y_log <= x_log))

    if tab_sbdb is not None and tab_ez is not None:
        if tab_sbdb.device.type != "cuda":
            raise ValueError("LNS Triton table backend requires tab_sbdb to be on a CUDA device.")

        tab_sbdb_size = tl.constexpr(tab_sbdb.size(1))
        tab_sbdb_data_ptr = tl.constexpr(tab_sbdb.data_ptr())
        tab_ez_item = tl.constexpr(tab_ez.item())

        @triton.jit
        def add(x, y):
            max_operand = tl.maximum(x, y)

            z = -tl.abs((x >> 1) - (y >> 1)).to(tl.int64)
            s = ((x ^ y) & 1).to(tl.int64)

            idx = (s + 1) * tab_sbdb_size + tl.where(z < tab_ez_item, tab_ez_item, tl.where(z == 0, -1, z))
            abs_ptr = tab_sbdb_data_ptr + idx * bytes_per_value

            sbdb = tl.inline_asm_elementwise(
                "{{\n   " + asm_load + "\n}}",
                asm_output_constraint,
                [abs_ptr],
                dtype=tl_int_dtype,
                is_pure=True,
                pack=1,
            )

            result = checked_add(max_operand, sbdb, max_operand & 1)
            return tl.where(x == ZERO, y, tl.where(y == ZERO, x, tl.where(x == neg(y), tl.cast(ZERO, tl_int_dtype), result)))

        _bump_triton_jit_hash(
            add,
            ZERO=ZERO,
            POS_INF=POS_INF,
            NEG_INF=NEG_INF,
            tab_sbdb=tab_sbdb.data_ptr(),
            tab_ez=tab_ez.item(),
            bitwidth=bitwidth,
        )

    else:
        @triton.jit
        def add(x, y):
            max_operand = tl.maximum(x, y)

            abs_diff = tl.abs((x >> 1) - (y >> 1)).to(tl.float64)
            sign_diff = ((x ^ y) & 1).to(tl.float64)

            power_term = tl.exp(LOG_BASE * -abs_diff)
            magnitude = tl.abs(1.0 - 2.0 * sign_diff + power_term)

            log_term = tl.log(magnitude) / LOG_BASE
            rounded = tl.where(log_term >= 0, tl.floor(log_term + 0.5), tl.ceil(log_term - 0.5))
            sbdb = rounded.to(tl_int_dtype) * 2

            result = checked_add(max_operand, sbdb, max_operand & 1)
            return tl.where(x == ZERO, y, tl.where(y == ZERO, x, tl.where(x == neg(y), tl.cast(ZERO, tl_int_dtype), result)))

        _bump_triton_jit_hash(add, LOG_BASE=LOG_BASE, ZERO=ZERO, POS_INF=POS_INF, NEG_INF=NEG_INF, bitwidth=bitwidth)

    _bump_triton_jit_hash(from_float, LOG_BASE=LOG_BASE, ZERO=ZERO, POS_INF=POS_INF, NEG_INF=NEG_INF, bitwidth=bitwidth)
    _bump_triton_jit_hash(to_float, LOG_BASE=LOG_BASE, ZERO=ZERO, bitwidth=bitwidth)
    _bump_triton_jit_hash(mul, ZERO=ZERO, POS_INF=POS_INF, NEG_INF=NEG_INF, bitwidth=bitwidth)
    _bump_triton_jit_hash(div, ZERO=ZERO, POS_INF=POS_INF, NEG_INF=NEG_INF, bitwidth=bitwidth)
    _bump_triton_jit_hash(sqrt, ZERO=ZERO, bitwidth=bitwidth)

    return TritonScalarOps(
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
    )
