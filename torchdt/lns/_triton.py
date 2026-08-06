import hashlib
import math

import torch
from torchdt.ops import TritonAccumulatorOps, TritonScalarOps, register_triton_ops, require_triton


def _bump_triton_jit_hash(fn, **values):
    items = tuple(sorted((key, repr(getattr(value, "value", value))) for key, value in values.items()))
    fn.hash = hashlib.sha256(repr(items).encode()).hexdigest()
    return fn


def _lns_triton_int_dtype(bitwidth: int, tl):
    if bitwidth == 16:
        return tl.int16
    if bitwidth == 32:
        return tl.int32
    if bitwidth == 64:
        return tl.int64
    raise ValueError(f"LNS Triton backend does not support bitwidth {bitwidth}.")


def _lpvip_essential_zero(precision):
    scale = 1 << precision
    half_ulp = 0.5 / scale
    one_minus_power = -math.expm1(-math.log(2.0) * half_ulp)
    boundary = -math.log2(one_minus_power)
    return math.floor(scale * boundary)


def make_lpvip_triton_add(precision, zero_value, pos_inf_value, neg_inf_value, tab_sbdb=None, tab_ez=None):
    triton, tl = require_triton()
    from triton.language.extra import libdevice

    F = tl.constexpr(precision)
    SCALE = tl.constexpr(1 << precision)
    FRACTION_MASK = tl.constexpr((1 << precision) - 1)
    ESSZER = tl.constexpr(_lpvip_essential_zero(precision))
    LN2 = tl.constexpr(math.log(2.0))

    # Through F25, values remains within int32
    WORK_INT = tl.int32 if precision <= 25 else tl.int64

    ZERO = tl.constexpr(zero_value)
    POS_INF = tl.constexpr(pos_inf_value)
    NEG_INF = tl.constexpr(neg_inf_value)
    MIN_LOG = tl.constexpr(zero_value >> 1)
    MAX_LOG = tl.constexpr(pos_inf_value >> 1)
    MIN_FINITE_LOG = tl.constexpr((zero_value >> 1) + 1)
    MAX_FINITE_LOG = tl.constexpr((pos_inf_value >> 1) - 1)

    SAME_PRE_LIMIT = tl.constexpr(-(7 * (1 << precision) // 2))
    SAME_PRE_CAP = tl.constexpr(7 * (1 << precision) // 16)
    SAME_POST_FAR = tl.constexpr(-3 * (1 << precision))
    SAME_POST_NEAR = tl.constexpr(-(3 * (1 << precision) // 4))
    POST_UNIT = tl.constexpr((1 << precision) // 64)

    OPP_PRE_LIMIT = tl.constexpr(-2 * (1 << precision))
    OPP_PRE_FAR = tl.constexpr(5 * (1 << precision) // 8)
    OPP_PRE_OFFSET = tl.constexpr(9 * (1 << precision) // 8)

    @triton.jit
    def mitchell(w):
        w = tl.cast(w, WORK_INT)
        integer_part = w >> F
        fractional_part = w & FRACTION_MASK
        shift = -integer_part
        safe_shift = tl.minimum(shift, F)
        approximation = (tl.cast(SCALE, WORK_INT) + fractional_part) >> safe_shift
        return tl.where(shift > F, tl.cast(0, WORK_INT), approximation)

    if tab_sbdb is not None:

        DB_TABLE_SIZE = tl.constexpr(tab_sbdb.size(1))
        DB_TABLE_DATA_PTR = tl.constexpr(tab_sbdb.data_ptr())
        DB_TABLE_EZ = tl.constexpr(tab_ez.item())

        @triton.jit
        def db_correction(d, z, use_db):
            table_z = tl.maximum(-d, tl.cast(DB_TABLE_EZ, WORK_INT))
            index = 2 * DB_TABLE_SIZE + table_z
            table_ptr = tl.cast(DB_TABLE_DATA_PTR, tl.pointer_type(tl.int32))
            gaussian_db = tl.cast(tl.load(table_ptr + index, mask=use_db, other=0), WORK_INT) >> 1
            return -gaussian_db

        db_mode = "table"
        db_table_ptr = tab_sbdb.data_ptr()
        db_table_ez = tab_ez.item()

    else:
        @triton.jit
        def db_correction(d, z, use_db):
            d_real = tl.cast(d, tl.float64) / tl.cast(SCALE, tl.float64)
            magnitude = libdevice.expm1(d_real * tl.cast(LN2, tl.float64))
            result = libdevice.log(magnitude) / tl.cast(LN2, tl.float64)
            scaled = result * tl.cast(SCALE, tl.float64)
            biased = scaled + 0.5
            truncated = tl.where(biased >= 0, tl.floor(biased), tl.ceil(biased))
            return -tl.cast(truncated, WORK_INT) - z

        db_mode = "ideal"
        db_table_ptr = None
        db_table_ez = None

    @triton.jit
    def add(x, y):
        log_x = tl.cast(x >> 1, WORK_INT)
        log_y = tl.cast(y >> 1, WORK_INT)
        x_is_large = log_x >= log_y

        large_log = tl.where(x_is_large, log_x, log_y)
        large_sign = tl.cast(tl.where(x_is_large, x & 1, y & 1), WORK_INT)
        z = tl.where(x_is_large, log_y - log_x, log_x - log_y)
        distance = -z
        subtract = ((x ^ y) & 1) != 0

        same_pre = tl.where(z > SAME_PRE_LIMIT, (-z) >> 3, SAME_PRE_CAP)
        same_post = tl.where(
            z <= SAME_POST_FAR,
            tl.cast(0, WORK_INT),
            tl.where(z >= SAME_POST_NEAR, -POST_UNIT, POST_UNIT),
        )
        opposite_pre = tl.where(
            z < OPP_PRE_LIMIT,
            OPP_PRE_FAR,
            (z >> 2) + OPP_PRE_OFFSET,
        )

        pre = tl.where(subtract, opposite_pre, same_pre)
        use_mitchell = (subtract == 0) | (distance >= SCALE)
        mitch = mitchell(tl.where(use_mitchell, z + pre, tl.cast(0, WORK_INT)))
        same_adjustment = tl.where(
            z == 0,
            tl.cast(SCALE, WORK_INT),
            mitch + same_post,
        )

        # db_correction is evaluated eagerly by tl.where. Keep its argument positive
        # for exact cancellation, which is handled separately below.
        safe_distance = tl.maximum(distance, 1)
        use_db = subtract & (distance < SCALE) & (z != 0) & (x != ZERO) & (y != ZERO)
        correction = tl.where(distance >= SCALE, mitch, db_correction(safe_distance, z, use_db))
        opposite_adjustment = -correction

        adjustment = tl.where(subtract, opposite_adjustment, same_adjustment)
        adjustment = tl.where(z < -ESSZER, tl.cast(0, WORK_INT), adjustment)
        result_log = large_log + adjustment

        finite_log = tl.minimum(
            tl.maximum(result_log, tl.cast(MIN_FINITE_LOG, WORK_INT)),
            tl.cast(MAX_FINITE_LOG, WORK_INT),
        )
        packed = tl.cast((finite_log << 1) | large_sign, tl.int32)
        inf = tl.where(
            large_sign == 0,
            tl.cast(POS_INF, tl.int32),
            tl.cast(NEG_INF, tl.int32),
        )
        result = tl.where(
            result_log >= tl.cast(MAX_LOG, WORK_INT),
            inf,
            tl.where(
                result_log <= tl.cast(MIN_LOG, WORK_INT),
                tl.cast(ZERO, tl.int32),
                packed,
            ),
        )

        exact_cancellation = subtract & (z == 0)
        return tl.where(
            x == ZERO, y,
            tl.where(
                y == ZERO, x,
                tl.where(exact_cancellation, tl.cast(ZERO, tl.int32), result),
            ),
        )

    _bump_triton_jit_hash(
        mitchell, F=F, SCALE=SCALE, FRACTION_MASK=FRACTION_MASK, WORK_INT=WORK_INT
    )
    _bump_triton_jit_hash(
        db_correction,
        F=F,
        SCALE=SCALE,
        LN2=LN2,
        mode=db_mode,
        WORK_INT=WORK_INT,
        table=db_table_ptr,
        table_ez=db_table_ez,
    )
    _bump_triton_jit_hash(
        add,
        F=F,
        SCALE=SCALE,
        ESSZER=ESSZER,
        ZERO=ZERO,
        POS_INF=POS_INF,
        NEG_INF=NEG_INF,
        db_mode=db_mode,
        db_table=db_table_ptr,
        db_table_ez=db_table_ez,
        WORK_INT=WORK_INT,
    )
    return add


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

    tl_int_dtype = _lns_triton_int_dtype(bitwidth, tl)

    # log_base_value = torch.log(base).item()
    # log2_base_value = torch.log2(base).item()
    # # limit to lower precision for boundary cases
    # use_fast_from_float = abs(1.0 / log2_base_value) <= 512

    # LOG_BASE = tl.constexpr(log_base_value)
    # LOG2_BASE = tl.constexpr(log2_base_value)
    # NAN_VALUE = tl.constexpr(0)

    LOG_BASE = tl.constexpr(torch.log(base).item())
    ZERO = tl.constexpr(zero_value)
    POS_INF = tl.constexpr(pos_inf_value)
    NEG_INF = tl.constexpr(neg_inf_value)
    MIN_LOG = tl.constexpr(zero_value >> 1)
    MAX_LOG = tl.constexpr(pos_inf_value >> 1)
    MIN_FINITE_LOG = tl.constexpr((zero_value >> 1) + 1)
    MAX_FINITE_LOG = tl.constexpr((pos_inf_value >> 1) - 1)

    # @triton.jit
    # def from_float(x):
    #     abs_x = tl.abs(tl.cast(x, tl.float32))
    #     bits = tl.cast(abs_x, tl.int32, bitcast=True)
    #     exponent_bits = (bits >> 23) & 0xff
    #     subnormal = exponent_bits == 0

    #     normalized = tl.where(subnormal, abs_x * 16777216.0, abs_x)
    #     normalized_bits = tl.cast(normalized, tl.int32, bitcast=True)
    #     exponent = ((normalized_bits >> 23) & 0xff) - 127 - tl.where(subnormal, 24, 0)
    #     mantissa_bits = (normalized_bits & 0x7fffff) | 0x3f800000
    #     mantissa = tl.cast(mantissa_bits, tl.float32, bitcast=True)

    #     fraction = tl.log2(mantissa) / LOG2_BASE
    #     lower = tl.floor(fraction)
    #     boundary = fraction == lower + 0.5
    #     threshold = tl.exp2(
    #         (tl.cast(lower, tl.float64) + 0.5) * tl.cast(LOG2_BASE, tl.float64)
    #     )
    #     rounded_fraction = tl.where(
    #         boundary,
    #         lower + tl.cast(tl.cast(mantissa, tl.float64) >= threshold, tl.float32),
    #         tl.floor(fraction + 0.5),
    #     )
    #     rounded = tl.cast(exponent, tl.float32) / LOG2_BASE + rounded_fraction

    #     sign_bit = tl.cast(x < 0, tl_int_dtype)
    #     finite_rounded = tl.minimum(
    #         tl.maximum(rounded, tl.cast(MIN_FINITE_LOG, tl.float32)),
    #         tl.cast(MAX_FINITE_LOG, tl.float32),
    #     )
    #     packed = (tl.cast(finite_rounded, tl_int_dtype) << 1) | sign_bit
    #     overflow = rounded >= tl.cast(MAX_LOG, tl.float32)
    #     underflow = rounded <= tl.cast(MIN_LOG, tl.float32)
    #     inf = tl.where(
    #         sign_bit == 0,
    #         tl.cast(POS_INF, tl_int_dtype),
    #         tl.cast(NEG_INF, tl_int_dtype),
    #     )
    #     result = tl.where(
    #         x == 0.0,
    #         tl.cast(ZERO, tl_int_dtype),
    #         tl.where(overflow, inf, tl.where(underflow, tl.cast(ZERO, tl_int_dtype), packed)),
    #     )
    #     return tl.where(
    #         x != x,
    #         tl.cast(NAN_VALUE, tl_int_dtype),
    #         tl.where(exponent_bits == 0xff, inf, result),
    #     )

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
            table_ptr = tl.cast(tab_sbdb_data_ptr, tl.pointer_type(tl_int_dtype))
            sbdb = tl.load(table_ptr + idx)

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
    # _bump_triton_jit_hash(
    #     from_float,
    #     LOG_BASE=LOG_BASE,
    #     LOG2_BASE=LOG2_BASE,
    #     ZERO=ZERO,
    #     POS_INF=POS_INF,
    #     NEG_INF=NEG_INF,
    #     NAN_VALUE=NAN_VALUE,
    #     bitwidth=bitwidth,
    #     use_fast_from_float=use_fast_from_float,
    # )

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
