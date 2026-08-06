import torch
from torch import Tensor
from torchdt import DType
from ._tables import lns_base, load_or_create_table, register_table_add, validate_precision

ZERO = torch.tensor(-32768, dtype=torch.int16) # smallest positive value in LNS
POS_INF = torch.tensor(32766, dtype=torch.int16) # largest positive value in LNS
NEG_INF = torch.tensor(32767, dtype=torch.int16) # largest negative value in LNS
MIN_LOG = ZERO.item() >> 1
MAX_LOG = POS_INF.item() >> 1
MIN_FINITE_LOG = MIN_LOG + 1
MAX_FINITE_LOG = MAX_LOG - 1
precision = 7
base = lns_base(precision)
tab_sbdb = None
tab_ez = None
tab_exp = None

class LNS16(DType, bitwidth=16, cpp_backend="lns"):

    @staticmethod
    def set_prec(prec: int, table: bool = False, table_device: str = None, filestem: str = "tab"):
        global base, precision, tab_sbdb, tab_ez, tab_exp

        validate_precision(prec, table)
        precision = prec
        base = lns_base(precision)
        tab_exp = None
        LNS16.ops.clear_scalar_cache()

        if table:
            tab_sbdb, tab_ez = load_or_create_table(
                bitwidth=16,
                prec=prec,
                int_dtype=torch.int16,
                base=base,
                table_device=table_device,
                filestem=filestem,
            )
            tab_exp = _make_lns16_exp_lookup_table(device=table_device)
            register_table_add(LNS16, zero=ZERO, tab_sbdb=tab_sbdb, tab_ez=tab_ez)

            @LNS16.register_op("exp")
            def lns16_exp(ops, x):
                idx = (x.to(torch.int32) - torch.iinfo(torch.int16).min).to(torch.long)
                return tab_exp[idx]

    @classmethod
    def enable_triton(cls, accumulator: bool | str = False):
        from dataclasses import replace

        from torchdt.ops import TritonAccumulatorOps, register_triton_ops, require_triton
        from ._triton import (
            _bump_triton_jit_hash,
            _lns_triton_int_dtype,
            make_lns_triton_scalar_ops,
            make_lpvip_triton_add,
        )

        if accumulator is False:
            accumulator_mode = "native"
        elif accumulator is True or accumulator == "lns32":
            accumulator_mode = "lns32"
        elif accumulator == "lpvip":
            accumulator_mode = "lpvip"
        else:
            raise ValueError(
                "accumulator must be False, True, 'lns32', or 'lpvip'."
            )

        use_accumulator = accumulator_mode != "native"
        if use_accumulator:
            from . import lns32

        fingerprint = (
            precision,
            accumulator_mode,
            tab_sbdb.data_ptr() if tab_sbdb is not None else None,
            tab_ez.data_ptr() if tab_ez is not None else None,
            tab_exp.data_ptr() if tab_exp is not None else None,
            lns32.precision if use_accumulator else None,
            lns32.tab_sbdb.data_ptr()
            if use_accumulator and lns32.tab_sbdb is not None else None,
            lns32.tab_ez.data_ptr()
            if use_accumulator and lns32.tab_ez is not None else None,
        )
        if getattr(cls.ops, "_triton_fingerprint", None) == fingerprint:
            return

        triton, tl = require_triton()

        VALUE_PRECISION = tl.constexpr(precision)
        VALUE_ZERO = tl.constexpr(ZERO.item())
        VALUE_POS_INF = tl.constexpr(POS_INF.item())
        VALUE_NEG_INF = tl.constexpr(NEG_INF.item())
        VALUE_MIN_LOG = tl.constexpr(MIN_LOG)
        VALUE_MAX_LOG = tl.constexpr(MAX_LOG)
        VALUE_MIN_FINITE_LOG = tl.constexpr(MIN_FINITE_LOG)
        VALUE_MAX_FINITE_LOG = tl.constexpr(MAX_FINITE_LOG)

        accumulator_ops = None
        if use_accumulator:
            ACC_PRECISION = tl.constexpr(lns32.precision)
            ACC_ZERO = tl.constexpr(lns32.ZERO.item())
            ACC_POS_INF = tl.constexpr(lns32.POS_INF.item())
            ACC_NEG_INF = tl.constexpr(lns32.NEG_INF.item())
            ACC_MIN_LOG = tl.constexpr(lns32.MIN_LOG)
            ACC_MAX_LOG = tl.constexpr(lns32.MAX_LOG)
            ACC_MIN_FINITE_LOG = tl.constexpr(lns32.MIN_FINITE_LOG)
            ACC_MAX_FINITE_LOG = tl.constexpr(lns32.MAX_FINITE_LOG)
            TO_ACC_SHIFT = tl.constexpr(lns32.precision - precision)
            FROM_ACC_SHIFT = tl.constexpr(precision - lns32.precision)
            # Every finite LNS16 log fits LNS32 after an upshift of at most 16,
            # so this common embedding cannot overflow or underflow.
            EXACT_TO_ACC = tl.constexpr(0 <= lns32.precision - precision <= 16)

            @triton.jit
            def to_lns32(x):
                log_x = tl.cast(x >> 1, tl.int32)
                sign_bit = tl.cast(x & 1, tl.int32)
                if EXACT_TO_ACC:
                    rounded = log_x << TO_ACC_SHIFT
                    converted = (rounded << 1) | sign_bit
                else:
                    if TO_ACC_SHIFT > 0:
                        rounded = log_x << TO_ACC_SHIFT
                    elif TO_ACC_SHIFT < 0:
                        downshift = -TO_ACC_SHIFT
                        half = 1 << (downshift - 1)
                        magnitude = tl.abs(log_x)
                        rounded_magnitude = (magnitude + half) >> downshift
                        rounded = tl.where(log_x < 0, -rounded_magnitude, rounded_magnitude)
                    else:
                        rounded = log_x
                    finite_rounded = tl.minimum(
                        tl.maximum(rounded, tl.cast(ACC_MIN_FINITE_LOG, tl.int32)),
                        tl.cast(ACC_MAX_FINITE_LOG, tl.int32),
                    )
                    packed = (tl.cast(finite_rounded, tl.int32) << 1) | sign_bit
                    overflow = rounded >= tl.cast(ACC_MAX_LOG, tl.int32)
                    underflow = rounded <= tl.cast(ACC_MIN_LOG, tl.int32)
                    inf = tl.where(sign_bit == 0, tl.cast(ACC_POS_INF, tl.int32), tl.cast(ACC_NEG_INF, tl.int32))
                    converted = tl.where(overflow, inf, tl.where(underflow, tl.cast(ACC_ZERO, tl.int32), packed))
                return tl.where(
                    x == VALUE_ZERO,
                    tl.cast(ACC_ZERO, tl.int32),
                    tl.where(x == VALUE_POS_INF, tl.cast(ACC_POS_INF, tl.int32),
                             tl.where(x == VALUE_NEG_INF, tl.cast(ACC_NEG_INF, tl.int32), converted)),
                )

            @triton.jit
            def from_lns32(x):
                log_x = x >> 1
                sign_bit = tl.cast(x & 1, tl.int16)
                if FROM_ACC_SHIFT > 0:
                    rounded = log_x << FROM_ACC_SHIFT
                elif FROM_ACC_SHIFT < 0:
                    downshift = -FROM_ACC_SHIFT
                    half = 1 << (downshift - 1)
                    magnitude = tl.abs(log_x)
                    rounded_magnitude = (magnitude + half) >> downshift
                    rounded = tl.where(log_x < 0, -rounded_magnitude, rounded_magnitude)
                else:
                    rounded = log_x
                finite_rounded = tl.minimum(
                    tl.maximum(rounded, tl.cast(VALUE_MIN_FINITE_LOG, tl.int32)),
                    tl.cast(VALUE_MAX_FINITE_LOG, tl.int32),
                )
                packed = (tl.cast(finite_rounded, tl.int16) << 1) | sign_bit
                overflow = rounded >= tl.cast(VALUE_MAX_LOG, tl.int32)
                underflow = rounded <= tl.cast(VALUE_MIN_LOG, tl.int32)
                inf = tl.where(sign_bit == 0, tl.cast(VALUE_POS_INF, tl.int16), tl.cast(VALUE_NEG_INF, tl.int16))
                converted = tl.where(overflow, inf, tl.where(underflow, tl.cast(VALUE_ZERO, tl.int16), packed))
                return tl.where(
                    x == ACC_ZERO,
                    tl.cast(VALUE_ZERO, tl.int16),
                    tl.where(x == ACC_POS_INF, tl.cast(VALUE_POS_INF, tl.int16),
                             tl.where(x == ACC_NEG_INF, tl.cast(VALUE_NEG_INF, tl.int16), converted)),
                )

            _bump_triton_jit_hash(
                to_lns32,
                VALUE_PRECISION=VALUE_PRECISION,
                VALUE_ZERO=VALUE_ZERO,
                VALUE_POS_INF=VALUE_POS_INF,
                VALUE_NEG_INF=VALUE_NEG_INF,
                ACC_PRECISION=ACC_PRECISION,
                EXACT_TO_ACC=EXACT_TO_ACC,
                ACC_ZERO=ACC_ZERO,
                ACC_POS_INF=ACC_POS_INF,
                ACC_NEG_INF=ACC_NEG_INF,
            )
            _bump_triton_jit_hash(
                from_lns32,
                VALUE_PRECISION=VALUE_PRECISION,
                VALUE_ZERO=VALUE_ZERO,
                VALUE_POS_INF=VALUE_POS_INF,
                VALUE_NEG_INF=VALUE_NEG_INF,
                ACC_PRECISION=ACC_PRECISION,
                ACC_ZERO=ACC_ZERO,
                ACC_POS_INF=ACC_POS_INF,
                ACC_NEG_INF=ACC_NEG_INF,
            )

            accumulator_scalar_ops = make_lns_triton_scalar_ops(
                bitwidth=32,
                base=lns32.base,
                zero_value=lns32.ZERO.item(),
                pos_inf_value=lns32.POS_INF.item(),
                neg_inf_value=lns32.NEG_INF.item(),
                tab_sbdb=lns32.tab_sbdb if accumulator_mode == "lns32" else None,
                tab_ez=lns32.tab_ez if accumulator_mode == "lns32" else None,
            )
            if accumulator_mode == "lpvip":
                accumulator_scalar_ops = replace(
                    accumulator_scalar_ops,
                    add=make_lpvip_triton_add(
                        precision=lns32.precision,
                        zero_value=lns32.ZERO.item(),
                        pos_inf_value=lns32.POS_INF.item(),
                        neg_inf_value=lns32.NEG_INF.item(),
                        tab_sbdb=lns32.tab_sbdb,
                        tab_ez=lns32.tab_ez,
                    ),
                )

            accumulator_ops = TritonAccumulatorOps(
                int_dtype=torch.int32,
                scalar_ops=accumulator_scalar_ops,
                to_accumulator=to_lns32,
                from_accumulator=from_lns32,
            )

        scalar_ops = make_lns_triton_scalar_ops(
            bitwidth=16,
            base=base,
            zero_value=ZERO.item(),
            pos_inf_value=POS_INF.item(),
            neg_inf_value=NEG_INF.item(),
            tab_sbdb=tab_sbdb,
            tab_ez=tab_ez,
        )

        if tab_exp is not None:

            tl_int_dtype = _lns_triton_int_dtype(16, tl)
            EXP_TABLE_DATA_PTR = tl.constexpr(tab_exp.data_ptr())

            @triton.jit
            def exp(x):
                idx = tl.cast(x, tl.int64) - tl.cast(VALUE_ZERO, tl.int64)
                table_ptr = tl.cast(EXP_TABLE_DATA_PTR, tl.pointer_type(tl_int_dtype))
                return tl.load(table_ptr + idx)

            _bump_triton_jit_hash(
                exp,
                VALUE_PRECISION=VALUE_PRECISION,
                VALUE_ZERO=VALUE_ZERO,
                VALUE_POS_INF=VALUE_POS_INF,
                VALUE_NEG_INF=VALUE_NEG_INF,
                exp_table=tab_exp.data_ptr(),
            )

            scalar_ops = replace(scalar_ops, exp=exp)

        register_triton_ops(
            cls,
            scalar_ops,
            accumulator_ops=accumulator_ops,
        )
        cls.ops._triton_fingerprint = fingerprint

def _make_lns16_exp_lookup_table(device=None) -> Tensor:
    info = torch.iinfo(torch.int16)
    codes = torch.arange(info.min, info.max + 1, dtype=torch.int32).to(torch.int16)
    exp_values = torch.exp(lns16_to_float(None, codes))
    table = lns16_from_float(None, exp_values).contiguous()
    if device is not None:
        table = table.to(device=device)
    return table

def _checked_add(x: Tensor, y: Tensor, overflow_sign: Tensor) -> Tensor:
    result = (x + y).to(torch.int16)

    zero = ZERO.to(device=result.device)
    pos_inf = POS_INF.to(device=result.device)
    neg_inf = NEG_INF.to(device=result.device)

    underflow = (x < 0) & (y < 0) & (result >= 0)
    overflow = (x >= 0) & (y >= 0) & (result < 0)
    inf_signed = torch.where((overflow_sign & 1) == 0, pos_inf, neg_inf)

    return torch.where(underflow, zero, torch.where(overflow, inf_signed, result))

@LNS16.register_op("from_float")
def lns16_from_float(ops, t: Tensor) -> Tensor:
    t = t.to(dtype=torch.float64)
    abs_t = torch.abs(t)

    log_t = torch.log(abs_t) / torch.log(base)
    rounded = torch.round(log_t)
    sign_bit = (t < 0).to(torch.int16)
    finite_rounded = rounded.clamp(MIN_FINITE_LOG, MAX_FINITE_LOG)
    packed = (finite_rounded.to(torch.int16) << 1) | sign_bit

    zero = ZERO.to(device=t.device)
    pos_inf = POS_INF.to(device=t.device)
    neg_inf = NEG_INF.to(device=t.device)
    overflow = rounded >= MAX_LOG
    underflow = rounded <= MIN_LOG
    inf = torch.where(sign_bit == 0, pos_inf, neg_inf)

    lns_t = torch.where(
        abs_t == 0,
        zero,
        torch.where(overflow, inf, torch.where(underflow, zero, packed.to(torch.int16))))
    return lns_t

@LNS16.register_op("to_float")
def lns16_to_float(ops, t: Tensor) -> Tensor:
    packed = t.view(torch.int16)
    log_t = (packed >> 1)
    sign_t = torch.where((packed & 1) == 1, -1.0, 1.0).to(torch.float64)

    abs_t = torch.pow(base, log_t)
    float_t = sign_t * abs_t

    float_t = torch.where(
        packed == ZERO, 0.0,
        torch.where(
            packed == POS_INF, float('inf'),
            torch.where(
                packed == NEG_INF, float('-inf'),
                float_t)))
    return float_t.to(torch.float64)

@LNS16.register_op("add")
def lns16_add(ops, x, y):
    max_operand = torch.max(x, y)

    abs_diff = torch.abs((x >> 1) - (y >> 1))
    sign_diff = (x ^ y) & 1

    power_term = torch.pow(base, -abs_diff)
    magnitude = torch.abs(1.0 - 2.0 * sign_diff + power_term)

    log_term = torch.log(magnitude) / torch.log(base)
    sbdb = torch.round(log_term).to(torch.int16) << 1

    return torch.where(
        x == ZERO,
        y, torch.where(
            y == ZERO,
            x, torch.where(
                x == ops.neg(y),
                ZERO, _checked_add(max_operand, sbdb, max_operand & 1))))

@LNS16.register_op("sub")
def lns16_sub(ops, x, y):
    return ops.add(x, ops.neg(y))

@LNS16.register_op("mul")
def lns16_mul(ops, x, y):
    zero = ZERO.to(device=x.device)
    prod_unsigned = _checked_add(x, y - (y & 1), x & 1)
    prod = torch.where(prod_unsigned == zero, zero, prod_unsigned ^ (y & 1))

    return torch.where(
        x == zero,
        zero, torch.where(
            y == zero,
            zero, prod))

@LNS16.register_op("div")
def lns16_div(ops, x, y):
    zero = ZERO.to(device=x.device)
    pos_inf = POS_INF.to(device=x.device)
    neg_inf = NEG_INF.to(device=x.device)
    safe_y = torch.where(y == zero, torch.zeros((), dtype=torch.int16, device=y.device), y)
    quotient_unsigned = _checked_add(x, -safe_y + (safe_y & 1), x & 1)
    quotient = torch.where(quotient_unsigned == zero, zero, quotient_unsigned ^ (safe_y & 1))
    div_by_zero = torch.where((x & 1) == 0, pos_inf, neg_inf)

    return torch.where(
        x == zero,
        zero, torch.where(
            y == zero,
            div_by_zero,
            quotient))

@LNS16.register_op("sqrt")
def lns16_sqrt(ops, x):
    zero = ZERO.to(device=x.device)
    pos_inf = POS_INF.to(device=x.device)

    result = ((x & (-2)) // 2) & (-2)
    return torch.where(
        x == zero,
        zero, torch.where(
            x == pos_inf,
            pos_inf,
            result
        )
    )

# todo: add support for negative bases
@LNS16.register_op("pow")
def lns16_pow(ops, x, y):
    zero = ZERO.to(device=x.device)
    pos_inf = POS_INF.to(device=x.device)

    y_float = ops.to_float(y).to(torch.float64)
    x_log = (x >> 1).to(torch.float64)

    scaled_log = x_log * y_float
    scaled_log = torch.where(
        x_log == 0,
        torch.zeros_like(scaled_log),
        scaled_log,
    )

    rounded_log = torch.round(scaled_log)
    finite_log = rounded_log.clamp(
        MIN_FINITE_LOG,
        MAX_FINITE_LOG,
    )
    finite_result = finite_log.to(torch.int16) << 1

    result = torch.where(
        rounded_log <= MIN_LOG,
        zero,
        torch.where(
            rounded_log >= MAX_LOG,
            pos_inf,
            finite_result,
        ),
    )

    # Handle zero and infinity
    result = torch.where(
        x == zero,
        torch.where(y_float < 0, pos_inf, zero),
        result,
    )
    result = torch.where(
        x == pos_inf,
        torch.where(y_float < 0, zero, pos_inf),
        result,
    )

    # Includes 0**0 and inf**0, consistent with torch.pow.
    return torch.where(y == zero, 0, result)

@LNS16.register_op("neg")
def lns16_neg(ops, x):
    return torch.where(x == ops.scalar_from_float(0.0, device=x.device), x, x ^ 1)

@LNS16.register_op("abs")
def lns16_abs(ops, x):
    return torch.where(x == ops.scalar_from_float(0.0, device=x.device), x, x & (-2)) # -2 is ~1

@LNS16.register_op("sign")
def lns16_sign(ops, x):
    return torch.where(
        x == ZERO, ZERO,
        torch.where(
            (x & 1) == 1,
            ops.scalar_from_float(-1.0, device=x.device),
            ops.scalar_from_float(1.0, device=x.device)))

@LNS16.register_op("ge")
def lns16_ge(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    both_neg = (x_sign == 1) & (y_sign == 1)

    return x_pos_y_neg | (both_pos & (x_log >= y_log)) | (both_neg & (y_log >= x_log))

@LNS16.register_op("gt")
def lns16_gt(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    both_neg = (x_sign == 1) & (y_sign == 1)

    return x_pos_y_neg | (both_pos & (x_log > y_log)) | (both_neg & (y_log > x_log))

@LNS16.register_op("le")
def lns16_le(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
    both_neg = (x_sign == 1) & (y_sign == 1)

    return x_neg_y_pos | (both_pos & (x_log <= y_log)) | (both_neg & (y_log <= x_log))

@LNS16.register_op("lt")
def lns16_lt(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
    both_neg = (x_sign == 1) & (y_sign == 1)

    return x_neg_y_pos | (both_pos & (x_log < y_log)) | (both_neg & (y_log < x_log))
