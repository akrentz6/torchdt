import torch
from torch import Tensor
from torchdt import DType
from ._tables import lns_base, load_or_create_table, register_table_add, validate_precision

ZERO = torch.tensor(-9_223_372_036_854_775_808, dtype=torch.int64) # smallest positive value in LNS
POS_INF = torch.tensor(9_223_372_036_854_775_806, dtype=torch.int64) # largest positive value in LNS
NEG_INF = torch.tensor(9_223_372_036_854_775_807, dtype=torch.int64) # largest negative value in LNS
MIN_LOG = ZERO.item() >> 1
MAX_LOG = POS_INF.item() >> 1
MIN_FINITE_LOG = MIN_LOG + 1
MAX_FINITE_LOG = MAX_LOG - 1
precision = 23
base = lns_base(precision)
tab_sbdb = None
tab_ez = None

class LNS64(DType, bitwidth=64):

    @staticmethod
    def set_prec(prec: int, table: bool = False, table_device: str = None, filestem: str = "tab"):
        global base, precision, tab_sbdb, tab_ez

        validate_precision(prec, table)
        precision = prec
        base = lns_base(precision)

        if table:
            tab_sbdb, tab_ez = load_or_create_table(
                bitwidth=64,
                prec=prec,
                int_dtype=torch.int64,
                base=base,
                table_device=table_device,
                filestem=filestem,
            )
            register_table_add(LNS64, zero=ZERO, tab_sbdb=tab_sbdb, tab_ez=tab_ez)

    @classmethod
    def enable_triton(cls):
        from ._triton import enable_lns_triton_backend

        enable_lns_triton_backend(
            cls,
            base=base,
            zero_value=ZERO.item(),
            pos_inf_value=POS_INF.item(),
            neg_inf_value=NEG_INF.item(),
            tab_sbdb=tab_sbdb,
            tab_ez=tab_ez,
        )

def _checked_add(x: Tensor, y: Tensor, overflow_sign: Tensor) -> Tensor:
    result = (x + y).to(torch.int64)

    zero = ZERO.to(device=result.device)
    pos_inf = POS_INF.to(device=result.device)
    neg_inf = NEG_INF.to(device=result.device)

    underflow = (x < 0) & (y < 0) & (result >= 0)
    overflow = (x >= 0) & (y >= 0) & (result < 0)
    inf_signed = torch.where((overflow_sign & 1) == 0, pos_inf, neg_inf)

    return torch.where(underflow, zero, torch.where(overflow, inf_signed, result))

@LNS64.register_op("from_float")
def lns64_from_float(ops, t: Tensor) -> Tensor:
    t = t.to(dtype=torch.float64)
    abs_t = torch.abs(t)

    log_t = torch.log(abs_t) / torch.log(base)
    rounded = torch.round(log_t)
    sign_bit = (t < 0).to(torch.int64)
    finite_rounded = rounded.clamp(MIN_FINITE_LOG, MAX_FINITE_LOG)
    packed = (finite_rounded.to(torch.int64) << 1) | sign_bit

    zero = ZERO.to(device=t.device)
    pos_inf = POS_INF.to(device=t.device)
    neg_inf = NEG_INF.to(device=t.device)
    overflow = rounded >= MAX_LOG
    underflow = rounded <= MIN_LOG
    inf = torch.where(sign_bit == 0, pos_inf, neg_inf)

    lns_t = torch.where(
        abs_t == 0,
        zero,
        torch.where(overflow, inf, torch.where(underflow, zero, packed.to(torch.int64))))
    return lns_t

@LNS64.register_op("to_float")
def lns64_to_float(ops, t: Tensor) -> Tensor:
    packed = t.view(torch.int64)
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

@LNS64.register_op("add")
def lns64_add(ops, x, y):
    max_operand = torch.max(x, y)

    abs_diff = torch.abs((x >> 1) - (y >> 1))
    sign_diff = (x ^ y) & 1

    power_term = torch.pow(base, -abs_diff)
    magnitude = torch.abs(1.0 - 2.0 * sign_diff + power_term)

    log_term = torch.log(magnitude) / torch.log(base)
    sbdb = torch.round(log_term).to(torch.int64) << 1

    return torch.where(
        x == ZERO,
        y, torch.where(
            y == ZERO,
            x, torch.where(
                x == ops.neg(y),
                ZERO, _checked_add(max_operand, sbdb, max_operand & 1))))

@LNS64.register_op("sub")
def lns64_sub(ops, x, y):
    return ops.add(x, ops.neg(y))

@LNS64.register_op("mul")
def lns64_mul(ops, x, y):
    zero = ZERO.to(device=x.device)
    prod_unsigned = _checked_add(x, y - (y & 1), x & 1)
    prod = torch.where(prod_unsigned == zero, zero, prod_unsigned ^ (y & 1))

    return torch.where(
        x == zero,
        zero, torch.where(
            y == zero,
            zero, prod))

@LNS64.register_op("div")
def lns64_div(ops, x, y):
    zero = ZERO.to(device=x.device)
    pos_inf = POS_INF.to(device=x.device)
    neg_inf = NEG_INF.to(device=x.device)
    safe_y = torch.where(y == zero, torch.zeros((), dtype=torch.int64, device=y.device), y)
    quotient_unsigned = _checked_add(x, -safe_y + (safe_y & 1), x & 1)
    quotient = torch.where(quotient_unsigned == zero, zero, quotient_unsigned ^ (safe_y & 1))
    div_by_zero = torch.where((x & 1) == 0, pos_inf, neg_inf)

    return torch.where(
        x == zero,
        zero, torch.where(
            y == zero,
            div_by_zero,
            quotient))

@LNS64.register_op("sqrt")
def lns64_sqrt(ops, x):
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

@LNS64.register_op("pow")
def lns64_pow(ops, x, y):
    y_float = ops.to_float(y)
    return ((x & (-2)) * y_float).to(torch.int64) & (-2)

@LNS64.register_op("neg")
def lns64_neg(ops, x):
    return torch.where(x == ops.scalar_from_float(0.0), x, x ^ 1)

@LNS64.register_op("abs")
def lns64_abs(ops, x):
    return torch.where(x == ops.scalar_from_float(0.0), x, x & (-2)) # -2 is ~1

@LNS64.register_op("sign")
def lns64_sign(ops, x):
    return torch.where(
        x == ZERO, ZERO,
        torch.where(
            (x & 1) == 1,
            ops.scalar_from_float(-1.0),
            ops.scalar_from_float(1.0)))

@LNS64.register_op("ge")
def lns64_ge(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    both_neg = (x_sign == 1) & (y_sign == 1)

    return x_pos_y_neg | (both_pos & (x_log >= y_log)) | (both_neg & (y_log >= x_log))

@LNS64.register_op("gt")
def lns64_gt(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    both_neg = (x_sign == 1) & (y_sign == 1)

    return x_pos_y_neg | (both_pos & (x_log > y_log)) | (both_neg & (y_log > x_log))

@LNS64.register_op("le")
def lns64_le(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
    both_neg = (x_sign == 1) & (y_sign == 1)

    return x_neg_y_pos | (both_pos & (x_log <= y_log)) | (both_neg & (y_log <= x_log))

@LNS64.register_op("lt")
def lns64_lt(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)
    both_neg = (x_sign == 1) & (y_sign == 1)

    return x_neg_y_pos | (both_pos & (x_log < y_log)) | (both_neg & (y_log < x_log))
