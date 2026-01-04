import torch
from torch import Tensor
from torchdt import DType

ZERO = torch.tensor(-32768, dtype=torch.int16) # smallest positive value in LNS
POS_INF = torch.tensor(32766, dtype=torch.int16) # largest positive value in LNS
NEG_INF = torch.tensor(32767, dtype=torch.int16) # largest negative value in LNS
base = 2.0 ** (2.0 ** torch.tensor(-10, dtype=torch.float64))

class LNS16(DType, bitwidth=16, cpp_backend="lns"):
    pass

@LNS16.register_op("from_float")
def lns16_from_float(ops, t: Tensor) -> Tensor:
    t = t.to(dtype=torch.float64)
    abs_t = torch.abs(t)

    log_t = torch.log(abs_t) / torch.log(base)
    # clamp to first 15 bits then bitshift to 16 bits
    packed = torch.round(log_t).to(torch.int16).clamp(-16384, 16383) << 1 | (t < 0)

    lns_t = torch.where(
        abs_t == 0, ZERO,
        torch.where(
            torch.isposinf(t), POS_INF,
            torch.where(
                torch.isneginf(t), NEG_INF,
                packed.to(torch.int16))))
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
                ZERO, max_operand + sbdb)))

@LNS16.register_op("sub")
def lns16_sub(ops, x, y):
    return ops.add(x, ops.neg(y))

@LNS16.register_op("mul")
def lns16_mul(ops, x, y):
    return torch.where(
        x == ZERO,
        ZERO, torch.where(
            y == ZERO,
            ZERO, (x + y - (y & 1)) ^ (y & 1)))

@LNS16.register_op("div")
def lns16_div(ops, x, y):
    return torch.where(
        x == ZERO,
        ZERO, torch.where(
            y == ZERO,
            torch.where(
                ops.gt(x, ZERO), POS_INF, NEG_INF),
                (x - y + (y & 1)) ^ (y & 1)))

@LNS16.register_op("pow")
def lns16_pow(ops, x, y):
    y_float = ops.to_float(y)
    return ((x & (-2)) * y_float).to(torch.int16) & (-2)

@LNS16.register_op("neg")
def lns16_neg(ops, x):
    return torch.where(x == ops.scalar_from_float(0.0), x, x ^ 1)

@LNS16.register_op("abs")
def lns16_abs(ops, x):
    return torch.where(x == ops.scalar_from_float(0.0), x, x & (-2)) # -2 is ~1

@LNS16.register_op("sign")
def lns16_sign(ops, x):
    return torch.where(
        x == ZERO, ZERO,
        torch.where(
            (x & 1) == 1,
            ops.scalar_from_float(-1.0),
            ops.scalar_from_float(1.0)))

@LNS16.register_op("ge")
def lns16_ge(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    result_both_pos = torch.ge(x_log, y_log)

    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.ge(y_log, x_log)

    return torch.where(both_pos, result_both_pos,
        torch.where(x_pos_y_neg, True,
        torch.where(x_neg_y_pos, False, result_both_neg)))

@LNS16.register_op("gt")
def lns16_gt(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    result_both_pos = torch.gt(x_log, y_log)

    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.gt(y_log, x_log)

    return torch.where(both_pos, result_both_pos,
        torch.where(x_pos_y_neg, True,
        torch.where(x_neg_y_pos, False, result_both_neg)))

@LNS16.register_op("le")
def lns16_le(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    result_both_pos = torch.le(x_log, y_log)

    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.le(y_log, x_log)

    return torch.where(both_pos, result_both_pos,
        torch.where(x_pos_y_neg, False,
        torch.where(x_neg_y_pos, True, result_both_neg)))

@LNS16.register_op("lt")
def lns16_lt(ops, x, y):
    x_log, y_log = x >> 1, y >> 1
    x_sign, y_sign = x & 1, y & 1

    both_pos = (x_sign == 0) & (y_sign == 0)
    result_both_pos = torch.lt(x_log, y_log)

    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.lt(y_log, x_log)

    return torch.where(both_pos, result_both_pos,
        torch.where(x_pos_y_neg, False,
        torch.where(x_neg_y_pos, True, result_both_neg)))