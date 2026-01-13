import torch
from torch import Tensor
from torchdt import DType
from torchdt.ops import register_triton_ops
import numpy as np
import os
import textwrap
import linecache

ZERO = torch.tensor(-9_223_372_036_854_775_807, dtype=torch.int64) # smallest positive value in LNS
POS_INF = torch.tensor(9_223_372_036_854_775_806, dtype=torch.int64) # largest positive value in LNS
NEG_INF = torch.tensor(9_223_372_036_854_775_807, dtype=torch.int64) # largest negative value in LNS
base = 2.0 ** (2.0 ** torch.tensor(-23, dtype=torch.float64))
tab_sbdb = None
tab_ez = None

def sbdb_ideal(z, s):
    power_term = torch.pow(base, z)
    magnitude = torch.abs(1.0 - 2.0 * s + power_term)

    log_term = torch.log(magnitude) / torch.log(base)
    result = torch.round(log_term).to(torch.int64) << 1

    return result

class LNS64(DType, bitwidth=64):

    @staticmethod
    def enable_triton():
        lns64_register_triton_ops()

    @staticmethod
    def set_prec(prec: int, table: bool = False, table_device: str = None, filestem: str = "tab"):
        global base

        if prec < 1 or prec > 50:
            raise ValueError("Precision must be between 1 and 52")
        if table and prec > 20:
            raise ValueError("Table-based LNS only supports precision up to 20")

        base = 2.0 ** (2.0 ** torch.tensor(-prec, dtype=torch.float64))

        if table:
            global tab_sbdb, tab_ez
            filename = f"./{filestem}_{prec}.npz"

            if os.path.isfile(filename):
                data = np.load(filename)
                tab_sbdb = torch.tensor(data["tab_sbdb"], dtype=torch.int64, device=table_device).contiguous()
                tab_ez = torch.tensor(data["tab_ez"], dtype=torch.int64, device=table_device)
                data.close()

            else:
                zero = torch.tensor(0, dtype=torch.int64, device=table_device)
                one = torch.tensor(1, dtype=torch.int64, device=table_device)

                tab_ez = sbdb_ideal(one, one)

                zrange = torch.arange(tab_ez, 0, dtype=torch.int64, device=table_device)
                sbt = sbdb_ideal(zrange, zero)
                dbt = sbdb_ideal(zrange, one)
                tab_sbdb = torch.vstack((sbt, dbt)).contiguous()

                np.savez(filename, tab_ez=tab_ez.cpu().numpy(), tab_sbdb=tab_sbdb.cpu().numpy())

            @LNS64.register_op("add")
            def lns64_add_table(ops, x, y):
                max_operand = torch.maximum(x, y)

                z = -torch.abs((x >> 1) - (y >> 1))
                s = (x ^ y) & 1

                sbdb = tab_sbdb[s, torch.maximum(tab_ez, torch.where(z == 0, -1, z))]
                return torch.where(
                    x == ZERO,
                    y, torch.where(
                        y == ZERO,
                        x, torch.where(
                            x == ops.neg(y),
                            ZERO, max_operand + sbdb)))

@LNS64.register_op("from_float")
def lns64_from_float(ops, t: Tensor) -> Tensor:
    t = t.to(dtype=torch.float64)
    abs_t = torch.abs(t)

    log_t = torch.log(abs_t) / torch.log(base)
    # clamp to first 63 bits then bitshift to 64 bits
    packed = torch.round(log_t).to(torch.int64).clamp(-4_611_686_018_427_387_904, 4_611_686_018_427_387_903) << 1 | (t < 0)

    lns_t = torch.where(
        abs_t == 0, ZERO,
        torch.where(
            torch.isposinf(t), POS_INF,
            torch.where(
                torch.isneginf(t), NEG_INF,
                packed.to(torch.int64))))
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
    return float_t.to(torch.float32)

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
                ZERO, max_operand + sbdb)))

@LNS64.register_op("sub")
def lns64_sub(ops, x, y):
    return ops.add(x, ops.neg(y))

@LNS64.register_op("mul")
def lns64_mul(ops, x, y):
    return torch.where(
        x == ZERO,
        ZERO, torch.where(
            y == ZERO,
            ZERO, (x + y - (y & 1)) ^ (y & 1)))

@LNS64.register_op("div")
def lns64_div(ops, x, y):
    return torch.where(
        x == ZERO,
        ZERO, torch.where(
            y == ZERO,
            torch.where(
                ops.gt(x, ZERO), POS_INF, NEG_INF),
                (x - y + (y & 1)) ^ (y & 1)))

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
    result_both_pos = torch.ge(x_log, y_log)

    x_pos_y_neg = (x_sign == 0) & (y_sign == 1)
    x_neg_y_pos = (x_sign == 1) & (y_sign == 0)

    # no need to check explicitly for both negative case, as it's the final case
    result_both_neg = torch.ge(y_log, x_log)

    return torch.where(both_pos, result_both_pos,
        torch.where(x_pos_y_neg, True,
        torch.where(x_neg_y_pos, False, result_both_neg)))

@LNS64.register_op("gt")
def lns64_gt(ops, x, y):
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

@LNS64.register_op("le")
def lns64_le(ops, x, y):
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

@LNS64.register_op("lt")
def lns64_lt(ops, x, y):
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

def lns64_register_triton_ops():
    import triton
    import triton.language as tl

    _LOG_BASE = torch.log(base).item()
    _ZERO = ZERO.item()
    _ONE = 0
    _NEG_INF = 9_223_372_036_854_775_807
    _POS_INF = 9_223_372_036_854_775_806

    src=f"""
import triton
import triton.language as tl

@triton.jit
def from_float(x):
    abs_x = tl.abs(tl.cast(x, tl.float64))
    log_x = tl.log(abs_x) / tl.cast({_LOG_BASE}, tl.float64)

    rounded = tl.where(log_x >= 0, tl.floor(log_x + 0.5), tl.ceil(log_x - 0.5))
    sign_bit = tl.cast(x < 0, tl.int64)
    packed = (tl.cast(rounded, tl.int64) << 1) | sign_bit

    return tl.where(x == 0.0, {_ZERO}, packed)

@triton.jit
def to_float(x):
    log_x = x >> 1
    sign = tl.where((x & 1) == 1, -1.0, 1.0)

    abs_x = tl.exp(tl.cast({_LOG_BASE}, tl.float64) * tl.cast(log_x, tl.float64))
    float_x = sign * abs_x

    return tl.where(x == {_ZERO}, 0.0, float_x.to(tl.float32))

@triton.jit
def mul(x, y):
    prod = (x + y - (y & 1)) ^ (y & 1)
    return tl.where(x == {_ZERO}, {_ZERO}, tl.where(y == {_ZERO}, {_ZERO}, prod))

@triton.jit
def div(x, y):
    div = (x - y + (y & 1)) ^ (y & 1)
    return tl.where(x == {_ZERO}, {_ZERO}, tl.where(y == {_ZERO}, {_POS_INF}, div))

@triton.jit
def sqrt(x):
    result = ((x & (-2)) // 2) & (-2)
    return tl.where(x == {_ZERO}, {_ZERO}, result)

@triton.jit
def neg(x):
    return tl.where(x == {_ZERO}, {_ZERO}, x ^ 1)
"""

    if tab_sbdb is not None and tab_ez is not None:
        src += f"""
@triton.jit
def add(x, y):
    max_operand = tl.maximum(x, y)

    z = -tl.abs((x >> 1) - (y >> 1)).to(tl.int64)
    s = ((x ^ y) & 1).to(tl.int64)

    idx = (s + 1) * {tab_sbdb.size(1)} + tl.where(z < {tab_ez.item()}, {tab_ez.item()}, tl.where(z == 0, -1, z))
    abs_ptr = {tab_sbdb.data_ptr()} + idx * 8 # int64 has 8 bytes

    # Using tl.load directly is impossible because tab_sbdb_ptr
    # is treated as a constant by triton.jit, not a pointer object.
    sbdb = tl.inline_asm_elementwise(
        '''
        {{
            ld.global.b64 $0, [$1];
        }}
        ''',
        "=l, l",
        [abs_ptr],
        dtype=tl.int64,
        is_pure=True,
        pack=1,
    )

    result = max_operand + sbdb
    return tl.where(x == {_ZERO}, y, tl.where(y == {_ZERO}, x, tl.where(x == neg(y), {_ZERO}, result)))
"""

    else:
        src += f"""
@triton.jit
def add(x, y):
    max_operand = tl.maximum(x, y)

    abs_diff = tl.abs((x >> 1) - (y >> 1)).to(tl.float64)
    sign_diff = ((x ^ y) & 1).to(tl.float64)

    power_term = tl.exp({_LOG_BASE} * -abs_diff)
    magnitude = tl.abs(1.0 - 2.0 * sign_diff + power_term)

    log_term = tl.log(magnitude) / {_LOG_BASE}
    rounded = tl.where(log_term >= 0, tl.floor(log_term + 0.5), tl.ceil(log_term - 0.5))
    sbdb = rounded.to(tl.int64) * 2

    result = max_operand + sbdb
    return tl.where(x == {_ZERO}, y, tl.where(y == {_ZERO}, x, tl.where(x == (y ^ 1), {_ZERO}, result)))
"""

    src = textwrap.dedent(src)
    filename = f"<triton_kernels>"
    codeobj = compile(src, filename, "exec")
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)

    ns = {"__name__": filename}
    exec(codeobj, ns, ns)

    from_float = ns["from_float"]
    to_float = ns["to_float"]
    add = ns["add"]
    mul = ns["mul"]
    div = ns["div"]
    sqrt = ns["sqrt"]
    neg = ns["neg"]

    @triton.jit
    def sub(x, y):
        return add(x, neg(y))

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

    register_triton_ops(
        LNS64, from_float, to_float,
        add, sub, mul, div, sqrt,
        gt, ge, lt, le, neg,
        tl.constexpr(_ZERO), tl.constexpr(_NEG_INF), tl.constexpr(_ONE)
    )