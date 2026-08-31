from __future__ import annotations

import torch
from torch import Tensor

from torchdt import DType


_LIMB_BITS = 30
_LIMB_BASE = 1 << _LIMB_BITS
_LIMB_MASK = _LIMB_BASE - 1
_LIMBS = 6
_WIDE_BITS = _LIMB_BITS * _LIMBS


def _wide_zero_like(x: Tensor) -> Tensor:
    return torch.zeros((*x.shape, _LIMBS), dtype=torch.int64, device=x.device)


def _wide_from_int(x: Tensor) -> Tensor:
    x = x.to(torch.int64)
    return torch.stack(
        [(x >> (i * _LIMB_BITS)) & _LIMB_MASK for i in range(_LIMBS)],
        dim=-1,
    )


def _wide_is_zero(x: Tensor) -> Tensor:
    return torch.all(x == 0, dim=-1)


def _wide_select(mask: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return torch.where(mask.unsqueeze(-1), x, y)


def _wide_add(x: Tensor, y: Tensor) -> Tensor:
    limbs = []
    carry = torch.zeros_like(x[..., 0])
    for i in range(_LIMBS):
        value = x[..., i] + y[..., i] + carry
        limbs.append(value & _LIMB_MASK)
        carry = value >> _LIMB_BITS
    return torch.stack(limbs, dim=-1)


def _wide_sub(x: Tensor, y: Tensor) -> Tensor:
    """Return x-y for unsigned wide integers with x >= y."""
    limbs = []
    borrow = torch.zeros_like(x[..., 0])
    for i in range(_LIMBS):
        value = x[..., i] - y[..., i] - borrow
        borrow = (value < 0).to(torch.int64)
        limbs.append(torch.where(value < 0, value + _LIMB_BASE, value))
    return torch.stack(limbs, dim=-1)


def _wide_ge(x: Tensor, y: Tensor) -> Tensor:
    greater = torch.zeros_like(x[..., 0], dtype=torch.bool)
    equal = torch.ones_like(greater)
    for i in range(_LIMBS - 1, -1, -1):
        greater |= equal & (x[..., i] > y[..., i])
        equal &= x[..., i] == y[..., i]
    return greater | equal


def _wide_shl_one(x: Tensor) -> Tensor:
    limbs = []
    carry = torch.zeros_like(x[..., 0])
    for i in range(_LIMBS):
        value = (x[..., i] << 1) | carry
        limbs.append(value & _LIMB_MASK)
        carry = value >> _LIMB_BITS
    return torch.stack(limbs, dim=-1)


def _wide_shr_one(x: Tensor) -> tuple[Tensor, Tensor]:
    limbs = [None] * _LIMBS
    carry = torch.zeros_like(x[..., 0])
    for i in range(_LIMBS - 1, -1, -1):
        value = x[..., i]
        limbs[i] = (value >> 1) | (carry << (_LIMB_BITS - 1))
        carry = value & 1
    return torch.stack(limbs, dim=-1), carry


def _wide_shl(x: Tensor, amount: int) -> Tensor:
    for _ in range(amount):
        x = _wide_shl_one(x)
    return x


def _wide_shr_jam(x: Tensor, amount: Tensor, steps: int = _WIDE_BITS) -> Tensor:
    amount = amount.clamp(0, steps)
    sticky = torch.zeros_like(x[..., 0], dtype=torch.bool)
    for i in range(steps):
        shifted, dropped = _wide_shr_one(x)
        active = amount > i
        sticky |= active & (dropped != 0)
        x = _wide_select(active, shifted, x)
    low = x[..., 0] | sticky.to(torch.int64)
    return torch.cat((low.unsqueeze(-1), x[..., 1:]), dim=-1)


def _wide_bit(x: Tensor, index: Tensor | int) -> Tensor:
    if not isinstance(index, Tensor):
        index = torch.full_like(x[..., 0], index)
    valid = (index >= 0) & (index < _WIDE_BITS)
    safe = index.clamp(0, _WIDE_BITS - 1)
    limb = torch.div(safe, _LIMB_BITS, rounding_mode="floor")
    offset = safe - limb * _LIMB_BITS
    value = torch.gather(x, -1, limb.unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, (value >> offset) & 1, 0)


def _wide_set_bit(x: Tensor, index: int, bit: Tensor) -> Tensor:
    limb, offset = divmod(index, _LIMB_BITS)
    parts = [x[..., i] for i in range(_LIMBS)]
    parts[limb] = parts[limb] | (bit.to(torch.int64) << offset)
    return torch.stack(parts, dim=-1)


def _wide_bit_length(x: Tensor) -> Tensor:
    length = torch.zeros_like(x[..., 0])
    for i in range(_WIDE_BITS):
        length = torch.where(_wide_bit(x, i) != 0, i + 1, length)
    return length


def _wide_any_below(x: Tensor, index: Tensor) -> Tensor:
    result = torch.zeros_like(index, dtype=torch.bool)
    for i in range(_LIMBS):
        low = i * _LIMB_BITS
        count = (index - low).clamp(0, _LIMB_BITS)
        mask = (torch.ones_like(count) << count) - 1
        result |= (x[..., i] & mask) != 0
    return result


def _wide_mul(x: Tensor, y: Tensor, input_limbs: int = 3) -> Tensor:
    accum = [torch.zeros_like(x[..., 0]) for _ in range(_LIMBS)]
    for i in range(input_limbs):
        for j in range(input_limbs):
            if i + j < _LIMBS:
                accum[i + j] = accum[i + j] + x[..., i] * y[..., j]

    carry = torch.zeros_like(x[..., 0])
    limbs = []
    for value in accum:
        value = value + carry
        limbs.append(value & _LIMB_MASK)
        carry = value >> _LIMB_BITS
    return torch.stack(limbs, dim=-1)


def _wide_divmod(numerator: Tensor, denominator: Tensor, top_bit: int) -> tuple[Tensor, Tensor]:
    quotient = _wide_zero_like(numerator[..., 0])
    remainder = _wide_zero_like(numerator[..., 0])
    for i in range(top_bit, -1, -1):
        remainder = _wide_shl_one(remainder)
        remainder = _wide_set_bit(remainder, 0, _wide_bit(numerator, i))
        take = _wide_ge(remainder, denominator)
        remainder = _wide_select(take, _wide_sub(remainder, denominator), remainder)
        quotient = _wide_set_bit(quotient, i, take)
    return quotient, remainder


def _wide_isqrt(value: Tensor, top_pair: int) -> tuple[Tensor, Tensor]:
    root = _wide_zero_like(value[..., 0])
    remainder = _wide_zero_like(value[..., 0])
    for pair in range(top_pair, -1, -1):
        remainder = _wide_shl(_wide_shl_one(remainder), 1)
        pair_bits = (_wide_bit(value, 2 * pair + 1) << 1) | _wide_bit(value, 2 * pair)
        remainder = _wide_set_bit(remainder, 0, pair_bits & 1)
        remainder = _wide_set_bit(remainder, 1, pair_bits >> 1)

        trial = _wide_shl(root, 2)
        trial = _wide_set_bit(trial, 0, torch.ones_like(pair_bits))
        take = _wide_ge(remainder, trial)
        remainder = _wide_select(take, _wide_sub(remainder, trial), remainder)
        root = _wide_shl_one(root)
        root = _wide_set_bit(root, 0, take)
    return root, remainder


def _normalise_scale(k: Tensor, exponent: Tensor, useed: int) -> tuple[Tensor, Tensor]:
    carry = torch.div(exponent, useed, rounding_mode="floor")
    return k + carry, exponent - carry * useed


def _add_to_scale(k: Tensor, exponent: Tensor, delta: Tensor | int, useed: int):
    return _normalise_scale(k, exponent + delta, useed)


def _scale_ge(kx: Tensor, ex: Tensor, ky: Tensor, ey: Tensor) -> Tensor:
    return (kx > ky) | ((kx == ky) & (ex >= ey))


def _scale_distance(kh: Tensor, eh: Tensor, kl: Tensor, el: Tensor, useed: int) -> Tensor:
    """Positive scale distance, saturated at the wide working precision."""
    kd = kh - kl
    if useed <= _WIDE_BITS:
        return (kd * useed + eh - el).clamp(0, _WIDE_BITS)

    adjacent = (useed + eh - el).clamp(0, _WIDE_BITS)
    return torch.where(kd == 0, (eh - el).clamp(0, _WIDE_BITS),
                       torch.where(kd == 1, adjacent, _WIDE_BITS))


class _PositMixin:
    conversion_dtype = torch.float64
    es: int

    @classmethod
    def set_es(cls, es: int) -> None:
        if isinstance(es, bool) or not isinstance(es, int):
            raise TypeError("es must be an integer")
        if not 0 <= es <= cls.bitwidth - 2:
            raise ValueError(f"es must be between 0 and {cls.bitwidth - 2}")
        cls.es = es
        cls.ops.clear_scalar_cache()


class Posit16(_PositMixin, DType, bitwidth=16):
    es = 1


class Posit32(_PositMixin, DType, bitwidth=32):
    es = 2


class Posit64(_PositMixin, DType, bitwidth=64):
    es = 3


def _register_posit_ops(dtype):
    nbits = dtype.bitwidth
    precision = nbits - 1
    int_dtype = dtype.int_dtype
    nar_value = -(1 << (nbits - 1))
    maxpos_value = (1 << (nbits - 1)) - 1
    one_value = 1 << (nbits - 2)

    def constants(device):
        return (
            torch.tensor(nar_value, dtype=int_dtype, device=device),
            torch.tensor(maxpos_value, dtype=int_dtype, device=device),
        )

    def decode(code: Tensor):
        code = code.to(int_dtype)
        nar = code == nar_value
        zero = code == 0
        negative = (code < 0) & ~nar
        safe = torch.where(nar, torch.zeros_like(code), code)
        magnitude = torch.where(negative, -safe, safe).to(torch.int64)

        regime_bit = (magnitude >> (nbits - 2)) & 1
        run = torch.zeros_like(magnitude)
        active = ~(zero | nar)
        for position in range(nbits - 2, -1, -1):
            same = ((magnitude >> position) & 1) == regime_bit
            take = active & same
            run += take.to(torch.int64)
            active &= same

        k = torch.where(regime_bit != 0, run - 1, -run)
        remaining = (nbits - 2 - run).clamp(min=0)
        exponent = torch.zeros_like(magnitude)
        for j in range(dtype.es):
            position = nbits - 3 - run - j
            valid = (j < remaining) & (position >= 0)
            bit = torch.where(valid, (magnitude >> position.clamp(min=0)) & 1, 0)
            exponent = (exponent << 1) | bit

        fraction_bits = (remaining - dtype.es).clamp(min=0)
        fraction_mask = (torch.ones_like(fraction_bits) << fraction_bits) - 1
        fraction = magnitude & fraction_mask
        significand = (torch.ones_like(fraction_bits) << fraction_bits) | fraction
        fixed = significand << (precision - 1 - fraction_bits)
        fixed = torch.where(zero | nar, torch.zeros_like(fixed), fixed)
        return nar, zero, negative, k, exponent, fixed

    def pack(negative: Tensor, k: Tensor, exponent: Tensor, magnitude: Tensor,
             sticky_below: Tensor | None = None) -> Tensor:
        useed = 1 << dtype.es
        bit_length = _wide_bit_length(magnitude)
        position = bit_length - 1
        nonzero = bit_length != 0
        if sticky_below is None:
            sticky_below = torch.zeros_like(nonzero)

        run = torch.where(k >= 0, k + 1, -k)
        total_regime = run + 1
        positive_regime = k >= 0
        code = torch.zeros_like(k)

        def stream_bit(j: int) -> Tensor:
            in_run = j < run
            at_terminator = j == run
            after = j - total_regime
            in_exponent = (after >= 0) & (after < dtype.es)
            exp_shift = (dtype.es - 1 - after).clamp(min=0)
            exp_bit = (exponent >> exp_shift) & 1
            fraction_index = after - dtype.es
            fraction_bit = _wide_bit(magnitude, position - 1 - fraction_index)
            return torch.where(
                in_run,
                positive_regime.to(torch.int64),
                torch.where(
                    at_terminator,
                    (~positive_regime).to(torch.int64),
                    torch.where(in_exponent, exp_bit, fraction_bit),
                ),
            )

        for j in range(nbits - 1):
            code = (code << 1) | stream_bit(j)

        guard_index = nbits - 1
        guard = stream_bit(guard_index) != 0
        after_guard = guard_index - total_regime
        fraction_nonzero = _wide_any_below(magnitude, position)

        future_regime = torch.where(
            positive_regime,
            guard_index + 1 < run,
            guard_index < run,
        )
        exp_lower_count = (dtype.es - 1 - after_guard).clamp(0, dtype.es)
        exp_mask = (torch.ones_like(exp_lower_count) << exp_lower_count) - 1
        future_exponent = (exponent & exp_mask) != 0
        guard_fraction_index = after_guard - dtype.es
        guard_magnitude_index = position - 1 - guard_fraction_index
        future_fraction = _wide_any_below(magnitude, guard_magnitude_index)

        guard_in_regime = guard_index <= run
        guard_in_exponent = (after_guard >= 0) & (after_guard < dtype.es)
        sticky = torch.where(
            guard_in_regime,
            future_regime | (exponent != 0) | fraction_nonzero,
            torch.where(
                guard_in_exponent,
                future_exponent | fraction_nonzero,
                future_fraction,
            ),
        ) | sticky_below

        increment = guard & (sticky | ((code & 1) != 0))
        code = torch.minimum(code + increment.to(torch.int64),
                             torch.full_like(code, maxpos_value))

        # Posits saturate rather than producing zero or infinity.
        above = (k > nbits - 2) | ((k == nbits - 2) &
                ((exponent != 0) | fraction_nonzero))
        below = k < -(nbits - 2)
        code = torch.where(above, maxpos_value, torch.where(below, 1, code))
        code = torch.where(nonzero, code, 0)
        signed = torch.where(negative & nonzero, -code, code)
        return signed.to(int_dtype)

    def unpack(code: Tensor):
        nar, zero, negative, k, exponent, fixed = decode(code)
        return nar, zero, negative, k, exponent, _wide_from_int(fixed)

    @dtype.register_op("from_float")
    def from_float(ops, value: Tensor):
        value = value.to(torch.float64)
        bits = value.view(torch.int64)
        negative = bits < 0
        absolute_bits = bits & torch.iinfo(torch.int64).max
        exp_field = (absolute_bits >> 52) & 0x7FF
        fraction = absolute_bits & ((1 << 52) - 1)
        special = exp_field == 0x7FF
        zero = absolute_bits == 0
        normal = exp_field != 0

        normal_sig = (1 << 52) | fraction
        sub_length = torch.zeros_like(fraction)
        for i in range(52):
            sub_length = torch.where(((fraction >> i) & 1) != 0, i + 1, sub_length)
        significand = torch.where(normal, normal_sig, fraction)
        leading_scale = torch.where(normal, exp_field - 1023, sub_length - 1075)

        useed = 1 << dtype.es
        k = torch.div(leading_scale, useed, rounding_mode="floor")
        exponent = leading_scale - k * useed
        encoded = pack(negative, k, exponent, _wide_from_int(significand))
        encoded = torch.where(zero, torch.zeros_like(encoded), encoded)
        nar, _ = constants(value.device)
        return torch.where(special, nar, encoded)

    @dtype.register_op("to_float")
    def to_float(ops, code: Tensor):
        nar, zero, negative, k, exponent, fixed = decode(code)
        useed = 1 << dtype.es
        if useed <= 2048:
            scale = k * useed + exponent
        else:
            scale = torch.where(
                k == 0,
                exponent,
                torch.where(k == -1, exponent - useed,
                            torch.where(k > 0, 2048, -2048)),
            )
        fraction = fixed.to(torch.float64) / float(1 << (precision - 1))
        result = torch.ldexp(fraction, scale.clamp(-2048, 2048).to(torch.int32))
        result = torch.where(negative, -result, result)
        result = torch.where(zero, 0.0, result)
        return torch.where(nar, float("nan"), result)

    def add_impl(x: Tensor, y: Tensor) -> Tensor:
        x, y = torch.broadcast_tensors(x, y)
        nx, zx, sx, kx, ex, mx = unpack(x)
        ny, zy, sy, ky, ey, my = unpack(y)
        nar, _ = constants(x.device)

        x_high = _scale_ge(kx, ex, ky, ey)
        same_scale = (kx == ky) & (ex == ey)
        x_high = x_high & (~same_scale | _wide_ge(mx, my))
        kh, eh = torch.where(x_high, kx, ky), torch.where(x_high, ex, ey)
        mh, ml = _wide_select(x_high, mx, my), _wide_select(x_high, my, mx)
        sh, sl = torch.where(x_high, sx, sy), torch.where(x_high, sy, sx)
        kl, el = torch.where(x_high, ky, kx), torch.where(x_high, ey, ex)

        # Leave one top bit for a same-sign carry.
        extra = _WIDE_BITS - precision - 1
        mh = _wide_shl(mh, extra)
        ml = _wide_shl(ml, extra)
        distance = _scale_distance(kh, eh, kl, el, 1 << dtype.es)
        ml = _wide_shr_jam(ml, distance)
        same_sign = sh == sl
        magnitude = _wide_select(same_sign, _wide_add(mh, ml), _wide_sub(mh, ml))
        length = _wide_bit_length(magnitude)
        k, exponent = _add_to_scale(kh, eh, length - 1 - (precision - 1 + extra), 1 << dtype.es)
        result = pack(sh, k, exponent, magnitude)
        result = torch.where(zx, y, torch.where(zy, x, result))
        return torch.where(nx | ny, nar, result)

    @dtype.register_op("add")
    def add(ops, x, y):
        return add_impl(x, y)

    @dtype.register_op("sub")
    def sub(ops, x, y):
        neg_y = torch.where(y == nar_value, y, -y)
        return add_impl(x, neg_y)

    @dtype.register_op("mul")
    def mul(ops, x, y):
        x, y = torch.broadcast_tensors(x, y)
        nx, zx, sx, kx, ex, mx = unpack(x)
        ny, zy, sy, ky, ey, my = unpack(y)
        nar, _ = constants(x.device)
        magnitude = _wide_mul(mx, my)
        length = _wide_bit_length(magnitude)
        k, exponent = _normalise_scale(kx + ky, ex + ey, 1 << dtype.es)
        k, exponent = _add_to_scale(
            k, exponent, length - 1 - 2 * (precision - 1), 1 << dtype.es
        )
        result = pack(sx ^ sy, k, exponent, magnitude)
        result = torch.where(zx | zy, torch.zeros_like(result), result)
        return torch.where(nx | ny, nar, result)

    @dtype.register_op("div")
    def div(ops, x, y):
        x, y = torch.broadcast_tensors(x, y)
        nx, zx, sx, kx, ex, mx = unpack(x)
        ny, zy, sy, ky, ey, my = unpack(y)
        nar, _ = constants(x.device)
        shift = precision + 4
        numerator = _wide_shl(mx, shift)
        safe_denominator = _wide_select(zy | ny, _wide_from_int(torch.ones_like(x)), my)
        quotient, remainder = _wide_divmod(numerator, safe_denominator, 2 * precision + 4)
        length = _wide_bit_length(quotient)
        k, exponent = _normalise_scale(kx - ky, ex - ey, 1 << dtype.es)
        k, exponent = _add_to_scale(
            k, exponent, length - 1 - shift, 1 << dtype.es
        )
        result = pack(sx ^ sy, k, exponent, quotient, ~_wide_is_zero(remainder))
        result = torch.where(zx & ~(zy | ny), torch.zeros_like(result), result)
        return torch.where(nx | ny | zy, nar, result)

    @dtype.register_op("sqrt")
    def sqrt(ops, x):
        nar_input, zero, negative, k, exponent, magnitude = unpack(x)
        nar, _ = constants(x.device)
        useed = 1 << dtype.es
        if dtype.es == 0:
            half_k = torch.div(k, 2, rounding_mode="floor")
            parity = k - 2 * half_k
            half_exponent = torch.zeros_like(exponent)
        else:
            half_k = torch.div(k, 2, rounding_mode="floor")
            combined = (k - 2 * half_k) * useed + exponent
            parity = combined & 1
            half_exponent = combined >> 1

        quotient_bits = precision + 4
        shift = precision + 9
        radicand = _wide_shl(magnitude, shift)
        radicand = _wide_select(parity != 0, _wide_shl_one(radicand), radicand)
        root, remainder = _wide_isqrt(radicand, (2 * precision + 10) // 2)
        length = _wide_bit_length(root)
        out_k, out_exp = _add_to_scale(
            half_k, half_exponent, length - 1 - quotient_bits, useed
        )
        result = pack(torch.zeros_like(negative), out_k, out_exp, root,
                      ~_wide_is_zero(remainder))
        result = torch.where(zero, torch.zeros_like(result), result)
        return torch.where(nar_input | negative, nar, result)

    @dtype.register_op("neg")
    def neg(ops, x):
        return torch.where(x == nar_value, x, -x)

    @dtype.register_op("abs")
    def abs_(ops, x):
        return torch.where(x < 0, -x, x)

    @dtype.register_op("sign")
    def sign(ops, x):
        return torch.where(
            x == nar_value,
            torch.full_like(x, nar_value),
            torch.where(x > 0, torch.full_like(x, one_value),
                        torch.where(x < 0, torch.full_like(x, -one_value), 0)),
        )

    @dtype.register_op("ge")
    def ge(ops, x, y):
        return x >= y

    @dtype.register_op("gt")
    def gt(ops, x, y):
        return x > y

    @dtype.register_op("le")
    def le(ops, x, y):
        return x <= y

    @dtype.register_op("lt")
    def lt(ops, x, y):
        return x < y

    def float_unary(ops, x, fn, *, domain=None):
        nar, _, _, _, _, _ = decode(x)
        value = ops.to_float(x)
        result_value = fn(value)
        result = ops.from_float(result_value)
        invalid = nar | torch.isnan(result_value)
        if domain is not None:
            invalid |= ~domain(value)
        return torch.where(invalid, torch.full_like(result, nar_value), result)

    @dtype.register_op("exp")
    def exp(ops, x):
        nar_input, _, _, _, _, _ = decode(x)
        value = ops.to_float(x)
        evaluated = torch.exp(value)
        result = ops.from_float(evaluated)
        overflow = torch.isinf(evaluated) & torch.isfinite(value)
        underflow = (evaluated == 0) & torch.isfinite(value)
        result = torch.where(overflow, torch.full_like(result, maxpos_value), result)
        result = torch.where(underflow, torch.ones_like(result), result)
        return torch.where(nar_input, torch.full_like(result, nar_value), result)

    @dtype.register_op("log")
    def log(ops, x):
        return float_unary(ops, x, torch.log, domain=lambda value: value > 0)

    @dtype.register_op("pow")
    def pow_(ops, x, y):
        x, y = torch.broadcast_tensors(x, y)
        nx, _, _, _, _, _ = decode(x)
        ny, _, _, _, _, _ = decode(y)
        xf, yf = ops.to_float(x), ops.to_float(y)
        value = torch.pow(xf, yf)
        result = ops.from_float(value)
        invalid = nx | ny | torch.isnan(value) | ((x == 0) & (y == 0))
        overflow = torch.isinf(value) & ~invalid
        signed_max = torch.where(
            torch.signbit(value),
            torch.full_like(result, -maxpos_value),
            torch.full_like(result, maxpos_value),
        )
        result = torch.where(overflow, signed_max, result)
        underflow = (value == 0) & (x != 0) & ~invalid
        signed_min = torch.where(
            torch.signbit(value), torch.full_like(result, -1), torch.ones_like(result)
        )
        result = torch.where(underflow, signed_min, result)
        return torch.where(invalid, torch.full_like(result, nar_value), result)


for _dtype in (Posit16, Posit32, Posit64):
    _register_posit_ops(_dtype)
