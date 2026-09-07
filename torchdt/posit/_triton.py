from __future__ import annotations

from torchdt.ops import TritonScalarOps, register_triton_ops, require_triton


_LIMB_BITS = 30
_LIMB_BASE = 1 << _LIMB_BITS
_LIMB_MASK = _LIMB_BASE - 1
_LIMBS = 6
_WIDE_BITS = _LIMB_BITS * _LIMBS


def enable_posit_triton_backend(dtype_cls: type) -> None:
    from torchdt.triton import _autotune_revision

    fingerprint = (dtype_cls.bitwidth, dtype_cls.es, _autotune_revision())
    if getattr(dtype_cls.ops, "_triton_fingerprint", None) == fingerprint:
        return
    register_triton_ops(dtype_cls, make_posit_triton_scalar_ops(dtype_cls))
    dtype_cls.ops._triton_fingerprint = fingerprint


def make_posit_triton_scalar_ops(dtype_cls: type) -> TritonScalarOps:
    triton, tl = require_triton()
    from triton.language.extra import libdevice

    nbits = dtype_cls.bitwidth
    precision = nbits - 1
    es = dtype_cls.es
    useed = 1 << es
    working_bits = {16: 60, 32: 90, 64: 150}[nbits]
    working_limbs = (working_bits + _LIMB_BITS - 1) // _LIMB_BITS
    nar_value = -(1 << (nbits - 1))
    maxpos_value = (1 << (nbits - 1)) - 1
    one_value = 1 << (nbits - 2)
    int_types = {16: tl.int16, 32: tl.int32, 64: tl.int64}
    tl_int_dtype = int_types[nbits]

    NBITS = tl.constexpr(nbits)
    PRECISION = tl.constexpr(precision)
    ES = tl.constexpr(es)
    USEED = tl.constexpr(useed)
    NAR = tl.constexpr(nar_value)
    MAXPOS = tl.constexpr(maxpos_value)
    ONE = tl.constexpr(one_value)
    LIMB_BITS = tl.constexpr(_LIMB_BITS)
    LIMB_BASE = tl.constexpr(_LIMB_BASE)
    LIMB_MASK = tl.constexpr(_LIMB_MASK)
    LIMBS = tl.constexpr(working_limbs)
    WIDE_BITS = tl.constexpr(working_bits)
    ADD_EXTRA = tl.constexpr(working_bits - precision - 1)
    DIV_SHIFT = tl.constexpr(precision + 4)
    DIV_TOP_BIT = tl.constexpr(2 * precision + 4)
    SQRT_BITS = tl.constexpr(precision + 4)
    SQRT_SHIFT = tl.constexpr(precision + 9)
    SQRT_TOP_PAIR = tl.constexpr((2 * precision + 10) // 2)
    FLOAT_SIGNIFICAND_SCALE = tl.constexpr(float(1 << (precision - 1)))

    @triton.jit
    def wide_zero(value):
        zero = tl.cast(value * 0, tl.int64)
        return zero, zero, zero, zero, zero, zero

    @triton.jit
    def wide_from_int(value):
        value = tl.cast(value, tl.int64)
        return (
            value & LIMB_MASK,
            (value >> 30) & LIMB_MASK,
            (value >> 60) & LIMB_MASK,
            value * 0,
            value * 0,
            value * 0,
        )

    @triton.jit
    def wide_select(mask, x, y):
        return (
            tl.where(mask, x[0], y[0]), tl.where(mask, x[1], y[1]),
            tl.where(mask, x[2], y[2]), tl.where(mask, x[3], y[3]),
            tl.where(mask, x[4], y[4]), tl.where(mask, x[5], y[5]),
        )

    @triton.jit
    def wide_is_zero(x):
        return ((x[0] | x[1] | x[2] | x[3] | x[4] | x[5]) == 0)

    @triton.jit
    def wide_add(x, y):
        v0 = x[0] + y[0]
        z0 = v0 & LIMB_MASK
        v1 = x[1] + y[1] + (v0 >> LIMB_BITS)
        z1 = v1 & LIMB_MASK
        v2 = x[2] + y[2] + (v1 >> LIMB_BITS)
        z2 = v2 & LIMB_MASK
        v3 = x[3] + y[3] + (v2 >> LIMB_BITS)
        z3 = v3 & LIMB_MASK
        v4 = x[4] + y[4] + (v3 >> LIMB_BITS)
        z4 = v4 & LIMB_MASK
        v5 = x[5] + y[5] + (v4 >> LIMB_BITS)
        return z0, z1, z2, z3, z4, v5 & LIMB_MASK

    @triton.jit
    def wide_sub(x, y):
        v0 = x[0] - y[0]
        b0 = v0 < 0
        z0 = tl.where(b0, v0 + LIMB_BASE, v0)
        v1 = x[1] - y[1] - b0.to(tl.int64)
        b1 = v1 < 0
        z1 = tl.where(b1, v1 + LIMB_BASE, v1)
        v2 = x[2] - y[2] - b1.to(tl.int64)
        b2 = v2 < 0
        z2 = tl.where(b2, v2 + LIMB_BASE, v2)
        v3 = x[3] - y[3] - b2.to(tl.int64)
        b3 = v3 < 0
        z3 = tl.where(b3, v3 + LIMB_BASE, v3)
        v4 = x[4] - y[4] - b3.to(tl.int64)
        b4 = v4 < 0
        z4 = tl.where(b4, v4 + LIMB_BASE, v4)
        v5 = x[5] - y[5] - b4.to(tl.int64)
        return z0, z1, z2, z3, z4, tl.where(v5 < 0, v5 + LIMB_BASE, v5)

    @triton.jit
    def wide_ge(x, y):
        greater = x[5] > y[5]
        equal = x[5] == y[5]
        greater |= equal & (x[4] > y[4]); equal &= x[4] == y[4]
        greater |= equal & (x[3] > y[3]); equal &= x[3] == y[3]
        greater |= equal & (x[2] > y[2]); equal &= x[2] == y[2]
        greater |= equal & (x[1] > y[1]); equal &= x[1] == y[1]
        greater |= equal & (x[0] > y[0]); equal &= x[0] == y[0]
        return greater | equal

    @triton.jit
    def wide_shl_one(x):
        v0 = x[0] << 1
        z0 = v0 & LIMB_MASK
        v1 = (x[1] << 1) | (v0 >> LIMB_BITS)
        z1 = v1 & LIMB_MASK
        v2 = (x[2] << 1) | (v1 >> LIMB_BITS)
        z2 = v2 & LIMB_MASK
        v3 = (x[3] << 1) | (v2 >> LIMB_BITS)
        z3 = v3 & LIMB_MASK
        v4 = (x[4] << 1) | (v3 >> LIMB_BITS)
        z4 = v4 & LIMB_MASK
        v5 = (x[5] << 1) | (v4 >> LIMB_BITS)
        return z0, z1, z2, z3, z4, v5 & LIMB_MASK

    @triton.jit
    def wide_shr_one(x):
        z5 = x[5] >> 1
        z4 = (x[4] >> 1) | ((x[5] & 1) << 29)
        z3 = (x[3] >> 1) | ((x[4] & 1) << 29)
        z2 = (x[2] >> 1) | ((x[3] & 1) << 29)
        z1 = (x[1] >> 1) | ((x[2] & 1) << 29)
        z0 = (x[0] >> 1) | ((x[1] & 1) << 29)
        return (z0, z1, z2, z3, z4, z5), x[0] & 1

    @triton.jit
    def wide_shl(x, amount: tl.constexpr):
        for _ in tl.static_range(0, amount):
            x = wide_shl_one(x)
        return x

    @triton.jit
    def wide_shr_jam(x, amount):
        amount = tl.minimum(tl.maximum(amount, 0), WIDE_BITS)
        sticky = amount < 0
        for i in tl.static_range(0, WIDE_BITS):
            shifted, dropped = wide_shr_one(x)
            active = amount > i
            sticky |= active & (dropped != 0)
            x = wide_select(active, shifted, x)
        return (x[0] | sticky.to(tl.int64), x[1], x[2], x[3], x[4], x[5])

    @triton.jit
    def wide_bit(x, index):
        valid = (index >= 0) & (index < WIDE_BITS)
        safe = tl.minimum(tl.maximum(index, 0), WIDE_BITS - 1)
        limb = safe // LIMB_BITS
        offset = safe - limb * LIMB_BITS
        value = tl.where(
            limb == 0, x[0], tl.where(
                limb == 1, x[1], tl.where(
                    limb == 2, x[2], tl.where(
                        limb == 3, x[3], tl.where(limb == 4, x[4], x[5])
                    )
                )
            )
        )
        return tl.where(valid, (value >> offset) & 1, 0)

    @triton.jit
    def wide_set_bit(x, index: tl.constexpr, bit):
        value = bit.to(tl.int64) << (index % LIMB_BITS)
        if index // LIMB_BITS == 0:
            return x[0] | value, x[1], x[2], x[3], x[4], x[5]
        elif index // LIMB_BITS == 1:
            return x[0], x[1] | value, x[2], x[3], x[4], x[5]
        elif index // LIMB_BITS == 2:
            return x[0], x[1], x[2] | value, x[3], x[4], x[5]
        elif index // LIMB_BITS == 3:
            return x[0], x[1], x[2], x[3] | value, x[4], x[5]
        elif index // LIMB_BITS == 4:
            return x[0], x[1], x[2], x[3], x[4] | value, x[5]
        else:
            return x[0], x[1], x[2], x[3], x[4], x[5] | value

    @triton.jit
    def wide_bit_length(x):
        length = x[0] * 0
        for i in tl.static_range(0, WIDE_BITS):
            length = tl.where(wide_bit(x, i) != 0, i + 1, length)
        return length

    @triton.jit
    def wide_any_below(x, index):
        result = index < 0
        result &= False
        for i in tl.static_range(0, LIMBS):
            count = tl.minimum(tl.maximum(index - i * LIMB_BITS, 0), LIMB_BITS)
            mask = (tl.cast(1, tl.int64) << count) - 1
            result |= (x[i] & mask) != 0
        return result

    @triton.jit
    def wide_mul(x, y):
        a0 = x[0] * y[0]
        a1 = x[0] * y[1] + x[1] * y[0]
        a2 = x[0] * y[2] + x[1] * y[1] + x[2] * y[0]
        a3 = x[1] * y[2] + x[2] * y[1]
        a4 = x[2] * y[2]
        v0 = a0
        z0 = v0 & LIMB_MASK
        v1 = a1 + (v0 >> LIMB_BITS)
        z1 = v1 & LIMB_MASK
        v2 = a2 + (v1 >> LIMB_BITS)
        z2 = v2 & LIMB_MASK
        v3 = a3 + (v2 >> LIMB_BITS)
        z3 = v3 & LIMB_MASK
        v4 = a4 + (v3 >> LIMB_BITS)
        z4 = v4 & LIMB_MASK
        return z0, z1, z2, z3, z4, (v4 >> LIMB_BITS) & LIMB_MASK

    @triton.jit
    def wide_divmod(numerator, denominator, top_bit: tl.constexpr):
        quotient = wide_zero(numerator[0])
        remainder = wide_zero(numerator[0])
        for step in tl.static_range(0, top_bit + 1):
            index = top_bit - step
            remainder = wide_shl_one(remainder)
            remainder = wide_set_bit(remainder, 0, wide_bit(numerator, index))
            take = wide_ge(remainder, denominator)
            remainder = wide_select(take, wide_sub(remainder, denominator), remainder)
            quotient = wide_set_bit(quotient, index, take)
        return quotient, remainder

    @triton.jit
    def wide_isqrt(value, top_pair: tl.constexpr):
        root = wide_zero(value[0])
        remainder = wide_zero(value[0])
        for step in tl.static_range(0, top_pair + 1):
            pair = top_pair - step
            remainder = wide_shl(wide_shl_one(remainder), 1)
            pair_bits = (wide_bit(value, 2 * pair + 1) << 1) | wide_bit(value, 2 * pair)
            remainder = wide_set_bit(remainder, 0, pair_bits & 1)
            remainder = wide_set_bit(remainder, 1, pair_bits >> 1)
            trial = wide_shl(root, 2)
            trial = wide_set_bit(trial, 0, pair_bits * 0 + 1)
            take = wide_ge(remainder, trial)
            remainder = wide_select(take, wide_sub(remainder, trial), remainder)
            root = wide_shl_one(root)
            root = wide_set_bit(root, 0, take)
        return root, remainder

    @triton.jit
    def floor_div(value, divisor: tl.constexpr):
        quotient = value // divisor
        remainder = value - quotient * divisor
        return quotient - ((value < 0) & (remainder != 0)).to(tl.int64)

    @triton.jit
    def normalise_scale(k, exponent):
        carry = floor_div(exponent, USEED)
        return k + carry, exponent - carry * USEED

    @triton.jit
    def add_to_scale(k, exponent, delta):
        return normalise_scale(k, exponent + delta)

    @triton.jit
    def scale_ge(kx, ex, ky, ey):
        return (kx > ky) | ((kx == ky) & (ex >= ey))

    @triton.jit
    def scale_distance(kh, eh, kl, el):
        kd = kh - kl
        if USEED <= WIDE_BITS:
            return tl.minimum(tl.maximum(kd * USEED + eh - el, 0), WIDE_BITS)
        adjacent = tl.minimum(tl.maximum(USEED + eh - el, 0), WIDE_BITS)
        return tl.where(
            kd == 0,
            tl.minimum(tl.maximum(eh - el, 0), WIDE_BITS),
            tl.where(kd == 1, adjacent, WIDE_BITS),
        )

    @triton.jit
    def decode(code):
        code = code.to(tl_int_dtype)
        nar = code == NAR
        zero = code == 0
        negative = (code < 0) & ~nar
        safe = tl.where(nar, 0, code).to(tl.int64)
        magnitude = tl.where(negative, -safe, safe)
        regime_bit = (magnitude >> (NBITS - 2)) & 1
        run = magnitude * 0
        active = ~(zero | nar)
        for step in tl.static_range(0, NBITS - 1):
            position = NBITS - 2 - step
            same = ((magnitude >> position) & 1) == regime_bit
            take = active & same
            run += take.to(tl.int64)
            active &= same
        k = tl.where(regime_bit != 0, run - 1, -run)
        remaining = tl.maximum(NBITS - 2 - run, 0)
        exponent = magnitude * 0
        for j in tl.static_range(0, ES):
            position = NBITS - 3 - run - j
            valid = (j < remaining) & (position >= 0)
            bit = tl.where(valid, (magnitude >> tl.maximum(position, 0)) & 1, 0)
            exponent = (exponent << 1) | bit
        fraction_bits = tl.maximum(remaining - ES, 0)
        fraction_mask = (tl.cast(1, tl.int64) << fraction_bits) - 1
        fraction = magnitude & fraction_mask
        significand = (tl.cast(1, tl.int64) << fraction_bits) | fraction
        fixed = significand << (PRECISION - 1 - fraction_bits)
        fixed = tl.where(zero | nar, 0, fixed)
        return nar, zero, negative, k, exponent, fixed

    @triton.jit
    def stream_bit(j: tl.constexpr, run, positive_regime, exponent, magnitude, position):
        in_run = j < run
        at_terminator = j == run
        after = j - (run + 1)
        in_exponent = (after >= 0) & (after < ES)
        exp_shift = tl.maximum(ES - 1 - after, 0)
        exp_bit = (exponent >> exp_shift) & 1
        fraction_index = after - ES
        fraction_bit = wide_bit(magnitude, position - 1 - fraction_index)
        return tl.where(
            in_run, positive_regime.to(tl.int64),
            tl.where(at_terminator, (~positive_regime).to(tl.int64),
                     tl.where(in_exponent, exp_bit, fraction_bit)),
        )

    @triton.jit
    def pack(negative, k, exponent, magnitude, sticky_below):
        bit_length = wide_bit_length(magnitude)
        position = bit_length - 1
        nonzero = bit_length != 0
        run = tl.where(k >= 0, k + 1, -k)
        total_regime = run + 1
        positive_regime = k >= 0
        code = k * 0
        for j in tl.static_range(0, NBITS - 1):
            code = (code << 1) | stream_bit(
                j, run, positive_regime, exponent, magnitude, position
            )

        guard_index = NBITS - 1
        guard = stream_bit(
            guard_index, run, positive_regime, exponent, magnitude, position
        ) != 0
        after_guard = guard_index - total_regime
        fraction_nonzero = wide_any_below(magnitude, position)
        future_regime = tl.where(
            positive_regime, guard_index + 1 < run, guard_index < run
        )
        exp_lower_count = tl.minimum(tl.maximum(ES - 1 - after_guard, 0), ES)
        exp_mask = (tl.cast(1, tl.int64) << exp_lower_count) - 1
        future_exponent = (exponent & exp_mask) != 0
        guard_fraction_index = after_guard - ES
        future_fraction = wide_any_below(
            magnitude, position - 1 - guard_fraction_index
        )
        guard_in_regime = guard_index <= run
        guard_in_exponent = (after_guard >= 0) & (after_guard < ES)
        sticky = tl.where(
            guard_in_regime,
            future_regime | (exponent != 0) | fraction_nonzero,
            tl.where(guard_in_exponent, future_exponent | fraction_nonzero,
                     future_fraction),
        ) | sticky_below
        increment = guard & (sticky | ((code & 1) != 0))
        code = tl.where((code == MAXPOS) & increment, MAXPOS,
                        code + increment.to(tl.int64))
        above = (k > NBITS - 2) | (
            (k == NBITS - 2) & ((exponent != 0) | fraction_nonzero)
        )
        below = k < -(NBITS - 2)
        code = tl.where(above, MAXPOS, tl.where(below, 1, code))
        code = tl.where(nonzero, code, 0)
        return tl.where(negative & nonzero, -code, code).to(tl_int_dtype)

    @triton.jit
    def unpack(code):
        nar, zero, negative, k, exponent, fixed = decode(code)
        return nar, zero, negative, k, exponent, wide_from_int(fixed)

    @triton.jit
    def from_float(value):
        value = value.to(tl.float64)
        bits = tl.cast(value, tl.int64, bitcast=True)
        negative = bits < 0
        absolute_bits = bits & 0x7FFFFFFFFFFFFFFF
        exp_field = (absolute_bits >> 52) & 0x7FF
        fraction = absolute_bits & 0xFFFFFFFFFFFFF
        special = exp_field == 0x7FF
        zero = absolute_bits == 0
        normal = exp_field != 0
        normal_sig = (tl.cast(1, tl.int64) << 52) | fraction
        sub_length = fraction * 0
        for i in tl.static_range(0, 52):
            sub_length = tl.where(((fraction >> i) & 1) != 0, i + 1, sub_length)
        significand = tl.where(normal, normal_sig, fraction)
        leading_scale = tl.where(normal, exp_field - 1023, sub_length - 1075)
        k = floor_div(leading_scale, USEED)
        exponent = leading_scale - k * USEED
        encoded = pack(
            negative, k, exponent, wide_from_int(significand), special & False
        )
        encoded = tl.where(zero, 0, encoded)
        return tl.where(special, NAR, encoded).to(tl_int_dtype)

    @triton.jit
    def to_float(code):
        nar, zero, negative, k, exponent, fixed = decode(code)
        if USEED <= 2048:
            scale = k * USEED + exponent
        else:
            scale = tl.where(
                k == 0, exponent,
                tl.where(k == -1, exponent - USEED,
                         tl.where(k > 0, 2048, -2048)),
            )
        fraction = fixed.to(tl.float64) / FLOAT_SIGNIFICAND_SCALE
        result = libdevice.ldexp(
            fraction, tl.minimum(tl.maximum(scale, -2048), 2048).to(tl.int32)
        )
        result = tl.where(negative, -result, result)
        result = tl.where(zero, 0.0, result)
        return tl.where(nar, float("nan"), result)

    @triton.jit
    def neg(x):
        return tl.where(x == NAR, x, -x).to(tl_int_dtype)

    @triton.jit
    def add(x, y):
        nx, zx, sx, kx, ex, mx = unpack(x)
        ny, zy, sy, ky, ey, my = unpack(y)
        x_high = scale_ge(kx, ex, ky, ey)
        same_scale = (kx == ky) & (ex == ey)
        x_high &= ~same_scale | wide_ge(mx, my)
        kh = tl.where(x_high, kx, ky); eh = tl.where(x_high, ex, ey)
        mh = wide_select(x_high, mx, my); ml = wide_select(x_high, my, mx)
        sh = tl.where(x_high, sx, sy); sl = tl.where(x_high, sy, sx)
        kl = tl.where(x_high, ky, kx); el = tl.where(x_high, ey, ex)
        mh = wide_shl(mh, ADD_EXTRA); ml = wide_shl(ml, ADD_EXTRA)
        ml = wide_shr_jam(ml, scale_distance(kh, eh, kl, el))
        same_sign = sh == sl
        magnitude = wide_select(same_sign, wide_add(mh, ml), wide_sub(mh, ml))
        length = wide_bit_length(magnitude)
        k, exponent = add_to_scale(
            kh, eh, length - 1 - (PRECISION - 1 + ADD_EXTRA)
        )
        result = pack(sh, k, exponent, magnitude, nx & False)
        result = tl.where(zx, y, tl.where(zy, x, result))
        return tl.where(nx | ny, NAR, result).to(tl_int_dtype)

    @triton.jit
    def sub(x, y):
        return add(x, neg(y))

    @triton.jit
    def mul(x, y):
        nx, zx, sx, kx, ex, mx = unpack(x)
        ny, zy, sy, ky, ey, my = unpack(y)
        magnitude = wide_mul(mx, my)
        length = wide_bit_length(magnitude)
        k, exponent = normalise_scale(kx + ky, ex + ey)
        k, exponent = add_to_scale(
            k, exponent, length - 1 - 2 * (PRECISION - 1)
        )
        result = pack(sx ^ sy, k, exponent, magnitude, nx & False)
        result = tl.where(zx | zy, 0, result)
        return tl.where(nx | ny, NAR, result).to(tl_int_dtype)

    @triton.jit
    def div(x, y):
        nx, zx, sx, kx, ex, mx = unpack(x)
        ny, zy, sy, ky, ey, my = unpack(y)
        numerator = wide_shl(mx, DIV_SHIFT)
        safe_denominator = wide_select(zy | ny, wide_from_int(x * 0 + 1), my)
        quotient, remainder = wide_divmod(
            numerator, safe_denominator, DIV_TOP_BIT
        )
        length = wide_bit_length(quotient)
        k, exponent = normalise_scale(kx - ky, ex - ey)
        k, exponent = add_to_scale(k, exponent, length - 1 - DIV_SHIFT)
        result = pack(
            sx ^ sy, k, exponent, quotient, ~wide_is_zero(remainder)
        )
        result = tl.where(zx & ~(zy | ny), 0, result)
        return tl.where(nx | ny | zy, NAR, result).to(tl_int_dtype)

    @triton.jit
    def sqrt(x):
        nar, zero, negative, k, exponent, magnitude = unpack(x)
        if ES == 0:
            half_k = floor_div(k, 2)
            parity = k - 2 * half_k
            half_exponent = exponent * 0
        else:
            half_k = floor_div(k, 2)
            combined = (k - 2 * half_k) * USEED + exponent
            parity = combined & 1
            half_exponent = combined >> 1
        radicand = wide_shl(magnitude, SQRT_SHIFT)
        radicand = wide_select(parity != 0, wide_shl_one(radicand), radicand)
        root, remainder = wide_isqrt(radicand, SQRT_TOP_PAIR)
        length = wide_bit_length(root)
        out_k, out_exp = add_to_scale(
            half_k, half_exponent, length - 1 - SQRT_BITS
        )
        result = pack(
            negative & False, out_k, out_exp, root, ~wide_is_zero(remainder)
        )
        result = tl.where(zero, 0, result)
        return tl.where(nar | negative, NAR, result).to(tl_int_dtype)

    @triton.jit
    def gt(x, y):
        return x > y

    @triton.jit
    def ge(x, y):
        return x >= y

    @triton.jit
    def lt(x, y):
        return x < y

    @triton.jit
    def le(x, y):
        return x <= y

    @triton.jit
    def sign(x):
        return tl.where(
            x == NAR, NAR,
            tl.where(x > 0, ONE, tl.where(x < 0, -ONE, 0)),
        ).to(tl_int_dtype)

    @triton.jit
    def exp(x):
        nar = x == NAR
        value = to_float(x)
        evaluated = libdevice.exp(value)
        result = from_float(evaluated)
        result = tl.where((evaluated == float("inf")) & ~nar, MAXPOS, result)
        result = tl.where((evaluated == 0.0) & ~nar, 1, result)
        return tl.where(nar, NAR, result).to(tl_int_dtype)

    @triton.jit
    def log(x):
        invalid = (x == NAR) | (x <= 0)
        safe = tl.where(invalid, ONE, x)
        result = from_float(libdevice.log(to_float(safe)))
        return tl.where(invalid, NAR, result).to(tl_int_dtype)

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
        exp=exp,
        log=log,
        sign=sign,
    )
