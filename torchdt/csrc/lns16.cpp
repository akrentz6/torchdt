#include <torchdt/registry.h>

int16_t zero = -32768;
int16_t pos_inf = 32766;
int16_t neg_inf = 32767;
double base = std::pow(2.0, std::pow(2.0, -10));
double inv_log_base = 1.0 / std::log(base);

int16_t from_float16(float x) {

    if (x == 0.0f) return zero;
    if (std::isinf(x)) return (x > 0.0f) ? pos_inf : neg_inf;

    int16_t sign_bit = (x < 0.0f) ? 1 : 0;
    int16_t exponent = static_cast<int16_t>(std::round(std::log(std::abs(x)) * inv_log_base));

    return (exponent << 1) | sign_bit;
}

float to_float16(int16_t x) {

    if (x == zero) return 0.0f;
    if (x == pos_inf) return std::numeric_limits<float>::infinity();
    if (x == neg_inf) return -std::numeric_limits<float>::infinity();

    float exponent = static_cast<float>(x >> 1);
    float sign = (x & 1) ? -1.0f : 1.0f;

    return sign * std::pow(base, exponent); 
}

int16_t neg(int16_t x) {
    return x ^ 1;
}

int16_t add16(int16_t x, int16_t y) {
    if ((x | 1LL) == zero) return y;
    if ((y | 1LL) == zero) return x;
    if ((neg(x) == y)) return zero;

    const int16_t max_operand = std::max(x, y);
    const int16_t abs_diff = std::abs((x >> 1) - (y >> 1));
    const int16_t sign_diff = (x ^ y) & 1;

    double power_term = std::pow(base, -abs_diff);
    double magnitude = std::abs(1.0 - 2.0 * sign_diff + power_term);
    double log_term = std::log(magnitude) * inv_log_base;
    double rounded_value = std::clamp(
        std::round(log_term),
        (double)(std::numeric_limits<int16_t>::min()),
        (double)(std::numeric_limits<int16_t>::max())
    );

    return max_operand + (static_cast<int16_t>(rounded_value) << 1);
}

int16_t sub16(int16_t x, int16_t y) {
    return add16(x, neg(y));
}

int16_t mul16(int16_t x, int16_t y) {
    if ((x | 1LL) == zero || (y | 1LL) == zero) return zero;
    return (x + y - (y & 1)) ^ (y & 1);
}

int16_t div16(int16_t x, int16_t y) {
    if ((x | 1LL) == zero) return zero;
    if ((y | 1LL) == zero) throw std::runtime_error("Division by zero");
    return (x - y + (y & 1)) ^ (y & 1);
}

bool ge16(int16_t x, int16_t y) {
    const int16_t x_log  = x >> 1;
    const int16_t y_log  = y >> 1;
    const int x_sign = x & 1;
    const int y_sign = y & 1;

    if ((x_sign | y_sign) == 0) return x_log >= y_log; // both positive
    if ((x_sign == 0) & (y_sign == 1)) return true; // x positive, y negative
    if ((x_sign == 1) & (y_sign == 0)) return false; // x negative, y positive
    return y_log >= x_log; // both negative
}

bool gt16(int16_t x, int16_t y) {
    const int16_t x_log = x >> 1;
    const int16_t y_log = y >> 1;
    const int x_sign = x & 1;
    const int y_sign = y & 1;

    if ((x_sign | y_sign) == 0) return x_log > y_log; // both positive
    if ((x_sign == 0) & (y_sign == 1)) return true; // x positive, y negative
    if ((x_sign == 1) & (y_sign == 0)) return false; // x negative, y positive
    return y_log > x_log; // both negative
}

bool le16(int16_t x, int16_t y) {
    const int16_t x_log = x >> 1;
    const int16_t y_log = y >> 1;
    const int x_sign = x & 1;
    const int y_sign = y & 1;

    if ((x_sign | y_sign) == 0) return x_log <= y_log; // both positive
    if ((x_sign == 0) & (y_sign == 1)) return false; // x positive, y negative
    if ((x_sign == 1) & (y_sign == 0)) return true; // x negative, y positive
    return y_log <= x_log; // both negative
}

bool lt16(int16_t x, int16_t y) {
    const int16_t x_log = x >> 1;
    const int16_t y_log = y >> 1;
    const int x_sign = x & 1;
    const int y_sign = y & 1;

    if ((x_sign | y_sign) == 0) return x_log < y_log; // both positive
    if ((x_sign == 0) & (y_sign == 1)) return false; // x positive, y negative
    if ((x_sign == 1) & (y_sign == 0)) return true; // x negative, y positive
    return y_log < x_log; // both negative
}

Ops<16> ops16 = []{
    Ops<16> o;
    o.from_float = from_float16;
    o.to_float = to_float16;
    o.add = add16;
    o.sub = sub16;
    o.mul = mul16;
    o.div = div16;
    o.ge = ge16;
    o.gt = gt16;
    o.le = le16;
    o.lt = lt16;
    return o;
}();

Ops<16> get_lns16_ops() {
    return ops16;
}