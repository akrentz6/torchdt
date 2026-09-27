// Compiled by opt-in CUDA builds; never registered or launched.
#include "../scalar.h"
#include "launch_context.cuh"
namespace torchdt::native {
template<int Bits>
__global__ void scalar_compile_probe(typename LnsTraits<Bits>::S* values, ScalarConfig config) {
    LnsScalar<Bits> ops{config};
    auto x = values[0], y = values[1];
    values[0] = ops.add(x,y);
    values[1] = ops.sub(x,y);
    values[2] = ops.mul(x,y);
    values[3] = ops.div(x,y);
    values[4] = ops.from_float(ops.to_float(x));
    values[5] = ops.pow(x,y);
    values[6] = ops.sqrt(x);
    values[7] = ops.neg(x);
    values[8] = ops.abs(x);
    values[9] = ops.sign(x);
    values[10] = ops.ge(x,y) + ops.gt(x,y) + ops.le(x,y) + ops.lt(x,y);
}
template __global__ void scalar_compile_probe<16>(int16_t*, ScalarConfig);
template __global__ void scalar_compile_probe<32>(int32_t*, ScalarConfig);
template __global__ void scalar_compile_probe<64>(int64_t*, ScalarConfig);
}
