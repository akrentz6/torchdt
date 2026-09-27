#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "../scalar.h"
namespace torchdt::native {
// Future CUDA kernels receive the snapshot and a device-owned correction table.
// Owners must remain alive through launch; table copies must be recorded on the
// launch stream. Never pass the CPU context's table pointer to a device kernel.
struct CudaLaunchContext {
    c10::cuda::CUDAGuard guard;
    c10::cuda::CUDAStream stream;
    ScalarConfig config;
    at::Tensor table_owner;
    CudaLaunchContext(const at::Tensor& input, ScalarConfig snapshot, at::Tensor table)
        : guard(input.device()), stream(c10::cuda::getCurrentCUDAStream(input.get_device())),
          config(snapshot), table_owner(std::move(table)) {
        TORCH_CHECK(!table_owner.defined() || table_owner.device() == input.device(),
                    "CUDA correction table must be on the input device");
    }
};
}
