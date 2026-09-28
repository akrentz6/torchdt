#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace torchdt::native {
// Adjacent pairs, left then right, odd tails carried. First reduce contiguous
// 256-leaf chunks, then their totals with the same tree. Padding by the scalar
// additive identity is equivalent to carrying a tail.
template<class Scalar,class Value,class Finish>
__global__ void reduce_outputs(typename Scalar::S* out,int64_t outputs,int64_t count,
                               Scalar scalar,Value value,Finish finish) {
    using S=typename Scalar::S;
    __shared__ S leaves[256];
    __shared__ S levels[64];
    for(int64_t i=blockIdx.x;i<outputs;i+=gridDim.x) {
        uint64_t occupied=0;
        for(int64_t offset=0;offset<count;offset+=256) {
            auto q=offset+threadIdx.x;
            leaves[threadIdx.x]=q<count ? value(i,q) : Scalar::zero;
            __syncthreads();
            for(int stride=1;stride<256;stride*=2) {
                if(threadIdx.x%(2*stride)==0)
                    leaves[threadIdx.x]=scalar.add(leaves[threadIdx.x],leaves[threadIdx.x+stride]);
                __syncthreads();
            }
            if(threadIdx.x==0) {
                S current=leaves[0];
                unsigned level=0;
                while(occupied & (uint64_t(1)<<level)) {
                    current=scalar.add(levels[level],current);
                    occupied &= ~(uint64_t(1)<<level++);
                }
                levels[level]=current;
                occupied |= uint64_t(1)<<level;
            }
            __syncthreads();
        }
        if(threadIdx.x==0) {
            S result=Scalar::zero;
            bool first=true;
            for(unsigned level=0;level<64;++level) if(occupied & (uint64_t(1)<<level)) {
                result=first ? levels[level] : scalar.add(levels[level],result);
                first=false;
            }
            out[i]=finish(i,result);
        }
        __syncthreads();
    }
}
template<class Scalar,class Value,class Finish>
void launch_reduce(const at::Tensor& out,int64_t count,Scalar scalar,Value value,Finish finish) {
    if(!out.numel()) return;
    auto stream=c10::cuda::getCurrentCUDAStream(out.get_device());
    reduce_outputs<<<std::min<int64_t>(out.numel(),65535),256,0,stream>>>(
        out.template data_ptr<typename Scalar::S>(),out.numel(),count,scalar,value,finish);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<class Scalar,class Value>
void launch_reduce(const at::Tensor& out,int64_t count,Scalar scalar,Value value) {
    launch_reduce(out,count,scalar,value,[] __device__(int64_t,typename Scalar::S x) {return x;});
}
}
