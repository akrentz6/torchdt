#include <torchdt/cuda/kernels.cuh>
#include "../scalar.h"
#include "../config.h"
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <map>

namespace torchdt::native {
template<int Bits> struct LnsCUDAArithmetic {
    using Scalar = LnsScalar<Bits>;
    using S = typename Scalar::S;
    static constexpr auto storage=c10::CppTypeToScalarType<S>::value;
    Scalar scalar;
    Tensor host_table;
    struct Entry { Tensor table; at::cuda::CUDAEvent ready; };
    struct Cache { std::mutex mutex; std::map<int,std::unique_ptr<Entry>> devices; };
    std::shared_ptr<Cache> cache=std::make_shared<Cache>();
    explicit LnsCUDAArithmetic(const LnsConfig& config)
        : scalar(ScalarConfig{int(config.precision),config.base,config.log_base}) {
        TORCH_CHECK(config.precision>=1 && config.precision<=50,"Precision must be between 1 and 50");
        TORCH_CHECK(std::isfinite(config.base) && config.base>1 && std::isfinite(config.log_base) && config.log_base>0,"Invalid LNS base");
        if(config.table) {
            TORCH_CHECK(config.precision<=20,"Table-based LNS only supports precision up to 20");
            TORCH_CHECK(config.table->device().is_cpu() && config.table->layout()==at::kStrided && config.table->scalar_type()==storage,"Invalid LNS correction table storage");
            TORCH_CHECK(config.table->dim()==2 && config.table->size(0)==2 && config.table_ez<0 && config.table_ez==-config.table->size(1),"Invalid LNS correction table");
            host_table=config.table->contiguous().clone();
            scalar.table_size=host_table.size(1); scalar.table_ez=config.table_ez;
        }
    }
    Scalar prepare(const Tensor& input) const {
        auto result=scalar;
        if(!host_table.defined()) return result;
        auto stream=c10::cuda::getCurrentCUDAStream(input.get_device());
        std::lock_guard<std::mutex> lock(cache->mutex);
        auto& entry=cache->devices[input.get_device()];
        if(!entry) {
            auto created=std::make_unique<Entry>();
            auto pinned=host_table.pin_memory();
            created->table=pinned.to(input.device(), /*non_blocking=*/true, /*copy=*/true);
            created->ready.record(stream);
            entry=std::move(created);
        }
        entry->ready.block(stream);
        // Contexts may be replaced/destroyed immediately after launch. The
        // allocator retains the table storage until every recorded use finishes.
        c10::cuda::CUDACachingAllocator::recordStream(entry->table.storage().data_ptr(),stream);
        result.table=entry->table.template data_ptr<S>();
        return result;
    }
    void validate_conversion(const Tensor& input) const {
        // Match Python's synchronous ValueError; only encoding synchronizes.
        TORCH_CHECK_VALUE(!at::isnan(input).any().item<bool>(),"LNS cannot encode NaN values");
    }
};
void register_lns_cuda() {
#define REGISTER(BITS) Registry::instance().register_factory("lns",BITS,"cuda",[](const Config& c) { return std::make_shared<CUDAKernels<LnsCUDAArithmetic<BITS>>>(LnsCUDAArithmetic<BITS>(LnsConfig(c))); });
    REGISTER(16) REGISTER(32) REGISTER(64)
#undef REGISTER
}
}
