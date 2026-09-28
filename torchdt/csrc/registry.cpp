#include <torchdt/registry.h>
namespace torchdt::native {
Registry& Registry::instance() { static Registry registry; return registry; }
static std::string key(const std::string& name, int64_t bits, const std::string& device) {
    return name + ":" + std::to_string(bits) + ":" + device;
}
void Registry::register_factory(const std::string& name, int64_t bits,
                               const std::string& device, Factory factory) {
    TORCH_CHECK(factory, "Empty native factory");
    std::lock_guard<std::mutex> lock(mutex_);
    factories_[key(name, bits, device)] = std::move(factory);
}
bool Registry::contains(const std::string& name, int64_t bits, const std::string& device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return factories_.count(key(name, bits, device)) != 0;
}
std::shared_ptr<const TensorKernels> Registry::create(const std::string& name,
                              const std::string& device, const Config& config) const {
    Factory factory;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = factories_.find(key(name, config.bitwidth, device));
        TORCH_CHECK(it != factories_.end(), "No native backend for ", name,
                    " (", config.bitwidth, " bits, ", device, ")");
        factory = it->second;
    }
    auto kernels = factory(config);
    TORCH_CHECK(kernels, "Native factory returned an empty registration");
    return kernels;
}
}
