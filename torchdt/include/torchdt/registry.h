#pragma once
#include <ATen/ATen.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace torchdt::native {
using Tensor = at::Tensor;
using OptionalTensor = c10::optional<Tensor>;
using Dims = std::vector<int64_t>;
struct Config {
    int64_t bitwidth;
    // Backend-specific options
    c10::Dict<std::string, c10::IValue> options;
};

// A registration is immutable and owned by each live context, so replacing
// a factory affects only newly created contexts, not existing ones.
struct TensorKernels {
    virtual ~TensorKernels() = default;
    virtual std::vector<std::string> capabilities() const = 0;
    virtual Tensor unary(const Tensor&, const std::string&) const {
        TORCH_CHECK(false, "Native unary operation is not registered");
    }
    virtual Tensor binary(const Tensor&, const Tensor&, const std::string&) const {
        TORCH_CHECK(false, "Native binary operation is not registered");
    }
    virtual Tensor sum(const Tensor&, c10::optional<Dims>, bool) const {
        TORCH_CHECK(false, "Native sum is not registered");
    }
    virtual Tensor matmul(const Tensor&, const Tensor&) const {
        TORCH_CHECK(false, "Native matmul is not registered");
    }
    virtual std::vector<Tensor> matmul_backward(const Tensor&, const Tensor&, const Tensor&) const {
        TORCH_CHECK(false, "Native matmul_backward is not registered");
    }
    virtual Tensor conv2d(const Tensor&, const Tensor&, const OptionalTensor&,
                         const Dims&, const Dims&, const Dims&, int64_t) const {
        TORCH_CHECK(false, "Native conv2d is not registered");
    }
    virtual std::vector<Tensor> conv2d_backward(const Tensor&, const Tensor&, const Tensor&,
                         const Dims&, const Dims&, const Dims&, bool, int64_t) const {
        TORCH_CHECK(false, "Native conv2d_backward is not registered");
    }
};
using Factory = std::function<std::shared_ptr<const TensorKernels>(const Config&)>;
class Registry {
public:
    static Registry& instance();
    void register_factory(const std::string& name, int64_t bits,
                          const std::string& device, Factory factory);
    bool contains(const std::string& name, int64_t bits, const std::string& device) const;
    std::shared_ptr<const TensorKernels> create(const std::string& name,
                          const std::string& device, const Config& config) const;
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Factory> factories_;
};
} // namespace torchdt::native
