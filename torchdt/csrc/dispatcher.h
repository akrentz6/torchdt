#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <torchdt/registry.h>
#include "ops_impl.h"

class Dispatcher {
public:
    static Dispatcher& instance() {
        static Dispatcher d;
        return d;
    }

    // Get cached OpsBase* for name and bitwidth.
    // If not cached and the public registry contains an Ops<bitwidth> for this name,
    // lazily construct an OpsImpl<bitwidth> from the stored Ops and cache it.
    // Returns nullptr if no Ops registered for (name,bitwidth).
    OpsBase* get(const std::string& name, size_t bitwidth) {
        const std::string key = make_key(name, bitwidth);

        // fast path: check cached map under lock
        {
            std::lock_guard<std::mutex> guard(mutex_);
            auto it = impl_map_.find(key);
            if (it != impl_map_.end()) return it->second.get();
        }

        // not cached: try to construct based on bitwidth
        std::unique_ptr<OpsBase> created = make_impl_from_registry(name, bitwidth);
        if (!created) return nullptr;

        // store & return
        {
            std::lock_guard<std::mutex> guard(mutex_);
            // double-check another thread didn't create it
            auto it = impl_map_.find(key);
            if (it == impl_map_.end()) {
                OpsBase* raw = created.get();
                impl_map_.emplace(key, std::move(created));
                return raw;
            } else {
                return it->second.get();
            }
        }
    }

    // Convenience template: typed get that returns pointer to OpsImpl<bitwidth>
    template <size_t bitwidth>
    OpsImpl<bitwidth>* get_typed(const std::string& name) {
        OpsBase* base = get(name, bitwidth);
        return static_cast<OpsImpl<bitwidth>*>(base);
    }

private:
    Dispatcher() = default;

    // Create OpsImpl for supported bitwidths by fetching the registered Ops<bitwidth> from public registry
    std::unique_ptr<OpsBase> make_impl_from_registry(const std::string& name, size_t bitwidth) {
        switch (bitwidth) {
        case 8: {
            auto p = Registry::instance().get_ops_typed<8>(name);
            if (!p) return nullptr;
            return std::make_unique<OpsImpl<8>>(*p);
        }
        case 16: {
            auto p = Registry::instance().get_ops_typed<16>(name);
            if (!p) return nullptr;
            return std::make_unique<OpsImpl<16>>(*p);
        }
        case 32: {
            auto p = Registry::instance().get_ops_typed<32>(name);
            if (!p) return nullptr;
            return std::make_unique<OpsImpl<32>>(*p);
        }
        case 64: {
            auto p = Registry::instance().get_ops_typed<64>(name);
            if (!p) return nullptr;
            return std::make_unique<OpsImpl<64>>(*p);
        }
        default:
            return nullptr;
        }
    }

    std::map<std::string, std::unique_ptr<OpsBase>> impl_map_;
    std::mutex mutex_;
};

// -----------------------------
// Helper functions (optional)
// -----------------------------
inline OpsBase* get_ops_impl(const std::string& name, size_t bitwidth) {
    return Dispatcher::instance().get(name, bitwidth);
}

template <size_t bitwidth>
inline OpsImpl<bitwidth>* get_ops_impl_typed(const std::string& name) {
    return Dispatcher::instance().get_typed<bitwidth>(name);
}

#endif // DISPATCHER_H