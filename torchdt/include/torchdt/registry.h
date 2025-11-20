#ifndef REGISTRY_H
#define REGISTRY_H

#include <cstdint>
#include <type_traits>
#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <iostream>
#include <cassert>

// map bitwidth to storage type
template <size_t bitwidth>
struct StorageFor {
    static_assert(
        bitwidth == 8 || bitwidth == 16 || bitwidth == 32 || bitwidth == 64,
        "Supported bitwidths: 8,16,32,64"
    );

    using type = typename std::conditional<
        bitwidth == 8, uint8_t,
        typename std::conditional<
            bitwidth == 16, uint16_t,
            typename std::conditional<
                bitwidth == 32, uint32_t,
                uint64_t
            >::type
        >::type
    >::type;
};

// runtime container of op-function pointers for a specific bitwidth/StorageT template
template<size_t bitwidth>
struct Ops {
    using StorageT = typename StorageFor<bitwidth>::type;
    using BinOp = StorageT(*)(StorageT, StorageT);

    BinOp add = nullptr;
    BinOp sub = nullptr;
    BinOp mul = nullptr;
    BinOp div = nullptr;
};

struct OpsBase {
    virtual ~OpsBase() = default;
};

template <size_t bitwidth>
struct OpsImpl : public OpsBase {
    OpsImpl(const Ops<bitwidth>& o) : ops(o) {}
    Ops<bitwidth> ops;
};

// simple key construction
inline std::string make_key(const std::string &name, size_t bitwidth) {
    std::ostringstream ss;
    ss << name << ":" << bitwidth;
    return ss.str();
}

// Registry singleton
class Registry {
public:
    static Registry &instance() {
        static Registry r;
        return r;
    }

    // register (take ownership)
    void register_ops(const std::string &name, size_t bitwidth, std::unique_ptr<OpsBase> impl) {
        std::lock_guard<std::mutex> guard(mutex_);
        std::string key = make_key(name, bitwidth);
        if (map_.count(key))
            std::cerr << "Warning: overriding registration for " << key << "\n";
        map_[key] = std::move(impl);
    }

    // typed getter, returns nullptr if not found or if wrong bitwidth
    template <size_t bitwidth>
    Ops<bitwidth>* get_ops_typed(const std::string &name) {
        std::lock_guard<std::mutex> guard(mutex_);
        std::string key = make_key(name, bitwidth);
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        OpsImpl<bitwidth>* p = dynamic_cast<OpsImpl<bitwidth>*>(it->second.get());
        if (!p) return nullptr;
        return &(p->ops);
    }

    // runtime getter by bitwidth: returns OpsBase* (less type-safe)
    OpsBase* get_ops_base(const std::string &name, size_t bitwidth) {
        std::lock_guard<std::mutex> guard(mutex_);
        auto it = map_.find(make_key(name, bitwidth));
        if (it == map_.end()) return nullptr;
        return it->second.get();
    }

private:
    Registry() = default;
    std::map<std::string, std::unique_ptr<OpsBase>> map_;
    std::mutex mutex_;
};

// helper macro for user registration
#define REGISTER_DTYPE(NAME_STR, BITWIDTH, OPS_VAR)                                     \
namespace {                                                                             \
    struct _reg_helper_##BITWIDTH##_##__LINE__ {                                        \
        _reg_helper_##BITWIDTH##_##__LINE__() {                                         \
            auto ptr = std::make_unique< OpsImpl<BITWIDTH> >((OPS_VAR));                \
            Registry::instance().register_ops((NAME_STR), (BITWIDTH), std::move(ptr));  \
        }                                                                               \
    };                                                                                  \
    static _reg_helper_##BITWIDTH##_##__LINE__ _reg_instance_##BITWIDTH##_##__LINE__;   \
}

#endif // REGISTRY_H