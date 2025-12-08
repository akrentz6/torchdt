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
        bitwidth == 8, int8_t,
        typename std::conditional<
            bitwidth == 16, int16_t,
            typename std::conditional<
                bitwidth == 32, int32_t,
                int64_t
            >::type
        >::type
    >::type;
};

// runtime container of op-function pointers for a specific bitwidth/StorageT template
template<size_t bitwidth>
struct Ops {
    using StorageT = typename StorageFor<bitwidth>::type;
    using BinOp = StorageT(*)(StorageT, StorageT);
    using BoolBinOp = bool(*)(StorageT, StorageT);

    StorageT(*from_float)(float) = nullptr;
    float(*to_float)(StorageT) = nullptr;

    BinOp add = nullptr;
    BinOp sub = nullptr;
    BinOp mul = nullptr;
    BinOp div = nullptr;

    BoolBinOp ge = nullptr;
    BoolBinOp gt = nullptr;
    BoolBinOp le = nullptr;
    BoolBinOp lt = nullptr;
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

    template <size_t bitwidth>
    void register_ops(const std::string &name, std::shared_ptr<Ops<bitwidth>> ops) {
        std::lock_guard<std::mutex> guard(mutex_);
        std::string key = make_key(name, bitwidth);
        if (map_.count(key))
            std::cerr << "Warning: overriding registration for " << key << "\n";
        map_[key] = ops;
    }

    // typed getter, returns nullptr if not found or if wrong bitwidth
    template <size_t bitwidth>
    Ops<bitwidth>* get_ops_typed(const std::string &name) {
        std::lock_guard<std::mutex> guard(mutex_);
        std::string key = make_key(name, bitwidth);
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        return static_cast<Ops<bitwidth>*>(it->second.get());
    }

private:
    Registry() = default;
    std::map<std::string, std::shared_ptr<void>> map_;
    std::mutex mutex_;
};

// helper macro for user registration
#define REGISTER_DTYPE(NAME_STR, BITWIDTH, OPS_VAR)                                     \
namespace {                                                                             \
    struct _reg_helper_##BITWIDTH##_##__LINE__ {                                        \
        _reg_helper_##BITWIDTH##_##__LINE__() {                                         \
            auto ptr = std::make_shared< Ops<BITWIDTH> >((OPS_VAR));                    \
            Registry::instance().register_ops<BITWIDTH>((NAME_STR), ptr);               \
        }                                                                               \
    };                                                                                  \
    static _reg_helper_##BITWIDTH##_##__LINE__ _reg_instance_##BITWIDTH##_##__LINE__;   \
}

#endif // REGISTRY_H