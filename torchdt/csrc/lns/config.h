#pragma once
#include <torchdt/registry.h>

namespace torchdt::native {
// Only LNS interprets these options; other native backends define their own.
struct LnsConfig {
    int64_t precision;
    double base;
    double log_base;
    OptionalTensor table;
    int64_t table_ez;

    explicit LnsConfig(const Config& config)
        : precision(config.options.at("precision").toInt()),
          base(config.options.at("base").toDouble()),
          log_base(config.options.at("log_base").toDouble()),
          table_ez(config.options.at("table_ez").toInt()) {
        const auto value = config.options.at("table");
        if (!value.isNone()) table = value.toTensor();
    }
};
}
