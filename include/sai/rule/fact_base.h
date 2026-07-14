#pragma once

#include <chrono>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sai/rule/value.h"

namespace sai::rule {

enum class FactSourceKind {
    Direct,
    GraphPath,
    VectorSearch,
    Computed,
    Default
};

struct FactSource {
    FactSourceKind kind;
    std::string description;
    std::chrono::microseconds elapsed{0};
    std::optional<std::string> sql;
    std::optional<int> top_k;
};

class FactBase {
public:
    auto Set(std::string_view key, Value value, FactSource source) -> void;
    auto SetDefault(std::string_view key, Value value) -> void;

    auto Get(std::string_view key) const -> std::optional<Value>;
    auto GetOr(std::string_view key, Value default_val) const -> Value;
    auto Has(std::string_view key) const -> bool;

    auto AddPathMapping(std::string_view graph_path, std::string_view flat_key) -> void;
    auto ResolvePath(std::string_view graph_path) const -> std::optional<std::string>;

    auto SourceOf(std::string_view key) const -> const FactSource&;
    auto AllEntries() const -> std::vector<std::pair<std::string, Value>>;
    auto AllSources() const -> std::vector<std::pair<std::string, FactSource>>;

    auto Snapshot() const -> FactBase;

private:
    struct Entry {
        Value value;
        FactSource source;
    };
    std::map<std::string, Entry, std::less<>> entries_;
    std::map<std::string, std::string, std::less<>> path_mappings_;

    static const FactSource kDefaultSource_;
};

}  // namespace sai::rule
