#include "sai/rule/fact_base.h"

namespace sai::rule {

const FactSource FactBase::kDefaultSource_{
    FactSourceKind::Default, "", std::chrono::microseconds{0}, std::nullopt, std::nullopt};

auto FactBase::Set(std::string_view key, Value value, FactSource source) -> void {
    entries_.insert_or_assign(std::string{key}, Entry{std::move(value), std::move(source)});
}

auto FactBase::SetDefault(std::string_view key, Value value) -> void {
    if (entries_.find(key) == entries_.end()) {
        FactSource src;
        src.kind = FactSourceKind::Default;
        entries_.emplace(std::string{key}, Entry{std::move(value), std::move(src)});
    }
}

auto FactBase::Get(std::string_view key) const -> std::optional<Value> {
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        return it->second.value;
    }
    return std::nullopt;
}

auto FactBase::GetOr(std::string_view key, Value default_val) const -> Value {
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        return it->second.value;
    }
    return default_val;
}

auto FactBase::Has(std::string_view key) const -> bool {
    return entries_.find(key) != entries_.end();
}

auto FactBase::AddPathMapping(std::string_view graph_path, std::string_view flat_key) -> void {
    path_mappings_.emplace(std::string{graph_path}, std::string{flat_key});
}

auto FactBase::ResolvePath(std::string_view graph_path) const -> std::optional<std::string> {
    auto it = path_mappings_.find(graph_path);
    if (it != path_mappings_.end()) {
        return it->second;
    }
    return std::nullopt;
}

auto FactBase::SourceOf(std::string_view key) const -> const FactSource& {
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        return it->second.source;
    }
    return kDefaultSource_;
}

auto FactBase::AllEntries() const -> std::vector<std::pair<std::string, Value>> {
    std::vector<std::pair<std::string, Value>> result;
    result.reserve(entries_.size());
    for (const auto& [k, entry] : entries_) {
        result.emplace_back(k, entry.value);
    }
    return result;
}

auto FactBase::AllSources() const -> std::vector<std::pair<std::string, FactSource>> {
    std::vector<std::pair<std::string, FactSource>> result;
    result.reserve(entries_.size());
    for (const auto& [k, entry] : entries_) {
        result.emplace_back(k, entry.source);
    }
    return result;
}

auto FactBase::Snapshot() const -> FactBase {
    return *this;
}

}  // namespace sai::rule
