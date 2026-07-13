// knowledge_record.cpp — KnowledgeRecord ↔ nlohmann::json 序列化
#include <sai/knowledge/knowledge_record.h>
#include <nlohmann/json.hpp>

namespace sai::knowledge {

namespace {

auto FieldValueToJson(const FieldValue& fv) -> nlohmann::json {
    return std::visit([](const auto& v) -> nlohmann::json {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::vector<std::uint8_t>>) {
            return nlohmann::json::binary(v);
        } else {
            return nlohmann::json(v);
        }
    }, fv);
}

auto JsonToFieldValue(const nlohmann::json& j) -> FieldValue {
    if (j.is_number_integer()) {
        return FieldValue{j.get<std::int64_t>()};
    }
    if (j.is_number_float()) {
        return FieldValue{j.get<double>()};
    }
    if (j.is_string()) {
        return FieldValue{j.get<std::string>()};
    }
    if (j.is_binary()) {
        auto bin = j.get_binary();
        return FieldValue{std::vector<std::uint8_t>(bin.begin(), bin.end())};
    }
    // Unsupported JSON types (bool, null, array, object) default to 0.
    // This is intentional: KnowledgeRecord only supports int64/double/string/binary.
    // Callers are expected to validate data before ingestion.
    return FieldValue{std::int64_t{0}};  // fallback
}

}  // anonymous namespace

auto RecordToJson(const KnowledgeRecord& record) -> nlohmann::json {
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [key, value] : record.fields) {
        j[key] = FieldValueToJson(value);
    }
    return j;
}

auto JsonToRecord(const nlohmann::json& j) -> KnowledgeRecord {
    KnowledgeRecord record;
    if (!j.is_object()) return record;
    for (auto it = j.begin(); it != j.end(); ++it) {
        record.fields[it.key()] = JsonToFieldValue(it.value());
    }
    return record;
}

}  // namespace sai::knowledge
