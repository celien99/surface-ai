// knowledge_record.h — 批次 4.1 类型化知识记录字段容器
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace sai::knowledge {

using FieldValue = std::variant<
    std::int64_t,
    double,
    std::string,
    std::vector<std::uint8_t>
>;

struct KnowledgeRecord {
    std::map<std::string, FieldValue> fields;
};

// JSON 序列化（实现在 knowledge_record.cpp，供 KnowledgeGraph 等 SQLite 存储使用）
auto RecordToJson(const KnowledgeRecord& record) -> nlohmann::json;
auto JsonToRecord(const nlohmann::json& j) -> KnowledgeRecord;

}  // namespace sai::knowledge
