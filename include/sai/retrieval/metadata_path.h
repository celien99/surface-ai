// metadata_path.h — 批次 4.2 SQLite 结构化元数据查询
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>
#include <sai/core/error.h>

struct sqlite3;

namespace sai::retrieval {

enum class FilterOp : std::uint8_t {
    Equal, NotEqual, LessThan, GreaterThan,
    LessOrEqual, GreaterOrEqual, Like, In,
};

struct FilterCondition {
    std::string field;
    FilterOp op;
    std::variant<std::int64_t, double, std::string, std::vector<std::int64_t>> value;
};

struct MetadataResult {
    std::int64_t node_id;
    float score;
};

class MetadataPath final {
public:
    struct Config {
        std::vector<FilterCondition> filters;
        std::vector<std::string> node_types;
        std::size_t max_results = 100;
    };

    explicit MetadataPath(sqlite3* db) noexcept;

    [[nodiscard]] auto Search(const Config& cfg) const noexcept
        -> Result<std::vector<MetadataResult>>;

private:
    sqlite3* db_;
};

}  // namespace sai::retrieval
