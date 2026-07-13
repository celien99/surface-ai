#include <sai/retrieval/metadata_path.h>
#include <sqlite3.h>
#include <regex>
#include <source_location>
#include <sstream>

namespace sai::retrieval {

MetadataPath::MetadataPath(sqlite3* db) noexcept : db_(db) {}

namespace {

auto OpToSql(FilterOp op) -> const char* {
    switch (op) {
        case FilterOp::Equal: return "=";
        case FilterOp::NotEqual: return "!=";
        case FilterOp::LessThan: return "<";
        case FilterOp::GreaterThan: return ">";
        case FilterOp::LessOrEqual: return "<=";
        case FilterOp::GreaterOrEqual: return ">=";
        case FilterOp::Like: return "LIKE";
        case FilterOp::In: return "IN";
    }
    return "=";
}

}  // anonymous namespace

auto MetadataPath::Search(const Config& cfg) const noexcept
    -> Result<std::vector<MetadataResult>> {
    // Validate field names against whitelist pattern before any SQL construction.
    // Field names appear in json_extract paths which cannot be parameterized,
    // so they must be validated to prevent injection.
    static const std::regex kFieldNamePattern("^[a-zA-Z_][a-zA-Z0-9_]*$");
    for (const auto& filter : cfg.filters) {
        if (!std::regex_match(filter.field, kFieldNamePattern)) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Infra_ConfigValidationFailed,
                "Invalid field name in filter: '" + filter.field +
                    "' does not match [a-zA-Z_][a-zA-Z0-9_]*",
                std::source_location::current(),
            });
        }
    }

    std::ostringstream sql;
    sql << "SELECT id FROM nodes WHERE 1=1";
    int param_index = 1;

    // Filter by node types — each type string bound individually
    if (!cfg.node_types.empty()) {
        sql << " AND type IN (";
        for (std::size_t i = 0; i < cfg.node_types.size(); ++i) {
            if (i > 0) sql << ", ";
            sql << "?" << param_index++;
        }
        sql << ")";
    }

    // Filter by field conditions via json_extract.
    // Field names are already validated above; comparison values are bound as parameters.
    for (const auto& filter : cfg.filters) {
        sql << " AND json_extract(properties_json, '$." << filter.field << "') "
            << OpToSql(filter.op) << " ";
        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::int64_t>) {
                sql << "?" << param_index++;
            } else if constexpr (std::is_same_v<T, double>) {
                sql << "?" << param_index++;
            } else if constexpr (std::is_same_v<T, std::string>) {
                sql << "?" << param_index++;
            } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
                sql << "(";
                for (std::size_t i = 0; i < v.size(); ++i) {
                    if (i > 0) sql << ", ";
                    sql << "?" << param_index++;
                }
                sql << ")";
            }
        }, filter.value);
    }

    sql << " LIMIT ?" << param_index++;

    // Prepare statement
    sqlite3_stmt* stmt = nullptr;
    auto query = sql.str();
    if (sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Infra_ConfigValidationFailed,
            sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }

    // Bind all parameters in the same order they appear in the SQL
    int bind_index = 1;

    for (const auto& node_type : cfg.node_types) {
        sqlite3_bind_text(stmt, bind_index++, node_type.c_str(),
                          -1, SQLITE_TRANSIENT);
    }

    for (const auto& filter : cfg.filters) {
        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::int64_t>) {
                sqlite3_bind_int64(stmt, bind_index++, v);
            } else if constexpr (std::is_same_v<T, double>) {
                sqlite3_bind_double(stmt, bind_index++, v);
            } else if constexpr (std::is_same_v<T, std::string>) {
                sqlite3_bind_text(stmt, bind_index++, v.c_str(),
                                  -1, SQLITE_TRANSIENT);
            } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
                for (auto elem : v) {
                    sqlite3_bind_int64(stmt, bind_index++, elem);
                }
            }
        }, filter.value);
    }

    sqlite3_bind_int64(stmt, bind_index++,
                       static_cast<std::int64_t>(cfg.max_results));

    // Execute query
    std::vector<MetadataResult> results;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        MetadataResult r;
        r.node_id = sqlite3_column_int64(stmt, 0);
        r.score = 1.0F;  // all matching results get full score (binary match)
        results.push_back(r);
    }
    sqlite3_finalize(stmt);
    return results;
}

}  // namespace sai::retrieval
