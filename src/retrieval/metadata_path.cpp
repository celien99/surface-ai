#include <sai/retrieval/metadata_path.h>
#include <sqlite3.h>
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

auto BindJsonExtract(sqlite3_stmt* stmt, int col, const std::string& field,
                     const FilterCondition& cond) -> void {
    // SQLite json_extract: json_extract(properties_json, '$.field')
    // We bind the json path as part of the WHERE clause
    (void)stmt; (void)col; (void)field; (void)cond;
}

}  // anonymous namespace

auto MetadataPath::Search(const Config& cfg) const noexcept
    -> Result<std::vector<MetadataResult>> {
    std::ostringstream sql;
    sql << "SELECT id FROM nodes WHERE 1=1";

    // Filter by node types
    if (!cfg.node_types.empty()) {
        sql << " AND type IN (";
        for (std::size_t i = 0; i < cfg.node_types.size(); ++i) {
            if (i > 0) sql << ", ";
            sql << "'" << cfg.node_types[i] << "'";
        }
        sql << ")";
    }

    // Filter by field conditions via json_extract
    for (const auto& filter : cfg.filters) {
        sql << " AND json_extract(properties_json, '$." << filter.field << "') "
            << OpToSql(filter.op) << " ";
        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::int64_t>) {
                sql << v;
            } else if constexpr (std::is_same_v<T, double>) {
                sql << v;
            } else if constexpr (std::is_same_v<T, std::string>) {
                sql << "'" << v << "'";
            } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
                sql << "(";
                for (std::size_t i = 0; i < v.size(); ++i) {
                    if (i > 0) sql << ", ";
                    sql << v[i];
                }
                sql << ")";
            }
        }, filter.value);
    }

    sql << " LIMIT " << cfg.max_results;

    sqlite3_stmt* stmt = nullptr;
    auto query = sql.str();
    if (sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }

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
