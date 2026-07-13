// knowledge_evolution.cpp — SQLite 变更日志 CRUD 实现
#include <sai/knowledge/knowledge_evolution.h>
#include <sqlite3.h>
#include <nlohmann/json.hpp>
#include <cstring>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace sai::knowledge {

namespace {

auto OpToString(EvolutionOp op) -> const char* {
    switch (op) {
        case EvolutionOp::Insert: return "Insert";
        case EvolutionOp::Update: return "Update";
        case EvolutionOp::Delete: return "Delete";
    }
    return "Unknown";
}

auto StringToOp(const char* s) -> EvolutionOp {
    if (std::strcmp(s, "Insert") == 0) return EvolutionOp::Insert;
    if (std::strcmp(s, "Update") == 0) return EvolutionOp::Update;
    return EvolutionOp::Delete;
}

auto ParseTimestamp(const char* ts) -> std::chrono::system_clock::time_point {
    std::tm tm = {};
    std::istringstream ss(ts);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

}  // anonymous namespace

KnowledgeEvolution::KnowledgeEvolution(sqlite3* db) noexcept : db_(db) {}

auto KnowledgeEvolution::Append(std::string entity_type, std::int64_t entity_id,
                                 EvolutionOp op, KnowledgeRecord before_image,
                                 std::string changed_by) noexcept -> Result<void> {
    const char* ver_sql = "SELECT COALESCE(MAX(version), 0) + 1 FROM evolution_log "
                           "WHERE entity_type = ? AND entity_id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, ver_sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, entity_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, entity_id);
    std::int64_t next_version = 1;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        next_version = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);

    std::string before_json = op == EvolutionOp::Insert
        ? "" : RecordToJson(before_image).dump();

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        now - std::chrono::system_clock::from_time_t(t)).count();
    std::ostringstream ts;
    ts << std::put_time(std::gmtime(&t), "%Y-%m-%d %H:%M:%S") << "."
       << std::setw(6) << std::setfill('0') << us;
    auto ts_str = ts.str();

    const char* sql =
        "INSERT INTO evolution_log (entity_type, entity_id, operation, version, "
        "changed_by, before_image_json, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)";
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, entity_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, entity_id);
    sqlite3_bind_text(stmt, 3, OpToString(op), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 4, next_version);
    sqlite3_bind_text(stmt, 5, changed_by.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 6, before_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 7, ts_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return {};
}

auto KnowledgeEvolution::GetHistory(std::string_view entity_type,
                                      std::int64_t entity_id) const noexcept
    -> Result<std::vector<EvolutionEntry>> {
    const char* sql =
        "SELECT entry_id, entity_type, entity_id, operation, version, timestamp, "
        "changed_by, before_image_json "
        "FROM evolution_log WHERE entity_type = ? AND entity_id = ? ORDER BY version ASC";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, entity_type.data(),
                      static_cast<int>(entity_type.size()), SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, entity_id);

    std::vector<EvolutionEntry> entries;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        EvolutionEntry e;
        e.entry_id = sqlite3_column_int64(stmt, 0);
        e.entity_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        e.entity_id = sqlite3_column_int64(stmt, 2);
        e.operation = StringToOp(
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
        e.version = sqlite3_column_int64(stmt, 4);
        e.timestamp = ParseTimestamp(
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
        e.changed_by = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        auto bj = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        if (bj && bj[0] != '\0') {
            e.before_image = JsonToRecord(nlohmann::json::parse(bj));
        }
        entries.push_back(std::move(e));
    }
    sqlite3_finalize(stmt);
    return entries;
}

auto KnowledgeEvolution::GetChangesSince(
    std::chrono::system_clock::time_point since) const noexcept
    -> Result<std::vector<EvolutionEntry>> {
    auto t = std::chrono::system_clock::to_time_t(since);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        since - std::chrono::system_clock::from_time_t(t)).count();
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&t), "%Y-%m-%d %H:%M:%S") << "."
        << std::setw(6) << std::setfill('0') << us;
    auto ts_str = oss.str();

    const char* sql =
        "SELECT entry_id, entity_type, entity_id, operation, version, timestamp, "
        "changed_by, before_image_json "
        "FROM evolution_log WHERE timestamp > ? ORDER BY timestamp ASC";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, ts_str.c_str(), -1, SQLITE_TRANSIENT);

    std::vector<EvolutionEntry> entries;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        EvolutionEntry e;
        e.entry_id = sqlite3_column_int64(stmt, 0);
        e.entity_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        e.entity_id = sqlite3_column_int64(stmt, 2);
        e.operation = StringToOp(
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
        e.version = sqlite3_column_int64(stmt, 4);
        e.timestamp = ParseTimestamp(
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
        e.changed_by = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        auto bj = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        if (bj && bj[0] != '\0') {
            e.before_image = JsonToRecord(nlohmann::json::parse(bj));
        }
        entries.push_back(std::move(e));
    }
    sqlite3_finalize(stmt);
    return entries;
}

}  // namespace sai::knowledge
