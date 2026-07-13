// knowledge_snapshot.cpp — SQLite SAVEPOINT 时间点快照 CRUD 实现
#include <sai/knowledge/knowledge_snapshot.h>
#include <sqlite3.h>
#include <source_location>
#include <sstream>
#include <ctime>
#include <iomanip>

namespace sai::knowledge {

KnowledgeSnapshot::KnowledgeSnapshot(sqlite3* db) noexcept : db_(db) {}

auto KnowledgeSnapshot::Create(std::string label) noexcept -> Result<std::int64_t> {
    // Insert snapshot metadata row to get an ID
    const char* ins_sql = "INSERT INTO snapshots (label, savepoint_name) VALUES (?, '')";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, ins_sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, label.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    auto snapshot_id = static_cast<std::int64_t>(sqlite3_last_insert_rowid(db_));
    sqlite3_finalize(stmt);

    // Create SAVEPOINT with ID-based name
    auto sp_name = "sp_" + std::to_string(snapshot_id);
    auto sql = "SAVEPOINT " + sp_name;
    char* err = nullptr;
    if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown";
        sqlite3_free(err);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed, msg,
            std::source_location::current(),
        });
    }

    // Count current nodes and edges
    std::int64_t node_count = 0, edge_count = 0;
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM nodes", -1, &stmt, nullptr);
    if (sqlite3_step(stmt) == SQLITE_ROW) node_count = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM edges", -1, &stmt, nullptr);
    if (sqlite3_step(stmt) == SQLITE_ROW) edge_count = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);

    // Update the savepoint_name and counts
    const char* upd_sql = "UPDATE snapshots SET savepoint_name = ?, node_count = ?, edge_count = ? WHERE snapshot_id = ?";
    sqlite3_prepare_v2(db_, upd_sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, sp_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, node_count);
    sqlite3_bind_int64(stmt, 3, edge_count);
    sqlite3_bind_int64(stmt, 4, snapshot_id);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return snapshot_id;
}

auto KnowledgeSnapshot::List() const noexcept -> Result<std::vector<SnapshotInfo>> {
    const char* sql = "SELECT snapshot_id, label, created_at, node_count, edge_count FROM snapshots ORDER BY snapshot_id";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    std::vector<SnapshotInfo> snapshots;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        SnapshotInfo info;
        info.snapshot_id = sqlite3_column_int64(stmt, 0);
        info.label = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        // parse timestamp
        std::tm tm = {};
        std::istringstream ss(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        info.created_at = std::chrono::system_clock::from_time_t(timegm(&tm));
        info.node_count = sqlite3_column_int64(stmt, 3);
        info.edge_count = sqlite3_column_int64(stmt, 4);
        snapshots.push_back(info);
    }
    sqlite3_finalize(stmt);
    return snapshots;
}

auto KnowledgeSnapshot::Restore(std::int64_t snapshot_id) noexcept -> Result<void> {
    const char* sql = "SELECT savepoint_name FROM snapshots WHERE snapshot_id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, snapshot_id);
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_SnapshotNotFound,
            "snapshot " + std::to_string(snapshot_id) + " not found",
            std::source_location::current(),
        });
    }
    auto sp_name = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    sqlite3_finalize(stmt);

    auto rollback_sql = "ROLLBACK TO SAVEPOINT " + sp_name;
    char* err = nullptr;
    if (sqlite3_exec(db_, rollback_sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown";
        sqlite3_free(err);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_SnapshotRestoreFailed, msg,
            std::source_location::current(),
        });
    }
    return {};
}

auto KnowledgeSnapshot::Delete(std::int64_t snapshot_id) noexcept -> Result<void> {
    const char* sql = "SELECT savepoint_name FROM snapshots WHERE snapshot_id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, snapshot_id);
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_SnapshotNotFound,
            "snapshot " + std::to_string(snapshot_id) + " not found",
            std::source_location::current(),
        });
    }
    auto sp_name = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    sqlite3_finalize(stmt);

    // Release the SAVEPOINT
    auto release_sql = "RELEASE SAVEPOINT " + sp_name;
    sqlite3_exec(db_, release_sql.c_str(), nullptr, nullptr, nullptr);

    // Delete metadata row
    const char* del_sql = "DELETE FROM snapshots WHERE snapshot_id = ?";
    sqlite3_prepare_v2(db_, del_sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, snapshot_id);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return {};
}

}  // namespace sai::knowledge
