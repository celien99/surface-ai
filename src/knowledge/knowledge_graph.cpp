// knowledge_graph.cpp — SQLite 属性图节点/边 CRUD 实现
#include <sai/knowledge/knowledge_graph.h>
#include <sqlite3.h>
#include <nlohmann/json.hpp>
#include <source_location>

namespace sai::knowledge {

KnowledgeGraph::KnowledgeGraph(sqlite3* db) noexcept : db_(db) {}

auto KnowledgeGraph::InsertNode(std::string type, KnowledgeRecord properties) noexcept
    -> Result<NodeId> {
    auto json_str = RecordToJson(properties).dump();
    const char* sql = "INSERT INTO nodes (type, properties_json) VALUES (?, ?)";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("prepare insert node: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_text(stmt, 1, type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, json_str.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("insert node: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    auto id = static_cast<NodeId>(sqlite3_last_insert_rowid(db_));
    sqlite3_finalize(stmt);
    return id;
}

auto KnowledgeGraph::InsertNodesBatch(
    std::vector<std::pair<std::string, KnowledgeRecord>> entries) noexcept
    -> Result<std::vector<NodeId>> {
    if (entries.empty()) return std::vector<NodeId>{};

    // BEGIN TRANSACTION
    char* err_msg = nullptr;
    if (sqlite3_exec(db_, "BEGIN IMMEDIATE", nullptr, nullptr, &err_msg) != SQLITE_OK) {
        std::string msg = err_msg ? err_msg : "unknown";
        sqlite3_free(err_msg);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            "BEGIN IMMEDIATE failed: " + msg,
            std::source_location::current(),
        });
    }

    const char* sql = "INSERT INTO nodes (type, properties_json) VALUES (?, ?)";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("prepare batch insert: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }

    std::vector<NodeId> ids;
    ids.reserve(entries.size());

    for (auto& [type, props] : entries) {
        auto json_str = RecordToJson(props).dump();
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
        sqlite3_bind_text(stmt, 1, type.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, json_str.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Knowledge_DbOpenFailed,
                std::string("batch insert row: ") + sqlite3_errmsg(db_),
                std::source_location::current(),
            });
        }
        ids.push_back(static_cast<NodeId>(sqlite3_last_insert_rowid(db_)));
    }
    sqlite3_finalize(stmt);

    // COMMIT
    if (sqlite3_exec(db_, "COMMIT", nullptr, nullptr, &err_msg) != SQLITE_OK) {
        std::string msg = err_msg ? err_msg : "unknown";
        sqlite3_free(err_msg);
        sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            "COMMIT failed: " + msg,
            std::source_location::current(),
        });
    }

    return ids;
}

auto KnowledgeGraph::UpdateNode(NodeId id, KnowledgeRecord properties) noexcept -> Result<void> {
    auto json_str = RecordToJson(properties).dump();
    const char* sql = "UPDATE nodes SET properties_json = ?, updated_at = datetime('now') WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed, sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_text(stmt, 1, json_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, id);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed, sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_finalize(stmt);
    if (sqlite3_changes(db_) == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_NodeNotFound,
            "node " + std::to_string(id) + " not found",
            std::source_location::current(),
        });
    }
    return {};
}

auto KnowledgeGraph::SetNodeField(NodeId id, std::string key, FieldValue value) noexcept -> Result<void> {
    // Read existing properties, update one field, write back
    auto node_result = GetNode(id);
    if (!node_result.has_value()) {
        return tl::make_unexpected(node_result.error());
    }
    node_result->properties.fields[std::move(key)] = std::move(value);
    return UpdateNode(id, std::move(node_result->properties));
}

auto KnowledgeGraph::DeleteNode(NodeId id) noexcept -> Result<void> {
    const char* sql = "DELETE FROM nodes WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("prepare delete node: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_int64(stmt, 1, id);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("delete node: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_finalize(stmt);
    return {};
}

auto KnowledgeGraph::GetNode(NodeId id) const noexcept -> Result<KnowledgeNode> {
    const char* sql = "SELECT id, type, properties_json FROM nodes WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_int64(stmt, 1, id);
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_NodeNotFound,
            "node " + std::to_string(id) + " not found",
            std::source_location::current(),
        });
    }
    KnowledgeNode node;
    node.id = sqlite3_column_int64(stmt, 0);
    node.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    node.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
    sqlite3_finalize(stmt);
    return node;
}

auto KnowledgeGraph::FindNodesByType(std::string_view type) const noexcept
    -> Result<std::vector<KnowledgeNode>> {
    const char* sql = "SELECT id, type, properties_json FROM nodes WHERE type = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_text(stmt, 1, type.data(), static_cast<int>(type.size()), SQLITE_TRANSIENT);
    std::vector<KnowledgeNode> nodes;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        KnowledgeNode node;
        node.id = sqlite3_column_int64(stmt, 0);
        node.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        node.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
        nodes.push_back(std::move(node));
    }
    sqlite3_finalize(stmt);
    return nodes;
}

auto KnowledgeGraph::InsertEdge(NodeId source, NodeId target,
                                 std::string relationship,
                                 KnowledgeRecord properties) noexcept -> Result<EdgeId> {
    if (relationship.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_InvalidRelationship,
            "relationship cannot be empty",
            std::source_location::current(),
        });
    }
    auto json_str = RecordToJson(properties).dump();
    const char* sql = "INSERT INTO edges (source_id, target_id, relationship, properties_json) VALUES (?, ?, ?, ?)";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("prepare insert edge: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_int64(stmt, 1, source);
    sqlite3_bind_int64(stmt, 2, target);
    sqlite3_bind_text(stmt, 3, relationship.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, json_str.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("insert edge: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    auto id = static_cast<EdgeId>(sqlite3_last_insert_rowid(db_));
    sqlite3_finalize(stmt);
    return id;
}

auto KnowledgeGraph::DeleteEdge(EdgeId id) noexcept -> Result<void> {
    const char* sql = "DELETE FROM edges WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("prepare delete edge: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_int64(stmt, 1, id);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("delete edge: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_finalize(stmt);
    return {};
}

auto KnowledgeGraph::GetEdge(EdgeId id) const noexcept -> Result<KnowledgeEdge> {
    const char* sql = "SELECT id, source_id, target_id, relationship, properties_json FROM edges WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_int64(stmt, 1, id);
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_EdgeNotFound,
            "edge " + std::to_string(id) + " not found",
            std::source_location::current(),
        });
    }
    KnowledgeEdge edge;
    edge.id = sqlite3_column_int64(stmt, 0);
    edge.source_id = sqlite3_column_int64(stmt, 1);
    edge.target_id = sqlite3_column_int64(stmt, 2);
    edge.relationship = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
    edge.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
    sqlite3_finalize(stmt);
    return edge;
}

auto KnowledgeGraph::Traverse(NodeId from, std::string_view relationship,
                               std::size_t max_depth) const noexcept
    -> Result<std::vector<GraphPath>> {
    std::unordered_set<NodeId> visited;
    return TraverseImpl(from, relationship, max_depth, visited);
}

auto KnowledgeGraph::TraverseImpl(NodeId from, std::string_view relationship,
                                   std::size_t max_depth,
                                   std::unordered_set<NodeId>& visited) const noexcept
    -> Result<std::vector<GraphPath>> {
    if (max_depth == 0) {
        return std::vector<GraphPath>{};
    }

    visited.insert(from);

    const char* sql =
        "SELECT e.id, n.id, n.type, n.properties_json "
        "FROM edges e JOIN nodes n ON e.target_id = n.id "
        "WHERE e.source_id = ? AND e.relationship = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("prepare traverse: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_int64(stmt, 1, from);
    sqlite3_bind_text(stmt, 2, relationship.data(), static_cast<int>(relationship.size()), SQLITE_TRANSIENT);

    // Collect direct targets and their IDs for recursion
    GraphPath path;
    path.source = from;
    path.relationship = relationship;
    std::vector<NodeId> neighbor_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        KnowledgeNode node;
        node.id = sqlite3_column_int64(stmt, 1);
        node.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        node.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
        neighbor_ids.push_back(node.id);
        path.targets.push_back(std::move(node));
    }
    sqlite3_finalize(stmt);

    std::vector<GraphPath> results;
    if (!path.targets.empty()) {
        results.push_back(std::move(path));
    }

    // Recursively traverse from each unvisited neighbor for deeper levels
    if (max_depth > 1) {
        for (auto neighbor_id : neighbor_ids) {
            if (visited.count(neighbor_id)) continue;
            auto deeper = TraverseImpl(neighbor_id, relationship, max_depth - 1, visited);
            if (!deeper.has_value()) {
                return tl::make_unexpected(deeper.error());
            }
            auto& deeper_paths = *deeper;
            results.insert(results.end(),
                           std::make_move_iterator(deeper_paths.begin()),
                           std::make_move_iterator(deeper_paths.end()));
        }
    }

    return results;
}

auto KnowledgeGraph::ReverseTraverse(NodeId to, std::string_view relationship,
                                      std::size_t max_depth) const noexcept
    -> Result<std::vector<GraphPath>> {
    std::unordered_set<NodeId> visited;
    return ReverseTraverseImpl(to, relationship, max_depth, visited);
}

auto KnowledgeGraph::ReverseTraverseImpl(NodeId to, std::string_view relationship,
                                          std::size_t max_depth,
                                          std::unordered_set<NodeId>& visited) const noexcept
    -> Result<std::vector<GraphPath>> {
    if (max_depth == 0) {
        return std::vector<GraphPath>{};
    }

    visited.insert(to);

    const char* sql =
        "SELECT e.id, n.id, n.type, n.properties_json "
        "FROM edges e JOIN nodes n ON e.source_id = n.id "
        "WHERE e.target_id = ? AND e.relationship = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("prepare reverse_traverse: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_int64(stmt, 1, to);
    sqlite3_bind_text(stmt, 2, relationship.data(), static_cast<int>(relationship.size()), SQLITE_TRANSIENT);

    GraphPath path;
    path.source = to;
    path.relationship = relationship;
    std::vector<NodeId> neighbor_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        KnowledgeNode node;
        node.id = sqlite3_column_int64(stmt, 1);
        node.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        node.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
        neighbor_ids.push_back(node.id);
        path.targets.push_back(std::move(node));
    }
    sqlite3_finalize(stmt);

    std::vector<GraphPath> results;
    if (!path.targets.empty()) {
        results.push_back(std::move(path));
    }

    // Recursively reverse-traverse from each unvisited neighbor for deeper levels
    if (max_depth > 1) {
        for (auto neighbor_id : neighbor_ids) {
            if (visited.count(neighbor_id)) continue;
            auto deeper = ReverseTraverseImpl(neighbor_id, relationship, max_depth - 1, visited);
            if (!deeper.has_value()) {
                return tl::make_unexpected(deeper.error());
            }
            auto& deeper_paths = *deeper;
            results.insert(results.end(),
                           std::make_move_iterator(deeper_paths.begin()),
                           std::make_move_iterator(deeper_paths.end()));
        }
    }

    return results;
}

auto KnowledgeGraph::NodeCount() const noexcept -> std::size_t {
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM nodes", -1, &stmt, nullptr) != SQLITE_OK) {
        return 0;
    }
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        auto count = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
        sqlite3_finalize(stmt);
        return count;
    }
    sqlite3_finalize(stmt);
    return 0;
}

auto KnowledgeGraph::EdgeCount() const noexcept -> std::size_t {
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM edges", -1, &stmt, nullptr) != SQLITE_OK) {
        return 0;
    }
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        auto count = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
        sqlite3_finalize(stmt);
        return count;
    }
    sqlite3_finalize(stmt);
    return 0;
}

}  // namespace sai::knowledge
