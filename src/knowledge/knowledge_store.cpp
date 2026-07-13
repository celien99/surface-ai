#include <sai/knowledge/knowledge_store.h>
#include <sqlite3.h>
#include <source_location>

namespace sai::knowledge {

namespace {

const char* kSchemaSQL = R"(
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        properties_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
    CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
        target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
        relationship TEXT NOT NULL,
        properties_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
    CREATE INDEX IF NOT EXISTS idx_edges_relationship ON edges(relationship);
    CREATE TABLE IF NOT EXISTS evolution_log (
        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type TEXT NOT NULL,
        entity_id INTEGER NOT NULL,
        operation TEXT NOT NULL,
        version INTEGER NOT NULL,
        timestamp TEXT NOT NULL DEFAULT (datetime('now')),
        changed_by TEXT NOT NULL DEFAULT 'system',
        before_image_json TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_evolution_entity ON evolution_log(entity_type, entity_id);
    CREATE INDEX IF NOT EXISTS idx_evolution_timestamp ON evolution_log(timestamp);
    CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT NOT NULL,
        savepoint_name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        node_count INTEGER NOT NULL DEFAULT 0,
        edge_count INTEGER NOT NULL DEFAULT 0
    );
    INSERT OR IGNORE INTO schema_version (version) VALUES (1);
)";

auto RunSchema(sqlite3* db) -> Result<void> {
    char* err = nullptr;
    if (sqlite3_exec(db, kSchemaSQL, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown";
        sqlite3_free(err);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_SchemaMigrationFailed, msg,
            std::source_location::current(),
        });
    }
    return {};
}

}  // anonymous namespace

auto KnowledgeStore::Sqlite3Deleter::operator()(sqlite3* db) const noexcept -> void {
    sqlite3_close(db);
}

auto KnowledgeStore::Create(const Config& cfg) noexcept -> Result<std::unique_ptr<KnowledgeStore>> {
    sqlite3* raw_db = nullptr;
    auto path_str = cfg.db_path.string();
    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
    if (path_str == ":memory:") {
        flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_MEMORY;
    }
    if (sqlite3_open_v2(path_str.c_str(), &raw_db, flags, nullptr) != SQLITE_OK) {
        std::string msg = sqlite3_errmsg(raw_db);
        sqlite3_close(raw_db);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed, msg,
            std::source_location::current(),
        });
    }
    // Enable WAL mode for concurrent reads
    sqlite3_exec(raw_db, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr);

    auto store = std::unique_ptr<KnowledgeStore>(new KnowledgeStore());
    store->db_.reset(raw_db);
    store->config_ = cfg;

    auto schema_result = RunSchema(store->db_.get());
    if (!schema_result.has_value()) {
        return tl::make_unexpected(schema_result.error());
    }

    store->graph_ = KnowledgeGraph(store->db_.get());
    store->evolution_ = KnowledgeEvolution(store->db_.get());
    store->snapshot_ = KnowledgeSnapshot(store->db_.get());

    return store;
}

KnowledgeStore::~KnowledgeStore() = default;

auto KnowledgeStore::InsertNode(std::string type, KnowledgeRecord properties,
                                 std::string changed_by) noexcept -> Result<NodeId> {
    auto prev = GetNode(0);  // dummy — Insert never has before_image
    KnowledgeRecord before;
    auto result = graph_.InsertNode(std::move(type), properties);
    if (result.has_value() && config_.enable_evolution) {
        (void)evolution_.Append("Node", *result, EvolutionOp::Insert, before, std::move(changed_by));
    }
    return result;
}

auto KnowledgeStore::UpdateNode(NodeId id, KnowledgeRecord properties,
                                 std::string changed_by) noexcept -> Result<void> {
    KnowledgeRecord before;
    if (config_.enable_evolution) {
        auto prev = graph_.GetNode(id);
        if (prev.has_value()) before = std::move(prev->properties);
    }
    auto result = graph_.UpdateNode(id, std::move(properties));
    if (result.has_value() && config_.enable_evolution) {
        (void)evolution_.Append("Node", id, EvolutionOp::Update, before, std::move(changed_by));
    }
    return result;
}

auto KnowledgeStore::DeleteNode(NodeId id, std::string changed_by) noexcept -> Result<void> {
    KnowledgeRecord before;
    if (config_.enable_evolution) {
        auto prev = graph_.GetNode(id);
        if (prev.has_value()) before = std::move(prev->properties);
    }
    auto result = graph_.DeleteNode(id);
    if (result.has_value() && config_.enable_evolution) {
        (void)evolution_.Append("Node", id, EvolutionOp::Delete, before, std::move(changed_by));
    }
    return result;
}

auto KnowledgeStore::GetNode(NodeId id) const noexcept -> Result<KnowledgeNode> {
    return graph_.GetNode(id);
}

auto KnowledgeStore::FindNodesByType(std::string_view type) const noexcept
    -> Result<std::vector<KnowledgeNode>> {
    return graph_.FindNodesByType(type);
}

auto KnowledgeStore::InsertEdge(NodeId source, NodeId target,
                                 std::string relationship, KnowledgeRecord properties,
                                 std::string changed_by) noexcept -> Result<EdgeId> {
    auto result = graph_.InsertEdge(source, target, std::move(relationship), std::move(properties));
    if (result.has_value() && config_.enable_evolution) {
        (void)evolution_.Append("Edge", *result, EvolutionOp::Insert, KnowledgeRecord{}, std::move(changed_by));
    }
    return result;
}

auto KnowledgeStore::GetEdge(EdgeId id) const noexcept -> Result<KnowledgeEdge> {
    return graph_.GetEdge(id);
}

auto KnowledgeStore::Traverse(NodeId from, std::string_view relationship) const noexcept
    -> Result<std::vector<GraphPath>> {
    return graph_.Traverse(from, relationship);
}

auto KnowledgeStore::CreateSnapshot(std::string label) noexcept -> Result<std::int64_t> {
    return snapshot_.Create(std::move(label));
}

auto KnowledgeStore::ListSnapshots() const noexcept -> Result<std::vector<SnapshotInfo>> {
    return snapshot_.List();
}

auto KnowledgeStore::RestoreSnapshot(std::int64_t snapshot_id) noexcept -> Result<void> {
    return snapshot_.Restore(snapshot_id);
}

auto KnowledgeStore::GetEntityHistory(std::string_view entity_type,
                                        std::int64_t entity_id) const noexcept
    -> Result<std::vector<EvolutionEntry>> {
    return evolution_.GetHistory(entity_type, entity_id);
}

auto KnowledgeStore::GetChangesSince(
    std::chrono::system_clock::time_point since) const noexcept
    -> Result<std::vector<EvolutionEntry>> {
    return evolution_.GetChangesSince(since);
}

}  // namespace sai::knowledge
