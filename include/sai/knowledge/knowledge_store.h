// knowledge_store.h — 批次 4.1 知识子系统统一门面
// Note: KnowledgeStore uses static Create() factory rather than IService DI — simplifies testing; Context registration deferred to M6.
#pragma once
#include <filesystem>
#include <memory>
#include <string>
#include <sai/core/error.h>
#include <sai/core/object.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/knowledge/knowledge_evolution.h>
#include <sai/knowledge/knowledge_snapshot.h>
#include <sai/knowledge/knowledge_record.h>

struct sqlite3;

namespace sai::knowledge {

class KnowledgeStore final : public Object {
public:
    struct Config {
        std::filesystem::path db_path;
        bool enable_evolution = true;
    };

    [[nodiscard]] static auto Create(const Config& cfg) noexcept
        -> Result<std::unique_ptr<KnowledgeStore>>;

    [[nodiscard]] auto InsertNode(std::string type, KnowledgeRecord properties,
                                    std::string changed_by = "system") noexcept -> Result<NodeId>;
    [[nodiscard]] auto UpdateNode(NodeId id, KnowledgeRecord properties,
                                    std::string changed_by = "system") noexcept -> Result<void>;
    [[nodiscard]] auto DeleteNode(NodeId id,
                                    std::string changed_by = "system") noexcept -> Result<void>;
    [[nodiscard]] auto GetNode(NodeId id) const noexcept -> Result<KnowledgeNode>;
    [[nodiscard]] auto FindNodesByType(std::string_view type) const noexcept
        -> Result<std::vector<KnowledgeNode>>;
    [[nodiscard]] auto InsertEdge(NodeId source, NodeId target,
                                    std::string relationship, KnowledgeRecord properties,
                                    std::string changed_by = "system") noexcept -> Result<EdgeId>;
    [[nodiscard]] auto GetEdge(EdgeId id) const noexcept -> Result<KnowledgeEdge>;
    [[nodiscard]] auto Traverse(NodeId from, std::string_view relationship,
                                std::size_t max_depth = 3) const noexcept
        -> Result<std::vector<GraphPath>>;

    [[nodiscard]] auto CreateSnapshot(std::string label) noexcept -> Result<std::int64_t>;
    [[nodiscard]] auto ListSnapshots() const noexcept -> Result<std::vector<SnapshotInfo>>;
    [[nodiscard]] auto RestoreSnapshot(std::int64_t snapshot_id) noexcept -> Result<void>;

    [[nodiscard]] auto GetEntityHistory(std::string_view entity_type,
                                          std::int64_t entity_id) const noexcept
        -> Result<std::vector<EvolutionEntry>>;
    [[nodiscard]] auto GetChangesSince(
        std::chrono::system_clock::time_point since) const noexcept
        -> Result<std::vector<EvolutionEntry>>;

    [[nodiscard]] auto LinkEmbedding(NodeId node_id, std::size_t vec_index) noexcept
        -> Result<void>;

    [[nodiscard]] auto DbHandle() const noexcept -> sqlite3* { return db_.get(); }
    [[nodiscard]] auto Graph() noexcept -> KnowledgeGraph& { return graph_; }
    [[nodiscard]] auto Graph() const noexcept -> const KnowledgeGraph& { return graph_; }
    [[nodiscard]] auto Evolution() noexcept -> KnowledgeEvolution& { return evolution_; }
    [[nodiscard]] auto Snapshot() noexcept -> KnowledgeSnapshot& { return snapshot_; }

    ~KnowledgeStore() override;

    KnowledgeStore(const KnowledgeStore&) = delete;
    auto operator=(const KnowledgeStore&) -> KnowledgeStore& = delete;
    KnowledgeStore(KnowledgeStore&&) = delete;
    auto operator=(KnowledgeStore&&) -> KnowledgeStore& = delete;

private:
    KnowledgeStore() noexcept = default;

    struct Sqlite3Deleter {
        auto operator()(sqlite3* db) const noexcept -> void;
    };
    std::unique_ptr<sqlite3, Sqlite3Deleter> db_;
    KnowledgeGraph graph_{nullptr};
    KnowledgeEvolution evolution_{nullptr};
    KnowledgeSnapshot snapshot_{nullptr};
    Config config_;
};

}  // namespace sai::knowledge
