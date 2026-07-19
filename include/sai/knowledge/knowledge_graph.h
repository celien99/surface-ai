// knowledge_graph.h — 批次 4.1 SQLite 属性图存储
#pragma once
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include <sai/core/error.h>
#include <sai/knowledge/knowledge_record.h>

struct sqlite3;

namespace sai::knowledge {

using NodeId = std::int64_t;
using EdgeId = std::int64_t;

struct KnowledgeNode {
    NodeId id = 0;
    std::string type;
    KnowledgeRecord properties;
};

struct KnowledgeEdge {
    EdgeId id = 0;
    NodeId source_id = 0;
    NodeId target_id = 0;
    std::string relationship;
    KnowledgeRecord properties;
};

struct GraphPath {
    NodeId source;
    std::string relationship;
    std::vector<KnowledgeNode> targets;
};

class KnowledgeGraph final {
public:
    explicit KnowledgeGraph(sqlite3* db) noexcept;

    [[nodiscard]] auto InsertNode(std::string type, KnowledgeRecord properties) noexcept
        -> Result<NodeId>;
    // Batch insert: wraps N InsertNode calls in a single SQLite transaction.
    // Returns all inserted node IDs on success; on failure the transaction is
    // rolled back and no partial inserts are visible.
    [[nodiscard]] auto InsertNodesBatch(
        std::vector<std::pair<std::string, KnowledgeRecord>> entries) noexcept
        -> Result<std::vector<NodeId>>;
    [[nodiscard]] auto UpdateNode(NodeId id, KnowledgeRecord properties) noexcept -> Result<void>;
    // Set a single field on a node (read-modify-write of the properties JSON).
    [[nodiscard]] auto SetNodeField(NodeId id, std::string key, FieldValue value) noexcept -> Result<void>;
    [[nodiscard]] auto DeleteNode(NodeId id) noexcept -> Result<void>;
    [[nodiscard]] auto GetNode(NodeId id) const noexcept -> Result<KnowledgeNode>;
    [[nodiscard]] auto FindNodesByType(std::string_view type) const noexcept
        -> Result<std::vector<KnowledgeNode>>;

    [[nodiscard]] auto InsertEdge(NodeId source, NodeId target,
                                   std::string relationship,
                                   KnowledgeRecord properties) noexcept -> Result<EdgeId>;
    [[nodiscard]] auto DeleteEdge(EdgeId id) noexcept -> Result<void>;
    [[nodiscard]] auto GetEdge(EdgeId id) const noexcept -> Result<KnowledgeEdge>;

    [[nodiscard]] auto Traverse(NodeId from, std::string_view relationship,
                                std::size_t max_depth = 3) const noexcept
        -> Result<std::vector<GraphPath>>;
    [[nodiscard]] auto ReverseTraverse(NodeId to, std::string_view relationship,
                                       std::size_t max_depth = 3) const noexcept
        -> Result<std::vector<GraphPath>>;

    [[nodiscard]] auto NodeCount() const noexcept -> std::size_t;
    [[nodiscard]] auto EdgeCount() const noexcept -> std::size_t;

    KnowledgeGraph(const KnowledgeGraph&) = delete;
    auto operator=(const KnowledgeGraph&) -> KnowledgeGraph& = delete;
    KnowledgeGraph(KnowledgeGraph&&) noexcept = default;
    auto operator=(KnowledgeGraph&&) noexcept -> KnowledgeGraph& = default;

private:
    sqlite3* db_;

    [[nodiscard]] auto TraverseImpl(NodeId from, std::string_view relationship,
                                    std::size_t max_depth,
                                    std::unordered_set<NodeId>& visited) const noexcept
        -> Result<std::vector<GraphPath>>;
    [[nodiscard]] auto ReverseTraverseImpl(NodeId to, std::string_view relationship,
                                           std::size_t max_depth,
                                           std::unordered_set<NodeId>& visited) const noexcept
        -> Result<std::vector<GraphPath>>;
};

}  // namespace sai::knowledge
