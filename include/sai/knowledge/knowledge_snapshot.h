// knowledge_snapshot.h — 批次 4.1 SQLite SAVEPOINT 时间点快照
#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <sai/core/error.h>

struct sqlite3;

namespace sai::knowledge {

struct SnapshotInfo {
    std::int64_t snapshot_id = 0;
    std::string label;
    std::chrono::system_clock::time_point created_at;
    std::int64_t node_count = 0;
    std::int64_t edge_count = 0;
};

class KnowledgeSnapshot final {
public:
    explicit KnowledgeSnapshot(sqlite3* db) noexcept;

    [[nodiscard]] auto Create(std::string label) noexcept -> Result<std::int64_t>;
    [[nodiscard]] auto List() const noexcept -> Result<std::vector<SnapshotInfo>>;
    [[nodiscard]] auto Restore(std::int64_t snapshot_id) noexcept -> Result<void>;
    [[nodiscard]] auto Delete(std::int64_t snapshot_id) noexcept -> Result<void>;

    KnowledgeSnapshot(const KnowledgeSnapshot&) = delete;
    auto operator=(const KnowledgeSnapshot&) -> KnowledgeSnapshot& = delete;
    KnowledgeSnapshot(KnowledgeSnapshot&&) noexcept = default;
    auto operator=(KnowledgeSnapshot&&) noexcept -> KnowledgeSnapshot& = default;

private:
    sqlite3* db_;
};

}  // namespace sai::knowledge
