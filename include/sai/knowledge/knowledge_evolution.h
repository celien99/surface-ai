// knowledge_evolution.h — 批次 4.1 知识变更日志
#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <sai/core/error.h>
#include <sai/knowledge/knowledge_record.h>

struct sqlite3;

namespace sai::knowledge {

enum class EvolutionOp : std::uint8_t { Insert, Update, Delete };

struct EvolutionEntry {
    std::int64_t entry_id = 0;
    std::string entity_type;
    std::int64_t entity_id;
    EvolutionOp operation;
    std::int64_t version = 0;
    std::chrono::system_clock::time_point timestamp;
    std::string changed_by;
    KnowledgeRecord before_image;
};

class KnowledgeEvolution final {
public:
    explicit KnowledgeEvolution(sqlite3* db) noexcept;

    [[nodiscard]] auto Append(std::string entity_type, std::int64_t entity_id,
                               EvolutionOp op, KnowledgeRecord before_image,
                               std::string changed_by) noexcept -> Result<void>;

    [[nodiscard]] auto GetHistory(std::string_view entity_type,
                                    std::int64_t entity_id) const noexcept
        -> Result<std::vector<EvolutionEntry>>;

    [[nodiscard]] auto GetChangesSince(
        std::chrono::system_clock::time_point since) const noexcept
        -> Result<std::vector<EvolutionEntry>>;

    KnowledgeEvolution(const KnowledgeEvolution&) = delete;
    auto operator=(const KnowledgeEvolution&) -> KnowledgeEvolution& = delete;
    KnowledgeEvolution(KnowledgeEvolution&&) noexcept = default;
    auto operator=(KnowledgeEvolution&&) noexcept -> KnowledgeEvolution& = default;

private:
    sqlite3* db_;
};

}  // namespace sai::knowledge
