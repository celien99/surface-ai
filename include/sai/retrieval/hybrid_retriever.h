// hybrid_retriever.h — 批次 4.2 混合检索引擎
#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <sai/core/error.h>
#include <sai/retrieval/score_fusion.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>

namespace sai::retrieval {

struct RetrievalItem {
    std::int64_t node_id;
    ScoreBreakdown scores;
    std::string node_type;
};

class HybridRetriever final {
public:
    struct Config {
        VectorPath::Config vector;
        MetadataPath::Config metadata;
    };

    HybridRetriever(VectorPath& vec_path, MetadataPath& meta_path,
                     std::unique_ptr<IScoreFusion> fusion) noexcept;

    [[nodiscard]] auto Retrieve(const float* query_vec,
                                  const Config& cfg,
                                  const std::vector<std::int64_t>& vec_to_node_ids) const noexcept
        -> Result<std::vector<RetrievalItem>>;

    auto SetFusion(std::unique_ptr<IScoreFusion> fusion) noexcept -> void;

    ~HybridRetriever();

    HybridRetriever(const HybridRetriever&) = delete;
    auto operator=(const HybridRetriever&) -> HybridRetriever& = delete;
    HybridRetriever(HybridRetriever&&) noexcept = default;
    auto operator=(HybridRetriever&&) noexcept -> HybridRetriever& = default;

private:
    VectorPath& vec_path_;
    MetadataPath& meta_path_;
    std::unique_ptr<IScoreFusion> fusion_;
};

}  // namespace sai::retrieval
