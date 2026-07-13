#include <sai/retrieval/hybrid_retriever.h>
#include <sqlite3.h>
#include <source_location>
#include <unordered_set>

namespace sai::retrieval {

HybridRetriever::HybridRetriever(VectorPath& vec_path, MetadataPath& meta_path,
                                   std::unique_ptr<IScoreFusion> fusion) noexcept
    : vec_path_(vec_path), meta_path_(meta_path), fusion_(std::move(fusion)) {}

HybridRetriever::~HybridRetriever() = default;

auto HybridRetriever::SetFusion(std::unique_ptr<IScoreFusion> fusion) noexcept -> void {
    fusion_ = std::move(fusion);
}

auto HybridRetriever::Retrieve(const float* query_vec,
                                 const Config& cfg,
                                 const std::vector<std::int64_t>& vec_to_node_ids) const noexcept
    -> Result<std::vector<RetrievalItem>> {
    return vec_path_.Search(query_vec, cfg.vector)
        .and_then([&](auto&& vec) -> Result<std::vector<RetrievalItem>> {
            return meta_path_.Search(cfg.metadata)
                .and_then([&](auto&& meta) -> Result<std::vector<RetrievalItem>> {
                    // Map vector results to node IDs, skipping unmapped indices
                    std::vector<std::int64_t> vec_node_ids;
                    vec_node_ids.reserve(vec.size());
                    for (const auto& vr : vec) {
                        if (vr.index >= vec_to_node_ids.size()) continue;
                        vec_node_ids.push_back(vec_to_node_ids[vr.index]);
                    }

                    // Fuse scores
                    auto fused = fusion_->Fuse(vec, vec_node_ids, meta);

                    // Convert to RetrievalItem (node_type placeholder — caller fills from KnowledgeGraph)
                    std::vector<RetrievalItem> items;
                    items.reserve(fused.size());
                    for (auto& [node_id, breakdown] : fused) {
                        RetrievalItem item;
                        item.node_id = node_id;
                        item.scores = std::move(breakdown);
                        items.push_back(std::move(item));
                    }
                    return items;
                });
        });
}

}  // namespace sai::retrieval
