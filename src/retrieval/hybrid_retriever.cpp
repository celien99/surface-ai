#include <sai/retrieval/hybrid_retriever.h>
#include <sqlite3.h>
#include <source_location>
#include <unordered_map>
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
    // Step 1: Run metadata search first so we can use its results to constrain
    // the vector search scope via id_subset when filters are active.
    return meta_path_.Search(cfg.metadata)
        .and_then([&](auto&& meta) -> Result<std::vector<RetrievalItem>> {
            // Step 2: Build vector search config. When metadata filters are
            // present and the caller requested TopK, wire the metadata-matched
            // node IDs as an id_subset so FAISS only searches within candidates.
            VectorPath::Config vec_cfg = cfg.vector;
            bool has_filters = !cfg.metadata.filters.empty() || !cfg.metadata.node_types.empty();

            if (has_filters && vec_cfg.mode == VectorPath::Mode::TopK) {
                if (meta.empty()) {
                    // Metadata filters matched nothing — final result is empty,
                    // no point running vector search.
                    return std::vector<RetrievalItem>{};
                }

                // Build reverse map: node_id → FAISS index
                std::unordered_map<std::int64_t, std::size_t> node_to_vec;
                for (std::size_t i = 0; i < vec_to_node_ids.size(); ++i) {
                    node_to_vec[vec_to_node_ids[i]] = i;
                }

                // Collect FAISS indices for metadata-matched nodes
                std::vector<std::size_t> id_subset;
                id_subset.reserve(meta.size());
                for (const auto& mr : meta) {
                    auto it = node_to_vec.find(mr.node_id);
                    if (it != node_to_vec.end()) {
                        id_subset.push_back(it->second);
                    }
                }

                if (!id_subset.empty()) {
                    vec_cfg.mode = VectorPath::Mode::Hybrid;
                    vec_cfg.id_subset = std::move(id_subset);
                }
                // If id_subset is empty (none of the metadata-matched nodes have
                // vectors), VectorPath::Mode::Hybrid falls back to TopK internally,
                // preserving the original behaviour.
            }

            // Step 3: Run vector search (constrained by id_subset when applicable)
            return vec_path_.Search(query_vec, vec_path_.Dim(), vec_cfg)
                .and_then([&](auto&& vec) -> Result<std::vector<RetrievalItem>> {
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
