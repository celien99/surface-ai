#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sai/core/error.h"
#include "sai/retrieval/vector_path.h"
#include "sai/rule/fact_base.h"

namespace sai::detection {
struct DetectionResult;
}  // namespace sai::detection

namespace sai::knowledge {
class KnowledgeGraph;
}  // namespace sai::knowledge

namespace sai::rule {

// FactBuilder bridges M3 Detection + M4 Knowledge into a FactBase.
//
// Design decisions:
// - MapDetection directly maps DetectionResult field values into FactBase
//   keys (adapted to real DetectionResult struct fields, not the speculative
//   field names from the original spec).
// - ResolveGraphPaths parses "->"-separated path strings and walks the
//   KnowledgeGraph property graph via Traverse.
// - RunVectorRetrieval looks up a query vector from FactBase (as a Value
//   list), calls VectorPath::Search with TopK mode, and expands results
//   into FactBase keys.
// - L2-distance-based retrieval cache: when a query vector is similar
//   (L2 < cache_epsilon_) to the previous query, cached results are reused
//   to avoid redundant FAISS searches. Default threshold 0.01.
class FactBuilder {
public:
    explicit FactBuilder(
        std::shared_ptr<knowledge::KnowledgeGraph> kg,
        std::shared_ptr<retrieval::VectorPath> vp);

    // Build populates a FactBase from DetectionResult and graph paths.
    // RunVectorRetrieval is NOT called automatically because Build does
    // not receive a query vector — call it separately when needed.
    auto Build(
        std::string_view surface_id,
        const detection::DetectionResult& detection,
        const std::vector<std::string>& graph_paths_to_resolve)
        -> Result<FactBase>;

    // RunVectorRetrieval uses vp_->Search (TopK mode) with a query vector
    // stored in FactBase under embedding_key (must be a Value list of doubles
    // representing the float query vector). Results are stored as:
    //   "retrieval.top1.index", "retrieval.top1.distance", ...
    //   "retrieval.topN.index", "retrieval.topN.distance"
    //   "retrieval.count"
    auto RunVectorRetrieval(FactBase& fb, std::string_view embedding_key) noexcept
        -> Result<void>;

    // Configure L2-distance threshold for retrieval cache hits.
    // Set to 0.0 to disable caching entirely. Default 0.01.
    auto SetRetrievalCacheEpsilon(double eps) noexcept -> void { cache_epsilon_ = eps; }

private:
    auto MapDetection(FactBase& fb, const detection::DetectionResult& dr) -> void;

    auto ResolveGraphPaths(FactBase& fb,
                           const std::vector<std::string>& paths) noexcept
        -> Result<void>;

    // Write VectorResults to FactBase — shared by cache-hit and cache-miss paths.
    auto WriteRetrievalResults(FactBase& fb,
                                const std::vector<retrieval::VectorResult>& results,
                                std::size_t k) -> void;

    std::shared_ptr<knowledge::KnowledgeGraph> kg_;
    std::shared_ptr<retrieval::VectorPath> vp_;

    // L2-distance-based retrieval cache
    struct RetrievalCache {
        std::vector<float> query;
        std::vector<retrieval::VectorResult> results;
    };
    mutable std::optional<RetrievalCache> retrieval_cache_;
    double cache_epsilon_ = 0.01;
};

}  // namespace sai::rule
