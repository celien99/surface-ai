#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "sai/core/error.h"
#include "sai/rule/fact_base.h"

namespace sai::detection {
struct DetectionResult;
}  // namespace sai::detection

namespace sai::knowledge {
class KnowledgeGraph;
}  // namespace sai::knowledge

namespace sai::retrieval {
class VectorPath;
}  // namespace sai::retrieval

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

private:
    auto MapDetection(FactBase& fb, const detection::DetectionResult& dr) -> void;

    auto ResolveGraphPaths(FactBase& fb,
                           const std::vector<std::string>& paths) noexcept
        -> Result<void>;

    std::shared_ptr<knowledge::KnowledgeGraph> kg_;
    std::shared_ptr<retrieval::VectorPath> vp_;
};

}  // namespace sai::rule
