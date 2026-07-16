#include "sai/rule/fact_builder.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "sai/core/error.h"
#include "sai/detection/detection_result.h"
#include "sai/knowledge/knowledge_graph.h"
#include "sai/knowledge/knowledge_record.h"
#include "sai/retrieval/vector_path.h"

namespace sai::rule {

namespace {

// Convert a knowledge FieldValue to a rule Value.
// Binary data (vector<uint8_t>) is skipped — returns Null.
auto FieldValueToRuleValue(const knowledge::FieldValue& fv) -> Value {
    if (auto* i = std::get_if<std::int64_t>(&fv)) {
        return Value::Of(static_cast<double>(*i));
    }
    if (auto* d = std::get_if<double>(&fv)) {
        return Value::Of(*d);
    }
    if (auto* s = std::get_if<std::string>(&fv)) {
        return Value::Of(*s);
    }
    return Value::Null();
}

// Split a string by a delimiter, returning non-empty tokens.
auto SplitString(std::string_view sv, std::string_view delim)
    -> std::vector<std::string>
{
    std::vector<std::string> result;
    std::string::size_type start = 0;
    while (true) {
        auto pos = sv.find(delim, start);
        if (pos == std::string_view::npos) {
            auto token = sv.substr(start);
            if (!token.empty()) {
                result.emplace_back(token);
            }
            break;
        }
        auto token = sv.substr(start, pos - start);
        if (!token.empty()) {
            result.emplace_back(token);
        }
        start = pos + delim.size();
    }
    return result;
}

// Build a dotted key for graph-path results, e.g.
// segments = ["DefectType", "has_cause"], property = "reject_rate"
// → "graph.DefectType.has_cause.reject_rate"
auto GraphPathFlatKey(const std::vector<std::string>& segments,
                      std::string_view property) -> std::string
{
    std::string key = "graph";
    for (const auto& seg : segments) {
        key += '.';
        key += seg;
    }
    key += '.';
    key += property;
    return key;
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// FactBuilder construction
// -----------------------------------------------------------------------

FactBuilder::FactBuilder(
    std::shared_ptr<knowledge::KnowledgeGraph> kg,
    std::shared_ptr<retrieval::VectorPath> vp)
    : kg_(std::move(kg)), vp_(std::move(vp)) {}

// -----------------------------------------------------------------------
// MapDetection — populate FactBase keys from DetectionResult fields.
//
// Adapted from the brief: the actual DetectionResult struct contains
// anomaly_map, regions, image_level_score, and inference_latency — not
// defect.type/score/confidence/passed. We map the real fields directly.
// -----------------------------------------------------------------------

auto FactBuilder::MapDetection(FactBase& fb,
                               const detection::DetectionResult& dr) -> void
{
    auto ts = std::chrono::steady_clock::now();

    fb.Set("detection.image_level_score",
           Value::Of(static_cast<double>(dr.image_level_score)),
           FactSource{FactSourceKind::Direct, "DetectionResult::image_level_score",
                      std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - ts)});

    fb.Set("detection.latency_ns",
           Value::Of(static_cast<double>(dr.inference_latency.count())),
           FactSource{FactSourceKind::Direct, "DetectionResult::inference_latency"});

    fb.Set("detection.anomaly_map.max_score",
           Value::Of(static_cast<double>(dr.anomaly_map.MaxScore())),
           FactSource{FactSourceKind::Direct, "DetectionResult::anomaly_map.MaxScore()"});

    fb.Set("detection.grid_h",
           Value::Of(static_cast<double>(dr.anomaly_map.grid_h)),
           FactSource{FactSourceKind::Direct, "DetectionResult::anomaly_map.grid_h"});

    fb.Set("detection.grid_w",
           Value::Of(static_cast<double>(dr.anomaly_map.grid_w)),
           FactSource{FactSourceKind::Direct, "DetectionResult::anomaly_map.grid_w"});

    fb.Set("detection.region_count",
           Value::Of(static_cast<double>(dr.regions.size())),
           FactSource{FactSourceKind::Direct, "DetectionResult::regions.size()"});

    // Map first region details if present
    if (!dr.regions.empty()) {
        const auto& r0 = dr.regions[0];
        fb.Set("detection.region.0.max_score",
               Value::Of(static_cast<double>(r0.max_anomaly_score)),
               FactSource{FactSourceKind::Direct, "regions[0].max_anomaly_score"});
        fb.Set("detection.region.0.mean_score",
               Value::Of(static_cast<double>(r0.mean_anomaly_score)),
               FactSource{FactSourceKind::Direct, "regions[0].mean_anomaly_score"});
        fb.Set("detection.region.0.area_pixels",
               Value::Of(static_cast<double>(r0.area_pixels)),
               FactSource{FactSourceKind::Direct, "regions[0].area_pixels"});
        fb.Set("detection.region.0.x",
               Value::Of(static_cast<double>(r0.bounding_box.x)),
               FactSource{FactSourceKind::Direct, "regions[0].bounding_box.x"});
        fb.Set("detection.region.0.y",
               Value::Of(static_cast<double>(r0.bounding_box.y)),
               FactSource{FactSourceKind::Direct, "regions[0].bounding_box.y"});
        fb.Set("detection.region.0.width",
               Value::Of(static_cast<double>(r0.bounding_box.width)),
               FactSource{FactSourceKind::Direct, "regions[0].bounding_box.width"});
        fb.Set("detection.region.0.height",
               Value::Of(static_cast<double>(r0.bounding_box.height)),
               FactSource{FactSourceKind::Direct, "regions[0].bounding_box.height"});
    }
}

// -----------------------------------------------------------------------
// ResolveGraphPaths — parse "->"-separated path strings and traverse the
// KnowledgeGraph property graph to extract property values.
//
// Path format:
//   "NodeType->Relationship.PropertyKey"
//   "NodeType->Relationship1->Relationship2.PropertyKey"
//   "NodeType.PropertyKey"
//
// Steps:
//   1. Split by "->" → segments.
//   2. First segment = source node type for FindNodesByType.
//   3. Last segment is split at the final ".":
//      - Prefix (optional) = final relationship name
//      - Suffix = property key to extract from target node's KnowledgeRecord
//   4. Walk segments[1..n-1] as relationships, calling Traverse at each hop.
//   5. Extract the property key from the final node set.
// -----------------------------------------------------------------------

auto FactBuilder::ResolveGraphPaths(FactBase& fb,
                                    const std::vector<std::string>& paths) noexcept
    -> Result<void>
{
    // If paths is empty, kg_ being null is fine — nothing to resolve
    if (paths.empty()) {
        return {};
    }
    if (!kg_) {
        return tl::unexpected(ErrorInfo{
            ErrorCode::Rule_InvalidPath,
            "ResolveGraphPaths: KnowledgeGraph is null"});
    }

    for (const auto& path : paths) {
        if (path.empty()) continue;

        // Split by "->"
        auto segments = SplitString(path, "->");
        if (segments.size() < 2) {
            return tl::unexpected(ErrorInfo{
                ErrorCode::Rule_InvalidPath,
                "Path must have at least 'NodeType->PropertyKey': " + path});
        }

        // First segment: source node type
        std::string node_type = segments[0];

        // Parse the last segment: "Relationship.PropertyKey" or "PropertyKey"
        std::string final_rel;
        std::string property_key;
        {
            const auto& last = segments.back();
            auto dot_pos = last.rfind('.');
            if (dot_pos != std::string::npos) {
                final_rel = last.substr(0, dot_pos);
                property_key = last.substr(dot_pos + 1);
            } else {
                property_key = last;
            }
        }

        // Build ordered relationship list:
        // segments[1..n-2] are intermediate relationships,
        // final_rel (from last segment) is the last relationship to traverse
        std::vector<std::string> relationships;
        if (segments.size() > 2) {
            for (std::size_t i = 1; i < segments.size() - 1; ++i) {
                relationships.push_back(segments[i]);
            }
        }
        if (!final_rel.empty()) {
            relationships.push_back(final_rel);
        }

        if (property_key.empty()) {
            return tl::unexpected(ErrorInfo{
                ErrorCode::Rule_InvalidPath,
                "Empty property key in path: " + path});
        }

        // --- Walk the graph ---

        // Step 1: Find start nodes by type
        auto nodes_result = kg_->FindNodesByType(node_type);
        if (!nodes_result) {
            // No nodes of this type → nothing to resolve for this path
            continue;
        }

        // Collect initial NodeIds
        std::vector<knowledge::NodeId> current_ids;
        current_ids.reserve(nodes_result->size());
        for (const auto& node : *nodes_result) {
            current_ids.push_back(node.id);
        }

        // Step 2: Traverse through each relationship
        for (const auto& rel : relationships) {
            std::vector<knowledge::NodeId> next_ids;

            for (auto id : current_ids) {
                auto traverse_result = kg_->Traverse(id, rel, 1);
                if (!traverse_result) continue;

                for (const auto& gp : *traverse_result) {
                    for (const auto& target : gp.targets) {
                        next_ids.push_back(target.id);
                    }
                }
            }

            // Deduplicate
            std::sort(next_ids.begin(), next_ids.end());
            next_ids.erase(
                std::unique(next_ids.begin(), next_ids.end()),
                next_ids.end());

            current_ids = std::move(next_ids);
            if (current_ids.empty()) break;  // No more nodes to traverse
        }

        // Step 3: Extract property value from final nodes
        for (auto id : current_ids) {
            auto node_result = kg_->GetNode(id);
            if (!node_result) continue;

            auto& fields = node_result->properties.fields;
            auto it = fields.find(property_key);
            if (it == fields.end()) continue;

            Value val = FieldValueToRuleValue(it->second);
            if (val.IsNull()) continue;  // Skip binary / unmappable values

            // Build a flat dotted key from traversal state
            std::string flat_key;
            {
                std::vector<std::string> key_segments;
                key_segments.push_back(node_type);
                for (const auto& rel : relationships) {
                    key_segments.push_back(rel);
                }
                flat_key = GraphPathFlatKey(key_segments, property_key);
            }

            fb.Set(flat_key, std::move(val),
                   FactSource{FactSourceKind::GraphPath, path});
        }
    }

    return {};
}

// -----------------------------------------------------------------------
// RunVectorRetrieval — uses VectorPath::Search (TopK mode) with a query
// vector stored in FactBase under embedding_key.
//
// The embedding_key must point to a Value list of doubles (the flattened
// float query vector). Results are stored back into FactBase:
//   "retrieval.top1.index"       double
//   "retrieval.top1.distance"    double
//   "retrieval.top2.index"       double
//   "retrieval.top2.distance"    double
//   ...
//   "retrieval.count"            double (N)
// -----------------------------------------------------------------------

auto FactBuilder::RunVectorRetrieval(FactBase& fb,
                                     std::string_view embedding_key) noexcept
    -> Result<void>
{
    if (!vp_) {
        return tl::unexpected(ErrorInfo{
            ErrorCode::Rule_InvalidPath,
            "RunVectorRetrieval: VectorPath is null"});
    }

    // Look up the query vector from FactBase
    auto query_val = fb.Get(embedding_key);
    if (!query_val) {
        return tl::unexpected(ErrorInfo{
            ErrorCode::Rule_TypeMismatch,
            std::string("RunVectorRetrieval: embedding_key '")
                .append(embedding_key).append("' not found in FactBase")});
    }

    auto* list = query_val->AsList();
    if (!list || list->empty()) {
        return tl::unexpected(ErrorInfo{
            ErrorCode::Rule_TypeMismatch,
            std::string("RunVectorRetrieval: embedding_key '")
                .append(embedding_key).append("' must be a non-empty Value list")});
    }

    // Convert Value list to float query vector
    auto dim = list->size();
    // Use stack allocation for small vectors, heap for larger ones
    std::vector<float> query_vec(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        auto d = (*list)[i].AsDouble();
        if (!d) {
            return tl::unexpected(ErrorInfo{
                ErrorCode::Rule_TypeMismatch,
                "RunVectorRetrieval: query vector element must be numeric"});
        }
        query_vec[i] = static_cast<float>(*d);
    }

    // Execute TopK search (default k=10, configurable via FactBase)
    retrieval::VectorPath::Config cfg;
    cfg.mode = retrieval::VectorPath::Mode::TopK;
    cfg.k = 10;  // default

    // Allow optional override of k from FactBase
    auto k_val = fb.Get("retrieval.config.k");
    if (k_val) {
        if (auto kd = k_val->AsDouble()) {
            cfg.k = static_cast<std::size_t>(*kd);
        }
    }

    // Allow optional id_subset override
    auto subset_val = fb.Get("retrieval.config.id_subset");
    if (subset_val) {
        if (auto* slist = subset_val->AsList()) {
            cfg.id_subset.clear();
            cfg.id_subset.reserve(slist->size());
            for (const auto& sv : *slist) {
                if (auto sd = sv.AsDouble()) {
                    cfg.id_subset.push_back(static_cast<std::size_t>(*sd));
                }
            }
        }
    }

    // L2-distance-based retrieval cache: skip FAISS search when the query
    // is nearly identical to the previous query. Reduces CPU load for
    // consecutive frames that share the same CLIP global embedding.
    if (retrieval_cache_.has_value() && cache_epsilon_ > 0.0) {
        const auto& cached = retrieval_cache_->query;
        if (cached.size() == dim) {
            double sq_dist = 0.0;
            for (std::size_t i = 0; i < dim; ++i) {
                double d = static_cast<double>(query_vec[i]) -
                           static_cast<double>(cached[i]);
                sq_dist += d * d;
            }
            if (sq_dist < cache_epsilon_ * cache_epsilon_) {
                WriteRetrievalResults(fb, retrieval_cache_->results, cfg.k);
                return {};
            }
        }
    }

    auto results = vp_->Search(query_vec.data(), dim, cfg);
    if (!results) {
        return tl::unexpected(results.error());
    }

    // Update cache
    retrieval_cache_ = RetrievalCache{query_vec, *results};

    WriteRetrievalResults(fb, *results, cfg.k);
    return {};
}

auto FactBuilder::WriteRetrievalResults(
    FactBase& fb,
    const std::vector<retrieval::VectorResult>& results,
    std::size_t k) -> void {
    auto result_count = static_cast<double>(results.size());
    fb.Set("retrieval.count", Value::Of(result_count),
           FactSource{FactSourceKind::VectorSearch,
                      std::string("VectorPath TopK k=").append(std::to_string(k))});

    for (std::size_t i = 0; i < results.size(); ++i) {
        auto rank = i + 1;
        auto prefix = std::string("retrieval.top")
                          .append(std::to_string(rank));

        fb.Set(prefix + ".index",
               Value::Of(static_cast<double>(results[i].index)),
               FactSource{FactSourceKind::VectorSearch, prefix + ".index"});
        fb.Set(prefix + ".distance",
               Value::Of(static_cast<double>(results[i].distance)),
               FactSource{FactSourceKind::VectorSearch, prefix + ".distance"});
    }
}

// -----------------------------------------------------------------------
// Build — orchestrate MapDetection + ResolveGraphPaths.
//
// RunVectorRetrieval is NOT called automatically because Build does not
// receive a query vector. Callers should invoke RunVectorRetrieval
// separately after Build, with the appropriate embedding key.
// -----------------------------------------------------------------------

auto FactBuilder::Build(
    std::string_view surface_id,
    const detection::DetectionResult& detection,
    const std::vector<std::string>& graph_paths_to_resolve)
    -> Result<FactBase>
{
    FactBase fb;

    // Store surface identifier
    fb.Set("surface.id", Value::Of(std::string(surface_id)),
           FactSource{FactSourceKind::Direct, "Build input surface_id"});

    // Step 1: Map detection result fields
    MapDetection(fb, detection);

    // Step 2: Resolve graph paths (may fail)
    auto path_result = ResolveGraphPaths(fb, graph_paths_to_resolve);
    if (!path_result) {
        return tl::unexpected(path_result.error());
    }

    return fb;
}

}  // namespace sai::rule
