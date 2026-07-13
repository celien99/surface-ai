#include <sai/retrieval/score_fusion.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <source_location>

namespace sai::retrieval {

// WeightedFusion

auto WeightedFusion::Configure(const YAML::Node& params) noexcept -> Result<void> {
    if (!params["alpha"]) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Retrieval_FusionConfigInvalid,
            "WeightedFusion: missing 'alpha' parameter",
            std::source_location::current(),
        });
    }
    alpha_ = params["alpha"].as<float>();
    if (alpha_ < 0.0F || alpha_ > 1.0F) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Retrieval_FusionConfigInvalid,
            "WeightedFusion: alpha must be in [0, 1]",
            std::source_location::current(),
        });
    }
    return {};
}

auto WeightedFusion::Fuse(
    const std::vector<VectorResult>& vec_results,
    const std::vector<std::int64_t>& vec_node_ids,
    const std::vector<MetadataResult>& meta_results) const noexcept
    -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> {

    // Find max vector distance for normalization
    float max_dist = 1.0F;
    for (const auto& v : vec_results) {
        if (v.distance > max_dist) max_dist = v.distance;
    }

    // Build metadata score lookup
    std::unordered_map<std::int64_t, float> meta_scores;
    for (const auto& m : meta_results) {
        meta_scores[m.node_id] = m.score;
    }

    // Compute fused scores for all vector results + unmatched metadata results
    std::unordered_map<std::int64_t, ScoreBreakdown> breakdowns;

    for (std::size_t i = 0; i < vec_results.size(); ++i) {
        auto node_id = vec_node_ids[i];
        float v_score = 1.0F - (vec_results[i].distance / (max_dist + 1e-6F));
        float m_score = meta_scores.count(node_id) ? meta_scores[node_id] : 0.0F;
        float fused = alpha_ * v_score + (1.0F - alpha_) * m_score;
        breakdowns[node_id] = {v_score, m_score, fused, "WeightedFusion"};
    }

    // Add metadata-only results not in vector results
    for (const auto& m : meta_results) {
        if (!breakdowns.count(m.node_id)) {
            float fused = (1.0F - alpha_) * m.score;
            breakdowns[m.node_id] = {0.0F, m.score, fused, "WeightedFusion"};
        }
    }

    // Sort by fused_score descending
    std::vector<std::pair<std::int64_t, ScoreBreakdown>> results(
        breakdowns.begin(), breakdowns.end());
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.second.fused_score > b.second.fused_score;
              });
    return results;
}

// RRFFusion

auto RRFFusion::Configure(const YAML::Node& params) noexcept -> Result<void> {
    if (params["k"]) {
        k_ = params["k"].as<float>();
        if (k_ <= 0.0F) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Retrieval_FusionConfigInvalid,
                "RRFFusion: k must be positive",
                std::source_location::current(),
            });
        }
    }
    return {};
}

auto RRFFusion::Fuse(
    const std::vector<VectorResult>& vec_results,
    const std::vector<std::int64_t>& vec_node_ids,
    const std::vector<MetadataResult>& meta_results) const noexcept
    -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> {

    std::unordered_map<std::int64_t, ScoreBreakdown> breakdowns;

    // Vector rankings (sorted by distance ascending, lower rank = better)
    std::vector<std::pair<std::size_t, float>> vec_ranked;
    for (std::size_t i = 0; i < vec_results.size(); ++i) {
        vec_ranked.push_back({i, vec_results[i].distance});
    }
    std::sort(vec_ranked.begin(), vec_ranked.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    for (std::size_t rank = 0; rank < vec_ranked.size(); ++rank) {
        auto idx = vec_ranked[rank].first;
        auto node_id = vec_node_ids[idx];
        float rrf = 1.0F / (k_ + static_cast<float>(rank + 1));
        auto& bd = breakdowns[node_id];
        bd.vector_score = rrf;
        bd.fused_score += rrf;
        bd.fusion_strategy = "RRFFusion";
    }

    // Metadata rankings (preserve input order as ranking)
    for (std::size_t rank = 0; rank < meta_results.size(); ++rank) {
        auto node_id = meta_results[rank].node_id;
        float rrf = 1.0F / (k_ + static_cast<float>(rank + 1));
        auto& bd = breakdowns[node_id];
        bd.metadata_score = rrf;
        bd.fused_score += rrf;
        bd.fusion_strategy = "RRFFusion";
    }

    std::vector<std::pair<std::int64_t, ScoreBreakdown>> results(
        breakdowns.begin(), breakdowns.end());
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.second.fused_score > b.second.fused_score;
              });
    return results;
}

}  // namespace sai::retrieval
