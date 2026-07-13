// score_fusion.h — 批次 4.2 可插拔分数融合策略
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <sai/core/error.h>

namespace YAML { class Node; }

namespace sai::retrieval {

struct VectorResult;
struct MetadataResult;

struct ScoreBreakdown {
    float vector_score = 0.0F;
    float metadata_score = 0.0F;
    float fused_score = 0.0F;
    std::string fusion_strategy;
};

class IScoreFusion {
public:
    virtual ~IScoreFusion() = default;
    [[nodiscard]] virtual auto Configure(const YAML::Node& params) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto Fuse(
        const std::vector<VectorResult>& vec_results,
        const std::vector<std::int64_t>& vec_node_ids,
        const std::vector<MetadataResult>& meta_results) const noexcept
        -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> = 0;
    [[nodiscard]] virtual auto Name() const noexcept -> std::string_view = 0;
};

class WeightedFusion final : public IScoreFusion {
public:
    [[nodiscard]] auto Configure(const YAML::Node& params) noexcept -> Result<void> override;
    [[nodiscard]] auto Fuse(
        const std::vector<VectorResult>& vec_results,
        const std::vector<std::int64_t>& vec_node_ids,
        const std::vector<MetadataResult>& meta_results) const noexcept
        -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> override;
    [[nodiscard]] auto Name() const noexcept -> std::string_view override { return "WeightedFusion"; }
private:
    float alpha_ = 0.5F;
};

class RRFFusion final : public IScoreFusion {
public:
    [[nodiscard]] auto Configure(const YAML::Node& params) noexcept -> Result<void> override;
    [[nodiscard]] auto Fuse(
        const std::vector<VectorResult>& vec_results,
        const std::vector<std::int64_t>& vec_node_ids,
        const std::vector<MetadataResult>& meta_results) const noexcept
        -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> override;
    [[nodiscard]] auto Name() const noexcept -> std::string_view override { return "RRFFusion"; }
private:
    float k_ = 60.0F;
};

}  // namespace sai::retrieval
