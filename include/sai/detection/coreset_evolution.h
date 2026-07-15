// coreset_evolution.h — 在线 Coreset 自进化
#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/object.h>

namespace sai::detection {

class FeatureBank;
class PatchCore;
struct DetectionResult;

// ── 正常度画像（coreset 自查询统计） ──

struct NormalityProfile {
    std::size_t k_nearest = 5;
    std::size_t dim = 0;
    std::size_t num_samples = 0;
    float p50 = 0.0F;
    float p95 = 0.0F;
    float p99 = 0.0F;
    float mean = 0.0F;
    float stddev = 0.0F;

    // 从 FeatureBank 计算自查询统计（O(N²·D)，初始化/FullRebuild 时调用）
    [[nodiscard]] static auto Compute(const FeatureBank& bank,
                                       std::size_t k = 5) noexcept -> NormalityProfile;

    // 对 standby bank 做采样自查询（O(sqrt(K)²·D) ≈ 50ms），用于运行时更新
    [[nodiscard]] static auto ComputeFast(const FeatureBank& bank,
                                            std::size_t k = 5,
                                            std::size_t sample_count = 100) noexcept
        -> NormalityProfile;

    [[nodiscard]] static auto LoadFromYaml(const std::filesystem::path& path) noexcept
        -> Result<NormalityProfile>;
    [[nodiscard]] auto SaveToYaml(const std::filesystem::path& path) const noexcept
        -> Result<void>;
};

// ── 正常度评估结果 ──

struct NormalityAssessment {
    float normalcy_score = 0.0F;      // 0~1
    float concentration_ratio = 0.0F; // median(query) / profile.P50
    float tail_ratio = 0.0F;          // 超过 P95 的 patch 比例
};

}  // namespace sai::detection
