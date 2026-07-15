// coreset_evolution.h — 在线 Coreset 自进化
#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
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

// ── 冗余检测结果 ──

struct NoveltyResult {
    bool is_novel = false;
    float coverage_ratio = 1.0F;     // 被 P50 覆盖的 patch 比例
    std::size_t novel_patch_count = 0;
};

// ── 候选帧 ──

struct EvolutionCandidate {
    std::shared_ptr<const float> patch_vectors;
    std::size_t grid_h = 0;
    std::size_t grid_w = 0;
    std::size_t dim = 0;
    float normalcy_score = 0.0F;
    std::chrono::steady_clock::time_point captured_at;
};

// ── 有界候选缓冲 ──

class CandidateBuffer {
public:
    struct Config {
        std::size_t max_frames = 50;
        std::size_t max_patches = 50000;
        std::size_t trigger_frames = 20;
        std::size_t trigger_patches = 20000;
    };

    explicit CandidateBuffer(Config cfg) noexcept : cfg_(cfg) {}

    // 检测线程调用。返回 true = 已加入。
    auto Append(EvolutionCandidate candidate) -> bool;

    // 任一触发条件满足？
    [[nodiscard]] auto IsTriggered() const -> bool;

    // 后台线程调用：取出全部候选并清空
    auto DrainAll() -> std::vector<EvolutionCandidate>;

    [[nodiscard]] auto FrameCount() const -> std::size_t;
    [[nodiscard]] auto PatchCount() const -> std::size_t;

private:
    Config cfg_;
    mutable std::mutex mutex_;
    std::vector<EvolutionCandidate> candidates_;
    std::size_t total_patches_ = 0;
};

// ── 多信号共识判定 ──

// 所有输入来自已有管线产出，无需额外计算。
// 返回 true = 所有信号一致确认"正常"。
[[nodiscard]] auto MultiSignalConsensusCheck(
    const NormalityAssessment& normalcy,
    const DetectionResult& detection,
    std::size_t matched_rules_count,
    const std::string& reasoner_verdict,
    float effective_threshold,
    float pca_image_score,       // 0.0F if PCA not enabled
    float pca_self_query_p95)    // PCA self-query threshold (0.0F if not enabled)
    noexcept -> bool;

}  // namespace sai::detection
