// coreset_evolution.h — 在线 Coreset 自进化
#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/object.h>

// Forward declarations
namespace YAML { class Node; }

// Forward declarations in sai namespace (outside sai::detection)
namespace sai::knowledge { class KnowledgeStore; }

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

    // 通过 coreset 自查询计算出的 normalcy_score。
    // 所有门控阈值均由此值推导，无需人类标定。
    float self_normalcy = 0.0F;

    // 从 FeatureBank 计算自查询统计（O(N²·D)，初始化/FullRebuild 时调用）
    [[nodiscard]] static auto Compute(const FeatureBank& bank,
                                       std::size_t k = 5,
                                       float tail_ratio_max = 0.10F) noexcept
        -> NormalityProfile;

    // 对 standby bank 做采样自查询（O(sqrt(K)²·D) ≈ 50ms），用于运行时更新
    [[nodiscard]] static auto ComputeFast(const FeatureBank& bank,
                                            std::size_t k = 5,
                                            std::size_t sample_count = 100,
                                            float tail_ratio_max = 0.10F) noexcept
        -> NormalityProfile;

    // 由 self_normalcy 推导 consensus 门控阈值。
    // 新帧 normalcy_score 低于此值时被共识判定拒绝。
    [[nodiscard]] auto ConsensusThreshold() const noexcept -> float;

    // 由 self_normalcy 推导进化安全门控阈值。
    // 候选帧 mean normalcy 低于此值时跳过整批进化。
    [[nodiscard]] auto EvolutionGate() const noexcept -> float;

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
    float normalcy_threshold,     // from NormalityProfile::ConsensusThreshold() — data-driven
    float pca_image_score,        // 0.0F if PCA not enabled
    float pca_self_query_p95)     // PCA self-query threshold (0.0F if not enabled)
    noexcept -> bool;

// ── 自进化配置 ──

struct EvolutionConfig {
    bool enabled = false;

    // 从 YAML 节点解析配置（self_evolution 子节点）
    [[nodiscard]] static auto FromYaml(const YAML::Node& node) -> Result<EvolutionConfig>;

    // Normality
    std::size_t normality_k = 5;
    float tail_ratio_max = 0.10F;

    // Novelty
    float coverage_threshold = 0.60F;

    // Buffer
    std::size_t trigger_frames = 20;
    std::size_t trigger_patches = 20000;
    std::size_t max_frames = 50;
    std::size_t max_patches = 50000;

    // Update
    std::size_t target_size = 10000;
    std::chrono::seconds min_update_interval{5};
    std::size_t candidate_sample_limit = 5000;

    // Persistence
    bool save_on_stop = true;
    bool backup_old_bank = true;
    std::size_t max_backups = 3;

    // ── Long-term maintenance ──
    // Coverage saturation: when coverage_ratio stays above this threshold for
    // saturation_window consecutive evolutions, the bank is considered "mature"
    // and evolution auto-pauses to avoid unnecessary CPU/disk churn.
    float coverage_saturation_threshold = 0.95F;
    std::size_t saturation_window = 10;

    // After this many incremental updates, request a full bounded rebuild.
    std::size_t max_incremental_updates = 100;

    // ── Self-validation ──
    // After each incremental swap, sample old bank patches and query the new
    // bank to verify coverage did not degrade. If new coverage drops below
    // old_coverage * validation_degradation_threshold, the swap is rejected
    // and the old bank is restored.
    bool evolution_self_validation = true;
    float validation_degradation_threshold = 0.80F;  // 20% degradation tolerance
    std::size_t validation_sample_count = 100;        // patches sampled from old bank
};

// ── 更新统计 ──

struct EvolutionStats {
    std::size_t frames_added = 0;
    std::size_t patches_added = 0;
    std::size_t patches_removed = 0;
    std::size_t size_before = 0;
    std::size_t size_after = 0;
    float mean_displacement = 0.0F;
    float coverage_gain = 0.0F;
    std::chrono::milliseconds update_duration{0};
    std::string last_error;
    std::size_t update_count = 0;
};

// ── CoresetEvolution 核心门面 ──

class CoresetEvolution final : public Object {
public:
    CoresetEvolution(EvolutionConfig cfg,
                     PatchCore& detector,
                     NormalityProfile profile) noexcept;

    // 每帧调用（检测线程，~微秒，零阻塞）
    // embedding_data: 原始 patch 向量（grid_h * grid_w * dim 个 float），
    //   通过 shared_ptr 共享，避免每帧 5+ MB 的额外拷贝。
    //   调用方（seat_aoi）从 PatchCore::DetectionContext 获取。
    auto AssessAndOffer(const float* distances,
                        std::size_t query_count,
                        std::size_t k,
                        std::shared_ptr<const std::vector<float>> embedding_data,
                        std::size_t grid_h,
                        std::size_t grid_w,
                        std::size_t dim,
                        const DetectionResult& det_result,
                        std::size_t matched_rules_count,
                        const std::string& reasoner_verdict,
                        float effective_threshold,
                        float pca_image_score,
                        float pca_self_query_p95) noexcept -> void;

    // 启动/停止后台更新线程
    auto Start(std::stop_token token) noexcept -> void;
    auto Stop() noexcept -> void;  // 阻塞直到线程退出 + FullRebuild

    [[nodiscard]] auto IsRunning() const noexcept -> bool;
    [[nodiscard]] auto LatestStats() const noexcept -> EvolutionStats;
    [[nodiscard]] auto Profile() const noexcept -> const NormalityProfile&;

    auto BindKnowledgeStore(std::shared_ptr<knowledge::KnowledgeStore> ks) noexcept -> void;

    // 显式全量重建 + 持久化（Pipeline Stop 时调用）
    [[nodiscard]] auto FullRebuild(const std::filesystem::path& save_path) noexcept
        -> Result<void>;

    // Object constraints
    CoresetEvolution(CoresetEvolution&&) noexcept = delete;
    CoresetEvolution(const CoresetEvolution&) = delete;
    ~CoresetEvolution() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sai::detection
