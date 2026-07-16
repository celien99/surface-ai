// multi_signal_consensus.cpp — 多信号联合判定
#include <sai/detection/coreset_evolution.h>
#include <sai/detection/detection_result.h>

#include <string>

namespace sai::detection {

namespace {
constexpr float kNormalcyScoreThreshold = 0.9F;
}  // namespace

auto MultiSignalConsensusCheck(
    const NormalityAssessment& normalcy,
    const DetectionResult& detection,
    std::size_t matched_rules_count,
    const std::string& reasoner_verdict,
    float effective_threshold,
    float pca_image_score,
    float pca_self_query_p95) noexcept -> bool {

    // 1. k-NN 正常度
    if (normalcy.normalcy_score < kNormalcyScoreThreshold) return false;

    // 2. 图像级异常分数在阈值以下
    if (detection.image_level_score >= effective_threshold) return false;

    // 3. 规则全部未命中
    if (matched_rules_count > 0) return false;

    // 4. Reasoner 最终裁决为 OK
    if (reasoner_verdict != "OK") return false;

    // 5. PCA 子空间正常（如果启用）
    if (pca_self_query_p95 > 0.0F && pca_image_score >= pca_self_query_p95) return false;

    return true;
}

}  // namespace sai::detection
