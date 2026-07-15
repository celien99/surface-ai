// detection_result.h — 批次 3.3 Detector 结构化检测产出
#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

#include <sai/device/device.h>  // Rect

namespace sai::detection {

// 异常热力图：grid_h × grid_w 的 patch-level 异常分数矩阵
struct AnomalyMap {
    std::vector<float> scores;
    std::size_t grid_h = 0;
    std::size_t grid_w = 0;

    // 以行主序索引访问 (y, x) 处的异常分数
    [[nodiscard]] auto At(std::size_t y, std::size_t x) const noexcept -> float {
        return scores[y * grid_w + x];
    }
    // 返回全局最大异常分数；空 map 返回 0.0F
    [[nodiscard]] auto MaxScore() const noexcept -> float;
    // 若有任意 patch 分数超过阈值则认为有缺陷
    [[nodiscard]] auto IsDefective(float threshold) const noexcept -> bool {
        return MaxScore() > threshold;
    }
};

// 候选缺陷区域：由连通分量标记从二值化 anomaly map 中提取
struct RegionProposal {
    sai::device::Rect bounding_box;
    float max_anomaly_score = 0.0F;
    float mean_anomaly_score = 0.0F;
    std::size_t area_pixels = 0;
};

// 检测结果：AnomalyMap + regions + 图像级分数 + 推理耗时
struct DetectionResult {
    AnomalyMap anomaly_map;
    std::vector<RegionProposal> regions;
    float image_level_score = 0.0F;
    std::chrono::nanoseconds inference_latency{0};

    // Global image-level features (CLIP embedding) for cross-modal retrieval.
    // Populated by DetectStage from the Embedding's global_features.
    // Consumed by RuleEvalStage for FactBuilder::RunVectorRetrieval.
    std::vector<float> global_features;

    // Surface identity — carried from image metadata through the pipeline.
    // Set by DetectStage from Embedding::SurfaceId, consumed by RuleEvalStage.
    std::string surface_id;

    // 图像级异常分数超过阈值则认为本帧存在缺陷
    [[nodiscard]] auto IsDefective(float threshold) const noexcept -> bool {
        return image_level_score > threshold;
    }
};

}  // namespace sai::detection
