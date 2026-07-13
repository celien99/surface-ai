// post_process_utils.h — 异常检测后处理工具函数（上采样、平滑、连通域标记）
#pragma once

#include <chrono>
#include <cstddef>
#include <span>
#include <vector>

#include <sai/core/error.h>
#include <sai/detection/detection_result.h>
#include <sai/embedding/embedding.h>

namespace sai::detection {

class IDetector;

// 双线性上采样：将 src (src_h × src_w) 上采样到 dst_h × dst_w。
// 像素中心映射：src_coord = (dst_coord + 0.5) * src_size / dst_size - 0.5。
[[nodiscard]] auto BilinearUpsample(const float* src, std::size_t src_h, std::size_t src_w,
                                     std::size_t dst_h, std::size_t dst_w) -> std::vector<float>;

// 可分离高斯模糊（sigma 以像素为单位）。
// sigma == 0 时直接返回副本。边界 clamp-to-edge。
[[nodiscard]] auto GaussianBlur(const float* src, std::size_t h, std::size_t w,
                                 std::size_t sigma) -> std::vector<float>;

// 4-连通分量标记：从二值 mask + anomaly scores 中提取 RegionProposal。
// binary: h×w 二值 mask（> 0 为前景），scores: h×w 异常分数矩阵。
// 返回按 max_anomaly_score 降序排列的候选缺陷区域。
[[nodiscard]] auto ConnectedComponents(const float* binary, std::size_t h, std::size_t w,
                                        const float* scores)
    -> std::vector<RegionProposal>;

// 验证 patch grid 与配置参数是否匹配。
// 检查 grid_h == image_height / patch_size 且 grid_w == image_width / patch_size。
[[nodiscard]] auto ValidatePatchGrid(std::size_t grid_h, std::size_t grid_w,
    std::size_t image_height, std::size_t image_width,
    std::size_t patch_size) noexcept -> Result<void>;

// 从 patch scores 构建 DetectionResult——标准后处理管线：
// AnomalyMap → BilinearUpsample → GaussianBlur → threshold → ConnectedComponents
// patch_scores 为已归一化到 [0,1] 的 (grid_h × grid_w) 异常分数。
// image_level_score 设为 anomaly_map.MaxScore()。
[[nodiscard]] auto BuildDetectionResult(
    std::vector<float>&& patch_scores,
    std::size_t grid_h, std::size_t grid_w,
    std::size_t image_height, std::size_t image_width,
    std::size_t gaussian_sigma,
    float threshold,
    std::chrono::nanoseconds latency) noexcept -> DetectionResult;

// 批量检测——遍历 embeddings 并逐个调用 detector.Detect()。
// PatchCore 和 PcaDetector 共享此实现（避免 DetectBatch 重复）。
[[nodiscard]] auto DetectBatchImpl(IDetector& detector,
    std::span<const sai::embedding::Embedding* const> embeddings) noexcept
    -> Result<std::vector<DetectionResult>>;

}  // namespace sai::detection
