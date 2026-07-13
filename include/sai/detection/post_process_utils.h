// post_process_utils.h — 异常检测后处理工具函数（上采样、平滑、连通域标记）
#pragma once

#include <cstddef>
#include <vector>

#include <sai/detection/detection_result.h>

namespace sai::detection {

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

}  // namespace sai::detection
