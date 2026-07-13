// specular_filter.h — 镜面反射滤波（工业 AOI 专用后处理）
#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

// NOTE: This filter is NOT automatically applied by PatchCore or PcaDetector.
// It must be invoked by the downstream pipeline orchestrator (M6 Pipeline/Scheduler)
// which has access to both the original sRGB image and the anomaly map.
// Usage: ComputeSpecularMask(rgb_image, H, W) → conf_map → FilterSpecularAnomalies(anomaly_map, conf_map, H, W)

namespace sai::detection {

// 从 sRGB 图像生成镜面反射 mask。
//
// 四线索融合：
//   1. sY（亮度）: sigmoid(kY * (Y - tY)) — 高亮度 → 镜面
//   2. sS（去饱和度）: sigmoid(kS * (tS - S)) — 低饱和度 → 镜面
//   3. sK（曲率）: Laplacian of Gaussian — 高曲率 → 镜面
//   4. clip_flag: 像素值 > 0.985 → 过曝
//
// 加权融合: Sspec = 0.5*sY + 0.3*sS + 0.2*sK + 0.3*clip_flag
//
// img_rgb: H×W 扁平 RGB 数组（值域 [0, 1]），row-major，连续存储 RGBRGB...
// tau: binarization threshold [0, 1]
// 返回 {bin_mask, soft_spec, confidence}
//   - bin_mask: H×W, bool-ish float (0=normal, 1=specular)
//   - soft_spec: H×W, [0,1] soft specular map
//   - confidence: H×W, 1-soft_spec (非镜面置信度)
[[nodiscard]] auto ComputeSpecularMask(const float* img_rgb, std::size_t H, std::size_t W,
                                        float tau = 0.6F)
    -> std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>;

// 用镜面置信度过滤异常图中的镜面反射伪影。
//
// 核心思想：如果一个像素的异常分数主要来自镜面反射（而非真实缺陷），
// 那么它的分数应该比其非镜面邻域的平均分数低得多。
//
// 算法：
//   1. 用 conf_map 掩盖异常图（仅保留非镜面区域）
//   2. 对掩盖后的图做 Gaussian 模糊 → 得到非镜面邻域的平均异常分数
//   3. 计算 context_score = avg / (pixel_score + ε)
//   4. 抑制：filtered = score * lerp(conf, 1.0, context_score)
//
// anomaly_map / conf_map: H×W 扁平数组
// blur_sigma: 邻域平滑 sigma（像素单位）
[[nodiscard]] auto FilterSpecularAnomalies(const float* anomaly_map, const float* conf_map,
                                            std::size_t H, std::size_t W,
                                            float blur_sigma = 5.0F)
    -> std::vector<float>;

}  // namespace sai::detection
