// saliency_mask.h — DINO Attention 背景显著性掩码
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sai::inference {

// 二值化方法
enum class MaskThresholdMethod : std::uint8_t {
    Percentile,  // 按百分比阈值：top (1-percentile) 为前景
    Otsu,        // 大津法自动阈值
};

// 从 DINO attention weights 提取前景显著性 mask。
//
// 使用 register tokens 对 patch tokens 的 attention 作为显著性信号。
// DINOv2/v3 的 register token 天然关注前景物体——不需要额外模型。
//
// attention_weights: [B, num_heads, register_tokens, patches] 扁平 float 数组
// B: batch size
// num_heads: attention heads 数量
// num_reg: register token 数量（DINOv2-giant: 4）
// num_patches: patch token 数量（不含 CLS 和 register）
// grid_h, grid_w: patch grid 尺寸（grid_h × grid_w = num_patches）
// 返回：[B, grid_h, grid_w] 显著性图（值域取决于 attention 实现，通常 [0, 1]）
[[nodiscard]] auto ComputeDinoSaliencyMask(
    const float* attention_weights, std::size_t B, std::size_t num_heads,
    std::size_t num_reg, std::size_t num_patches,
    std::size_t grid_h, std::size_t grid_w) -> std::vector<float>;

// 二值化显著性 mask
// saliency: H×W 扁平数组
// method: 阈值方法（Percentile 或 Otsu）
// percentile: 仅在 Percentile 模式下使用，前景阈值分位数
// 返回：H×W 二值 mask（1.0f = 前景，0.0f = 背景）
[[nodiscard]] auto BinarizeSaliencyMask(const float* saliency, std::size_t H, std::size_t W,
                                         MaskThresholdMethod method,
                                         float percentile = 0.15F) -> std::vector<float>;

}  // namespace sai::inference
