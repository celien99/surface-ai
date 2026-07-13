// saliency_mask.cpp — DINO Attention 背景显著性掩码实现
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <sai/inference/saliency_mask.h>

namespace sai::inference {

auto ComputeDinoSaliencyMask(
    const float* attention_weights, std::size_t B, std::size_t num_heads,
    std::size_t num_reg, std::size_t num_patches,
    std::size_t grid_h, std::size_t grid_w) -> std::vector<float> {
    std::vector<float> saliency(B * grid_h * grid_w, 0.0f);

    // register tokens 对每个 patch 的 attention 取平均（跨 heads 和 register tokens）
    // attention shape: [B, num_heads, num_reg, num_patches]
    float inv_h = 1.0F / static_cast<float>(num_heads);
    float inv_r = 1.0F / static_cast<float>(num_reg);

    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t h = 0; h < num_heads; ++h) {
            for (std::size_t r = 0; r < num_reg; ++r) {
                for (std::size_t p = 0; p < num_patches; ++p) {
                    // 扁平索引：b * (num_heads * num_reg * num_patches)
                    //          + h * (num_reg * num_patches)
                    //          + r * num_patches
                    //          + p
                    auto idx = ((b * num_heads + h) * num_reg + r) * num_patches + p;
                    saliency[b * num_patches + p] += attention_weights[idx] * inv_h * inv_r;
                }
            }
        }
    }

    return saliency;
}

namespace {

auto OtsuThreshold(const float* data, std::size_t N) -> float {
    if (N == 0) return 0.0F;

    float d_min = *std::min_element(data, data + N);
    float d_max = *std::max_element(data, data + N);
    if (d_max - d_min < 1e-6F) return d_max;

    // 构建直方图（256 bins）
    constexpr int n_bins = 256;
    std::vector<std::size_t> hist(n_bins, 0);
    float scale = static_cast<float>(n_bins - 1) / (d_max - d_min);
    for (std::size_t i = 0; i < N; ++i) {
        int bin = static_cast<int>((data[i] - d_min) * scale);
        bin = std::max(0, std::min(n_bins - 1, bin));
        ++hist[static_cast<std::size_t>(bin)];
    }

    // Otsu's method: maximize between-class variance
    std::size_t total = N;
    float sum_all = 0.0F;
    for (int i = 0; i < n_bins; ++i) {
        sum_all += static_cast<float>(i) * static_cast<float>(hist[static_cast<std::size_t>(i)]);
    }

    float best_thresh = 0.0F;
    float best_var = -1.0F;
    std::size_t w_fg = 0;
    float sum_fg = 0.0F;

    for (int t = 0; t < n_bins; ++t) {
        w_fg += hist[static_cast<std::size_t>(t)];
        if (w_fg == 0 || w_fg == total) continue;

        sum_fg += static_cast<float>(t) * static_cast<float>(hist[static_cast<std::size_t>(t)]);
        std::size_t w_bg = total - w_fg;
        float m_fg = sum_fg / static_cast<float>(w_fg);
        float m_bg = (sum_all - sum_fg) / static_cast<float>(w_bg);
        float var = static_cast<float>(w_fg) * static_cast<float>(w_bg)
                    * (m_fg - m_bg) * (m_fg - m_bg);

        if (var > best_var) {
            best_var = var;
            best_thresh = d_min + static_cast<float>(t) / scale;
        }
    }

    return best_thresh;
}

}  // namespace

auto BinarizeSaliencyMask(const float* saliency, std::size_t H, std::size_t W,
                           MaskThresholdMethod method,
                           float percentile) -> std::vector<float> {
    auto total = H * W;
    std::vector<float> mask(total, 0.0F);

    float threshold = 0.0F;
    if (method == MaskThresholdMethod::Percentile) {
        std::vector<float> sorted(saliency, saliency + total);
        std::sort(sorted.begin(), sorted.end());
        auto idx = static_cast<std::size_t>(static_cast<float>(total - 1) * (1.0F - percentile));
        threshold = sorted[std::min(idx, total - 1)];
    } else {
        threshold = OtsuThreshold(saliency, total);
    }

    for (std::size_t i = 0; i < total; ++i) {
        mask[i] = (saliency[i] >= threshold) ? 1.0F : 0.0F;
    }

    return mask;
}

}  // namespace sai::inference
