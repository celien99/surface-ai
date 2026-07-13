// post_process_utils.cpp — 异常检测后处理工具函数实现
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <source_location>
#include <vector>

#include <sai/core/error.h>
#include <sai/detection/detector.h>
#include <sai/detection/post_process_utils.h>

namespace sai::detection {

// ── BilinearUpsample ────────────────────────────────────────────────

auto BilinearUpsample(const float* src, std::size_t src_h, std::size_t src_w,
                      std::size_t dst_h, std::size_t dst_w) -> std::vector<float> {
    if (src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0) return {};

    std::vector<float> dst(dst_h * dst_w);
    auto sf_h = static_cast<float>(src_h);
    auto sf_w = static_cast<float>(src_w);
    auto df_h = static_cast<float>(dst_h);
    auto df_w = static_cast<float>(dst_w);

    for (std::size_t y = 0; y < dst_h; ++y) {
        float src_y = (static_cast<float>(y) + 0.5F) * sf_h / df_h - 0.5F;
        src_y = std::max(0.0F, std::min(src_y, sf_h - 1.0F));
        auto y0 = static_cast<std::size_t>(src_y);
        auto y1 = std::min(y0 + 1, src_h - 1);
        float wy = src_y - static_cast<float>(y0);

        for (std::size_t x = 0; x < dst_w; ++x) {
            float src_x = (static_cast<float>(x) + 0.5F) * sf_w / df_w - 0.5F;
            src_x = std::max(0.0F, std::min(src_x, sf_w - 1.0F));
            auto x0 = static_cast<std::size_t>(src_x);
            auto x1 = std::min(x0 + 1, src_w - 1);
            float wx = src_x - static_cast<float>(x0);

            float v00 = src[y0 * src_w + x0];
            float v01 = src[y0 * src_w + x1];
            float v10 = src[y1 * src_w + x0];
            float v11 = src[y1 * src_w + x1];

            float v0 = v00 * (1.0F - wx) + v01 * wx;
            float v1 = v10 * (1.0F - wx) + v11 * wx;
            dst[y * dst_w + x] = v0 * (1.0F - wy) + v1 * wy;
        }
    }
    return dst;
}

// ── GaussianBlur ────────────────────────────────────────────────────

auto GaussianBlur(const float* src, std::size_t h, std::size_t w,
                  std::size_t sigma) -> std::vector<float> {
    if (h == 0 || w == 0) return {};
    if (sigma == 0) return std::vector<float>(src, src + h * w);

    auto radius = static_cast<std::size_t>(std::ceil(3.0F * static_cast<float>(sigma)));
    auto kernel_size = 2 * radius + 1;
    std::vector<float> kernel(kernel_size);
    float sigma_f = static_cast<float>(sigma);
    float sum = 0.0F;
    for (std::size_t i = 0; i < kernel_size; ++i) {
        auto r = static_cast<float>(i) - static_cast<float>(radius);
        kernel[i] = std::exp(-0.5F * r * r / (sigma_f * sigma_f));
        sum += kernel[i];
    }
    for (auto& k : kernel) k /= sum;

    std::vector<float> temp(h * w);
    std::vector<float> dst(h * w);

    // 水平 pass
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            float val = 0.0F;
            for (std::size_t k_idx = 0; k_idx < kernel_size; ++k_idx) {
                auto sx = static_cast<std::ptrdiff_t>(x + k_idx) - static_cast<std::ptrdiff_t>(radius);
                sx = std::max(std::ptrdiff_t{0}, std::min(sx, static_cast<std::ptrdiff_t>(w - 1)));
                val += src[y * w + static_cast<std::size_t>(sx)] * kernel[k_idx];
            }
            temp[y * w + x] = val;
        }
    }

    // 垂直 pass
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            float val = 0.0F;
            for (std::size_t k_idx = 0; k_idx < kernel_size; ++k_idx) {
                auto sy = static_cast<std::ptrdiff_t>(y + k_idx) - static_cast<std::ptrdiff_t>(radius);
                sy = std::max(std::ptrdiff_t{0}, std::min(sy, static_cast<std::ptrdiff_t>(h - 1)));
                val += temp[static_cast<std::size_t>(sy) * w + x] * kernel[k_idx];
            }
            dst[y * w + x] = val;
        }
    }
    return dst;
}

// ── ConnectedComponents ─────────────────────────────────────────────

auto ConnectedComponents(const float* binary, std::size_t h, std::size_t w,
                         const float* scores)
    -> std::vector<RegionProposal> {
    if (h == 0 || w == 0) return {};

    auto total = h * w;
    std::vector<std::ptrdiff_t> labels(total, -1);
    std::vector<std::ptrdiff_t> parent;

    // 并查集
    auto find_root = [&](std::ptrdiff_t label) -> std::ptrdiff_t {
        while (parent[static_cast<std::size_t>(label)] != label) {
            parent[static_cast<std::size_t>(label)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(label)])];
            label = parent[static_cast<std::size_t>(label)];
        }
        return label;
    };

    // 第一遍：分配临时标签
    std::ptrdiff_t next_label = 0;
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            auto idx = y * w + x;
            if (binary[idx] <= 0.0F) continue;

            std::ptrdiff_t up_label = (y > 0 && labels[idx - w] >= 0) ? labels[idx - w] : -1;
            std::ptrdiff_t left_label = (x > 0 && labels[idx - 1] >= 0) ? labels[idx - 1] : -1;

            if (up_label < 0 && left_label < 0) {
                labels[idx] = next_label;
                parent.push_back(next_label);
                ++next_label;
            } else if (up_label >= 0 && left_label >= 0) {
                auto root_up = find_root(up_label);
                auto root_left = find_root(left_label);
                auto min_label = std::min(root_up, root_left);
                labels[idx] = min_label;
                if (root_up != root_left) {
                    parent[static_cast<std::size_t>(std::max(root_up, root_left))] = min_label;
                }
            } else {
                labels[idx] = (up_label >= 0) ? up_label : left_label;
            }
        }
    }

    auto num_components = static_cast<std::size_t>(next_label);
    if (num_components == 0) return {};

    // 统计每个分量的信息
    struct ComponentInfo {
        float max_score = 0.0F;
        float sum_score = 0.0F;
        std::size_t pixel_count = 0;
        std::size_t min_x = ~std::size_t{0};
        std::size_t min_y = ~std::size_t{0};
        std::size_t max_x = 0;
        std::size_t max_y = 0;
    };
    std::vector<ComponentInfo> comps(num_components);

    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            auto idx = y * w + x;
            if (labels[idx] < 0) continue;

            auto root = find_root(labels[idx]);
            labels[idx] = root;

            auto& ci = comps[static_cast<std::size_t>(root)];
            ci.max_score = std::max(ci.max_score, scores[idx]);
            ci.sum_score += scores[idx];
            ci.pixel_count += 1;
            ci.min_x = std::min(ci.min_x, x);
            ci.min_y = std::min(ci.min_y, y);
            ci.max_x = std::max(ci.max_x, x);
            ci.max_y = std::max(ci.max_y, y);
        }
    }

    // 构建 RegionProposal
    std::vector<RegionProposal> regions;
    regions.reserve(num_components);
    for (std::size_t i = 0; i < num_components; ++i) {
        const auto& ci = comps[i];
        if (ci.pixel_count == 0) continue;

        sai::device::Rect bbox{
            ci.min_x,
            ci.min_y,
            ci.max_x - ci.min_x + 1,
            ci.max_y - ci.min_y + 1,
        };
        regions.push_back(RegionProposal{
            bbox,
            ci.max_score,
            ci.sum_score / static_cast<float>(ci.pixel_count),
            ci.pixel_count,
        });
    }

    // 按 max_score 降序排列
    std::sort(regions.begin(), regions.end(),
              [](const RegionProposal& a, const RegionProposal& b) {
                  return a.max_anomaly_score > b.max_anomaly_score;
              });

    return regions;
}

// ── ValidatePatchGrid ────────────────────────────────────────────────

auto ValidatePatchGrid(std::size_t grid_h, std::size_t grid_w,
                        std::size_t image_height, std::size_t image_width,
                        std::size_t patch_size) noexcept -> Result<void> {
    auto expected_grid_h = image_height / patch_size;
    auto expected_grid_w = image_width / patch_size;
    if (grid_h != expected_grid_h || grid_w != expected_grid_w) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Detection_InvalidPatchGrid,
            "patch grid " + std::to_string(grid_h) + "×" + std::to_string(grid_w)
                + " does not match config " + std::to_string(expected_grid_h)
                + "×" + std::to_string(expected_grid_w),
            std::source_location::current()});
    }
    return {};
}

// ── BuildDetectionResult ─────────────────────────────────────────────

auto BuildDetectionResult(
    std::vector<float>&& patch_scores,
    std::size_t grid_h, std::size_t grid_w,
    std::size_t image_height, std::size_t image_width,
    std::size_t gaussian_sigma,
    float threshold,
    std::chrono::nanoseconds latency) noexcept -> DetectionResult {
    // 构建 AnomalyMap
    AnomalyMap anomaly_map;
    anomaly_map.grid_h = grid_h;
    anomaly_map.grid_w = grid_w;
    anomaly_map.scores = std::move(patch_scores);
    auto image_level_score = anomaly_map.MaxScore();

    // 双线性上采样到图像分辨率
    auto upsampled = BilinearUpsample(anomaly_map.scores.data(),
                                      grid_h, grid_w,
                                      image_height, image_width);

    // Gaussian 平滑
    auto blurred = GaussianBlur(upsampled.data(), image_height, image_width,
                                gaussian_sigma);

    // 阈值 → 二值 mask → 连通分量
    std::vector<float> binary(image_height * image_width);
    for (std::size_t i = 0; i < binary.size(); ++i) {
        binary[i] = (blurred[i] > threshold) ? 1.0F : 0.0F;
    }
    auto regions = ConnectedComponents(binary.data(), image_height, image_width,
                                       blurred.data());

    return DetectionResult{
        std::move(anomaly_map),
        std::move(regions),
        image_level_score,
        latency,
    };
}

// ── DetectBatchImpl ──────────────────────────────────────────────────

auto DetectBatchImpl(IDetector& detector,
    std::span<const sai::embedding::Embedding* const> embeddings) noexcept
    -> Result<std::vector<DetectionResult>> {
    std::vector<DetectionResult> results;
    results.reserve(embeddings.size());

    for (const auto* emb : embeddings) {
        if (emb == nullptr) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Detection_InvalidPatchGrid,
                "null embedding pointer in DetectBatch",
                std::source_location::current()});
        }
        auto result = detector.Detect(*emb);
        if (!result.has_value()) {
            return tl::make_unexpected(result.error());
        }
        results.push_back(std::move(result.value()));
    }
    return results;
}

}  // namespace sai::detection
