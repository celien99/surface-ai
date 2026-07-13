// specular_filter.cpp — 镜面反射滤波实现
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include <sai/detection/post_process_utils.h>
#include <sai/detection/specular_filter.h>

namespace sai::detection {

namespace {

constexpr float kEps = 1e-6F;

// sRGB → 线性 RGB
inline auto Linearize(float c) -> float {
    return std::pow(std::max(kEps, std::min(1.0F, c)), 2.2F);
}

// 线性 RGB → 亮度 Y (ITU-R BT.709)
inline auto Luminance(float r, float g, float b) -> float {
    return 0.2126F * r + 0.7152F * g + 0.0722F * b;
}

// HSV Saturation 近似：max(r,g,b) - min(r,g,b) / max(r,g,b)
inline auto SaturationApprox(float r, float g, float b) -> float {
    float max_c = std::max({r, g, b});
    float min_c = std::min({r, g, b});
    return (max_c < kEps) ? 0.0F : (max_c - min_c) / max_c;
}

}  // namespace

auto ComputeSpecularMask(const float* img_rgb, std::size_t H, std::size_t W,
                          float tau) -> std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> {
    auto total = H * W;
    std::vector<float> sY_v(total), sS_v(total), sK_v(total), clip_v(total);

    constexpr float kY = 15.0F, tY = 0.85F;
    constexpr float kS = 10.0F, tS = 0.25F;

    // 亮度 & 饱和度 & 过曝
    for (std::size_t i = 0; i < total; ++i) {
        float r = img_rgb[i * 3 + 0];
        float g = img_rgb[i * 3 + 1];
        float b = img_rgb[i * 3 + 2];

        float r_lin = Linearize(r);
        float g_lin = Linearize(g);
        float b_lin = Linearize(b);

        float Y = Luminance(r_lin, g_lin, b_lin);
        float S = SaturationApprox(r, g, b);  // sRGB 空间计算饱和度

        sY_v[i] = 1.0F / (1.0F + std::exp(-kY * (Y - tY)));    // sigmoid
        sS_v[i] = 1.0F / (1.0F + std::exp(-kS * (tS - S)));    // sigmoid(负向)
        clip_v[i] = (std::max({r, g, b}) > 0.985F) ? 1.0F : 0.0F;
    }

    // 曲率 (Laplacian of Gaussian on Y)
    // 简化：用 3×3 Laplacian 核近似 LoG
    auto pixel_Y = [&](std::size_t py, std::size_t px) -> float {
        auto p = (py * W + px) * 3;
        return Luminance(Linearize(img_rgb[p]), Linearize(img_rgb[p + 1]), Linearize(img_rgb[p + 2]));
    };

    for (std::size_t y = 1; y < H - 1; ++y) {
        for (std::size_t x = 1; x < W - 1; ++x) {
            float Y = pixel_Y(y, x);
            float lap = -4.0F * Y
                + pixel_Y(y - 1, x) + pixel_Y(y + 1, x)
                + pixel_Y(y, x - 1) + pixel_Y(y, x + 1);
            sK_v[y * W + x] = 1.0F / (1.0F + std::exp(-4.0F * (std::abs(lap) - 0.1F)));
        }
    }

    // 加权融合
    constexpr float w1 = 0.5F, w2 = 0.3F, w3 = 0.2F, w4 = 0.3F;
    std::vector<float> soft_spec(total), bin_mask(total), conf(total);
    for (std::size_t i = 0; i < total; ++i) {
        soft_spec[i] = std::max(0.0F, std::min(1.0F,
            w1 * sY_v[i] + w2 * sS_v[i] + w3 * sK_v[i] + w4 * clip_v[i]));
        bin_mask[i] = (soft_spec[i] > tau) ? 1.0F : 0.0F;
        conf[i] = 1.0F - soft_spec[i];
    }

    return {std::move(bin_mask), std::move(soft_spec), std::move(conf)};
}

auto FilterSpecularAnomalies(const float* anomaly_map, const float* conf_map,
                              std::size_t H, std::size_t W, float blur_sigma)
    -> std::vector<float> {
    auto total = H * W;

    // 1. 非镜面区域的异常分数
    std::vector<float> anomaly_non_spec(total);
    for (std::size_t i = 0; i < total; ++i) {
        anomaly_non_spec[i] = anomaly_map[i] * conf_map[i];
    }

    // 2. Gaussian 模糊 → 非镜面邻域的平均异常分数
    auto avg_anomaly = GaussianBlur(anomaly_non_spec.data(), H, W,
                                     static_cast<std::size_t>(blur_sigma));
    auto avg_conf = GaussianBlur(conf_map, H, W,
                                  static_cast<std::size_t>(blur_sigma));

    // 3 & 4. Context score → lerp suppression
    std::vector<float> filtered(total);
    for (std::size_t i = 0; i < total; ++i) {
        // context_score: how anomalous this pixel is relative to its non-specular neighborhood
        float ctx = std::max(0.0F, std::min(1.0F,
            avg_anomaly[i] / (anomaly_map[i] + kEps)));
        // suppress: lerp between conf (no suppression) and 1.0 (full suppression)
        float suppression = conf_map[i] + (1.0F - conf_map[i]) * ctx;
        filtered[i] = anomaly_map[i] * suppression;
    }

    return filtered;
}

}  // namespace sai::detection
