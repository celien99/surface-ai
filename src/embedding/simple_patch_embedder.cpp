// simple_patch_embedder.cpp — CPU patch embedder implementation
#include <sai/embedding/simple_patch_embedder.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <source_location>

#include <sai/image/image.h>

namespace sai::embedding {

namespace {

// ── utility helpers ──────────────────────────────────────────────────

constexpr auto kPi = 3.14159265358979323846F;

// RGB → luminance (ITU-R BT.601)
inline auto Luminance(std::uint8_t r, std::uint8_t g, std::uint8_t b) noexcept -> float {
    return 0.299F * static_cast<float>(r)
         + 0.587F * static_cast<float>(g)
         + 0.114F * static_cast<float>(b);
}

// Clamp value to [lo, hi]
inline auto Clamp(float v, float lo, float hi) noexcept -> float {
    return v < lo ? lo : (v > hi ? hi : v);
}

// L2-normalize a vector in-place. Zero-vector is left as-is.
auto L2Normalize(std::vector<float>& v) noexcept -> void {
    float norm_sq = 0.0F;
    for (auto x : v) norm_sq += x * x;
    if (norm_sq > 1e-12F) {
        float inv = 1.0F / std::sqrt(norm_sq);
        for (auto& x : v) x *= inv;
    }
}

// Uniform histogram: `n_bins` bins spanning [0, 256) for uint8 values.
// Input: `values` array of `count` uint8 samples.
// Output: `n_bins` floats appended to `out`, each = count in bin / count (L1-normalized).
auto Uint8Histogram(const std::uint8_t* values, std::size_t count,
                    std::size_t n_bins,
                    std::vector<float>& out) noexcept -> void {
    if (count == 0) {
        out.insert(out.end(), n_bins, 0.0F);
        return;
    }
    auto base = out.size();
    out.insert(out.end(), n_bins, 0.0F);
    float scale = 256.0F / static_cast<float>(n_bins);
    for (std::size_t i = 0; i < count; ++i) {
        auto bin = static_cast<std::size_t>(static_cast<float>(values[i]) / scale);
        if (bin >= n_bins) bin = n_bins - 1;
        out[base + bin] += 1.0F;
    }
    float inv = 1.0F / static_cast<float>(count);
    for (std::size_t b = 0; b < n_bins; ++b) out[base + b] *= inv;
}

// Float histogram: `n_bins` bins spanning [min_val, max_val].
// Input: `values` array of `count` floats.
// Output: `n_bins` floats appended to `out`, L1-normalized.
auto FloatHistogram(const float* values, std::size_t count,
                    std::size_t n_bins,
                    std::vector<float>& out) noexcept -> void {
    if (count == 0) {
        out.insert(out.end(), n_bins, 0.0F);
        return;
    }
    auto base = out.size();
    out.insert(out.end(), n_bins, 0.0F);
    float min_v = values[0], max_v = values[0];
    for (std::size_t i = 1; i < count; ++i) {
        if (values[i] < min_v) min_v = values[i];
        if (values[i] > max_v) max_v = values[i];
    }
    float range = max_v - min_v;
    if (range < 1e-6F) {
        out[base] = 1.0F;  // all in first bin
        return;
    }
    float scale = static_cast<float>(n_bins) / range;
    for (std::size_t i = 0; i < count; ++i) {
        auto bin = static_cast<std::size_t>((values[i] - min_v) * scale);
        if (bin >= n_bins) bin = n_bins - 1;
        out[base + bin] += 1.0F;
    }
    float inv = 1.0F / static_cast<float>(count);
    for (std::size_t b = 0; b < n_bins; ++b) out[base + b] *= inv;
}

}  // anonymous namespace

// ── Move operations ──────────────────────────────────────────────────

SimplePatchEmbedder::SimplePatchEmbedder(SimplePatchEmbedder&& other) noexcept
    : cfg_(other.cfg_)
    , grid_h_(other.grid_h_)
    , grid_w_(other.grid_w_)
    , raw_feature_dim_(other.raw_feature_dim_)
{
}

auto SimplePatchEmbedder::operator=(SimplePatchEmbedder&& other) noexcept
    -> SimplePatchEmbedder& {
    if (this != &other) {
        cfg_ = other.cfg_;
        grid_h_ = other.grid_h_;
        grid_w_ = other.grid_w_;
        raw_feature_dim_ = other.raw_feature_dim_;
    }
    return *this;
}

// ── ExtractPatch — core feature computation for one patch ────────────

auto SimplePatchEmbedder::ExtractPatch(
    const std::uint8_t* img_data,
    std::size_t img_stride_bytes,
    std::size_t gy, std::size_t gx,
    std::vector<float>& out_features) const noexcept -> void {

    const auto ps = cfg_.patch_size;
    const auto ps2 = ps * ps;
    const auto img_w = cfg_.image_width;
    const auto channels = std::size_t{3};

    // Gather patch pixels: R, G, B planar arrays + interleaved raw
    std::vector<std::uint8_t> r_vals(ps2), g_vals(ps2), b_vals(ps2);
    std::vector<std::uint8_t> gray_vals(ps2);
    std::vector<float> raw_norm(ps2 * 3);

    std::size_t px = 0;
    for (std::size_t dy = 0; dy < ps; ++dy) {
        for (std::size_t dx = 0; dx < ps; ++dx) {
            auto src_y = gy * ps + dy;
            auto src_x = gx * ps + dx;
            auto offset = src_y * img_stride_bytes + src_x * channels;
            auto r = img_data[offset];
            auto g = img_data[offset + 1];
            auto b = img_data[offset + 2];
            r_vals[px] = r;
            g_vals[px] = g;
            b_vals[px] = b;
            gray_vals[px] = static_cast<std::uint8_t>(Luminance(r, g, b));
            raw_norm[px * 3 + 0] = static_cast<float>(r) / 255.0F;
            raw_norm[px * 3 + 1] = static_cast<float>(g) / 255.0F;
            raw_norm[px * 3 + 2] = static_cast<float>(b) / 255.0F;
            ++px;
        }
    }

    // ── 1. Raw normalized pixels (ps2 * 3 dims) ──
    out_features.insert(out_features.end(), raw_norm.begin(), raw_norm.end());

    // ── 2. RGB histograms (16 bins × 3 = 48 dims) ──
    constexpr auto kHistBins = std::size_t{16};
    Uint8Histogram(r_vals.data(), ps2, kHistBins, out_features);
    Uint8Histogram(g_vals.data(), ps2, kHistBins, out_features);
    Uint8Histogram(b_vals.data(), ps2, kHistBins, out_features);

    // ── 3. Color moments (mean, variance, skewness, kurtosis) × 3 = 12 dims ──
    for (const auto& chan : {r_vals, g_vals, b_vals}) {
        double sum = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
        for (auto v : chan) {
            double fv = static_cast<double>(v);
            sum += fv;
            sum2 += fv * fv;
            sum3 += fv * fv * fv;
            sum4 += fv * fv * fv * fv;
        }
        double n = static_cast<double>(ps2);
        double mean = sum / n;
        double variance = sum2 / n - mean * mean;
        double std_dev = std::sqrt(std::max(variance, 0.0));
        double skewness = (std_dev > 1e-9)
            ? ((sum3 / n - 3.0 * mean * variance - mean * mean * mean)
               / (std_dev * std_dev * std_dev))
            : 0.0;
        double kurtosis = (variance > 1e-9)
            ? ((sum4 / n - 4.0 * mean * (sum3 / n)
                + 6.0 * mean * mean * (sum2 / n)
                - 3.0 * mean * mean * mean * mean)
               / (variance * variance) - 3.0)
            : 0.0;
        out_features.push_back(static_cast<float>(mean / 255.0));
        out_features.push_back(static_cast<float>(std_dev / 255.0));
        out_features.push_back(static_cast<float>(Clamp(static_cast<float>(skewness), -5.0F, 5.0F) / 5.0F));
        out_features.push_back(static_cast<float>(Clamp(static_cast<float>(kurtosis), -5.0F, 5.0F) / 5.0F));
    }

    // ── 4. Gradient magnitude histogram (16 bins) ──
    std::vector<float> grad_mag(ps2);
    std::vector<float> grad_orient(ps2);
    for (std::size_t dy = 1; dy + 1 < ps; ++dy) {
        for (std::size_t dx = 1; dx + 1 < ps; ++dx) {
            auto idx = dy * ps + dx;
            // Sobel 3×3 on luminance
            float gx = -1.0F * gray_vals[(dy-1)*ps+(dx-1)]
                       + 1.0F * gray_vals[(dy-1)*ps+(dx+1)]
                       - 2.0F * gray_vals[dy*ps+(dx-1)]
                       + 2.0F * gray_vals[dy*ps+(dx+1)]
                       - 1.0F * gray_vals[(dy+1)*ps+(dx-1)]
                       + 1.0F * gray_vals[(dy+1)*ps+(dx+1)];
            float gy_v = -1.0F * gray_vals[(dy-1)*ps+(dx-1)]
                         - 2.0F * gray_vals[(dy-1)*ps+dx]
                         - 1.0F * gray_vals[(dy-1)*ps+(dx+1)]
                         + 1.0F * gray_vals[(dy+1)*ps+(dx-1)]
                         + 2.0F * gray_vals[(dy+1)*ps+dx]
                         + 1.0F * gray_vals[(dy+1)*ps+(dx+1)];
            grad_mag[idx] = std::sqrt(gx * gx + gy_v * gy_v);
            grad_orient[idx] = std::atan2(gy_v, gx);  // [-π, π]
        }
    }
    FloatHistogram(grad_mag.data(), ps2, 16, out_features);

    // ── 5. Gradient orientation histogram (9 bins, [0, π)) ──
    {
        auto base = out_features.size();
        out_features.insert(out_features.end(), 9, 0.0F);
        float total_weight = 0.0F;
        for (std::size_t i = 0; i < ps2; ++i) {
            float mag = grad_mag[i];
            if (mag < 1e-6F) continue;
            float orient = grad_orient[i];
            if (orient < 0.0F) orient += kPi;  // map to [0, π)
            auto bin = static_cast<std::size_t>(orient / kPi * 9.0F);
            if (bin >= 9) bin = 8;
            out_features[base + bin] += mag;
            total_weight += mag;
        }
        if (total_weight > 1e-9F) {
            float inv = 1.0F / total_weight;
            for (std::size_t b = 0; b < 9; ++b) out_features[base + b] *= inv;
        }
    }

    // ── 6. Simplified LBP histogram (10 bins) ──
    // Uniform LBP-like: compare 8 neighbors, count how many are brighter.
    // This captures local texture coarseness.
    {
        auto base = out_features.size();
        out_features.insert(out_features.end(), 10, 0.0F);
        std::size_t interior_count = 0;
        for (std::size_t dy = 1; dy + 1 < ps; ++dy) {
            for (std::size_t dx = 1; dx + 1 < ps; ++dx) {
                auto center = gray_vals[dy * ps + dx];
                int brighter = 0;
                for (int ndy = -1; ndy <= 1; ++ndy) {
                    for (int ndx = -1; ndx <= 1; ++ndx) {
                        if (ndy == 0 && ndx == 0) continue;
                        if (gray_vals[(dy + ndy) * ps + (dx + ndx)] > center) ++brighter;
                    }
                }
                if (brighter < 10) {
                    out_features[base + static_cast<std::size_t>(brighter)] += 1.0F;
                }
                ++interior_count;
            }
        }
        if (interior_count > 0) {
            float inv = 1.0F / static_cast<float>(interior_count);
            for (std::size_t b = 0; b < 10; ++b) out_features[base + b] *= inv;
        }
    }

    // ── 7. Spatial downsample (3×3 grid × 3 channels = 27 dims) ──
    {
        auto cell_h = ps / 3;
        auto cell_w = ps / 3;
        for (std::size_t cy = 0; cy < 3; ++cy) {
            for (std::size_t cx = 0; cx < 3; ++cx) {
                float sum_r = 0.0F, sum_g = 0.0F, sum_b = 0.0F;
                std::size_t n_cell = 0;
                for (std::size_t dy = cy * cell_h; dy < (cy + 1) * cell_h && dy < ps; ++dy) {
                    for (std::size_t dx = cx * cell_w; dx < (cx + 1) * cell_w && dx < ps; ++dx) {
                        sum_r += r_vals[dy * ps + dx];
                        sum_g += g_vals[dy * ps + dx];
                        sum_b += b_vals[dy * ps + dx];
                        ++n_cell;
                    }
                }
                float inv = 1.0F / (static_cast<float>(n_cell) * 255.0F);
                out_features.push_back(sum_r * inv);
                out_features.push_back(sum_g * inv);
                out_features.push_back(sum_b * inv);
            }
        }
    }
}

// ── Extract ──────────────────────────────────────────────────────────

auto SimplePatchEmbedder::Extract(const sai::image::Image& image) noexcept
    -> Result<Embedding> {
    const auto& meta = image.Meta();

    // Validate dimensions
    if (meta.width != cfg_.image_width || meta.height != cfg_.image_height) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "SimplePatchEmbedder: image size mismatch (expected "
                + std::to_string(cfg_.image_width) + "x" + std::to_string(cfg_.image_height)
                + ", got " + std::to_string(meta.width) + "x" + std::to_string(meta.height) + ")",
        });
    }

    if (meta.channels != 3) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Embedding_DimensionMismatch,
            "SimplePatchEmbedder: expected 3-channel RGB8 image, got "
                + std::to_string(meta.channels) + " channels",
        });
    }

    auto total_patches = grid_h_ * grid_w_;
    auto stride = meta.width * meta.channels;  // bytes per row

    // Pre-allocate feature vector
    std::vector<float> features;
    features.reserve(total_patches * cfg_.feature_dim);

    // Extract per-patch raw features, then reduce to feature_dim
    std::vector<float> patch_raw;
    patch_raw.reserve(raw_feature_dim_);

    for (std::size_t gy = 0; gy < grid_h_; ++gy) {
        for (std::size_t gx = 0; gx < grid_w_; ++gx) {
            patch_raw.clear();
            ExtractPatch(image.Data(), stride, gy, gx, patch_raw);

            // Reduce raw features → feature_dim via strided averaging.
            // This is a simple but effective dimensionality reduction:
            // group adjacent feature dimensions and average them.
            auto raw_dim = patch_raw.size();
            if (raw_dim <= cfg_.feature_dim) {
                // Pad with zeros
                features.insert(features.end(), patch_raw.begin(), patch_raw.end());
                features.insert(features.end(), cfg_.feature_dim - raw_dim, 0.0F);
            } else {
                // Strided averaging: each output dim is the mean of a contiguous block
                float ratio = static_cast<float>(raw_dim) / static_cast<float>(cfg_.feature_dim);
                for (std::size_t d = 0; d < cfg_.feature_dim; ++d) {
                    auto start = static_cast<std::size_t>(static_cast<float>(d) * ratio);
                    auto end = static_cast<std::size_t>(static_cast<float>(d + 1) * ratio);
                    if (end <= start) end = start + 1;
                    if (end > raw_dim) end = raw_dim;
                    float acc = 0.0F;
                    for (auto i = start; i < end; ++i) acc += patch_raw[i];
                    features.push_back(acc / static_cast<float>(end - start));
                }
            }
        }
    }

    // L2-normalize each patch vector
    if (cfg_.normalize) {
        for (std::size_t p = 0; p < total_patches; ++p) {
            float norm_sq = 0.0F;
            auto* start = features.data() + p * cfg_.feature_dim;
            for (std::size_t d = 0; d < cfg_.feature_dim; ++d) {
                norm_sq += start[d] * start[d];
            }
            if (norm_sq > 1e-12F) {
                float inv = 1.0F / std::sqrt(norm_sq);
                for (std::size_t d = 0; d < cfg_.feature_dim; ++d) start[d] *= inv;
            }
        }
    }

    EmbeddingMeta emb_meta;
    emb_meta.model_name = "SimplePatch";
    emb_meta.type = EmbeddingType::Patch;
    emb_meta.dim = cfg_.feature_dim;
    emb_meta.count = total_patches;
    emb_meta.grid = {grid_h_, grid_w_};

    return Embedding::FromCpu(std::move(features), std::move(emb_meta));
}

// ── ExtractBatch ─────────────────────────────────────────────────────

auto SimplePatchEmbedder::ExtractBatch(
    std::span<const sai::image::Image* const> images) noexcept
    -> Result<std::vector<Embedding>> {
    std::vector<Embedding> results;
    results.reserve(images.size());
    for (const auto* img : images) {
        if (img == nullptr) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Embedding_DimensionMismatch,
                "SimplePatchEmbedder::ExtractBatch: null image pointer",
            });
        }
        auto result = Extract(*img);
        if (!result) return tl::make_unexpected(result.error());
        results.push_back(std::move(*result));
    }
    return results;
}

}  // namespace sai::embedding
