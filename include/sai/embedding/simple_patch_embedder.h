// simple_patch_embedder.h — CPU patch embedder with hand-crafted visual features
#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include <sai/embedding/embedder.h>

namespace sai::embedding {

// SimplePatchEmbedder: CPU-only patch embedding using hand-crafted visual features.
//
// Unlike PatchEmbedder (which requires CUDA + GPU image + DINOv3 TensorRT engine),
// this embedder operates on CPU-backed SurfaceImage and computes real, deterministic
// feature vectors from per-patch statistics: raw pixel values, RGB histograms,
// gradient magnitude/orientation histograms, color moments, LBP, and spatial
// downsampling. This enables end-to-end testing and development without a GPU.
//
// The feature space is designed to be discriminative for surface defect detection:
// different patches produce meaningfully different vectors, and similar patches
// (same material region) cluster in L2 space. The features capture color shifts
// (stains), edge anomalies (scratches/cracks), and texture irregularities (dents).
//
// Configuration:
//   feature_dim:   Target feature dimensionality per patch (default 128).
//                   Hand-crafted features are ~766 dims raw; internal PCA-like
//                   reduction compresses to feature_dim via channel-wise averaging.
//   patch_size:    Edge length of each square patch in pixels (default 14).
//   image_width:   Expected input image width in pixels (default 1024).
//   image_height:  Expected input image height in pixels (default 1024).
//   normalize:     If true, L2-normalize each patch feature vector (default true).

struct SimplePatchEmbedderConfig {
    std::size_t image_width = 1024;
    std::size_t image_height = 1024;
    std::size_t patch_size = 14;
    std::size_t feature_dim = 128;
    bool normalize = true;
};

class SimplePatchEmbedder final : public IEmbedder {
public:
    [[nodiscard]] static auto Create(SimplePatchEmbedderConfig cfg) noexcept
        -> Result<SimplePatchEmbedder>;

    [[nodiscard]] auto Extract(const sai::image::Image& image) noexcept
        -> Result<Embedding> override;
    [[nodiscard]] auto ExtractBatch(
        std::span<const sai::image::Image* const> images) noexcept
        -> Result<std::vector<Embedding>> override;
    [[nodiscard]] auto ModelName() const noexcept -> std::string_view override
    { return "SimplePatch"; }

    SimplePatchEmbedder(SimplePatchEmbedder&&) noexcept;
    auto operator=(SimplePatchEmbedder&&) noexcept -> SimplePatchEmbedder&;
    SimplePatchEmbedder(const SimplePatchEmbedder&) = delete;
    auto operator=(const SimplePatchEmbedder&) -> SimplePatchEmbedder& = delete;

private:
    explicit SimplePatchEmbedder(SimplePatchEmbedderConfig cfg) noexcept;

    // Compute per-patch features for a single patch at grid position (gy, gx).
    // Input: pointer to image data (RGB8, interleaved), image stride in bytes.
    // Output: feature_dim floats appended to `out_features`.
    auto ExtractPatch(const std::uint8_t* img_data,
                      std::size_t img_stride_bytes,
                      std::size_t gy, std::size_t gx,
                      std::vector<float>& out_features) const noexcept -> void;

    SimplePatchEmbedderConfig cfg_;
    std::size_t grid_h_ = 0;
    std::size_t grid_w_ = 0;
    std::size_t raw_feature_dim_ = 0;  // hand-crafted dim before reduction to feature_dim
};

// ── inline Create / constructor ──────────────────────────────────────

inline auto SimplePatchEmbedder::Create(SimplePatchEmbedderConfig cfg) noexcept
    -> Result<SimplePatchEmbedder> {
    if (cfg.image_width == 0 || cfg.image_height == 0 || cfg.patch_size == 0) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Embedding_InvalidPatchParameters, "SimplePatchEmbedder: image dimensions and patch size must be > 0"});
    }
    if (cfg.image_width % cfg.patch_size != 0 || cfg.image_height % cfg.patch_size != 0) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Embedding_InvalidPatchParameters, "SimplePatchEmbedder: image size must be divisible by patch size"});
    }
    if (cfg.feature_dim == 0) {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Embedding_InvalidPatchParameters, "SimplePatchEmbedder: feature_dim must be > 0"});
    }
    return SimplePatchEmbedder{cfg};
}

inline SimplePatchEmbedder::SimplePatchEmbedder(SimplePatchEmbedderConfig cfg) noexcept
    : cfg_(cfg)
    , grid_h_(cfg.image_height / cfg.patch_size)
    , grid_w_(cfg.image_width / cfg.patch_size)
{
    // Raw feature components (before reduction):
    //   raw pixels:   patch_size^2 * 3
    //   RGB histogram: 16 bins * 3 channels = 48
    //   color moments: 4 stats * 3 channels = 12
    //   gradient mag histogram: 16 bins
    //   gradient orient histogram: 9 bins
    //   LBP histogram: 10 bins
    //   spatial downsample: 3x3 * 3 = 27
    auto ps2 = cfg.patch_size * cfg.patch_size;
    raw_feature_dim_ = ps2 * 3 + 48 + 12 + 16 + 9 + 10 + 27;
}

}  // namespace sai::embedding
