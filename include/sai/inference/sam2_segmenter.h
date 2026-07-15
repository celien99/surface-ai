// sam2_segmenter.h — SAM2 pipeline wrapper (M5 placeholder → ReasonStage consumer)
#pragma once

#include <cstddef>
#include <vector>

#include <sai/core/error.h>
#include <sai/inference/sam2_adapter.h>

namespace sai::detection {
struct DetectionResult;
struct RegionProposal;
}  // namespace sai::detection

namespace sai::image {
class GpuImage;
}  // namespace sai::image

namespace sai::inference {

// Per-region refined mask output from SAM2.
struct RefinedRegion {
    std::size_t region_index = 0;     // index into DetectionResult::regions
    std::vector<float> mask_data;     // flattened mask [H*W], values 0.0–1.0
    std::size_t mask_height = 0;
    std::size_t mask_width = 0;
    float mean_confidence = 0.0F;     // average mask confidence
};

// Sam2Segmenter wraps Sam2Adapter for pipeline consumption.
//
// Unlike the three Embedder types (PatchEmbedder, GlobalEmbedder,
// SimplePatchEmbedder), this class does NOT implement IEmbedder because
// SAM2 produces segmentation masks, not feature embeddings.  It is held
// by ReasonStage and invoked after detection to refine anomaly region
// boundaries.
//
// Limit: M3 only supports mask prompts (not point/box).  M5 will extend
// the prompt type to variant<PointPrompt, BoxPrompt, MaskPrompt>.
class Sam2Segmenter {
public:
    [[nodiscard]] static auto Create(Sam2Adapter adapter) noexcept
        -> Result<Sam2Segmenter>;

    // Refine anomaly regions from DetectionResult using SAM2.
    // Each region in DetectionResult::regions becomes a mask prompt;
    // SAM2 produces a refined segmentation mask per region.
    //
    // On CUDA builds the actual GPU inference runs; on non-CUDA builds
    // returns an empty vector (stub — no refinement, not an error).
    [[nodiscard]] auto Refine(
        const sai::image::GpuImage& image,
        const sai::detection::DetectionResult& detection) noexcept
        -> Result<std::vector<RefinedRegion>>;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view
    { return "SAM2"; }

    Sam2Segmenter(Sam2Segmenter&&) noexcept = default;
    auto operator=(Sam2Segmenter&&) noexcept -> Sam2Segmenter& = default;
    Sam2Segmenter(const Sam2Segmenter&) = delete;
    auto operator=(const Sam2Segmenter&) -> Sam2Segmenter& = delete;

private:
    explicit Sam2Segmenter(Sam2Adapter adapter) noexcept;
    Sam2Adapter adapter_;
    bool has_adapter_ = true;
};

// ── inline Create / constructor ────────────────────────────────────────

inline auto Sam2Segmenter::Create(Sam2Adapter adapter) noexcept
    -> Result<Sam2Segmenter> {
    return Sam2Segmenter{std::move(adapter)};
}

inline Sam2Segmenter::Sam2Segmenter(Sam2Adapter adapter) noexcept
    : adapter_(std::move(adapter)) {}

}  // namespace sai::inference
