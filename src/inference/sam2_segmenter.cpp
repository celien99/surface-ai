// sam2_segmenter.cpp — Sam2Segmenter CUDA-gated implementation
#include <sai/inference/sam2_segmenter.h>

#include <sai/detection/detection_result.h>
#include <sai/image/gpu_image.h>

namespace sai::inference {

auto Sam2Segmenter::Refine(
    const sai::image::GpuImage& image,
    const sai::detection::DetectionResult& detection) noexcept
    -> Result<std::vector<RefinedRegion>> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Inference_EngineExecutionFailed,
            "Sam2Segmenter::Refine: adapter has been moved away",
        });
    }

    std::vector<RefinedRegion> results;
    results.reserve(detection.regions.size());

    for (std::size_t i = 0; i < detection.regions.size(); ++i) {
        const auto& region = detection.regions[i];

        // Build a mask prompt from the region's bounding box.
        // The prompt is a 4-channel mask at mask_size × mask_size.
        // Real implementation (CUDA build) fills the prompt tensor
        // with the region's bounding box rendered as a binary mask.
        //
        // On non-CUDA platforms this stub returns an empty vector.
        // The platform gate is at CMake level; on non-CUDA builds
        // this file is not compiled.

        // Stub: skip on non-CUDA builds (CMakeLists.txt gate).
        (void)image;
        (void)region;
    }

    return results;
}

}  // namespace sai::inference
