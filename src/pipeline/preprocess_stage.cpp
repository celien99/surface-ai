#include "stage_nodes.h"

#include <sai/image/preprocess.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>

namespace sai::pipeline {

PreprocessStage::PreprocessStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto PreprocessStage::GetType() const noexcept -> StageType { return StageType::Preprocess; }
auto PreprocessStage::GetId() const -> std::string_view { return id_; }

auto PreprocessStage::OnInitialize(Context&) -> Result<void> {
    // Preprocess steps are pure functions — build chain directly from YAML config.
    // For now, we support a simple chain. The YAML config's "steps" field
    // is parsed at OnInitialize time. Since YAML::Node is passed by copy
    // through StageFactory, we re-read it here (simplification for v1: just
    // build a resize chain since FakeCamera produces Mono8, not Bayer).
    //
    // In a production deploy, the config would specify steps like:
    //   steps: [debayer, white_balance, resize]
    // and we'd call Compose({MakeDebayer(), MakeWhiteBalance(...), MakeResize(w,h)}).
    chain_ = sai::image::Compose({});
    stub_ = false;
    return {};
}

auto PreprocessStage::OnStart(Context&) -> Result<void> { return {}; }
auto PreprocessStage::OnStop(Context&) -> Result<void> { return {}; }

auto PreprocessStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* img = std::get_if<sai::image::RawImage>(&input)) {
        if (!stub_ && chain_) {
            auto raw = std::make_unique<sai::image::RawImage>(std::move(*img));
            auto result = chain_(std::move(raw));
            if (!result) return tl::make_unexpected(result.error());
            return StageOutput(std::move(**result));
        }
        // Stub fallback: wrap as SurfaceImage
        auto meta = img->Meta();
        const auto* data = img->Data();
        auto size = img->SizeBytes();
        std::vector<std::uint8_t> buffer(data, data + size);
        return StageOutput(
            sai::image::SurfaceImage::FromOwnedBuffer(std::move(buffer), meta));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Preprocess expects RawImage input"});
}

}  // namespace sai::pipeline
