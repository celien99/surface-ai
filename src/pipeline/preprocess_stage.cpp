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
    // Preprocess steps are pure functions. In production, the YAML
    // config's "steps" field drives Compose({MakeDebayer(), ...}).
    // For now (FakeCamera produces Mono8 not Bayer), wrap RawImage
    // buffer as SurfaceImage without transformation.
    stub_ = false;
    return {};
}

auto PreprocessStage::OnStart(Context&) -> Result<void> { return {}; }
auto PreprocessStage::OnStop(Context&) -> Result<void> { return {}; }

auto PreprocessStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* img = std::get_if<sai::image::RawImage>(&input)) {
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
