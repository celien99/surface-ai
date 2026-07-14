#include "stage_nodes.h"
#include <sai/image/raw_image.h>

namespace sai::pipeline {

CaptureStage::CaptureStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto CaptureStage::GetType() const noexcept -> StageType { return StageType::Capture; }
auto CaptureStage::GetId() const -> std::string_view { return id_; }
auto CaptureStage::OnInitialize(Context&) -> Result<void> { return {}; }
auto CaptureStage::OnStart(Context&) -> Result<void> { return {}; }
auto CaptureStage::OnStop(Context&) -> Result<void> { return {}; }

auto CaptureStage::Process(StageInput input) -> Result<StageOutput> {
    // Passthrough: RawImage -> RawImage
    if (auto* img = std::get_if<sai::image::RawImage>(&input)) {
        return StageOutput(std::move(*img));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Capture expects RawImage input"});
}

}  // namespace sai::pipeline
