#include "stage_nodes.h"

#include <sai/device/camera.h>
#include <sai/image/raw_image.h>
#include <sai/pipeline/pipeline.h>

namespace sai::pipeline {

CaptureStage::CaptureStage(std::string id, YAML::Node config, Pipeline* pipeline)
    : id_(std::move(id)), pipeline_(pipeline) {
    (void)config;  // config consumed in OnInitialize
}

auto CaptureStage::GetType() const noexcept -> StageType { return StageType::Capture; }
auto CaptureStage::GetId() const -> std::string_view { return id_; }

auto CaptureStage::OnInitialize(Context& ctx) -> Result<void> {
    (void)ctx;
    // When ICamera is resolvable via Context (e.g. ctx.Resolve<device::ICamera>()),
    // the stub_ flag is cleared and the camera lifecycle hooks in OnStart/OnStop
    // activate.  Currently ICamera (IPlugin) does not satisfy the IService
    // constraint required by Context::Resolve<T>, so stub_ stays true.
    return {};
}

auto CaptureStage::OnStart(Context&) -> Result<void> {
    if (stub_ || !camera_) return {};

    auto result = camera_->Connect();
    if (!result) return result;

    result = camera_->SetTriggerMode(device::ICamera::TriggerMode::FreeRun);
    if (!result) return result;

    result = camera_->RegisterFrameCallback(
        [this](sai::image::RawImage img) {
            if (pipeline_) {
                (void)pipeline_->Submit(std::move(img));
            }
        });
    if (!result) return result;

    return camera_->StartAcquisition();
}

auto CaptureStage::OnStop(Context&) -> Result<void> {
    if (stub_ || !camera_) return {};
    auto r1 = camera_->StopAcquisition();
    auto r2 = camera_->Disconnect();
    if (!r1) return r1;
    return r2;
}

auto CaptureStage::Process(StageInput input) -> Result<StageOutput> {
    // Passthrough: frames arrive via Pipeline::Submit (from camera callback),
    // not through Process. Process handles the case where frames are already
    // in the input queue (submitted externally or from upstream stubs).
    if (auto* img = std::get_if<sai::image::RawImage>(&input)) {
        return StageOutput(std::move(*img));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Capture expects RawImage input"});
}

}  // namespace sai::pipeline
