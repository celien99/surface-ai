#include "stage_nodes.h"

#include <sai/device/camera.h>
#include <sai/image/raw_image.h>
#include <sai/pipeline/pipeline.h>

namespace sai::pipeline {

CaptureStage::CaptureStage(std::string id, YAML::Node config, Pipeline* pipeline)
    : id_(std::move(id)), pipeline_(pipeline) {
    auto mode_str = config["trigger_mode"].as<std::string>("freerun");
    if (mode_str == "hardware") {
        trigger_mode_ = device::ICamera::TriggerMode::Hardware;
    } else if (mode_str == "software") {
        trigger_mode_ = device::ICamera::TriggerMode::Software;
    } else {
        trigger_mode_ = device::ICamera::TriggerMode::FreeRun;
    }
}

auto CaptureStage::GetType() const noexcept -> StageType { return StageType::Capture; }
auto CaptureStage::GetId() const -> std::string_view { return id_; }

auto CaptureStage::OnInitialize(Context& ctx) -> Result<void> {
    (void)ctx;
    // ICamera (via IPlugin) is injected externally via SetCamera() before Start().
    // When no camera is set, stub_ remains true and OnStart/Process operate in
    // passthrough mode.
    return {};
}

auto CaptureStage::OnStart(Context&) -> Result<void> {
    if (stub_ || !camera_) return {};

    auto result = camera_->Connect();
    if (!result) return result;

    result = camera_->SetTriggerMode(trigger_mode_);
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
    if (auto* failure = input.GetIf<PipelineFailure>()) {
        return StageOutput::MakeWithContext(input, std::move(*failure));
    }

    // Passthrough: frames arrive via Pipeline::Submit (from camera callback or
    // external submission) into this stage's input queue. The Capture worker
    // dequeues and passes RawImage through to the downstream Preprocess stage.
    if (auto* img = input.GetIf<sai::image::RawImage>()) {
        return StageOutput::MakeWithContext(input, std::move(*img));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Capture expects RawImage input"});
}

}  // namespace sai::pipeline
