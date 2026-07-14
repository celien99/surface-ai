#include "stage_nodes.h"

#include <sai/detection/detection_result.h>
#include <sai/image/surface_image.h>
#include <sai/inference/inference_engine.h>

namespace sai::pipeline {

InferenceStage::InferenceStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    model_name_ = config["model"].as<std::string>("");
}

auto InferenceStage::GetType() const noexcept -> StageType { return StageType::Inference; }
auto InferenceStage::GetId() const -> std::string_view { return id_; }

auto InferenceStage::OnInitialize(Context& ctx) -> Result<void> {
    auto engine = ctx.Resolve<inference::IInferenceEngine>();
    if (engine) {
        engine_ = *engine;
        stub_ = false;
    }
    return {};
}

auto InferenceStage::OnStart(Context&) -> Result<void> { return {}; }
auto InferenceStage::OnStop(Context&) -> Result<void> { return {}; }

auto InferenceStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* img = std::get_if<sai::image::SurfaceImage>(&input)) {
        if (!stub_ && engine_) {
            // IInferenceEngine operates on raw tensor bindings, not SurfaceImage.
            // For MockEngine, Infer() produces no-op and detection happens
            // downstream in DetectStage. Return an empty DetectionResult
            // as a placeholder that downstream detect stage enriches.
            auto result = engine_->Infer();
            if (!result) return tl::make_unexpected(result.error());
        }
        // Return placeholder DetectionResult (will be enriched by DetectStage)
        return StageOutput(sai::detection::DetectionResult{});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Inference expects SurfaceImage input"});
}

}  // namespace sai::pipeline
