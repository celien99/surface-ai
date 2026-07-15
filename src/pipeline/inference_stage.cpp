#include "stage_nodes.h"

#include <sai/embedding/embedding.h>
#include <sai/image/surface_image.h>
#include <sai/inference/inference_engine.h>

namespace sai::pipeline {

InferenceStage::InferenceStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    model_name_ = config["model"].as<std::string>("");
}

auto InferenceStage::GetType() const noexcept -> StageType { return StageType::Inference; }
auto InferenceStage::GetId() const -> std::string_view { return id_; }

auto InferenceStage::OnInitialize(Context& /*ctx*/) -> Result<void> {
    // IInferenceEngine set externally via SetEngine()
    return {};
}

auto InferenceStage::OnStart(Context&) -> Result<void> { return {}; }
auto InferenceStage::OnStop(Context&) -> Result<void> { return {}; }

auto InferenceStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* img = std::get_if<sai::image::SurfaceImage>(&input)) {
        // Prefer IEmbedder path (CPU SimplePatchEmbedder or GPU PatchEmbedder).
        if (!stub_ && embedder_) {
            auto result = embedder_->Extract(*img);
            if (!result) return tl::make_unexpected(result.error());
            return StageOutput(std::move(*result));
        }
        // Fallback: raw IInferenceEngine path (GPU TensorRtEngine without adapter).
        if (!stub_ && engine_) {
            auto result = engine_->Infer();
            if (!result) return tl::make_unexpected(result.error());
        }
        // Output Embedding for downstream DetectStage.
        // Real adapter (CLIP/DINOv3/SAM2) or SimplePatchEmbedder populates this
        // with features extracted from the SurfaceImage.
        return StageOutput(sai::embedding::Embedding::FromCpu(
            std::vector<float>{}, sai::embedding::EmbeddingMeta{}));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Inference expects SurfaceImage input"});
}

}  // namespace sai::pipeline
