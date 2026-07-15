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
        // Primary: Patch embedder (DINOv3 → patch features for PatchCore)
        sai::embedding::Embedding embedding = [&]() -> sai::embedding::Embedding {
            if (!stub_ && embedder_) {
                auto result = embedder_->Extract(*img);
                if (result) return std::move(*result);
            }
            return sai::embedding::Embedding::FromCpu(
                std::vector<float>{}, sai::embedding::EmbeddingMeta{});
        }();

        // Secondary: Global embedder (CLIP → global features for retrieval)
        if (!stub_ && global_embedder_) {
            auto global_result = global_embedder_->Extract(*img);
            if (global_result) {
                const auto& global_meta = global_result->Meta();
                auto count = global_meta.count;
                auto dim = global_meta.dim;
                if (count > 0 && dim > 0) {
                    const float* src = global_result->Data();
                    std::vector<float> features(src, src + count * dim);
                    embedding.SetGlobalFeatures(std::move(features));
                }
            }
        }

        // Carry surface identity from image metadata through the pipeline.
        const auto& img_meta = img->Meta();
        if (!img_meta.surface_id.empty()) {
            embedding.SetSurfaceId(img_meta.surface_id);
        }

        return StageOutput(std::move(embedding));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Inference expects SurfaceImage input"});
}

}  // namespace sai::pipeline
