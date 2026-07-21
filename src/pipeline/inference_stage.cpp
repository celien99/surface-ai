#include "stage_nodes.h"

#include <sai/embedding/embedding.h>
#include <sai/image/surface_image.h>
#include <sai/image/gpu_image.h>
#include <sai/inference/inference_engine.h>
#include <sai/memory/gpu_pool.h>
#include <cuda_runtime.h>

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
    if (auto* img = input.GetIf<sai::image::SurfaceImage>()) {
        sai::embedding::Embedding embedding = [&]() -> sai::embedding::Embedding {
            if (stub_ || !embedder_) {
                return sai::embedding::Embedding::FromCpu(
                    std::vector<float>{}, sai::embedding::EmbeddingMeta{});
            }

            // HtoD: upload SurfaceImage to GPU before feeding to PatchEmbedder.
            // PatchEmbedder::Extract requires GpuImage (DINOv2 adapter uses GPU memory).
            if (!gpu_pool_) {
                return sai::embedding::Embedding::FromCpu(
                    std::vector<float>{}, sai::embedding::EmbeddingMeta{});
            }

            auto gpu_img_result = sai::image::GpuImage::FromPool(
                *gpu_pool_, img->Meta());
            if (!gpu_img_result) {
                return sai::embedding::Embedding::FromCpu(
                    std::vector<float>{}, sai::embedding::EmbeddingMeta{});
            }
            auto gpu_img = std::move(*gpu_img_result);

            auto cuda_err = cudaMemcpy(
                gpu_img.Data(), img->Data(), img->SizeBytes(),
                cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
                return sai::embedding::Embedding::FromCpu(
                    std::vector<float>{}, sai::embedding::EmbeddingMeta{});
            }

            auto ext_result = embedder_->Extract(gpu_img);
            if (!ext_result) {
                return sai::embedding::Embedding::FromCpu(
                    std::vector<float>{}, sai::embedding::EmbeddingMeta{});
            }

            // DtoH: download patch features to CPU for downstream stages
            // (PatchCore uses FAISS CPU index, k-NN search reads host memory).
            if (ext_result->IsOnGpu()) {
                auto byte_size = ext_result->SizeBytes();
                auto& meta = ext_result->Meta();
                std::vector<float> cpu_data(meta.count * meta.dim);
                auto cuda_err2 = cudaMemcpy(
                    cpu_data.data(), ext_result->Data(), byte_size,
                    cudaMemcpyDeviceToHost);
                if (cuda_err2 != cudaSuccess) {
                    return sai::embedding::Embedding::FromCpu(
                        std::vector<float>{}, sai::embedding::EmbeddingMeta{});
                }
                return sai::embedding::Embedding::FromCpu(
                    std::move(cpu_data), meta);
            }
            return std::move(*ext_result);
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
        // Carry position identity from image metadata through the pipeline.
        if (img_meta.position_id != 0) {
            embedding.SetPositionId(img_meta.position_id);
        }

        return StageOutput::Make(std::move(embedding));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Inference expects SurfaceImage input"});
}

}  // namespace sai::pipeline
