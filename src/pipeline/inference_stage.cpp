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
    if (auto* failure = input.GetIf<PipelineFailure>()) {
        return StageOutput::MakeWithContext(input, std::move(*failure));
    }

    if (auto* img = input.GetIf<sai::image::SurfaceImage>()) {
        const auto& meta = img->Meta();
        auto failure = [&](const sai::ErrorInfo& error) -> Result<StageOutput> {
            return StageOutput::MakeWithContext(input, PipelineFailure{
                .code = error.code,
                .stage_id = id_,
                .message = error.message,
                .surface_id = meta.surface_id,
                .position_id = meta.position_id,
            });
        };

        if (stub_ || !embedder_) {
            return failure(sai::ErrorInfo{
                sai::ErrorCode::Inference_EngineExecutionFailed,
                "InferenceStage: patch embedder is not configured"});
        }
        if (!gpu_pool_) {
            return failure(sai::ErrorInfo{
                sai::ErrorCode::Memory_PoolExhausted,
                "InferenceStage: GPU pool is not configured"});
        }

        auto gpu_img_result = sai::image::GpuImage::FromPool(
            *gpu_pool_, meta);
        if (!gpu_img_result) return failure(gpu_img_result.error());
        auto gpu_img = std::move(*gpu_img_result);

        auto cuda_err = cudaMemcpy(
            gpu_img.Data(), img->Data(), img->SizeBytes(),
            cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            return failure(sai::ErrorInfo{
                sai::ErrorCode::Runtime_GpuError,
                std::string("InferenceStage: HtoD copy failed: ") +
                    cudaGetErrorString(cuda_err)});
        }

        auto ext_result = embedder_->Extract(gpu_img);
        if (!ext_result) return failure(ext_result.error());

        auto embedding_result = [&]() -> Result<sai::embedding::Embedding> {
            if (ext_result->IsOnGpu()) {
            // PatchCore currently consumes CPU FAISS vectors. Keep this
            // explicit transfer, but never turn a failed copy into an empty
            // embedding that could be mistaken for a normal frame.
            auto byte_size = ext_result->SizeBytes();
            auto& embedding_meta = ext_result->Meta();
            std::vector<float> cpu_data(embedding_meta.count * embedding_meta.dim);
            auto cuda_err2 = cudaMemcpy(
                cpu_data.data(), ext_result->Data(), byte_size,
                cudaMemcpyDeviceToHost);
            if (cuda_err2 != cudaSuccess) {
                return tl::make_unexpected(sai::ErrorInfo{
                    sai::ErrorCode::Runtime_GpuError,
                    std::string("InferenceStage: DtoH copy failed: ") +
                        cudaGetErrorString(cuda_err2)});
            }
            return sai::embedding::Embedding::FromCpu(
                std::move(cpu_data), embedding_meta);
            }
            return std::move(*ext_result);
        }();
        if (!embedding_result) return failure(embedding_result.error());
        auto embedding = std::move(*embedding_result);

        // Secondary: Global embedder (CLIP → global features for retrieval)
        if (!stub_ && global_embedder_) {
            auto global_result = global_embedder_->Extract(*img);
            if (!global_result) return failure(global_result.error());
            const auto& global_meta = global_result->Meta();
            auto count = global_meta.count;
            auto dim = global_meta.dim;
            if (count > 0 && dim > 0) {
                const float* src = global_result->Data();
                std::vector<float> features(src, src + count * dim);
                embedding.SetGlobalFeatures(std::move(features));
            }
        }

        // Carry surface identity from image metadata through the pipeline.
        const auto& img_meta = meta;
        if (!img_meta.surface_id.empty()) {
            embedding.SetSurfaceId(img_meta.surface_id);
        }
        // Carry position identity from image metadata through the pipeline.
        if (img_meta.position_id != 0) {
            embedding.SetPositionId(img_meta.position_id);
        }

        return StageOutput::MakeWithContext(input, std::move(embedding));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Inference expects SurfaceImage input"});
}

}  // namespace sai::pipeline
