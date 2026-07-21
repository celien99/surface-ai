#include "stage_nodes.h"

#include <sai/detection/detection_result.h>
#include <sai/detection/detector.h>
#include <sai/detection/feature_bank.h>
#include <sai/detection/patch_core.h>
#include <sai/embedding/embedding.h>
#include <sai/infra/logger.h>

#if defined(SAI_CUDA_ENABLED)
#include <cuda_runtime.h>
#endif

namespace sai::pipeline {

DetectStage::DetectStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto DetectStage::GetType() const noexcept -> StageType { return StageType::Detect; }
auto DetectStage::GetId() const -> std::string_view { return id_; }

auto DetectStage::OnInitialize(Context& ctx) -> Result<void> {
    ctx_ = &ctx;
    return {};
}

auto DetectStage::OnStart(Context&) -> Result<void> { return {}; }
auto DetectStage::OnStop(Context&) -> Result<void> { return {}; }

auto DetectStage::SetBootstrapConfig(bool enabled, std::size_t min_frames,
                                      std::size_t target_size) -> void {
    bootstrap_enabled_ = enabled;
    bootstrap_min_frames_ = min_frames;
    bootstrap_target_size_ = target_size;
}

auto DetectStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* failure = input.GetIf<PipelineFailure>()) {
        return StageOutput::MakeWithContext(input, std::move(*failure));
    }

    if (auto* emb = input.GetIf<sai::embedding::Embedding>()) {
        sai::detection::DetectionResult result;

        // Select detector by (surface_id, position_id)
        BankKey key{emb->SurfaceId(), emb->PositionId()};
        auto it = detectors_.find(key);

        if (it == detectors_.end()) {
            // ── Cold-start bootstrap ──
            if (bootstrap_enabled_) {
                if (bootstrap_min_frames_ == 0
                    || bootstrap_target_size_ < bootstrap_min_frames_) {
                    return StageOutput::MakeWithContext(input, PipelineFailure{
                        .code = sai::ErrorCode::Pipeline_InvalidConfig,
                        .stage_id = id_,
                        .message = "DetectStage: bootstrap target_size must be "
                                   "at least min_frames and both must be non-zero",
                        .surface_id = emb->SurfaceId(),
                        .position_id = emb->PositionId(),
                    });
                }

                auto& state = bootstrap_states_[key];
                auto grid_h = emb->Meta().grid[0];
                auto grid_w = emb->Meta().grid[1];
                auto dim = emb->Meta().dim;
                if (dim == 0) {
                    return StageOutput::MakeWithContext(input, PipelineFailure{
                        .code = sai::ErrorCode::Embedding_DimensionMismatch,
                        .stage_id = id_,
                        .message = "DetectStage: bootstrap embedding dimension is zero",
                        .surface_id = emb->SurfaceId(),
                        .position_id = emb->PositionId(),
                    });
                }
                if (grid_h == 0 || grid_w == 0) {
                    return StageOutput::MakeWithContext(input, PipelineFailure{
                        .code = sai::ErrorCode::Detection_InvalidPatchGrid,
                        .stage_id = id_,
                        .message = "DetectStage: bootstrap patch grid is empty",
                        .surface_id = emb->SurfaceId(),
                        .position_id = emb->PositionId(),
                    });
                }
                auto patch_count = grid_h * grid_w;

                if (state.dim == 0) {
                    state.dim = dim;
                    state.grid_h = grid_h;
                    state.grid_w = grid_w;
                } else if (state.dim != dim) {
                    return StageOutput::MakeWithContext(input, PipelineFailure{
                        .code = sai::ErrorCode::Embedding_DimensionMismatch,
                        .stage_id = id_,
                        .message = "DetectStage: bootstrap embedding dimension changed",
                        .surface_id = emb->SurfaceId(),
                        .position_id = emb->PositionId(),
                    });
                } else if (state.grid_h != grid_h || state.grid_w != grid_w) {
                    return StageOutput::MakeWithContext(input, PipelineFailure{
                        .code = sai::ErrorCode::Detection_InvalidPatchGrid,
                        .stage_id = id_,
                        .message = "DetectStage: bootstrap patch grid changed",
                        .surface_id = emb->SurfaceId(),
                        .position_id = emb->PositionId(),
                    });
                }

                // Uniform-sample patches into the bootstrap buffer.
                // Stride adapts so that total accumulated patches ≈ target_size
                // after bootstrap_min_frames_ frames.
                auto target_per_frame = bootstrap_target_size_ / bootstrap_min_frames_;
                auto stride = (patch_count > target_per_frame)
                    ? (patch_count / target_per_frame) : 1;
                if (stride < 1) stride = 1;

                std::vector<float> host_data;
                const float* data = emb->Data();
                if (emb->IsOnGpu()) {
#if defined(SAI_CUDA_ENABLED)
                    host_data.resize(patch_count * dim);
                    auto cuda_err = cudaMemcpy(
                        host_data.data(), data, host_data.size() * sizeof(float),
                        cudaMemcpyDeviceToHost);
                    if (cuda_err != cudaSuccess) {
                        bootstrap_states_.erase(key);
                        return StageOutput::MakeWithContext(input, PipelineFailure{
                            .code = sai::ErrorCode::Runtime_GpuError,
                            .stage_id = id_,
                            .message = std::string("DetectStage: bootstrap DtoH copy failed: ")
                                + cudaGetErrorString(cuda_err),
                            .surface_id = emb->SurfaceId(),
                            .position_id = emb->PositionId(),
                        });
                    }
                    data = host_data.data();
#endif
                }
                for (std::size_t i = 0; i < patch_count; i += stride) {
                    auto offset = i * dim;
                    state.vectors.insert(state.vectors.end(),
                                         data + offset, data + offset + dim);
                }
                ++state.frame_count;

                if (state.frame_count >= bootstrap_min_frames_) {
                    // Build initial FeatureBank via greedy coreset
                    std::vector<float> vecs = std::move(state.vectors);
                    auto sample_count = vecs.size() / dim;

                    sai::embedding::EmbeddingMeta meta;
                    meta.model_name = "bootstrap";
                    meta.type = sai::embedding::EmbeddingType::Patch;
                    meta.dim = dim;
                    meta.count = sample_count;
                    meta.grid = {1, sample_count};
                    auto tmp_emb = sai::embedding::Embedding::FromCpu(
                        std::move(vecs), std::move(meta));
                    std::vector<const sai::embedding::Embedding*> ptrs{&tmp_emb};
                    auto fb_result = detection::FeatureBank::BuildWithGreedyCoreset(
                        ptrs, dim, std::min(bootstrap_target_size_, sample_count));
                    if (!fb_result) {
                        auto error = fb_result.error();
                        bootstrap_states_.erase(key);
                        return StageOutput::MakeWithContext(input, PipelineFailure{
                            .code = error.code,
                            .stage_id = id_,
                            .message = error.message,
                            .surface_id = emb->SurfaceId(),
                            .position_id = emb->PositionId(),
                        });
                    }

                    detection::PatchCore::Config pc_cfg;
                    pc_cfg.embed_dim = dim;
                    pc_cfg.k_nearest = 5;
                    pc_cfg.patch_size = 14;
                    pc_cfg.image_width = state.grid_w * pc_cfg.patch_size;
                    pc_cfg.image_height = state.grid_h * pc_cfg.patch_size;
                    auto pc = std::make_shared<detection::PatchCore>(pc_cfg);
                    auto bootstrap_bank = std::make_unique<detection::FeatureBank>(
                        std::move(*fb_result));
#if defined(SAI_CUDA_ENABLED) && defined(SAI_FAISS_GPU_ENABLED)
                    auto gpu_result = bootstrap_bank->ToGpu();
                    if (!gpu_result) {
                        auto error = gpu_result.error();
                        bootstrap_states_.erase(key);
                        return StageOutput::MakeWithContext(input, PipelineFailure{
                            .code = error.code,
                            .stage_id = id_,
                            .message = error.message,
                            .surface_id = emb->SurfaceId(),
                            .position_id = emb->PositionId(),
                        });
                    }
#endif
                    pc->SetFeatureBank(std::move(bootstrap_bank));
                    if (ctx_ == nullptr) {
                        return StageOutput::MakeWithContext(input, PipelineFailure{
                            .code = sai::ErrorCode::Pipeline_StageInitFailed,
                            .stage_id = id_,
                            .message = "DetectStage: context is not initialized",
                            .surface_id = emb->SurfaceId(),
                            .position_id = emb->PositionId(),
                        });
                    }
                    auto init_result = pc->Initialize(*ctx_);
                    if (!init_result) {
                        bootstrap_states_.erase(key);
                        return StageOutput::MakeWithContext(input, PipelineFailure{
                            .code = init_result.error().code,
                            .stage_id = id_,
                            .message = init_result.error().message,
                            .surface_id = emb->SurfaceId(),
                            .position_id = emb->PositionId(),
                        });
                    }
                    detectors_[key] = pc;
                    stub_ = false;

                    sai::infra::Logger::Get("pipeline").Log(
                        sai::infra::LogLevel::Info,
                        "DetectStage: bootstrap complete — BankKey({}, {}) "
                        "registered with {} samples from {} frames",
                        key.first, key.second, sample_count, state.frame_count);

                    // Notify app so it can create CoresetEvolution etc.
                    if (on_bootstrap_) on_bootstrap_(key, pc);
                    bootstrap_states_.erase(key);
                }

                if (detectors_.find(key) == detectors_.end()) {
                    return StageOutput::MakeWithContext(input, PipelineFailure{
                        .code = sai::ErrorCode::Detection_FeatureBankLoadFailed,
                        .stage_id = id_,
                        .message = "DetectStage: detector bootstrap is in progress",
                        .surface_id = emb->SurfaceId(),
                        .position_id = emb->PositionId(),
                    });
                }
            }
        }

        it = detectors_.find(key);

        auto* detector = (it != detectors_.end()) ? it->second.get() : default_detector_.get();

        if (stub_ || detector == nullptr) {
            return StageOutput::MakeWithContext(input, PipelineFailure{
                .code = sai::ErrorCode::Detection_FeatureBankLoadFailed,
                .stage_id = id_,
                .message = "DetectStage: detector is not configured",
                .surface_id = emb->SurfaceId(),
                .position_id = emb->PositionId(),
            });
        }

        auto det_result = detector->Detect(*emb);
        if (!det_result) {
            return StageOutput::MakeWithContext(input, PipelineFailure{
                .code = det_result.error().code,
                .stage_id = id_,
                .message = det_result.error().message,
                .surface_id = emb->SurfaceId(),
                .position_id = emb->PositionId(),
            });
        }
        result = std::move(*det_result);
        // Carry forward CLIP global features for RuleEvalStage.
        if (emb->HasGlobalFeatures()) {
            result.global_features = emb->GlobalFeatures();
        }
        // Carry forward surface and position identity.
        if (!emb->SurfaceId().empty()) {
            result.surface_id = emb->SurfaceId();
        }
        result.position_id = emb->PositionId();
        return StageOutput::MakeWithContext(input, std::move(result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects Embedding input"});
}

}  // namespace sai::pipeline
