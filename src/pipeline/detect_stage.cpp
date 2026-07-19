#include "stage_nodes.h"

#include <sai/detection/detection_result.h>
#include <sai/detection/detector.h>
#include <sai/detection/feature_bank.h>
#include <sai/detection/patch_core.h>
#include <sai/embedding/embedding.h>
#include <sai/infra/logger.h>

namespace sai::pipeline {

DetectStage::DetectStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto DetectStage::GetType() const noexcept -> StageType { return StageType::Detect; }
auto DetectStage::GetId() const -> std::string_view { return id_; }

auto DetectStage::OnInitialize(Context& /*ctx*/) -> Result<void> {
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
    if (auto* emb = input.GetIf<sai::embedding::Embedding>()) {
        sai::detection::DetectionResult result;

        // Select detector by (surface_id, position_id)
        BankKey key{emb->SurfaceId(), emb->PositionId()};
        auto it = detectors_.find(key);

        if (it == detectors_.end()) {
            // ── Cold-start bootstrap ──
            if (bootstrap_enabled_) {
                auto& state = bootstrap_states_[key];
                auto patch_count = emb->Meta().grid[0] * emb->Meta().grid[1];
                auto dim = emb->Meta().dim;

                if (state.dim == 0) {
                    state.dim = dim;
                    state.grid_h = emb->Meta().grid[0];
                    state.grid_w = emb->Meta().grid[1];
                }

                // Uniform-sample patches into the bootstrap buffer.
                // Stride adapts so that total accumulated patches ≈ target_size
                // after bootstrap_min_frames_ frames.
                auto target_per_frame = bootstrap_target_size_ / bootstrap_min_frames_;
                auto stride = (patch_count > target_per_frame)
                    ? (patch_count / target_per_frame) : 1;
                if (stride < 1) stride = 1;

                const float* data = emb->Data();
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
                    if (fb_result.has_value()) {
                        detection::PatchCore::Config pc_cfg;
                        pc_cfg.k_nearest = 5;
                        pc_cfg.image_width = state.grid_w;
                        pc_cfg.image_height = state.grid_h;
                        auto pc = std::make_shared<detection::PatchCore>(pc_cfg);
                        pc->SetFeatureBank(
                            std::make_unique<detection::FeatureBank>(
                                std::move(*fb_result)));
                        detectors_[key] = pc;
                        stub_ = false;

                        sai::infra::Logger::Get("pipeline").Log(
                            sai::infra::LogLevel::Info,
                            "DetectStage: bootstrap complete — BankKey({}, {}) "
                            "registered with {} samples from {} frames",
                            key.first, key.second, sample_count, state.frame_count);

                        // Notify app so it can create CoresetEvolution etc.
                        if (on_bootstrap_) on_bootstrap_(key, pc);
                    }
                    bootstrap_states_.erase(key);
                }

                // Return empty result during bootstrap — no model yet.
                result.surface_id = emb->SurfaceId();
                result.position_id = emb->PositionId();
                return StageOutput::Make(std::move(result));
            }
        }

        auto* detector = (it != detectors_.end()) ? it->second.get() : default_detector_.get();

        if (!stub_ && detector) {
            auto det_result = detector->Detect(*emb);
            if (det_result) result = std::move(*det_result);
        }
        // Carry forward CLIP global features for RuleEvalStage.
        if (emb->HasGlobalFeatures()) {
            result.global_features = emb->GlobalFeatures();
        }
        // Carry forward surface and position identity.
        if (!emb->SurfaceId().empty()) {
            result.surface_id = emb->SurfaceId();
        }
        result.position_id = emb->PositionId();
        return StageOutput::Make(std::move(result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects Embedding input"});
}

}  // namespace sai::pipeline
