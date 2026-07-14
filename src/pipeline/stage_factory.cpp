#include "stage_factory.h"
#include "stage_nodes.h"
#include <sai/core/error.h>

namespace sai::pipeline {

auto StageFactory::Create(const StageConfig& config, Context& ctx)
    -> Result<std::unique_ptr<IStageNode>> {
    // For M6: all stages return mock implementations.
    // Production implementations wire to M2/M3/M5 real objects
    // and are implemented in the same *_stage.cpp files.

    switch (config.type) {
        case StageType::Capture: {
            auto stage = std::make_unique<CaptureStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Capture init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Preprocess: {
            auto stage = std::make_unique<PreprocessStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Preprocess init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Inference: {
            auto stage = std::make_unique<InferenceStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Inference init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Detect: {
            auto stage = std::make_unique<DetectStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Detect init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::RuleEval: {
            auto stage = std::make_unique<RuleEvalStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "RuleEval init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Reason: {
            auto stage = std::make_unique<ReasonStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Reason init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Export: {
            auto stage = std::make_unique<ExportStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Export init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Custom:
            return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                "Custom stages not yet supported"});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig, "Unknown stage type"});
}

}  // namespace sai::pipeline
