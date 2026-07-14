#pragma once

#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/object.h>
#include <sai/core/context.h>
#include <sai/pipeline/pipeline_config.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>
#include <sai/embedding/embedding.h>
#include <sai/detection/detection_result.h>
#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_base.h>
#include <sai/reasoner/reasoner.h>

namespace sai::pipeline {

struct RuleEvalOutput {
    sai::rule::FactBase facts;
    std::vector<sai::rule::ResolvedRule> rules;
};

using StageInput = std::variant<
    sai::image::RawImage,               // 0: Capture
    sai::image::SurfaceImage,           // 1: Preprocess
    sai::embedding::Embedding,          // 2: Inference → Detect
    sai::detection::DetectionResult,    // 3: Detect → RuleEval
    sai::pipeline::RuleEvalOutput,      // 4: RuleEval → Reason
    std::vector<sai::rule::ResolvedRule>,// 5: (reserved)
    sai::reasoner::ReasoningResult      // 6: Reason → Export
>;

using StageOutput = StageInput;

class IStageNode : public Object {
public:
    virtual auto GetType() const noexcept -> StageType = 0;
    virtual auto GetId() const -> std::string_view = 0;
    virtual auto OnInitialize(Context&) -> Result<void> = 0;
    virtual auto OnStart(Context&) -> Result<void> = 0;
    virtual auto OnStop(Context&) -> Result<void> = 0;
    virtual auto Process(StageInput) -> Result<StageOutput> = 0;

    // M7: optional hot-reload of stage parameters. Default: no-op (returns success
    // but does nothing; caller should check logs, not rely on behavioural change).
    virtual auto ReloadConfig(const YAML::Node& /*config*/) -> Result<void> {
        return {};
    }
};

}  // namespace sai::pipeline
