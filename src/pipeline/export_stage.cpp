#include "stage_nodes.h"
#include <sai/reasoner/reasoner.h>

namespace sai::pipeline {

ExportStage::ExportStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto ExportStage::GetType() const noexcept -> StageType { return StageType::Export; }
auto ExportStage::GetId() const -> std::string_view { return id_; }
auto ExportStage::OnInitialize(Context&) -> Result<void> { return {}; }
auto ExportStage::OnStart(Context&) -> Result<void> { return {}; }
auto ExportStage::OnStop(Context&) -> Result<void> { return {}; }

auto ExportStage::Process(StageInput input) -> Result<StageOutput> {
    // Passthrough: ReasoningResult -> ReasoningResult
    if (auto* result = std::get_if<sai::reasoner::ReasoningResult>(&input)) {
        // In production: serialize to JSON/PPM via IExporter
        return StageOutput(std::move(*result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Export expects ReasoningResult input"});
}

}  // namespace sai::pipeline
