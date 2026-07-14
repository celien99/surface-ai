#pragma once

// Internal header: declares all M6 concrete stage classes.
// Each class is defined in its own *_stage.cpp file.
// stage_factory.cpp includes this to construct stages by type.

#include <string>
#include <yaml-cpp/yaml.h>

#include <sai/pipeline/stage_node.h>

namespace sai::pipeline {

class CaptureStage final : public IStageNode {
public:
    CaptureStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
};

class PreprocessStage final : public IStageNode {
public:
    PreprocessStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
};

class InferenceStage final : public IStageNode {
public:
    InferenceStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
};

class DetectStage final : public IStageNode {
public:
    DetectStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
};

class RuleEvalStage final : public IStageNode {
public:
    RuleEvalStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
};

class ReasonStage final : public IStageNode {
public:
    ReasonStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
};

class ExportStage final : public IStageNode {
public:
    ExportStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
};

}  // namespace sai::pipeline
