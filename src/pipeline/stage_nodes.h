#pragma once

// Internal header: declares all M6 concrete stage classes.
// Each class is defined in its own *_stage.cpp file.
// stage_factory.cpp includes this to construct stages by type.

#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <sai/pipeline/stage_node.h>
#include <sai/device/camera.h>
#include <sai/image/preprocess.h>
#include <sai/inference/inference_engine.h>
#include <sai/detection/detector.h>
#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_builder.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/retrieval/vector_path.h>
#include <sai/reasoner/reasoner.h>
#include <sai/io/exporter.h>

namespace sai::pipeline {

// Pipeline defined in <sai/pipeline/pipeline.h> — forward-declared here
// for CaptureStage's non-owning pointer (submits frames from camera callback).
class Pipeline;

class CaptureStage final : public IStageNode {
public:
    CaptureStage(std::string id, YAML::Node config, Pipeline* pipeline);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
    Pipeline* pipeline_ = nullptr;
    std::shared_ptr<sai::device::ICamera> camera_;
    bool stub_ = true;
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
    sai::image::PreprocessFn chain_;
    bool stub_ = true;
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
    std::shared_ptr<sai::inference::IInferenceEngine> engine_;
    std::string model_name_;
    bool stub_ = true;
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
    std::shared_ptr<sai::detection::IDetector> detector_;
    bool stub_ = true;
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
    std::shared_ptr<sai::rule::RuleEngine> rule_engine_;
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg_;
    std::shared_ptr<sai::retrieval::VectorPath> vp_;
    std::unique_ptr<sai::rule::FactBuilder> fact_builder_;
    std::string rule_file_;
    bool stub_ = true;
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
    std::shared_ptr<sai::reasoner::IReasoner> reasoner_;
    std::string tree_file_;
    bool stub_ = true;
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
    std::shared_ptr<sai::io::IExporter> exporter_;
    std::filesystem::path output_dir_;
    bool stub_ = true;
};

}  // namespace sai::pipeline
