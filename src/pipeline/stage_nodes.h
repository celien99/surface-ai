#pragma once

// Internal header: declares all M6 concrete stage classes.
// Each class is defined in its own *_stage.cpp file.
// stage_factory.cpp includes this to construct stages by type.

#include <functional>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <sai/pipeline/stage_node.h>
#include <sai/device/camera.h>
#include <sai/image/preprocess.h>
#include <sai/inference/inference_engine.h>
#include <sai/embedding/embedder.h>
#include <sai/detection/detector.h>
#include <sai/detection/bounded_patch_sampler.h>
#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_builder.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/retrieval/vector_path.h>
#include <sai/reasoner/reasoner.h>
#include <sai/inference/sam2_segmenter.h>
#include <sai/io/exporter.h>
#include <sai/pipeline/inspection_recorder.h>

namespace sai::memory { class IMemoryPool; }

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
    auto SetCamera(std::shared_ptr<sai::device::ICamera> cam) -> void {
        camera_ = std::move(cam); stub_ = false;
    }
private:
    std::string id_;
    Pipeline* pipeline_ = nullptr;
    std::shared_ptr<sai::device::ICamera> camera_;
    sai::device::ICamera::TriggerMode trigger_mode_ = sai::device::ICamera::TriggerMode::FreeRun;
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
    auto SetEngine(std::shared_ptr<sai::inference::IInferenceEngine> eng) -> void {
        engine_ = std::move(eng); stub_ = false;
    }
    auto SetEmbedder(std::shared_ptr<sai::embedding::IEmbedder> emb) -> void {
        embedder_ = std::move(emb); stub_ = false;
    }
    auto SetGpuPool(sai::memory::IMemoryPool* pool) -> void { gpu_pool_ = pool; }
    auto SetGlobalEmbedder(std::shared_ptr<sai::embedding::IEmbedder> emb) -> void {
        global_embedder_ = std::move(emb); stub_ = false;
    }
private:
    std::string id_;
    std::shared_ptr<sai::inference::IInferenceEngine> engine_;
    std::shared_ptr<sai::embedding::IEmbedder> embedder_;
    std::shared_ptr<sai::embedding::IEmbedder> global_embedder_;
    sai::memory::IMemoryPool* gpu_pool_ = nullptr;
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

    // Register a detector for a specific (surface_id, position_id) pair.
    auto AddDetector(std::string surface_id, std::uint16_t position_id,
                     std::shared_ptr<sai::detection::IDetector> det) -> void {
        detectors_[{std::move(surface_id), position_id}] = std::move(det);
        stub_ = false;
    }

    // Backward compat: single-detector convenience (position=0, any surface).
    auto SetDetector(std::shared_ptr<sai::detection::IDetector> det) -> void {
        default_detector_ = std::move(det);
        stub_ = false;
    }

    // Access for evolution wiring.
    [[nodiscard]] auto GetDetector(const std::string& surface_id,
                                    std::uint16_t position_id) const
        -> std::shared_ptr<sai::detection::IDetector> {
        auto it = detectors_.find({surface_id, position_id});
        if (it != detectors_.end()) return it->second;
        return default_detector_;
    }

    // ── Cold-start bootstrap ──
    // When enabled, unseen BankKeys trigger bootstrap instead of falling back
    // to default_detector_. Patches are accumulated into a buffer; once enough
    // frames have been collected, a FeatureBank + PatchCore are built and
    // registered as a new detector. The caller is notified via callback.
    using BootstrapCallback = std::function<void(
        BankKey, std::shared_ptr<sai::detection::IDetector>)>;
    auto SetBootstrapCallback(BootstrapCallback cb) -> void { on_bootstrap_ = std::move(cb); }
    auto SetBootstrapConfig(bool enabled, std::size_t min_frames,
                            std::size_t target_size) -> void;

    [[nodiscard]] auto HasDetector(const std::string& surface_id,
                                    std::uint16_t position_id) const -> bool {
        return detectors_.count({surface_id, position_id}) > 0;
    }

private:
    std::string id_;
    Context* ctx_ = nullptr;
    std::map<BankKey, std::shared_ptr<sai::detection::IDetector>> detectors_;
    std::shared_ptr<sai::detection::IDetector> default_detector_;
    bool stub_ = true;

    // Bootstrap state
    bool bootstrap_enabled_ = false;
    std::size_t bootstrap_min_frames_ = 50;
    std::size_t bootstrap_target_size_ = 10000;

    struct BootstrapState {
        std::optional<sai::detection::BoundedPatchSampler> sampler;
        std::size_t dim = 0;
        std::size_t grid_h = 0;
        std::size_t grid_w = 0;
        std::size_t frame_count = 0;
    };
    std::map<BankKey, BootstrapState> bootstrap_states_;
    BootstrapCallback on_bootstrap_;
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
    auto SetRuleEngine(std::shared_ptr<sai::rule::RuleEngine> re) -> void {
        rule_engine_ = std::move(re);
        stub_ = false;
    }
    auto SetRuleFile(std::filesystem::path path) -> void {
        rule_file_ = path.string();
    }
    auto SetKnowledgeGraph(std::shared_ptr<sai::knowledge::KnowledgeGraph> kg) -> void {
        kg_ = std::move(kg);
        TryCreateFactBuilder();
    }
    auto SetVectorPath(std::shared_ptr<sai::retrieval::VectorPath> vp) -> void {
        vp_ = std::move(vp);
        TryCreateFactBuilder();
    }
    auto SetInspectionRecorder(std::shared_ptr<InspectionRecorder> rec) -> void {
        recorder_ = std::move(rec);
    }
private:
    std::string id_;
    std::shared_ptr<sai::rule::RuleEngine> rule_engine_;
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg_;
    std::shared_ptr<sai::retrieval::VectorPath> vp_;
    std::unique_ptr<sai::rule::FactBuilder> fact_builder_;
    std::shared_ptr<InspectionRecorder> recorder_;
    std::string rule_file_;
    bool stub_ = true;

    // Setters are called before Start; construct only after both dependencies
    // are present so the builder never captures a null VectorPath.
    auto TryCreateFactBuilder() -> void {
        if (kg_ && vp_ && !fact_builder_)
            fact_builder_ = std::make_unique<sai::rule::FactBuilder>(kg_, vp_);
    }
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
    auto SetReasoner(std::shared_ptr<sai::reasoner::IReasoner> r) -> void {
        reasoner_ = std::move(r); stub_ = false;
    }
    auto SetSam2Segmenter(std::shared_ptr<sai::inference::Sam2Segmenter> seg) -> void {
        sam2_segmenter_ = std::move(seg);
    }
private:
    std::string id_;
    std::shared_ptr<sai::reasoner::IReasoner> reasoner_;
    std::shared_ptr<sai::inference::Sam2Segmenter> sam2_segmenter_;
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
    auto SetExporter(std::shared_ptr<sai::io::IExporter> exp) -> void {
        exporter_ = std::move(exp); stub_ = false;
    }
    auto SetOutputDir(std::filesystem::path dir) -> void { output_dir_ = std::move(dir); }
private:
    std::string id_;
    std::shared_ptr<sai::io::IExporter> exporter_;
    std::filesystem::path output_dir_;
    bool stub_ = true;
};

}  // namespace sai::pipeline
