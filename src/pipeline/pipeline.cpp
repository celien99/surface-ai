#include <sai/pipeline/pipeline.h>

#include <chrono>
#include <set>
#include <thread>

#include <sai/core/error.h>
#include <sai/core/registry.h>
#include <sai/core/type_id.h>
#include <sai/runtime/task_graph.h>
#include <sai/runtime/pipeline_executor.h>
#include <sai/runtime/task_scheduler.h>
#include <sai/runtime/worker_pool.h>
#include <sai/pipeline/stage_queue.h>

#include "pipeline_builder.h"
#include "stage_factory.h"

using namespace std::chrono_literals;

namespace sai::pipeline {

// ---------------------------------------------------------------------------
// Concrete ErasedStageQueue backed by StageQueue<StageOutput>
// ---------------------------------------------------------------------------
namespace detail {
class ConcreteStageQueue final : public ErasedStageQueue {
public:
    explicit ConcreteStageQueue(std::unique_ptr<StageQueue<StageOutput>> queue,
                                BackpressurePolicy bp)
        : queue_(std::move(queue)), bp_(bp) {}

    auto Depth() const -> size_t override { return queue_->Depth(); }
    auto Capacity() const -> size_t override { return queue_->Capacity(); }

    auto PushBlocking(std::unique_ptr<StageOutput> item) -> void override {
        queue_->PushBlocking(std::move(item));
    }
    auto TryPush(std::unique_ptr<StageOutput> item) -> bool override {
        return queue_->TryPush(std::move(item));
    }
    auto TryPop() -> std::unique_ptr<StageOutput> override {
        return queue_->TryPop();
    }
    auto PopBlocking() -> std::unique_ptr<StageOutput> override {
        return queue_->PopBlocking();
    }

private:
    std::unique_ptr<StageQueue<StageOutput>> queue_;
    BackpressurePolicy bp_;
};
}  // namespace detail

namespace {

// ---------------------------------------------------------------------------
// Maps StageType -> M1 stage_id string (used as WorkerPool registry key)
// ---------------------------------------------------------------------------
std::string StageTypeToStr(StageType t) {
    switch (t) {
        case StageType::Capture:     return "Capture";
        case StageType::Preprocess:  return "Capture";
        case StageType::Inference:   return "Inference";
        case StageType::Detect:      return "Inference";
        case StageType::RuleEval:    return "Reason";
        case StageType::Reason:      return "Reason";
        case StageType::Export:      return "IO";
        case StageType::Custom:      return "Background";
    }
    return "Background";
}

// ---------------------------------------------------------------------------
// Default pool sizes per StageType (string key)
// ---------------------------------------------------------------------------
struct PoolConfig {
    size_t threads;
    size_t queue_capacity;
};

PoolConfig PoolConfigFor(std::string_view stage_str) {
    if (stage_str == "Capture")    return {2, 8};
    if (stage_str == "Inference")  return {1, 4};
    if (stage_str == "Reason")     return {2, 16};
    if (stage_str == "IO")         return {1, 32};
    if (stage_str == "Background") return {1, 16};
    return {1, 8};
}

}  // anonymous namespace

// =========================================================================
// Pipeline::LoadFromYAML
// =========================================================================

auto Pipeline::LoadFromYAML(std::filesystem::path yaml_path, Context& ctx)
    -> Result<std::unique_ptr<Pipeline>> {
    // Step 0: Parse + validate YAML
    auto config = PipelineBuilder::ParseFromYAML(yaml_path);
    if (!config.has_value()) return tl::make_unexpected(std::move(config).error());

    auto valid = PipelineBuilder::Validate(*config);
    if (!valid.has_value()) return tl::make_unexpected(std::move(valid).error());

    auto pipeline = std::unique_ptr<Pipeline>(new Pipeline());

    // Step 1: Create worker pools for each unique stage type string
    std::set<std::string> stage_strs;
    for (auto& s : config->stages) {
        stage_strs.insert(StageTypeToStr(s.type));
    }

    pipeline->worker_pools_ =
        std::make_unique<Registry<runtime::WorkerPool>>();

    for (auto& str : stage_strs) {
        auto pool_cfg = PoolConfigFor(str);
        auto pool = std::make_shared<runtime::WorkerPool>(
            pool_cfg.threads, pool_cfg.queue_capacity);
        TypeId type_id = sai::detail::Fnv1aHash(str);
        auto reg_result = pipeline->worker_pools_->Register(type_id, std::move(pool));
        if (!reg_result.has_value()) return tl::make_unexpected(std::move(reg_result).error());
    }

    // Step 2: Create TaskScheduler + PipelineExecutor
    pipeline->scheduler_ = std::make_unique<runtime::TaskScheduler>(
        *pipeline->worker_pools_);
    pipeline->executor_ = std::make_unique<runtime::PipelineExecutor>(
        *pipeline->scheduler_);

    // Step 3: Create stage nodes via StageFactory
    for (auto& stage_cfg : config->stages) {
        auto node = StageFactory::Create(stage_cfg, ctx);
        if (!node.has_value()) return tl::make_unexpected(std::move(node).error());

        // metrics entry constructed in-place (std::atomic is not copyable)
        pipeline->metrics_.try_emplace(stage_cfg.id, stage_cfg.id, stage_cfg.type);

        pipeline->nodes_[stage_cfg.id] = std::move(*node);
    }

    // Step 4: Wire queues between stages
    auto wire_result = pipeline->BuildQueueWiring(*config);
    if (!wire_result.has_value()) return tl::make_unexpected(std::move(wire_result).error());

    // Step 5: Build TaskGraph (after queue wiring so lambdas have valid refs)
    pipeline->graph_ = std::make_unique<runtime::TaskGraph>();

    for (auto& stage_cfg : config->stages) {
        auto stage_id = sai::detail::Fnv1aHash(StageTypeToStr(stage_cfg.type));

        runtime::TaskId task_id = sai::detail::Fnv1aHash(stage_cfg.id);

        std::vector<runtime::TaskId> dep_ids;
        for (auto& dep : stage_cfg.depends_on) {
            dep_ids.push_back(sai::detail::Fnv1aHash(dep));
        }

        // References into pipeline-owned maps (stable after emplace)
        auto& metrics = pipeline->metrics_.at(stage_cfg.id);

        runtime::TaskNode task_node;
        task_node.id = task_id;
        task_node.stage_id = stage_id;
        task_node.dependencies = dep_ids;
        task_node.work = [node = pipeline->nodes_[stage_cfg.id].get(),
                          &metrics, &p = *pipeline,
                          stage_id_str = stage_cfg.id]()
            -> runtime::Task<void> {
            auto input = p.DequeueInput(stage_id_str);
            if (!input) {
                metrics.frames_failed.fetch_add(1, std::memory_order_relaxed);
                co_return Result<void>{};
            }

            auto result = node->Process(std::move(*input));
            if (result.has_value()) {
                metrics.frames_processed.fetch_add(1, std::memory_order_relaxed);
                auto output = std::make_unique<StageOutput>(
                    std::move(result.value()));
                p.EnqueueOutputs(stage_id_str, std::move(output));
                co_return Result<void>{};
            } else {
                metrics.frames_failed.fetch_add(1, std::memory_order_relaxed);
                co_return tl::make_unexpected(result.error());
            }
        };

        auto add_result = pipeline->graph_->AddNode(std::move(task_node));
        if (!add_result.has_value()) return tl::make_unexpected(std::move(add_result).error());
    }

    return pipeline;
}

// =========================================================================
// Pipeline::Start
// =========================================================================

auto Pipeline::Start() -> Result<void> {
    if (running_.exchange(true)) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Pipeline_InvalidState,
            "Pipeline already running"});
    }
    return {};
}

// =========================================================================
// Pipeline::Submit
// =========================================================================

auto Pipeline::Submit(sai::image::RawImage image) -> Result<void> {
    if (!running_.load()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Pipeline_InvalidState,
            "Pipeline not running — call Start() first"});
    }

    auto it = input_queues_.find(entry_stage_id_);
    if (it == input_queues_.end()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Pipeline_InvalidConfig,
            "No entry stage queue found for: " + entry_stage_id_});
    }

    auto output = std::make_unique<StageOutput>(std::move(image));
    it->second->PushBlocking(std::move(output));
    return {};
}

// =========================================================================
// Pipeline::DequeueInput
// =========================================================================

auto Pipeline::DequeueInput(const std::string& stage_id)
    -> std::unique_ptr<StageOutput> {
    auto it = input_queues_.find(stage_id);
    if (it == input_queues_.end()) return nullptr;
    return it->second->PopBlocking();
}

// =========================================================================
// Pipeline::EnqueueOutputs
// =========================================================================

auto Pipeline::EnqueueOutputs(const std::string& stage_id,
                              std::unique_ptr<StageOutput> output) -> void {
    auto adj = downstreams_.find(stage_id);
    if (adj == downstreams_.end()) return;

    // StageOutput is a variant containing move-only types (RawImage),
    // so it cannot be copied. For M6 v1, forward to the first downstream
    // only. Production: each downstream gets a shared_ptr<const StageOutput>.
    for (size_t i = 0; i < adj->second.size(); ++i) {
        auto& downstream_id = adj->second[i];
        auto q = input_queues_.find(downstream_id);
        if (q == input_queues_.end()) continue;

        if (i == 0) {
            // Move the original output to the first downstream
            q->second->PushBlocking(std::move(output));
        }
        // For v1: subsequent downstreams are skipped (output already moved)
    }
}

// =========================================================================
// Pipeline::BuildQueueWiring
// =========================================================================

auto Pipeline::BuildQueueWiring(const PipelineConfig& config) -> Result<void> {
    // Create an input queue for each stage
    for (auto& stage : config.stages) {
        size_t q_capacity = stage.queue_capacity.value_or(8);
        auto sq_result = StageQueue<StageOutput>::Create(
            q_capacity, stage.backpressure);
        if (!sq_result.has_value()) return tl::make_unexpected(std::move(sq_result).error());

        auto concrete = std::make_unique<detail::ConcreteStageQueue>(
            std::move(*sq_result), stage.backpressure);
        input_queues_[stage.id] = std::move(concrete);
    }

    // Build adjacency map (upstream -> downstreams)
    for (auto& stage : config.stages) {
        for (auto& dep : stage.depends_on) {
            downstreams_[dep].push_back(stage.id);
        }
    }

    // Identify entry stage (first stage with empty depends_on)
    for (auto& stage : config.stages) {
        if (stage.depends_on.empty()) {
            entry_stage_id_ = stage.id;
            break;
        }
    }

    return {};
}

// =========================================================================
// Pipeline::Drain
// =========================================================================

auto Pipeline::Drain() -> Result<void> {
    draining_.store(true);
    // Wait for all queues to drain.
    // For M6 v1 mock stages, processing is instantaneous; a brief sleep
    // gives any in-flight work time to complete. Production implementation
    // should poll queue depths until all are zero.
    std::this_thread::sleep_for(10ms);
    draining_.store(false);
    return {};
}

// =========================================================================
// Pipeline::Stop
// =========================================================================

auto Pipeline::Stop() -> Result<void> {
    stop_source_.request_stop();
    running_.store(false);

    // Drain remaining frames
    auto drain_result = Drain();
    if (!drain_result.has_value()) return drain_result;

    return {};
}

// =========================================================================
// Pipeline::Metrics
// =========================================================================

auto Pipeline::Metrics() const -> std::vector<StageMetrics> {
    std::vector<StageMetrics> result;
    result.reserve(metrics_.size());
    for (const auto& [id, m] : metrics_) {
        // StageMetrics contains std::atomic members which are not copyable,
        // so construct each returned struct field-by-field.
        StageMetrics sm(m.stage_id, m.type);
        sm.frames_processed.store(m.frames_processed.load());
        sm.frames_failed.store(m.frames_failed.load());
        sm.frames_dropped.store(m.frames_dropped.load());
        sm.set_queue_depth(m.queue_depth());
        result.push_back(std::move(sm));
    }
    return result;
}

}  // namespace sai::pipeline
