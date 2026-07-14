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
// Pipeline::~Pipeline
// =========================================================================

Pipeline::~Pipeline() = default;

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
    pipeline->ctx_ = &ctx;  // stored for lifecycle hooks (I2)

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
        // TaskGraph node holds a no-op placeholder; real persistent work
        // loops are launched in Start() via fire-and-forget coroutines.
        task_node.work = []() -> runtime::Task<void> {
            co_return Result<void>{};
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

    // I2: call OnStart lifecycle hooks on each stage node
    if (ctx_) {
        for (auto& [id, node] : nodes_) {
            auto result = node->OnStart(*ctx_);
            if (!result.has_value()) return tl::make_unexpected(result.error());
        }
    }

    // C1 + I1: launch one persistent worker thread per stage. Each thread
    // continuously dequeues from its input queue, processes, and enqueues to
    // downstream queues until stop is requested AND the input queue is drained.
    //
    // Uses std::jthread (not coroutines) to avoid coroutine frame heap-elision
    // issues observed on Apple Clang 21.

    for (auto& [stage_id, node] : nodes_) {
        StageMetrics* metrics_ptr = &metrics_[stage_id];
        auto stage_id_sp = std::make_shared<std::string>(stage_id);
        IStageNode* node_ptr = node.get();

        worker_threads_.emplace_back(
            [this, stage_id_sp, node_ptr, metrics_ptr](std::stop_token st) {
                using Clock = std::chrono::steady_clock;

                while (true) {
                    auto input = DequeueInput(*stage_id_sp, st);
                    if (!input) break;  // stop requested AND queue drained

                    auto t_start = Clock::now();
                    auto result = node_ptr->Process(std::move(*input));
                    auto elapsed =
                        std::chrono::duration<double, std::micro>(
                            Clock::now() - t_start)
                            .count();

                    if (result.has_value()) {
                        // M7: invoke result callback if set (Export stage only, last stage in chain)
                        if (result_callback_ && node_ptr->GetType() == StageType::Export) {
                            auto& variant = result.value();
                            if (auto* rr = std::get_if<sai::reasoner::ReasoningResult>(&variant)) {
                                int frame_id = frame_counter_.fetch_add(
                                    1, std::memory_order_relaxed);
                                result_callback_(frame_id, *rr);
                            }
                        }
                        metrics_ptr->frames_processed.fetch_add(
                            1, std::memory_order_relaxed);
                        metrics_ptr->avg_latency_us = elapsed;
                        EnqueueOutputs(
                            *stage_id_sp,
                            std::make_unique<StageOutput>(
                                std::move(result.value())));
                    } else {
                        metrics_ptr->frames_failed.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
            });
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

auto Pipeline::DequeueInput(const std::string& stage_id,
                              std::stop_token st)
    -> std::unique_ptr<StageOutput> {
    auto it = input_queues_.find(stage_id);
    if (it == input_queues_.end()) return nullptr;

    // Poll with short sleep, checking stop_token each iteration.
    while (!st.stop_requested()) {
        auto item = it->second->TryPop();
        if (item) return item;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }

    // Stop requested: one last non-blocking try to drain remaining items.
    return it->second->TryPop();
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

    // Poll all input queues until all are zero, with a 30 s timeout.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);

    while (std::chrono::steady_clock::now() < deadline) {
        bool all_empty = true;
        for (auto& [id, q] : input_queues_) {
            if (q->Depth() > 0) {
                all_empty = false;
                break;
            }
        }
        if (all_empty) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // After timeout (or early exit), force-drain any remaining items.
    for (auto& [id, q] : input_queues_) {
        while (q->Depth() > 0) {
            q->TryPop();
        }
    }

    draining_.store(false);
    return {};
}

// =========================================================================
// Pipeline::Stop
// =========================================================================

auto Pipeline::Stop() -> Result<void> {
    stop_source_.request_stop();
    running_.store(false);

    // Request stop on all worker jthreads (they share the same stop_source
    // via the lambdas' stop_token, but each jthread also has its own
    // stop_source — we request stop on ours so DequeueInput sees it).
    for (auto& t : worker_threads_) {
        t.request_stop();
    }

    // Wait for all worker threads to finish
    for (auto& t : worker_threads_) {
        t.join();
    }
    worker_threads_.clear();

    // Drain any remaining frames that may have been left in queues
    auto drain_result = Drain();
    if (!drain_result.has_value()) return drain_result;

    // I2: call OnStop lifecycle hooks in reverse order
    if (ctx_) {
        std::vector<std::string> reverse_ids;
        reverse_ids.reserve(nodes_.size());
        for (auto& [id, node] : nodes_) reverse_ids.push_back(id);
        for (auto it = reverse_ids.rbegin(); it != reverse_ids.rend(); ++it) {
            auto result = nodes_[*it]->OnStop(*ctx_);
            if (!result.has_value()) return tl::make_unexpected(result.error());
        }
    }

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
        sm.avg_latency_us = m.avg_latency_us;
        sm.p99_latency_us = m.p99_latency_us;
        sm.set_queue_depth(m.queue_depth());
        result.push_back(std::move(sm));
    }
    return result;
}

// =========================================================================
// Pipeline::SetResultCallback (M7)
// =========================================================================

auto Pipeline::SetResultCallback(ResultCallback callback) -> void {
    result_callback_ = std::move(callback);
}

}  // namespace sai::pipeline
