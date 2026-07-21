#include <sai/pipeline/pipeline.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <set>
#include <thread>

#include <sai/core/error.h>
#include <sai/core/registry.h>
#include <sai/core/type_id.h>
#include <sai/detection/detection_result.h>
#include <sai/reasoner/reasoner.h>
#include <sai/runtime/task_graph.h>
#include <sai/runtime/pipeline_executor.h>
#include <sai/runtime/task_scheduler.h>
#include <sai/runtime/worker_pool.h>
#include <sai/pipeline/stage_queue.h>

#include "pipeline_builder.h"
#include "stage_factory.h"
#include <sai/scheduler/scheduler.h>

using namespace std::chrono_literals;

namespace {
constexpr std::size_t kDefaultQueueCapacity = 8;
constexpr auto kDrainTimeout = std::chrono::seconds(30);
}  // namespace

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
    auto DroppedCount() const -> size_t override { return queue_->DroppedCount(); }

    auto PushBlocking(std::unique_ptr<StageOutput> item) -> void override {
        queue_->PushBlocking(std::move(item));
    }
    auto PushBlockingWithStop(std::unique_ptr<StageOutput> item,
                              std::stop_token st) -> bool override {
        return queue_->PushBlockingWithStop(std::move(item), st);
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
    auto PopBlockingWithStop(std::stop_token st) -> std::unique_ptr<StageOutput> override {
        return queue_->PopBlockingWithStop(st);
    }

private:
    std::unique_ptr<StageQueue<StageOutput>> queue_;
    BackpressurePolicy bp_;
};
}  // namespace detail

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

    // Step 1: Create pools via Scheduler (single source of truth for
    // StageType → pool mapping, replacing the duplicated logic previously
    // in the anonymous namespace above).
    pipeline->stage_scheduler_ = std::make_unique<sai::scheduler::Scheduler>();

    // Apply YAML pool_config overrides before Allocate()
    for (const auto& [pool_key, po] : config->pool_overrides) {
        pipeline->stage_scheduler_->OverridePoolConfig(
            pool_key, po.threads, po.queue_capacity);
    }

    BackpressureConfig bp_cfg;
    auto alloc_result = pipeline->stage_scheduler_->Allocate(
        config->stages, bp_cfg);
    if (!alloc_result.has_value()) {
        return tl::make_unexpected(std::move(alloc_result.error()));
    }

    // Step 2: Create TaskScheduler + PipelineExecutor using the Scheduler's
    // pool registry (formerly pipeline->worker_pools_).
    pipeline->task_scheduler_ = std::make_unique<runtime::TaskScheduler>(
        pipeline->stage_scheduler_->Pools());
    pipeline->executor_ = std::make_unique<runtime::PipelineExecutor>(
        *pipeline->task_scheduler_);

    // Step 3: Create stage nodes via StageFactory
    for (auto& stage_cfg : config->stages) {
        auto node = StageFactory::Create(stage_cfg, ctx, pipeline.get());
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
        auto stage_id = sai::detail::Fnv1aHash(
            std::string(sai::scheduler::Scheduler::StageTypeToPoolKey(stage_cfg.type)));

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

        // Determine thread count from scheduler pool config.
        // Multiple threads share the same input queue (StageQueue mutex-serialized
        // for both push and pop), turning each stage into a work-conserving pool.
        auto pool_key = std::string(
            sai::scheduler::Scheduler::StageTypeToPoolKey(node_ptr->GetType()));
        auto pool_cfg = stage_scheduler_->GetEffectivePoolConfig(pool_key);
        std::size_t num_threads = pool_cfg.threads;

        // Shared drop-count tracker across all threads of this stage.
        // Each thread reads the queue's DroppedCount() and atomically
        // accumulates only the delta via compare-exchange, avoiding
        // double-counting when multiple threads observe the same drop.
        auto shared_drop_tracker = std::make_shared<std::atomic<std::size_t>>(0);

        for (std::size_t t = 0; t < num_threads; ++t) {
            auto thread_label = (num_threads > 1)
                ? std::to_string(t) : std::string{};
            worker_threads_.emplace_back(
            [this, stage_id_sp, node_ptr, metrics_ptr, thread_label,
             shared_drop_tracker](std::stop_token st) {
                using Clock = std::chrono::steady_clock;

                // P99 latency reservoir (fixed-size, 100 samples)
                std::array<double, 100> latency_reservoir{};
                std::size_t reservoir_idx = 0;
                std::size_t reservoir_count = 0;

                while (true) {
                    auto input = DequeueInput(*stage_id_sp, st);
                    if (!input) break;  // stop requested AND queue drained

                    // (A) Snapshot queue depth + accumulate dropped frames
                    auto qit = input_queues_.find(*stage_id_sp);
                    if (qit != input_queues_.end()) {
                        metrics_ptr->set_queue_depth(qit->second->Depth());
                        // Atomically accumulate drop count delta across threads
                        std::size_t dc = qit->second->DroppedCount();
                        std::size_t prev =
                            shared_drop_tracker->load(std::memory_order_relaxed);
                        while (dc > prev &&
                               !shared_drop_tracker->compare_exchange_weak(
                                   prev, dc, std::memory_order_relaxed)) {
                            // prev reloaded by CAS; retry
                        }
                        if (dc > prev) {
                            metrics_ptr->frames_dropped.fetch_add(
                                dc - prev, std::memory_order_relaxed);
                        }
                    }

                    auto t_start = Clock::now();
                    auto result = node_ptr->Process(std::move(*input));
                    auto elapsed =
                        std::chrono::duration<double, std::micro>(
                            Clock::now() - t_start)
                            .count();

                    // (B) Update P99 latency reservoir
                    latency_reservoir[reservoir_idx % 100] = elapsed;
                    ++reservoir_idx;
                    if (reservoir_count < 100) ++reservoir_count;
                    if (reservoir_count > 0) {
                        std::array<double, 100> sorted;
                        std::copy_n(latency_reservoir.begin(), reservoir_count,
                                    sorted.begin());
                        auto end = sorted.begin() + reservoir_count;
                        std::sort(sorted.begin(), end);
                        auto p99_idx = static_cast<std::size_t>(
                            static_cast<double>(reservoir_count) * 0.99);
                        if (p99_idx >= reservoir_count)
                            p99_idx = reservoir_count - 1;
                        metrics_ptr->p99_latency_us = sorted[p99_idx];
                    }

                    if (result.has_value()) {
                        // Store SurfaceImage pixel snapshot for Export stage
                        // (per-frame side channel). SurfaceImage is move-only
                        // and forwarded to Inference, so we snapshot raw bytes.
                        if (node_ptr->GetType() == StageType::Preprocess) {
                            auto& variant = result.value();
                            if (auto* si = variant.GetIf<sai::image::SurfaceImage>()) {
                                SetFrameImage(*si);
                            }
                        }

                        // M7: invoke detection callback for live defect overlay
                        if (detection_callback_ && node_ptr->GetType() == StageType::Detect) {
                            auto& variant = result.value();
                            if (auto* dr = variant.GetIf<sai::detection::DetectionResult>()) {
                                detection_callback_(*dr);
                            }
                        }

                        // M7: invoke result callback if set (Export stage only, last stage in chain)
                        if (result_callback_ && node_ptr->GetType() == StageType::Export) {
                            auto& variant = result.value();
                            if (auto* rr = variant.GetIf<sai::reasoner::ReasoningResult>()) {
                                int frame_id = frame_counter_.fetch_add(
                                    1, std::memory_order_relaxed);
                                result_callback_(frame_id, *rr);
                            }
                        }
                        metrics_ptr->frames_processed.fetch_add(
                            1, std::memory_order_relaxed);
                        metrics_ptr->avg_latency_us = elapsed;
                        // Stop-token-aware enqueue: if shutdown was requested
                        // while we were processing, drop the output and exit.
                        if (!EnqueueOutputs(
                                *stage_id_sp,
                                std::make_unique<StageOutput>(
                                    std::move(result).value()),
                                st)) {
                            break;  // stop requested, exit worker loop
                        }
                    } else {
                        metrics_ptr->frames_failed.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
            });
        }  // for each thread in this stage's pool
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

    auto output = std::make_unique<StageOutput>(StageOutput::Make(std::move(image)));
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

    // Block on the queue's condition_variable, waking on data OR stop.
    // No busy-polling — the CV handles both signals efficiently.
    return it->second->PopBlockingWithStop(st);
}

// =========================================================================
// Pipeline::EnqueueOutputs
// =========================================================================

auto Pipeline::EnqueueOutputs(const std::string& stage_id,
                              std::unique_ptr<StageOutput> output) -> void {
    auto adj = downstreams_.find(stage_id);
    if (adj == downstreams_.end()) return;

    for (size_t i = 0; i < adj->second.size(); ++i) {
        auto& downstream_id = adj->second[i];
        auto q = input_queues_.find(downstream_id);
        if (q == input_queues_.end()) continue;

        if (i == 0) {
            q->second->PushBlocking(std::move(output));
        }
    }
}

// Stop-token-aware overload — used by worker threads so they can exit
// gracefully when shutdown is requested, even if the downstream queue is full.
auto Pipeline::EnqueueOutputs(const std::string& stage_id,
                              std::unique_ptr<StageOutput> output,
                              std::stop_token st) -> bool {
    auto adj = downstreams_.find(stage_id);
    if (adj == downstreams_.end()) return true;  // terminal stage — item consumed

    for (size_t i = 0; i < adj->second.size(); ++i) {
        auto& downstream_id = adj->second[i];
        auto q = input_queues_.find(downstream_id);
        if (q == input_queues_.end()) continue;

        if (i == 0) {
            return q->second->PushBlockingWithStop(std::move(output), st);
        }
    }
    return true;
}

// =========================================================================
// Pipeline::BuildQueueWiring
// =========================================================================

auto Pipeline::BuildQueueWiring(const PipelineConfig& config) -> Result<void> {
    // Create an input queue for each stage
    for (auto& stage : config.stages) {
        size_t q_capacity = stage.queue_capacity.value_or(kDefaultQueueCapacity);
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
    auto deadline = std::chrono::steady_clock::now() + kDrainTimeout;

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

auto Pipeline::SetDetectionCallback(DetectionCallback callback) -> void {
    detection_callback_ = std::move(callback);
}

auto Pipeline::SetFrameImage(const sai::image::SurfaceImage& image) -> void {
    // Snapshot pixel data — SurfaceImage is move-only, but we need a
    // copy for Export stage that runs later in the pipeline.
    const auto& meta = image.Meta();
    const auto* data = image.Data();
    std::size_t size = image.SizeBytes();
    if (data && size > 0) {
        std::vector<std::uint8_t> snapshot(data, data + size);
        current_frame_image_.emplace(std::move(snapshot), meta);
    }
}

auto Pipeline::TakeFrameImage() -> std::optional<FrameImageSnapshot> {
    auto result = std::move(current_frame_image_);
    current_frame_image_.reset();
    return result;
}

auto Pipeline::GetStage(std::string_view id) const -> IStageNode* {
    // Heterogeneous lookup via TransparentStringHash — no temporary std::string.
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return nullptr;
    return it->second.get();
}

}  // namespace sai::pipeline
