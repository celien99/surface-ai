#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/context.h>
#include <sai/image/raw_image.h>
#include <sai/pipeline/pipeline_config.h>
#include <sai/pipeline/stage_node.h>

namespace sai {
template <typename T>
class Registry;
}  // namespace sai

namespace sai::runtime {
class TaskGraph;
class PipelineExecutor;
class TaskScheduler;
class WorkerPool;
}  // namespace sai::runtime

namespace sai::pipeline {

// M7: result callback type — invoked in Export worker thread per completed frame
using ResultCallback = std::function<void(int frame_id, const sai::reasoner::ReasoningResult&)>;

struct StageMetrics {
    std::string stage_id;
    StageType type = StageType::Custom;
    std::atomic<size_t> frames_processed{0};
    std::atomic<size_t> frames_failed{0};
    std::atomic<size_t> frames_dropped{0};

    // Per-frame latency (microseconds) of the most recently processed frame.
    // Written by the worker loop, read by Metrics(). Non-atomic double is
    // safe for this single-writer / eventual-read pattern.
    double avg_latency_us = 0.0;
    double p99_latency_us = 0.0;

    StageMetrics() = default;
    StageMetrics(std::string id, StageType t)
        : stage_id(std::move(id)), type(t) {}

    // Custom move: std::atomic is not copyable, so the implicit move
    // constructor is deleted. Load-and-store each atomic to provide
    // movable semantics for std::map storage.
    StageMetrics(StageMetrics&& other) noexcept
        : stage_id(std::move(other.stage_id))
        , type(other.type)
        , frames_processed(other.frames_processed.load())
        , frames_failed(other.frames_failed.load())
        , frames_dropped(other.frames_dropped.load())
        , avg_latency_us(other.avg_latency_us)
        , p99_latency_us(other.p99_latency_us)
        , queue_depth_(other.queue_depth_.load()) {}

    StageMetrics& operator=(StageMetrics&& other) noexcept {
        if (this != &other) {
            stage_id = std::move(other.stage_id);
            type = other.type;
            frames_processed.store(other.frames_processed.load());
            frames_failed.store(other.frames_failed.load());
            frames_dropped.store(other.frames_dropped.load());
            avg_latency_us = other.avg_latency_us;
            p99_latency_us = other.p99_latency_us;
            queue_depth_.store(other.queue_depth_.load());
        }
        return *this;
    }

    size_t queue_depth() const { return queue_depth_.load(); }
    void set_queue_depth(size_t d) { queue_depth_.store(d); }

private:
    std::atomic<size_t> queue_depth_{0};
};

namespace detail {
class ErasedStageQueue {
public:
    virtual ~ErasedStageQueue() = default;
    virtual auto Depth() const -> size_t = 0;
    virtual auto Capacity() const -> size_t = 0;
    virtual auto PushBlocking(std::unique_ptr<StageOutput>) -> void = 0;
    virtual auto TryPush(std::unique_ptr<StageOutput>) -> bool = 0;
    virtual auto TryPop() -> std::unique_ptr<StageOutput> = 0;
    virtual auto PopBlocking() -> std::unique_ptr<StageOutput> = 0;
};
}  // namespace detail

class Pipeline {
public:
    static auto LoadFromYAML(std::filesystem::path yaml_path, Context& ctx)
        -> Result<std::unique_ptr<Pipeline>>;

    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;

    ~Pipeline();

    auto Start() -> Result<void>;
    auto Submit(sai::image::RawImage image) -> Result<void>;
    auto Drain() -> Result<void>;
    auto Stop() -> Result<void>;
    auto Metrics() const -> std::vector<StageMetrics>;

    auto SetResultCallback(ResultCallback callback) -> void;

private:
    Pipeline() = default;

    // DequeueInput: pop from the stage's input queue. When stop_token is
    // set, returns nullptr after the stop is observed (short-polling loop).
    // For entry stages (no depends_on), Submit() pushes directly to their queue.
    auto DequeueInput(const std::string& stage_id, std::stop_token st)
        -> std::unique_ptr<StageOutput>;
    // EnqueueOutputs: push output to all downstream stages' input queues.
    // Uses the adjacency map built during LoadFromYAML.
    auto EnqueueOutputs(const std::string& stage_id, std::unique_ptr<StageOutput>) -> void;
    // BuildQueueWiring: after parsing config, create input queues for each stage
    // and populate adjacency_ (upstream_id -> downstream_ids).
    auto BuildQueueWiring(const PipelineConfig& config) -> Result<void>;

    std::unique_ptr<runtime::TaskGraph> graph_;
    std::unique_ptr<runtime::PipelineExecutor> executor_;
    std::unique_ptr<runtime::TaskScheduler> scheduler_;
    std::unique_ptr<Registry<runtime::WorkerPool>> worker_pools_;
    std::map<std::string, std::unique_ptr<IStageNode>> nodes_;
    // One input queue per stage (keyed by stage id). Entry stages receive
    // frames from Submit(); downstream stages receive from upstream EnqueueOutputs.
    std::map<std::string, std::unique_ptr<detail::ErasedStageQueue>> input_queues_;
    // Adjacency: for each stage, the list of downstream stage ids
    std::map<std::string, std::vector<std::string>> downstreams_;
    std::map<std::string, StageMetrics> metrics_;
    std::stop_source stop_source_;
    std::atomic<bool> running_{false};
    std::atomic<bool> draining_{false};
    ResultCallback result_callback_;
    std::atomic<int> frame_counter_{0};
    std::vector<std::unique_ptr<runtime::WorkerPool>> pools_;
    std::string entry_stage_id_;  // first stage with empty depends_on
    Context* ctx_ = nullptr;      // stored during LoadFromYAML for lifecycle hooks
    // Per-stage worker threads (one jthread per stage).
    // Each thread runs a persistent loop: dequeue, process, enqueue.
    // jthread's built-in stop_token signals shutdown via stop_source_.
    std::vector<std::jthread> worker_threads_;
};

}  // namespace sai::pipeline
