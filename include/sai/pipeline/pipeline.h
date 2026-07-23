#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/context.h>
#include <sai/image/image.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>
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

namespace sai::detection {
struct DetectionResult;
}  // namespace sai::detection

namespace sai::reasoner {
struct ReasoningResult;
}  // namespace sai::reasoner

namespace sai::scheduler {
class Scheduler;
}  // namespace sai::scheduler

namespace sai::pipeline {

// M7: result callback type — invoked in Export worker thread per completed frame
using ResultCallback = std::function<void(
    const std::shared_ptr<const FrameContext>&,
    const sai::reasoner::ReasoningResult&)>;

// M7: detection callback — invoked in Detect worker thread per frame.
// Enables live defect overlay in QML via DefectModel::UpdateDefects().
using DetectionCallback = std::function<void(const sai::detection::DetectionResult&)>;

struct StageMetrics {
    std::string stage_id;
    StageType type = StageType::Custom;
    std::atomic<size_t> frames_processed{0};
    std::atomic<size_t> frames_failed{0};
    std::atomic<size_t> frames_dropped{0};
    std::atomic<size_t> frames_ng{0};

    // Per-frame latency (microseconds) of the most recently processed frame.
    // Stage pools may have multiple workers, so both values are atomic.
    std::atomic<double> avg_latency_us{0.0};
    std::atomic<double> p99_latency_us{0.0};

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
        , frames_ng(other.frames_ng.load())
        , avg_latency_us(other.avg_latency_us.load())
        , p99_latency_us(other.p99_latency_us.load())
        , queue_depth_(other.queue_depth_.load()) {}

    StageMetrics& operator=(StageMetrics&& other) noexcept {
        if (this != &other) {
            stage_id = std::move(other.stage_id);
            type = other.type;
            frames_processed.store(other.frames_processed.load());
            frames_failed.store(other.frames_failed.load());
            frames_dropped.store(other.frames_dropped.load());
            frames_ng.store(other.frames_ng.load());
            avg_latency_us.store(other.avg_latency_us.load());
            p99_latency_us.store(other.p99_latency_us.load());
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

class FrameCompletionState final {
public:
    auto Accept() -> void;
    auto Complete() -> void;
    [[nodiscard]] auto WaitUntil(std::chrono::steady_clock::time_point deadline)
        -> bool;

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t outstanding_ = 0;
};

class FrameCompletionToken final {
public:
    explicit FrameCompletionToken(
        std::shared_ptr<FrameCompletionState> state) noexcept;
    ~FrameCompletionToken();

    FrameCompletionToken(const FrameCompletionToken&) = delete;
    auto operator=(const FrameCompletionToken&) -> FrameCompletionToken& = delete;

private:
    std::shared_ptr<FrameCompletionState> state_;
};

// Transparent hash for heterogeneous lookup: find(std::string_view) on
// unordered_map<std::string, T> without constructing a temporary std::string.
struct TransparentStringHash {
    using is_transparent = void;
    [[nodiscard]] auto operator()(std::string_view sv) const noexcept -> std::size_t {
        return std::hash<std::string_view>{}(sv);
    }
};

template <typename T>
using StringMap = std::unordered_map<std::string, T, TransparentStringHash, std::equal_to<>>;

class ErasedStageQueue {
public:
    virtual ~ErasedStageQueue() = default;
    virtual auto Depth() const -> size_t = 0;
    virtual auto Capacity() const -> size_t = 0;
    virtual auto DroppedCount() const -> size_t = 0;
    virtual auto PushBlocking(std::unique_ptr<StageOutput>) -> void = 0;
    virtual auto PushBlockingWithStop(std::unique_ptr<StageOutput>, std::stop_token) -> bool = 0;
    virtual auto TryPush(std::unique_ptr<StageOutput>) -> bool = 0;
    virtual auto TryPop() -> std::unique_ptr<StageOutput> = 0;
    virtual auto PopBlocking() -> std::unique_ptr<StageOutput> = 0;
    virtual auto PopBlockingWithStop(std::stop_token st) -> std::unique_ptr<StageOutput> = 0;
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
    auto SetDetectionCallback(DetectionCallback callback) -> void;

    // Returns the stage node with the given id, or nullptr if not found.
    // Caller must cast to the concrete stage type (e.g. CaptureStage) and
    // call setter methods (SetEngine, SetCamera, ...) before Start().
    auto GetStage(std::string_view id) const -> IStageNode*;

private:
    enum class State {
        Stopped,
        Starting,
        Running,
        Draining,
    };

    Pipeline() = default;

    // DequeueInput: pop from the stage's input queue. When stop_token is
    // set, returns nullptr after the stop is observed (short-polling loop).
    // For entry stages (no depends_on), Submit() pushes directly to their queue.
    auto DequeueInput(const std::string& stage_id, std::stop_token st)
        -> std::unique_ptr<StageOutput>;
    // EnqueueOutputs: push output to the downstream stage's input queue.
    // Returns false if stop was requested and the item was not pushed.
    // Uses the adjacency map built during LoadFromYAML.
    // Caller discards the item on false.
    auto EnqueueOutputs(const std::string& stage_id, std::unique_ptr<StageOutput>,
                        std::stop_token st) -> bool;
    auto WaitForIdle() -> Result<void>;
    // BuildQueueWiring: after parsing config, create input queues for each stage
    // and populate adjacency_ (upstream_id -> downstream_ids).
    auto BuildQueueWiring(const PipelineConfig& config) -> Result<void>;

    std::unique_ptr<runtime::TaskGraph> graph_;
    std::unique_ptr<runtime::PipelineExecutor> executor_;
    std::unique_ptr<runtime::TaskScheduler> task_scheduler_;
    // Stage->pool mapper (sai::scheduler::Scheduler, owns the WorkerPool registry)
    std::unique_ptr<sai::scheduler::Scheduler> stage_scheduler_;
    detail::StringMap<std::unique_ptr<IStageNode>> nodes_;
    // One input queue per stage (keyed by stage id). Entry stages receive
    // frames from Submit(); downstream stages receive from upstream EnqueueOutputs.
    detail::StringMap<std::unique_ptr<detail::ErasedStageQueue>> input_queues_;
    // Adjacency: for each stage, the list of downstream stage ids
    detail::StringMap<std::vector<std::string>> downstreams_;
    detail::StringMap<StageMetrics> metrics_;
    std::mutex admission_mutex_;
    std::stop_source stop_source_;
    State state_ = State::Stopped;
    std::shared_ptr<detail::FrameCompletionState> frame_completion_ =
        std::make_shared<detail::FrameCompletionState>();
    ResultCallback result_callback_;
    DetectionCallback detection_callback_;
    std::atomic<int> frame_counter_{0};
    std::string entry_stage_id_;  // first stage with empty depends_on
    Context* ctx_ = nullptr;      // stored during LoadFromYAML for lifecycle hooks
    std::stop_source ingress_stop_source_;
    // Per-stage worker threads (one jthread per stage).
    // Each thread runs a persistent loop: dequeue, process, enqueue.
    // jthread's built-in stop_token signals shutdown via stop_source_.
    std::vector<std::jthread> worker_threads_;
};

}  // namespace sai::pipeline
