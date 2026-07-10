#include <sai/runtime/task_graph.h>
#include <sai/runtime/pipeline_executor.h>

#include <atomic>
#include <coroutine>
#include <functional>
#include <mutex>
#include <optional>
#include <source_location>
#include <stop_token>
#include <vector>

#include <gtest/gtest.h>

#include <sai/core/error.h>
#include <sai/core/registry.h>
#include <sai/core/type_id.h>
#include <sai/runtime/task.h>
#include <sai/runtime/task_scheduler.h>
#include <sai/runtime/worker_pool.h>

namespace {

using sai::ErrorCode;
using sai::ErrorInfo;
using sai::Registry;
using sai::Result;
using sai::TypeId;
using sai::runtime::PipelineExecutor;
using sai::runtime::Task;
using sai::runtime::TaskGraph;
using sai::runtime::TaskId;
using sai::runtime::TaskNode;
using sai::runtime::TaskScheduler;
using sai::runtime::WorkerPool;

const TypeId kStage = sai::detail::Fnv1aHash("Compute");

// Thread-safe append-only execution log: each node's work() records a tag when
// it runs, so tests can assert both ordering and single-execution properties.
// Node work runs on a WorkerPool thread, but PipelineExecutor::Dispatch blocks
// the driver thread until each node is signalled complete, so writes are fully
// serialized; the mutex is belt-and-braces around that happens-before chain.
struct ExecutionLog {
    std::mutex mutex;
    std::vector<int> tags;

    void Append(int tag) {
        std::unique_lock lock(mutex);
        tags.push_back(tag);
    }

    [[nodiscard]] auto Snapshot() -> std::vector<int> {
        std::unique_lock lock(mutex);
        return tags;
    }

    [[nodiscard]] auto Count(int tag) -> int {
        std::unique_lock lock(mutex);
        int n = 0;
        for (int t : tags) {
            n += (t == tag);
        }
        return n;
    }
};

Task<void> RecordAndSucceed(ExecutionLog& log, int tag) {
    log.Append(tag);
    co_return Result<void>{};
}

Task<void> RecordAndFail(ExecutionLog& log, int tag) {
    log.Append(tag);
    co_return tl::make_unexpected(ErrorInfo{
        ErrorCode::Core_Unknown,
        "deliberate node failure",
        std::source_location::current(),
    });
}

Task<void> IncrementCounter(std::atomic<int>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
    co_return Result<void>{};
}

// Bundles the assembly-phase objects (registry -> scheduler -> executor) whose
// lifetimes must span a whole RunToCompletion/Dispatch drive: the registry owns
// the WorkerPool, and the scheduler/executor only hold references into it.
struct Runtime {
    Registry<WorkerPool> registry;
    TaskScheduler scheduler{registry};
    PipelineExecutor executor{scheduler};

    explicit Runtime(std::size_t thread_count = 1) {
        [[maybe_unused]] Result<void> registered =
            registry.Register(kStage, std::make_shared<WorkerPool>(thread_count, 16));
    }
};

TaskNode MakeNode(TaskId id, std::function<Task<void>()> work, std::vector<TaskId> deps = {}) {
    return TaskNode{id, kStage, std::move(work), std::move(deps)};
}

// Drives a Task<void> to completion on this thread, takes the result, and
// destroys the handle (manual-destroy discipline). Used to run both
// RunToCompletion and Dispatch handles from the test body.
auto DriveToResult(Task<void> handle) -> Result<void> {
    handle.resume();
    EXPECT_TRUE(handle.done());
    Result<void> result = handle.promise().GetResult();
    handle.destroy();
    return result;
}

// ---------------------------------------------------------------------------
// Step 1: TaskGraph / TaskNode
// ---------------------------------------------------------------------------

TEST(TaskGraphTest, AddNodeThenNodeAtRoundTrips) {
    TaskGraph graph;
    ASSERT_TRUE(graph.AddNode(MakeNode(7, []() -> Task<void> { co_return Result<void>{}; }, {1, 2}))
                    .has_value());

    Result<const TaskNode*> found = graph.NodeAt(7);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ((*found)->id, 7u);
    EXPECT_EQ((*found)->stage_id, kStage);
    EXPECT_EQ((*found)->dependencies, (std::vector<TaskId>{1, 2}));
}

TEST(TaskGraphTest, NodeAtUnknownIdReturnsNodeNotFound) {
    TaskGraph graph;
    Result<const TaskNode*> found = graph.NodeAt(999);

    ASSERT_FALSE(found.has_value());
    EXPECT_EQ(found.error().code, ErrorCode::Runtime_NodeNotFound);
}

TEST(TaskGraphTest, AddNodeRejectsDuplicateId) {
    TaskGraph graph;
    ASSERT_TRUE(graph.AddNode(MakeNode(1, []() -> Task<void> { co_return Result<void>{}; }))
                    .has_value());

    Result<void> again = graph.AddNode(MakeNode(1, []() -> Task<void> { co_return Result<void>{}; }));
    ASSERT_FALSE(again.has_value());
    EXPECT_EQ(again.error().code, ErrorCode::Core_TypeAlreadyRegistered);
}

// ---------------------------------------------------------------------------
// Step 2: PipelineExecutor::Dispatch
// ---------------------------------------------------------------------------

TEST(PipelineExecutorTest, DispatchRunsWorkViaWorkerPool) {
    Runtime runtime;
    std::atomic<int> counter{0};

    Result<void> result =
        DriveToResult(runtime.executor.Dispatch(kStage, IncrementCounter(counter),
                                                 std::stop_token{}));

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(counter.load(std::memory_order_relaxed), 1);
}

TEST(PipelineExecutorTest, DispatchToUnregisteredStagePropagatesResolveError) {
    Registry<WorkerPool> registry;  // Nothing registered.
    TaskScheduler scheduler(registry);
    PipelineExecutor executor(scheduler);

    std::atomic<int> counter{0};
    const TypeId missing = sai::detail::Fnv1aHash("Missing");

    Result<void> result =
        DriveToResult(executor.Dispatch(missing, IncrementCounter(counter), std::stop_token{}));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Core_TypeNotFound);
    EXPECT_EQ(counter.load(std::memory_order_relaxed), 0);  // work never ran
}

TEST(PipelineExecutorTest, DispatchWithAlreadyRequestedStopReturnsCancelled) {
    Runtime runtime;
    std::atomic<int> counter{0};
    std::stop_source source;
    source.request_stop();

    Result<void> result =
        DriveToResult(runtime.executor.Dispatch(kStage, IncrementCounter(counter),
                                                 source.get_token()));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Runtime_Cancelled);
    EXPECT_EQ(counter.load(std::memory_order_relaxed), 0);  // never submitted
}

// ---------------------------------------------------------------------------
// Step 3: TaskGraph::RunToCompletion
// ---------------------------------------------------------------------------

TEST(RunToCompletionTest, LinearChainExecutesInDependencyOrder) {
    // A(1) depends on B(2), B(2) depends on C(3) — so C then B then A.
    Runtime runtime;
    ExecutionLog log;
    TaskGraph graph;
    ASSERT_TRUE(graph.AddNode(MakeNode(1, [&] { return RecordAndSucceed(log, 1); }, {2}))
                    .has_value());
    ASSERT_TRUE(graph.AddNode(MakeNode(2, [&] { return RecordAndSucceed(log, 2); }, {3}))
                    .has_value());
    ASSERT_TRUE(graph.AddNode(MakeNode(3, [&] { return RecordAndSucceed(log, 3); }))
                    .has_value());

    Result<void> result = DriveToResult(graph.RunToCompletion(runtime.executor, std::stop_token{}));

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(log.Snapshot(), (std::vector<int>{3, 2, 1}));
}

TEST(RunToCompletionTest, DiamondRunsSharedDependencyExactlyOnce) {
    // A(1) depends on B(2) and C(3); both depend on D(4). D must run once.
    Runtime runtime;
    ExecutionLog log;
    std::atomic<int> d_runs{0};
    TaskGraph graph;
    ASSERT_TRUE(graph.AddNode(MakeNode(1, [&] { return RecordAndSucceed(log, 1); }, {2, 3}))
                    .has_value());
    ASSERT_TRUE(graph.AddNode(MakeNode(2, [&] { return RecordAndSucceed(log, 2); }, {4}))
                    .has_value());
    ASSERT_TRUE(graph.AddNode(MakeNode(3, [&] { return RecordAndSucceed(log, 3); }, {4}))
                    .has_value());
    ASSERT_TRUE(graph.AddNode(MakeNode(4, [&] { return IncrementCounter(d_runs); }, {}))
                    .has_value());

    Result<void> result = DriveToResult(graph.RunToCompletion(runtime.executor, std::stop_token{}));

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(d_runs.load(std::memory_order_relaxed), 1);  // shared dep runs exactly once
    // B and C both precede A, and both follow D — assert relative ordering.
    std::vector<int> tags = log.Snapshot();
    EXPECT_EQ(tags.size(), 3u);  // 1, 2, 3 logged (D used the counter, not the log)
    EXPECT_EQ(tags.back(), 1);   // A runs last
}

TEST(RunToCompletionTest, FailedNodeShortCircuitsDependentsAndSurfacesError) {
    // A(1) depends on B(2); B fails, so A must never run, and the error surfaces.
    Runtime runtime;
    ExecutionLog log;
    TaskGraph graph;
    ASSERT_TRUE(graph.AddNode(MakeNode(1, [&] { return RecordAndSucceed(log, 1); }, {2}))
                    .has_value());
    ASSERT_TRUE(graph.AddNode(MakeNode(2, [&] { return RecordAndFail(log, 2); }))
                    .has_value());

    Result<void> result = DriveToResult(graph.RunToCompletion(runtime.executor, std::stop_token{}));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Core_Unknown);  // B's error, surfaced verbatim
    EXPECT_EQ(log.Count(2), 1);  // B ran
    EXPECT_EQ(log.Count(1), 0);  // A short-circuited, never ran
}

TEST(RunToCompletionTest, MissingDependencyNodeSurfacesNodeNotFound) {
    // A(1) depends on a node id (99) that was never added — structural error.
    Runtime runtime;
    ExecutionLog log;
    TaskGraph graph;
    ASSERT_TRUE(graph.AddNode(MakeNode(1, [&] { return RecordAndSucceed(log, 1); }, {99}))
                    .has_value());

    Result<void> result = DriveToResult(graph.RunToCompletion(runtime.executor, std::stop_token{}));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Runtime_NodeNotFound);
    EXPECT_EQ(log.Count(1), 0);  // dependent never ran
}

}  // namespace
