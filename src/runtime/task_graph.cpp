#include <sai/runtime/task_graph.h>

#include <coroutine>
#include <source_location>
#include <utility>
#include <vector>

#include <sai/runtime/pipeline_executor.h>

namespace sai::runtime {
namespace {

// SyncAwaiter drives a child Task<T> to completion inline on the awaiting
// thread and hands back its Result<T>, then destroys the child frame (§11
// manual-destroy discipline). await_suspend runs the child synchronously and
// returns false so the awaiting coroutine is never actually suspended — it
// falls straight through to await_resume. This is what turns the mutually
// recursive EnsureCompleted/ResolveDependencies coroutines into a depth-first
// walk that unwinds on the driver thread, while each node's real work still
// runs on its stage's WorkerPool inside PipelineExecutor::Dispatch.
template <typename T>
struct SyncAwaiter {
    Task<T> task;

    [[nodiscard]] auto await_ready() const noexcept -> bool { return false; }

    auto await_suspend(std::coroutine_handle<>) const noexcept -> bool {
        task.resume();
        return false;
    }

    [[nodiscard]] auto await_resume() const noexcept -> Result<T> {
        Result<T> result = task.promise().GetResult();
        task.destroy();
        return result;
    }
};

// Found by ADL on Task<T> = std::coroutine_handle<TaskPromise<T>> (associated
// namespace sai::runtime), so `co_await some_task` inside this TU resolves to a
// SyncAwaiter without Task<T> itself needing a member operator co_await.
template <typename T>
[[nodiscard]] auto operator co_await(Task<T> task) noexcept -> SyncAwaiter<T> {
    return SyncAwaiter<T>{task};
}

// Note: TaskPromise<void> already carries a Result<void> as its payload
// (return_value(Result<void>) / GetResult() -> Result<void>), so §5's
// conceptual "Task<Result<void>>" is exactly Task<void> here.
auto EnsureCompleted(TaskId id, TaskGraph& graph, PipelineExecutor& executor,
                     std::unordered_map<TaskId, Result<void>>& completed,
                     std::stop_token stop_token) -> Task<void>;

// Head/tail recursion over a list of node ids: ensure each in order, short-circuit
// on the first failure. Serves both as §5's ResolveDependencies (over one node's
// dependency list) and as RunToCompletion's driver over every node id.
auto EnsureEach(std::vector<TaskId> ids, TaskGraph& graph, PipelineExecutor& executor,
                std::unordered_map<TaskId, Result<void>>& completed,
                std::stop_token stop_token) -> Task<void> {
    if (ids.empty()) {
        co_return Result<void>{};  // 递归基：没有更多依赖
    }
    auto head_result =
        co_await EnsureCompleted(ids.front(), graph, executor, completed, stop_token);
    if (!head_result) {
        co_return head_result;  // 表头依赖失败，短路，不再递归处理表尾
    }
    co_return co_await EnsureEach(std::vector<TaskId>(ids.begin() + 1, ids.end()), graph, executor,
                                  completed, stop_token);
}

auto EnsureCompleted(TaskId id, TaskGraph& graph, PipelineExecutor& executor,
                     std::unordered_map<TaskId, Result<void>>& completed,
                     std::stop_token stop_token) -> Task<void> {
    if (auto it = completed.find(id); it != completed.end()) {
        co_return it->second;  // 递归基：已完成，幂等返回缓存结果
    }
    if (stop_token.stop_requested()) {
        co_return tl::make_unexpected(ErrorInfo{
            ErrorCode::Runtime_Cancelled,
            "task graph execution cancelled",
            std::source_location::current(),
        });
    }

    auto node_result = graph.NodeAt(id);
    if (!node_result) {
        // 图结构错误（依赖引用了不存在的节点），原样保留底层 ErrorInfo 并上传。
        Result<void> err(tl::unexpect, node_result.error());
        completed.emplace(id, err);
        co_return err;
    }
    const TaskNode& node = *node_result.value();

    auto deps_result =
        co_await EnsureEach(node.dependencies, graph, executor, completed, stop_token);
    if (!deps_result) {
        completed.emplace(id, deps_result);
        co_return deps_result;  // 依赖失败，本节点不再调度，原样向上传播
    }

    auto result = co_await executor.Dispatch(node.stage_id, node.work(), stop_token);
    completed.emplace(id, result);
    co_return result;
}

}  // namespace

auto TaskGraph::AddNode(TaskNode node) noexcept -> Result<void> {
    const TaskId id = node.id;
    if (!nodes_.emplace(id, std::move(node)).second) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeAlreadyRegistered,
            "task node id already added",
            std::source_location::current(),
        });
    }
    return {};
}

auto TaskGraph::NodeAt(TaskId id) const noexcept -> Result<const TaskNode*> {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Runtime_NodeNotFound,
            "task node id not found in graph",
            std::source_location::current(),
        });
    }
    return &it->second;
}

auto TaskGraph::RunToCompletion(PipelineExecutor& executor, std::stop_token stop_token) noexcept
    -> Task<void> {
    std::unordered_map<TaskId, Result<void>> completed;

    std::vector<TaskId> ids;
    ids.reserve(nodes_.size());
    for (const auto& entry : nodes_) {
        ids.push_back(entry.first);
    }

    co_return co_await EnsureEach(std::move(ids), *this, executor, completed, stop_token);
}

}  // namespace sai::runtime
