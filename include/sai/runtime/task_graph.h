#pragma once

// -----------------------------------------------------------------------
// <sai/runtime/task_graph.h>  (1.4-runtime.md §4)
// -----------------------------------------------------------------------

#include <cstdint>
#include <functional>
#include <stop_token>
#include <unordered_map>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/type_id.h>
#include <sai/runtime/task.h>

namespace sai::runtime {

class PipelineExecutor;

using TaskId = std::uint64_t;

// TaskGraph 的节点表示；stage_id 是本节点应分派到的 Pipeline 阶段
// （PipelineExecutor::Dispatch 的 stage_id 参数同源，均由阶段名字符串的
// Fnv1aHash 结果得到），dependencies 列出本节点必须等待哪些节点先完成；边隐含
// 在 dependencies 字段中，不设独立的边集合数据结构（见 6. Data Structure）。
struct TaskNode {
    TaskId id;
    TypeId stage_id;
    std::function<Task<void>()> work;
    std::vector<TaskId> dependencies;
};

// DAG 任务图本身：只持有节点集合与拓扑遍历所需的最小状态，不持有任何
// WorkerPool 引用（分派到哪个 WorkerPool 是 PipelineExecutor 的职责）。
class TaskGraph final {
public:
    // 装配阶段逐个插入节点；节点 id 已存在时返回 Core_TypeAlreadyRegistered
    // （同一 id 重复插入属于装配逻辑错误，不是运行期可恢复路径）。
    [[nodiscard]] auto AddNode(TaskNode node) noexcept -> Result<void>;

    // 只读查询：按 TaskId 取回节点内容，供递归拓扑遍历下钻每一层依赖时使用。
    // id 不对应任何已 AddNode 过的节点时返回 Runtime_NodeNotFound。返回值是指向
    // 内部 nodes_ 存储的裸指针而非拷贝，指针生命周期与 TaskGraph 本身一致。
    [[nodiscard]] auto NodeAt(TaskId id) const noexcept -> Result<const TaskNode*>;

    // 递归拓扑遍历驱动执行：见 5. Workflow 的完整算法与递归表达理由；本方法不对外
    // 暴露迭代版本的实现细节。
    [[nodiscard]] auto RunToCompletion(PipelineExecutor& executor,
                                        std::stop_token stop_token) noexcept -> Task<void>;

private:
    std::unordered_map<TaskId, TaskNode> nodes_;
};

}  // namespace sai::runtime
