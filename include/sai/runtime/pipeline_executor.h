#pragma once

// -----------------------------------------------------------------------
// <sai/runtime/pipeline_executor.h>  (1.4-runtime.md §4)
// -----------------------------------------------------------------------

#include <stop_token>

#include <sai/core/error.h>
#include <sai/core/type_id.h>
#include <sai/runtime/task.h>

namespace sai::runtime {

class TaskScheduler;

// 把 TaskGraph 的单个节点分派到其所属阶段的 WorkerPool；节点到阶段的映射由
// 调用方在装配阶段通过配置文件确定（例如 Capture 类节点分派到 "Capture"
// 阶段的 WorkerPool），本类不做业务语义判断。
class PipelineExecutor final {
public:
    explicit PipelineExecutor(TaskScheduler& scheduler) noexcept;

    // stage_id 由调用方在构造 TaskNode 时一并确定并传入（本签名与
    // TaskScheduler::Submit 的 stage_id 参数含义相同），返回的 Task<void> 在该
    // 节点的 work() 产出的协程被对应 WorkerPool 执行完成后恢复。stop_token 在
    // 发起前已请求停止时不提交，直接返回 Runtime_Cancelled；Submit 失败
    // （Core_TypeNotFound / Runtime_QueueFull）原样向上传播。
    [[nodiscard]] auto Dispatch(TypeId stage_id, Task<void> work,
                                std::stop_token stop_token) noexcept -> Task<void>;

private:
    TaskScheduler& scheduler_;
};

}  // namespace sai::runtime
