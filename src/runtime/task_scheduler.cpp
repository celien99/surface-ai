#include <sai/runtime/task_scheduler.h>

namespace sai::runtime {

TaskScheduler::TaskScheduler(Registry<WorkerPool>& worker_pools) noexcept
    : worker_pools_(worker_pools) {}

auto TaskScheduler::Submit(TypeId stage_id, std::coroutine_handle<> handle) noexcept
    -> Result<void> {
    // Registry<WorkerPool>::Resolve already returns Core_TypeNotFound for an
    // unregistered stage_id (1.2's frozen Registry<TInterface> contract);
    // and_then chains straight into TryEnqueue without introducing a second
    // error code for the same "not found" condition.
    return worker_pools_.Resolve(stage_id).and_then([handle](const std::shared_ptr<WorkerPool>& pool) {
        return pool->TryEnqueue(handle);
    });
}

}  // namespace sai::runtime
