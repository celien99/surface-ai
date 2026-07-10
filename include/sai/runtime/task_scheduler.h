#pragma once

// -----------------------------------------------------------------------
// <sai/runtime/task_scheduler.h>  (1.4-runtime.md §4)
// -----------------------------------------------------------------------

#include <coroutine>

#include <sai/core/error.h>
#include <sai/core/registry.h>
#include <sai/core/type_id.h>
#include <sai/runtime/worker_pool.h>

namespace sai::runtime {

// External task submission entry point; holds a reference to a
// Registry<WorkerPool> (1.2's Registry<TInterface> instantiated for
// WorkerPool) keyed by stage TypeId, so Submit is a lookup, not a fresh
// per-call construction path.
class TaskScheduler final {
public:
    explicit TaskScheduler(Registry<WorkerPool>& worker_pools) noexcept;

    // stage_id: TypeId of the target stage (Capture/Inference/Retrieval/
    // Reason/IO etc.), obtained by the caller via 1.1's Fnv1aHash over the
    // stage name. Propagates Registry<WorkerPool>::Resolve's error verbatim
    // when stage_id is not registered (Core_TypeNotFound) — this class does
    // not invent its own "stage not found" code. Returns Runtime_QueueFull
    // when the resolved WorkerPool's queue is full (§3 Design); does not
    // retry or block either way.
    [[nodiscard]] auto Submit(TypeId stage_id, std::coroutine_handle<> handle) noexcept
        -> Result<void>;

private:
    Registry<WorkerPool>& worker_pools_;
};

}  // namespace sai::runtime
