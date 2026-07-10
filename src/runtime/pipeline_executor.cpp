#include <sai/runtime/pipeline_executor.h>

#include <condition_variable>
#include <coroutine>
#include <memory>
#include <mutex>
#include <source_location>
#include <utility>

#include <sai/runtime/task_scheduler.h>

namespace sai::runtime {
namespace {

// Cross-thread rendezvous between Dispatch (the waiter, running on whatever
// thread drives TaskGraph::RunToCompletion) and the WorkerPool thread that
// actually runs the node's work coroutine. Heap-allocated and held by
// shared_ptr so it outlives both sides regardless of who finishes first.
struct CompletionState {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;
    Result<void> result;  // Result<void> default-constructs to success.

    void Set(Result<void> value) noexcept {
        {
            std::unique_lock lock(mutex);
            result = std::move(value);
            ready = true;
        }
        cv.notify_one();
    }

    [[nodiscard]] auto Wait() noexcept -> Result<void> {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return ready; });
        return std::move(result);
    }
};

// Fire-and-forget coroutine type used only as the payload submitted to the
// WorkerPool. Its final_suspend is suspend_never, so the frame destroys itself
// on the pool thread once RunAndSignal reaches co_return — nothing else ever
// needs to (or is allowed to) destroy it, which is exactly what lets the waiter
// walk away after Set() without touching this frame from a second thread.
struct DetachedTask {
    struct promise_type {
        [[nodiscard]] auto get_return_object() noexcept -> DetachedTask {
            return DetachedTask{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        [[nodiscard]] auto initial_suspend() noexcept -> std::suspend_always { return {}; }
        [[nodiscard]] auto final_suspend() noexcept -> std::suspend_never { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;
};

// Runs the node's work coroutine to completion on the pool thread, takes and
// destroys it (manual-destroy discipline, §11), then hands the result to the
// waiting Dispatch via `state`. Everything this coroutine touches after the
// Set() signal is frame-local (the compiler-generated final_suspend
// transition), so the waiter never races this frame.
DetachedTask RunAndSignal(Task<void> work, std::shared_ptr<CompletionState> state) {
    work.resume();
    Result<void> result = work.promise().GetResult();
    work.destroy();
    state->Set(std::move(result));
    co_return;
}

}  // namespace

PipelineExecutor::PipelineExecutor(TaskScheduler& scheduler) noexcept : scheduler_(scheduler) {}

auto PipelineExecutor::Dispatch(TypeId stage_id, Task<void> work,
                                std::stop_token stop_token) noexcept -> Task<void> {
    // Checkpoint before submission: an already-requested stop means the work
    // coroutine is never resumed, so destroy it here and short-circuit.
    if (stop_token.stop_requested()) {
        work.destroy();
        co_return tl::make_unexpected(ErrorInfo{
            ErrorCode::Runtime_Cancelled,
            "dispatch cancelled before submission",
            std::source_location::current(),
        });
    }

    auto state = std::make_shared<CompletionState>();
    std::coroutine_handle<> wrapper = RunAndSignal(work, state).handle;

    Result<void> submitted = scheduler_.Submit(stage_id, wrapper);
    if (!submitted) {
        // Submit failed (Core_TypeNotFound / Runtime_QueueFull): the wrapper was
        // never resumed, so it neither ran nor destroyed `work`. Tear both down
        // here — destroying the never-run wrapper frame does not touch the
        // captured `work` handle, so `work` must be destroyed separately.
        wrapper.destroy();
        work.destroy();
        co_return Result<void>(tl::unexpect, submitted.error());
    }

    // Submitted: a pool thread will run the wrapper, which resumes `work`,
    // destroys it, and signals `state`. Block until that signal arrives.
    co_return state->Wait();
}

}  // namespace sai::runtime
