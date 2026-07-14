#pragma once

// -----------------------------------------------------------------------
// <sai/runtime/task.h>  (1.4-runtime.md §4)
// -----------------------------------------------------------------------
//
// Task<T> is the framework's universal async return type: a bare
// std::coroutine_handle<TaskPromise<T>>. final_suspend() returns
// suspend_always, so the coroutine frame is NEVER destroyed automatically.
// Whoever holds a Task<T> handle must, after confirming handle.done() and
// taking the result via TaskPromise<T>::GetResult(), call handle.destroy()
// exactly once. This header does not provide an RAII wrapper for that —
// see 1.4-runtime.md §3/§11 for why a manual-destroy contract was chosen
// over wrapping Task<T> in a second, owning type.

#include <coroutine>
#include <optional>
#include <stop_token>
#include <type_traits>
#include <utility>

#include <sai/core/error.h>

namespace sai::runtime {

template <typename T>
class TaskPromise;

template <typename T>
using Task = std::coroutine_handle<TaskPromise<T>>;

template <typename T>
class TaskPromise {
public:
    // No-argument coroutines (e.g. TaskNode::work()'s std::function<Task<void>()>)
    // use this constructor; TaskPromise's default stop_token_ is then the
    // default-constructed std::stop_token (never-stop-requested, no
    // associated stop_source).
    TaskPromise() noexcept = default;

    // Promise-type constructor matching the coroutine function's own
    // parameter list: this is standard C++20 promise-type machinery for
    // adopting a std::stop_token passed to the coroutine at its call site
    // (see 3. Design's Cancellation section) into GetStopToken(). Only the
    // argument that is actually a std::stop_token is adopted; every other
    // parameter type is ignored here (coroutine bodies read their own
    // parameters normally, this constructor exists solely to intercept the
    // cancellation token).
    template <typename... Args>
    explicit TaskPromise(const Args&... args) noexcept {
        (AdoptIfStopToken(args), ...);
    }

    // co_return value; where value is a Result<T> — this is the only
    // return channel this promise type supports (no bare co_return;), see
    // 5. Workflow's EnsureCompleted/ResolveDependencies examples.
    void return_value(Result<T> value) noexcept { result_ = std::move(value); }

    // Coroutine bodies in this framework never let an exception escape
    // uncaught (see 1.1 batch §3 Design); this method only exists to
    // satisfy the promise-type protocol.
    void unhandled_exception() noexcept { std::terminate(); }

    [[nodiscard]] auto GetStopToken() const noexcept -> std::stop_token { return stop_token_; }

    // Not part of §4's frozen declaration list, but required for any
    // consumer of Task<T> to actually take the Result<T> out of a completed
    // coroutine frame before calling handle.destroy() (see §11 Memory's
    // manual-destroy contract: "已经取走结果之后手动调用 handle.destroy()").
    // Moves the stored result out; only valid to call once handle.done() is
    // true.
    [[nodiscard]] auto GetResult() noexcept -> Result<T> { return std::move(*result_); }

    [[nodiscard]] auto get_return_object() noexcept -> Task<T> {
        return Task<T>::from_promise(*this);
    }

    [[nodiscard]] auto initial_suspend() noexcept -> std::suspend_always { return {}; }
    [[nodiscard]] auto final_suspend() noexcept -> std::suspend_always { return {}; }

    // User-provided operator new/delete disable heap-allocation elision
    // (C++20 [dcl.fct.def.coroutine] p11).  Without these, compilers may
    // place the coroutine frame on the caller's stack when the handle
    // appears to not escape — which is incorrect when the handle is
    // submitted to a WorkerPool running on a different thread.
    static auto operator new(std::size_t size) -> void* {
        return ::operator new(size);
    }
    static auto operator delete(void* ptr, std::size_t size) -> void {
        ::operator delete(ptr, size);
    }

private:
    template <typename U>
    void AdoptIfStopToken(const U& value) noexcept {
        if constexpr (std::is_same_v<std::decay_t<U>, std::stop_token>) {
            stop_token_ = value;
        }
    }

    std::optional<Result<T>> result_;
    std::stop_token stop_token_;
};

}  // namespace sai::runtime

// Task<T> is a bare std::coroutine_handle<TaskPromise<T>> alias (§4), which
// has no nested ::promise_type of its own — std::coroutine_traits's primary
// template only picks up a promise type from R::promise_type. Functions
// declared to return Task<T> directly and use co_await/co_return (as §5
// Workflow's EnsureCompleted/ResolveDependencies pseudocode does) therefore
// need this specialization to resolve to TaskPromise<T>; without it, no
// function can actually be a coroutine returning the bare Task<T> alias as
// specified.
namespace std {
template <typename T, typename... Args>
struct coroutine_traits<::sai::runtime::Task<T>, Args...> {
    using promise_type = ::sai::runtime::TaskPromise<T>;
};
}  // namespace std
