#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <spdlog/async_logger.h>
#include <spdlog/common.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <sai/core/error.h>

namespace sai::infra {

enum class LogLevel : std::uint8_t {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical,
};

namespace detail {

[[nodiscard]] constexpr auto ToSpdlogLevel(LogLevel level) noexcept
    -> spdlog::level::level_enum {
    switch (level) {
        case LogLevel::Trace:
            return spdlog::level::trace;
        case LogLevel::Debug:
            return spdlog::level::debug;
        case LogLevel::Info:
            return spdlog::level::info;
        case LogLevel::Warning:
            return spdlog::level::warn;
        case LogLevel::Error:
            return spdlog::level::err;
        case LogLevel::Critical:
            return spdlog::level::critical;
    }
    return spdlog::level::info;
}

}  // namespace detail

// 按类别（category）持有的日志入口；同一类别名重复调用 Logger::Get 返回同一实例，
// 不重复创建底层 spdlog logger 或后台线程。类别名通常对应模块名，用于运维排障时
// 按类别单独调整日志级别。
class Logger final {
public:
    [[nodiscard]] static auto Get(std::string_view category) -> Logger&;

    [[nodiscard]] static auto InitializeGlobalSinks(std::filesystem::path log_dir)
        -> Result<void>;

    void SetLevel(LogLevel level) noexcept;

    // 级别过滤先行（低于 min_level_ 直接返回，不发生格式化与入队）；达到 Warning
    // 的条目路由到 block_tier_（overflow_policy::block，绝不丢弃），其余路由到
    // drop_tier_（overflow_policy::overrun_oldest，队列满时丢弃最旧条目）。
    template <typename... Args>
    void Log(LogLevel level, fmt::format_string<Args...> fmt_str,
             Args&&... args) noexcept {
        if (level < min_level_.load(std::memory_order_relaxed)) {
            return;
        }
        const auto& tier = (level >= LogLevel::Warning) ? block_tier_ : drop_tier_;
        if (!tier) {
            return;
        }
        tier->log(detail::ToSpdlogLevel(level), fmt_str, std::forward<Args>(args)...);
    }

    [[nodiscard]] auto DroppedCount() const noexcept -> std::uint64_t;

private:
    Logger(std::string category,
           std::shared_ptr<spdlog::async_logger> block_tier,
           std::shared_ptr<spdlog::async_logger> drop_tier) noexcept;

    std::string category_;
    // Read on the Log() hot path, written by SetLevel() from another thread;
    // atomic makes concurrent level changes a well-defined operation, not a race.
    std::atomic<LogLevel> min_level_{LogLevel::Info};
    std::shared_ptr<spdlog::async_logger> block_tier_;  // Warning 及以上，block
    std::shared_ptr<spdlog::async_logger> drop_tier_;   // Trace/Debug，overrun_oldest
    std::atomic<std::uint64_t> dropped_count_{0};
};

}  // namespace sai::infra
