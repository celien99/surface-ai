#include <sai/infra/logger.h>

#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <spdlog/async.h>
#include <spdlog/details/thread_pool.h>

#include <sai/infra/daily_and_size_sink.h>

namespace sai::infra {

namespace {

// Process-global shared sink set + init guard. All tiers of every category
// share this sink set; the block tier shares one spdlog::thread_pool initialized
// below, while each category's drop tier gets its own per-category pool (D2).
struct GlobalState {
    std::mutex init_mutex;
    bool initialized = false;
    std::vector<spdlog::sink_ptr> sinks;
};

auto Global() noexcept -> GlobalState& {
    static GlobalState state;
    return state;
}

// Category name -> Logger lookup table. shared_mutex read/write split mirrors
// 1.1's TypeRegistry: reads take shared_lock, first-time inserts take
// unique_lock. unique_ptr gives each Logger a stable address so Get can hand
// back a reference.
struct CategoryTable {
    std::shared_mutex mutex;
    std::unordered_map<std::string, std::unique_ptr<Logger>> entries;
};

auto Categories() noexcept -> CategoryTable& {
    static CategoryTable table;
    return table;
}

auto MakeTier(std::string name,
              std::shared_ptr<spdlog::details::thread_pool> pool,
              spdlog::async_overflow_policy policy)
    -> std::shared_ptr<spdlog::async_logger> {
    auto& global = Global();
    auto logger = std::make_shared<spdlog::async_logger>(
        std::move(name), global.sinks.begin(), global.sinks.end(),
        std::move(pool), policy);
    // Level gating is owned by Logger::min_level_; let spdlog pass everything
    // through so the two policies are the only routing spdlog performs.
    logger->set_level(spdlog::level::trace);
    return logger;
}

constexpr std::size_t kQueueCapacity = 8192;
constexpr std::size_t kBackgroundThreads = 1;
constexpr std::size_t kMaxFileSize = 100 * 1024 * 1024;  // 100MB rotation cap
constexpr std::size_t kMaxFiles = 10;

}  // namespace

Logger::Logger(std::string category,
               std::shared_ptr<spdlog::async_logger> block_tier,
               std::shared_ptr<spdlog::async_logger> drop_tier,
               std::shared_ptr<spdlog::details::thread_pool> drop_pool) noexcept
    : category_(std::move(category)),
      block_tier_(std::move(block_tier)),
      drop_tier_(std::move(drop_tier)),
      drop_pool_(std::move(drop_pool)) {}

auto Logger::InitializeGlobalSinks(std::filesystem::path log_dir) -> Result<void> {
    auto& global = Global();
    std::lock_guard lock(global.init_mutex);
    if (global.initialized) {
        return {};  // idempotent: do not rebuild an existing sink
    }
    try {
        std::filesystem::create_directories(log_dir);
        spdlog::init_thread_pool(kQueueCapacity, kBackgroundThreads);
        auto file = (log_dir / "surface-ai.log").string();
        global.sinks.push_back(std::make_shared<DailyAndSizeRotatingFileSink_mt>(
            file, kMaxFileSize, kMaxFiles));
        global.initialized = true;
        return {};
    } catch (const std::exception& ex) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Infra_LogSinkInitFailed,
            std::string("log sink init failed: ") + ex.what(),
            std::source_location::current(),
        });
    }
}

auto Logger::Get(std::string_view category) -> Logger& {
    auto& table = Categories();
    std::string key(category);
    {
        std::shared_lock lock(table.mutex);
        if (auto it = table.entries.find(key); it != table.entries.end()) {
            return *it->second;
        }
    }
    std::unique_lock lock(table.mutex);
    if (auto it = table.entries.find(key); it != table.entries.end()) {
        return *it->second;
    }
    auto block_pool = spdlog::thread_pool();  // shared pool for block tier
    auto drop_pool = std::make_shared<spdlog::details::thread_pool>(
        kQueueCapacity, kBackgroundThreads);
    auto block = MakeTier(key + ":block", block_pool,
                          spdlog::async_overflow_policy::block);
    auto drop = MakeTier(key + ":drop", drop_pool,
                         spdlog::async_overflow_policy::overrun_oldest);
    // Warning+ is low-frequency and must survive a crash: flush it to disk on
    // every message. Trace/Debug stays buffered (high-frequency, loss-tolerant).
    block->flush_on(spdlog::level::warn);
    auto* raw = new Logger(key, std::move(block), std::move(drop),
                           std::move(drop_pool));
    table.entries.emplace(std::move(key), std::unique_ptr<Logger>(raw));
    return *raw;
}

void Logger::SetLevel(LogLevel level) noexcept {
    min_level_.store(level, std::memory_order_relaxed);
}

auto Logger::DroppedCount() const noexcept -> std::uint64_t {
    // Each Logger instance owns its drop tier's thread_pool, so overrun_counter()
    // is genuinely per-category. (Milestone-1 used a single shared pool, which
    // made the counter process-wide — debt D2.)
    if (drop_pool_) {
        return static_cast<std::uint64_t>(drop_pool_->overrun_counter());
    }
    return 0;
}

auto Logger::DroppedCount(std::string_view category) -> std::uint64_t {
    return Get(category).DroppedCount();
}

}  // namespace sai::infra
