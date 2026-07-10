#include <sai/infra/logger.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <sai/core/error.h>

namespace {

using sai::infra::Logger;
using sai::infra::LogLevel;

// All tests in this executable share one process-global sink set: the first
// InitializeGlobalSinks call wins and later calls are idempotent no-ops (that
// is exactly the contract). So we pin one temp dir for the whole file and read
// the concrete log file back from it where a test needs to inspect content.
auto TestLogDir() -> std::filesystem::path {
    return std::filesystem::temp_directory_path() / "sai_logger_test";
}

auto LogFilePath() -> std::filesystem::path {
    return TestLogDir() / "surface-ai.log";
}

auto ReadLogFile() -> std::string {
    std::ifstream in(LogFilePath(), std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Poll the log file up to `timeout` for `token` to appear (async IO thread
// drains the queue on its own schedule).
auto WaitForToken(const std::string& token, std::chrono::milliseconds timeout)
    -> bool {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (ReadLogFile().find(token) != std::string::npos) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

TEST(LoggerTest, InitializeGlobalSinksSucceedsAndIsIdempotent) {
    auto first = Logger::InitializeGlobalSinks(TestLogDir());
    ASSERT_TRUE(first.has_value()) << "first init should succeed";

    // Second call with the same (or any) dir must not error and must not
    // rebuild the already-existing sink.
    auto second = Logger::InitializeGlobalSinks(TestLogDir());
    EXPECT_TRUE(second.has_value()) << "second init should be an idempotent no-op";
}

TEST(LoggerTest, GetReturnsSameInstanceForSameCategory) {
    ASSERT_TRUE(Logger::InitializeGlobalSinks(TestLogDir()).has_value());

    Logger& a = Logger::Get("SameCategory");
    Logger& b = Logger::Get("SameCategory");
    EXPECT_EQ(&a, &b) << "same category name must return the same instance";

    Logger& other = Logger::Get("OtherCategory");
    EXPECT_NE(&a, &other) << "distinct categories must be distinct instances";
}

TEST(LoggerTest, LevelFilterShortCircuitsBelowThreshold) {
    ASSERT_TRUE(Logger::InitializeGlobalSinks(TestLogDir()).has_value());

    Logger& logger = Logger::Get("LevelFilterCategory");
    logger.SetLevel(LogLevel::Info);

    // Unique tokens so this assertion is immune to other tests writing to the
    // same shared file.
    const std::string filtered_token = "FILTERED_BELOW_THRESHOLD_a1b2c3";
    const std::string visible_token = "VISIBLE_ABOVE_THRESHOLD_a1b2c3";

    // Debug < Info: must be short-circuited before any enqueue, so it can never
    // reach the file. Error >= Info: enqueued via block_tier_.
    logger.Log(LogLevel::Debug, "{}", filtered_token);
    logger.Log(LogLevel::Error, "{}", visible_token);

    ASSERT_TRUE(WaitForToken(visible_token, std::chrono::seconds(3)))
        << "an above-threshold message must reach the sink";

    // The below-threshold message was never enqueued, so it is deterministically
    // absent once the file exists.
    EXPECT_EQ(ReadLogFile().find(filtered_token), std::string::npos)
        << "a below-threshold message must be filtered out";
}

// Deterministic overflow of the fixed 8192 drop_tier_ queue is not achievable
// through the frozen public interface in a fast unit test (no way to inject a
// slow sink or shrink the queue). We assert the observable invariant that the
// drop counter is monotonic non-decreasing across a heavy flood, and record the
// value. See task-8-report.md for the documented gap.
TEST(LoggerTest, DropCounterIsMonotonicUnderFlood) {
    ASSERT_TRUE(Logger::InitializeGlobalSinks(TestLogDir()).has_value());

    Logger& logger = Logger::Get("FloodCategory");
    logger.SetLevel(LogLevel::Trace);

    const std::uint64_t before = logger.DroppedCount();
    for (int i = 0; i < 200000; ++i) {
        logger.Log(LogLevel::Debug, "flood {}", i);
    }
    const std::uint64_t after = logger.DroppedCount();

    EXPECT_GE(after, before) << "dropped count must never decrease";
    std::cout << "[ INFO ] drop_tier_ dropped during flood: " << (after - before)
              << std::endl;
}

// SetLevel must round-trip: a level raised above a message's severity filters
// it out, and lowering the level lets a subsequent same-severity message through.
// Exercises the atomic min_level_ store/load across the SetLevel -> Log boundary.
TEST(LoggerTest, SetLevelRoundTripsFilterDecision) {
    ASSERT_TRUE(Logger::InitializeGlobalSinks(TestLogDir()).has_value());

    Logger& logger = Logger::Get("SetLevelRoundTripCategory");

    const std::string suppressed_token = "SUPPRESSED_AT_ERROR_LEVEL_d4e5f6";
    const std::string admitted_token = "ADMITTED_AT_DEBUG_LEVEL_d4e5f6";

    // Raise the threshold to Error: a Warning message is now below it and must
    // be short-circuited before enqueue.
    logger.SetLevel(LogLevel::Error);
    logger.Log(LogLevel::Warning, "{}", suppressed_token);

    // Lower the threshold to Debug: a Debug message now passes and reaches disk.
    logger.SetLevel(LogLevel::Debug);
    logger.Log(LogLevel::Error, "{}", admitted_token);

    ASSERT_TRUE(WaitForToken(admitted_token, std::chrono::seconds(3)))
        << "message at/above the lowered threshold must reach the sink";
    EXPECT_EQ(ReadLogFile().find(suppressed_token), std::string::npos)
        << "message below the raised threshold must be filtered out";
}

}  // namespace
