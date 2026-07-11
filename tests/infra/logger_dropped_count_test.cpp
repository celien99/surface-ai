#include <sai/infra/logger.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <sai/core/error.h>

namespace {

using sai::infra::Logger;
using sai::infra::LogLevel;

auto MakeTempDir() -> std::filesystem::path {
    return std::filesystem::temp_directory_path() / "sai_logger_dropped_count_test";
}

TEST(LoggerDroppedCount, PerCategoryAttribution) {
    ASSERT_TRUE(Logger::InitializeGlobalSinks(MakeTempDir()).has_value());
    auto& a = Logger::Get("A");
    a.SetLevel(LogLevel::Trace);
    for (int i = 0; i < 200'000; ++i) a.Log(LogLevel::Debug, "flood {}", i);
    EXPECT_GT(Logger::DroppedCount("A"), 0U);
    EXPECT_EQ(Logger::DroppedCount("B"), 0U);
}

}  // namespace
