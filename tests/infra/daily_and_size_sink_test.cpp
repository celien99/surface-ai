#include <sai/infra/daily_and_size_sink.h>

#include <atomic>
#include <ctime>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include <gtest/gtest.h>

#include <spdlog/details/os.h>
#include <spdlog/logger.h>

namespace {

using sai::infra::DailyAndSizeRotatingFileSink;

// Each test gets a private temp dir so rotated-file assertions never collide
// with a sibling test's files.
auto MakeTempDir() -> std::filesystem::path {
    static std::atomic<int> counter{0};
    auto dir = std::filesystem::temp_directory_path() /
               ("sai_daily_size_sink_" + std::to_string(counter.fetch_add(1)));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

// Overrides the NowTm() seam so a test can pretend the calendar day advanced
// without waiting for real wall-clock time to cross midnight.
class TestableSink final : public DailyAndSizeRotatingFileSink<std::mutex> {
public:
    TestableSink(std::string base_filename, std::size_t max_size, std::size_t max_files)
        : DailyAndSizeRotatingFileSink<std::mutex>(std::move(base_filename), max_size,
                                                   max_files),
          now_tm_(spdlog::details::os::localtime()) {}

    // Advance the simulated clock by `days`. tm_yday/tm_year are what the sink
    // compares against, so bumping mday and letting NowTm expose the raw tm is
    // enough — but we normalize through timegm/localtime to keep tm_yday honest.
    void AdvanceDays(int days) {
        now_tm_.tm_mday += days;
        std::time_t as_time = std::mktime(&now_tm_);
        now_tm_ = spdlog::details::os::localtime(as_time);
    }

protected:
    auto NowTm() const noexcept -> std::tm override { return now_tm_; }

private:
    std::tm now_tm_;
};

TEST(DailyAndSizeSink, RotatesWhenSizeExceeded) {
    auto dir = MakeTempDir();
    auto sink = std::make_shared<TestableSink>((dir / "app.log").string(),
                                               /*max_size=*/64, /*max_files=*/3);
    spdlog::logger log("t", sink);
    for (int i = 0; i < 50; ++i) {
        log.info("0123456789");  // each formatted line >> 64 bytes total quickly
    }
    log.flush();
    EXPECT_TRUE(std::filesystem::exists(dir / "app.1.log"))
        << "exceeding max_size must produce a rotated file";
}

TEST(DailyAndSizeSink, RotatesOnDayChangeUnderSizeCap) {
    auto dir = MakeTempDir();
    auto sink = std::make_shared<TestableSink>((dir / "app.log").string(),
                                               /*max_size=*/1'000'000, /*max_files=*/3);
    spdlog::logger log("t", sink);
    log.info("day1");
    log.flush();
    EXPECT_FALSE(std::filesystem::exists(dir / "app.1.log"))
        << "under the size cap and same day, no rotation should have happened yet";

    sink->AdvanceDays(1);  // simulate tomorrow
    log.info("day2");
    log.flush();
    EXPECT_TRUE(std::filesystem::exists(dir / "app.1.log"))
        << "a day change must rotate even when far under max_size";
}

TEST(DailyAndSizeSink, MaxFilesCapDeletesOldest) {
    auto dir = MakeTempDir();
    auto sink = std::make_shared<TestableSink>((dir / "app.log").string(),
                                               /*max_size=*/64, /*max_files=*/3);
    spdlog::logger log("t", sink);
    for (int i = 0; i < 500; ++i) {
        log.info("0123456789");  // force many size rotations
    }
    log.flush();

    EXPECT_TRUE(std::filesystem::exists(dir / "app.3.log"))
        << "max_files=3 must retain app.3.log";
    EXPECT_FALSE(std::filesystem::exists(dir / "app.4.log"))
        << "max_files=3 must never keep a 4th rotated file";
}

}  // namespace
