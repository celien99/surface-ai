#pragma once

#include <cstddef>
#include <ctime>
#include <mutex>
#include <string>
#include <utility>

#include <spdlog/common.h>
#include <spdlog/details/file_helper.h>
#include <spdlog/details/log_msg.h>
#include <spdlog/details/os.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/base_sink.h>

namespace sai::infra {

// File sink that rotates on EITHER of two conditions, whichever comes first:
//   * the active file would exceed max_size bytes, or
//   * the local calendar day changes.
// Milestone 1 shipped a size-only rotating_file_sink; design mandate D3 requires
// "daily OR 100MB". Rather than stack spdlog's size-only and date-only sinks
// (which would keep two independent file sets), this single sink owns one
// file_helper and applies both triggers before every write. Filename layout and
// the N-file rename cascade mirror spdlog::sinks::rotating_file_sink so ops
// tooling that already understands base.log / base.1.log keeps working.
//
// NOTE: not marked `final` (the brief's interface line shows `final`) because the
// sanctioned test seam NowTm() requires a subclass to override it. See the task
// report's deviation log. Everything except NowTm() is otherwise closed.
template <typename Mutex>
class DailyAndSizeRotatingFileSink : public spdlog::sinks::base_sink<Mutex> {
public:
    DailyAndSizeRotatingFileSink(spdlog::filename_t base_filename, std::size_t max_size,
                                 std::size_t max_files)
        : base_filename_(std::move(base_filename)),
          max_size_(max_size),
          max_files_(max_files) {
        file_helper_.open(base_filename_);
        current_size_ = file_helper_.size();
        const std::tm now = NowTm();
        current_day_ = now.tm_yday;
        current_year_ = now.tm_year;
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);

        const std::tm now = NowTm();
        const bool day_changed =
            now.tm_yday != current_day_ || now.tm_year != current_year_;
        const bool size_exceeded = current_size_ + formatted.size() > max_size_;
        if (day_changed || size_exceeded) {
            Rotate_();
            current_day_ = now.tm_yday;
            current_year_ = now.tm_year;
            current_size_ = 0;
        }
        file_helper_.write(formatted);
        current_size_ += formatted.size();
    }

    void flush_() override { file_helper_.flush(); }

    // Test seam: production returns the real local time; a test subclass can
    // override this to simulate a day change without waiting for midnight.
    [[nodiscard]] virtual auto NowTm() const noexcept -> std::tm {
        return spdlog::details::os::localtime();
    }

private:
    // Rotate cascade (identical shape to rotating_file_sink):
    //   base.(max_files-1).log -> base.max_files.log (oldest overwritten/dropped)
    //   ...
    //   base.log -> base.1.log
    // then reopen a fresh, truncated base.log.
    void Rotate_() {
        file_helper_.close();
        for (auto i = max_files_; i > 0; --i) {
            const spdlog::filename_t src = CalcFilename(base_filename_, i - 1);
            if (!spdlog::details::os::path_exists(src)) {
                continue;
            }
            const spdlog::filename_t target = CalcFilename(base_filename_, i);
            (void)spdlog::details::os::remove(target);
            (void)spdlog::details::os::rename(src, target);
        }
        file_helper_.reopen(true);
    }

    // "logs/app.log", 0 -> "logs/app.log"; "logs/app.log", 3 -> "logs/app.3.log".
    [[nodiscard]] static auto CalcFilename(const spdlog::filename_t& filename,
                                           std::size_t index) -> spdlog::filename_t {
        if (index == 0U) {
            return filename;
        }
        auto [basename, ext] =
            spdlog::details::file_helper::split_by_extension(filename);
        return fmt::format(SPDLOG_FMT_STRING(SPDLOG_FILENAME_T("{}.{}{}")), basename,
                           index, ext);
    }

    spdlog::filename_t base_filename_;
    std::size_t max_size_;
    std::size_t max_files_;
    std::size_t current_size_ = 0;
    int current_day_ = 0;
    int current_year_ = 0;
    spdlog::details::file_helper file_helper_;
};

using DailyAndSizeRotatingFileSink_mt = DailyAndSizeRotatingFileSink<std::mutex>;

}  // namespace sai::infra
