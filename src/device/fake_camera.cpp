#include <sai/device/fake_camera.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#include <sai/core/error.h>
#include <sai/image/image.h>

namespace sai::device {

FakeCamera::FakeCamera(Config cfg) : cfg_(std::move(cfg)) {}

FakeCamera::~FakeCamera() {
    if (acquiring_.load()) {
        StopAcquisition();
    }
    if (state_.load() == State::Connected || state_.load() == State::Acquiring) {
        Disconnect();
    }
}

auto FakeCamera::Connect() noexcept -> Result<void> {
    if (state_.load() == State::Connected) return {};
    state_.store(State::Connected);
    return {};
}

auto FakeCamera::Disconnect() noexcept -> Result<void> {
    StopAcquisition();
    state_.store(State::Disconnected);
    return {};
}

auto FakeCamera::IsConnected() const noexcept -> bool {
    return state_.load() == State::Connected || state_.load() == State::Acquiring;
}

auto FakeCamera::SerialNumber() const noexcept -> std::string_view {
    return serial_;
}

auto FakeCamera::CurrentState() const noexcept -> State {
    return state_.load();
}

auto FakeCamera::SetTriggerMode(TriggerMode mode) noexcept -> Result<void> {
    trigger_mode_ = mode;
    return {};
}

auto FakeCamera::StartAcquisition() noexcept -> Result<void> {
    if (state_.load() != State::Connected) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            "FakeCamera: not connected"});
    }
    if (acquiring_.exchange(true)) return {};
    state_.store(State::Acquiring);

    acquisition_thread_ = std::jthread([this](std::stop_token st) {
        using Clock = std::chrono::steady_clock;
        auto frame_interval = std::chrono::microseconds(
            static_cast<long long>(1'000'000.0 / cfg_.fps));
        auto next_frame = Clock::now();

        while (!st.stop_requested()) {
            std::this_thread::sleep_until(next_frame);
            if (st.stop_requested()) break;

            auto frame = GenerateFrame();
            if (callback_) {
                callback_(std::move(frame));
            }
            next_frame += frame_interval;
            // Prevent drift accumulation
            if (Clock::now() > next_frame) {
                next_frame = Clock::now() + frame_interval;
            }
        }
    });
    return {};
}

auto FakeCamera::StopAcquisition() noexcept -> Result<void> {
    if (!acquiring_.exchange(false)) return {};
    acquisition_thread_.request_stop();
    if (acquisition_thread_.joinable()) {
        acquisition_thread_.join();
    }
    if (state_.load() == State::Acquiring) {
        state_.store(State::Connected);
    }
    return {};
}

auto FakeCamera::RegisterFrameCallback(FrameCallback callback) noexcept -> Result<void> {
    callback_ = std::move(callback);
    return {};
}

auto FakeCamera::SetExposureTime(std::chrono::microseconds) noexcept -> Result<void> {
    return {};
}

auto FakeCamera::SetGain(float) noexcept -> Result<void> {
    return {};
}

auto FakeCamera::SetROI(Rect) noexcept -> Result<void> {
    return {};
}

auto FakeCamera::GetManifest() const noexcept -> const PluginManifest& {
    return manifest_;
}

auto FakeCamera::GenerateFrame() -> sai::image::RawImage {
    sai::image::ImageMeta meta;
    meta.width = cfg_.width;
    meta.height = cfg_.height;
    meta.channels = 1;
    meta.pixel_format = cfg_.pixel_format;

    std::vector<std::uint8_t> buffer(cfg_.width * cfg_.height, 0x80);

    // Horizontal sine wave stripes (period=128, amplitude=24)
    const double period = 128.0;
    const double amplitude = 24.0;
    for (std::size_t y = 0; y < cfg_.height; ++y) {
        double sine = std::sin(2.0 * M_PI * static_cast<double>(y) / period);
        std::uint8_t stripe = static_cast<std::uint8_t>(128.0 + amplitude * sine);
        for (std::size_t x = 0; x < cfg_.width; ++x) {
            buffer[y * cfg_.width + x] = stripe;
        }
    }

    // Random dark spot defects (2-3 spots, radius 15-35)
    static thread_local std::mt19937 rng(
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<std::size_t> cx_dist(40, cfg_.width - 40);
    std::uniform_int_distribution<std::size_t> cy_dist(40, cfg_.height - 40);
    std::uniform_int_distribution<int> r_dist(15, 35);
    std::uniform_int_distribution<int> count_dist(2, 3);

    int num_spots = count_dist(rng);
    for (int i = 0; i < num_spots; ++i) {
        std::size_t cx = cx_dist(rng);
        std::size_t cy = cy_dist(rng);
        int radius = r_dist(rng);
        int r2 = radius * radius;

        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy > r2) continue;
                std::size_t x = static_cast<std::size_t>(
                    static_cast<int>(cx) + dx);
                std::size_t y = static_cast<std::size_t>(
                    static_cast<int>(cy) + dy);
                if (x >= cfg_.width || y >= cfg_.height) continue;
                // Darken the spot
                std::uint8_t& pixel = buffer[y * cfg_.width + x];
                double dist = std::sqrt(static_cast<double>(dx * dx + dy * dy));
                double factor = 1.0 - (1.0 - dist / radius) * 0.6;
                pixel = static_cast<std::uint8_t>(static_cast<double>(pixel) * factor);
            }
        }
    }

    return sai::image::RawImage::FromOwnedBuffer(std::move(buffer), meta);
}

}  // namespace sai::device
