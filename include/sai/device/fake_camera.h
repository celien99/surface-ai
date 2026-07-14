#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <sai/device/camera.h>
#include <sai/image/raw_image.h>

namespace sai::device {

class FakeCamera : public ICamera {
public:
    struct Config {
        std::size_t width = 1024;
        std::size_t height = 1024;
        double fps = 10.0;
        sai::image::PixelFormat pixel_format = sai::image::PixelFormat::Mono8;
    };

    explicit FakeCamera(Config cfg);
    ~FakeCamera() override;

    SAI_DECLARE_TYPE_ID(sai::device::FakeCamera)

    [[nodiscard]] auto Connect() noexcept -> Result<void> override;
    [[nodiscard]] auto Disconnect() noexcept -> Result<void> override;
    [[nodiscard]] auto IsConnected() const noexcept -> bool override;
    [[nodiscard]] auto SerialNumber() const noexcept -> std::string_view override;
    [[nodiscard]] auto CurrentState() const noexcept -> State override;

    [[nodiscard]] auto SetTriggerMode(TriggerMode mode) noexcept -> Result<void> override;
    [[nodiscard]] auto StartAcquisition() noexcept -> Result<void> override;
    [[nodiscard]] auto StopAcquisition() noexcept -> Result<void> override;
    [[nodiscard]] auto RegisterFrameCallback(FrameCallback callback) noexcept -> Result<void> override;
    [[nodiscard]] auto SetExposureTime(std::chrono::microseconds us) noexcept -> Result<void> override;
    [[nodiscard]] auto SetGain(float db) noexcept -> Result<void> override;
    [[nodiscard]] auto SetROI(Rect region) noexcept -> Result<void> override;

    // Manifest (IPlugin overrides)
    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override;
    [[nodiscard]] auto OnInitialize(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStart(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStop(Context&) -> Result<void> override { return {}; }

private:
    auto GenerateFrame() -> sai::image::RawImage;

    Config cfg_;
    std::string serial_{"FAKE-001"};
    std::atomic<State> state_{State::Disconnected};
    FrameCallback callback_;
    std::jthread acquisition_thread_;
    std::atomic<bool> acquiring_{false};
    TriggerMode trigger_mode_{TriggerMode::FreeRun};
    PluginManifest manifest_{};
};

}  // namespace sai::device
