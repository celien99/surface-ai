#pragma once

#include <sai/device/camera.h>

#include <string>

namespace sai::test {

class FakeCamera final : public device::ICamera {
public:
    SAI_DECLARE_TYPE_ID(sai.test.fake-camera)

    FakeCamera();
    ~FakeCamera() override = default;

    // --- IModule ---
    auto OnInitialize(Context& context) -> Result<void> override;
    auto OnStart(Context& context) -> Result<void> override;
    auto OnStop(Context& context) -> Result<void> override;

    // --- IPlugin ---
    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override;

    // --- IDevice ---
    [[nodiscard]] auto Connect() noexcept -> Result<void> override;
    [[nodiscard]] auto Disconnect() noexcept -> Result<void> override;
    [[nodiscard]] auto IsConnected() const noexcept -> bool override;
    [[nodiscard]] auto SerialNumber() const noexcept -> std::string_view override;
    [[nodiscard]] auto CurrentState() const noexcept -> State override;

    // --- ICamera ---
    [[nodiscard]] auto SetTriggerMode(TriggerMode mode) noexcept -> Result<void> override;
    [[nodiscard]] auto StartAcquisition() noexcept -> Result<void> override;
    [[nodiscard]] auto StopAcquisition() noexcept -> Result<void> override;
    [[nodiscard]] auto RegisterFrameCallback(FrameCallback callback) noexcept -> Result<void> override;
    [[nodiscard]] auto SetExposureTime(std::chrono::microseconds us) noexcept -> Result<void> override;
    [[nodiscard]] auto SetGain(float db) noexcept -> Result<void> override;
    [[nodiscard]] auto SetROI(device::Rect region) noexcept -> Result<void> override;

    // --- Test hook ---
    [[nodiscard]] auto TriggerSoftware() noexcept -> Result<void>;

    // --- Test accessors ---
    [[nodiscard]] auto GetTriggerMode() const noexcept -> TriggerMode { return trigger_mode_; }
    [[nodiscard]] auto GetExposureTime() const noexcept -> std::chrono::microseconds { return exposure_time_; }
    [[nodiscard]] auto GetGain() const noexcept -> float { return gain_; }
    [[nodiscard]] auto GetROI() const noexcept -> device::Rect { return roi_; }

private:
    State state_{State::Disconnected};
    std::string serial_;
    TriggerMode trigger_mode_{TriggerMode::Software};
    std::chrono::microseconds exposure_time_{1000};
    float gain_{0.0f};
    device::Rect roi_{};
    FrameCallback frame_callback_;
};

}  // namespace sai::test
