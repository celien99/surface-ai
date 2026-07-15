// genicam_camera.h — GenICam / GigE Vision camera via Aravis (Linux-gated)
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>

#include <sai/device/camera.h>
#include <sai/image/raw_image.h>
#include <sai/plugin/plugin.h>

namespace sai::device {

// GenICamCamera: GigE Vision / USB3 Vision camera implementation using the
// Aravis library (aravis-0.8). Provides device discovery, GenICam feature
// control, and streaming frame capture.
//
// Only compiled on Linux when aravis is available (CMake-gated via
// find_package(aravis QUIET)). On other platforms, FakeCamera serves as
// the development substitute.
//
// Usage:
//   GenICamCamera::Config cfg;
//   cfg.serial_number = "BGxxxxxxxx";  // or user_id for USB3
//   auto cam = GenICamCamera(cfg);
//   cam.Connect();
//   cam.RegisterFrameCallback([](RawImage img) { ... });
//   cam.SetExposureTime(std::chrono::microseconds(5000));
//   cam.StartAcquisition();

class GenICamCamera final : public ICamera {
public:
    struct Config {
        // Device identification: if serial_number is non-empty, search by
        // serial; otherwise use user_id (GigE user-defined name or USB3 ID).
        std::string serial_number;
        std::string user_id;

        // Stream configuration.
        std::size_t packet_size = 9000;   // GigE jumbo frame
        std::size_t packet_delay = 1000;  // inter-packet delay (ns)
        std::size_t frame_retention = 8;  // frames to buffer in stream

        // Trigger defaults.
        ICamera::TriggerMode trigger_mode = ICamera::TriggerMode::FreeRun;
    };

    explicit GenICamCamera(Config cfg);
    ~GenICamCamera() override;

    // ICamera interface
    [[nodiscard]] auto SetTriggerMode(TriggerMode mode) noexcept -> Result<void> override;
    [[nodiscard]] auto StartAcquisition() noexcept -> Result<void> override;
    [[nodiscard]] auto StopAcquisition() noexcept -> Result<void> override;
    [[nodiscard]] auto RegisterFrameCallback(FrameCallback callback) noexcept
        -> Result<void> override;
    [[nodiscard]] auto SetExposureTime(std::chrono::microseconds us) noexcept
        -> Result<void> override;
    [[nodiscard]] auto SetGain(float db) noexcept -> Result<void> override;
    [[nodiscard]] auto SetROI(Rect region) noexcept -> Result<void> override;

    // IDevice interface
    [[nodiscard]] auto Connect() noexcept -> Result<void> override;
    [[nodiscard]] auto Disconnect() noexcept -> Result<void> override;
    [[nodiscard]] auto IsConnected() const noexcept -> bool override;
    [[nodiscard]] auto SerialNumber() const noexcept -> std::string_view override;
    [[nodiscard]] auto CurrentState() const noexcept -> State override;

    // IPlugin interface
    [[nodiscard]] auto OnInitialize(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStart(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStop(Context&) -> Result<void> override;
    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override
    { return manifest_; }

    // Move-only, non-copyable (owns hardware connection)
    GenICamCamera(GenICamCamera&&) = delete;
    auto operator=(GenICamCamera&&) -> GenICamCamera& = delete;
    GenICamCamera(const GenICamCamera&) = delete;
    auto operator=(const GenICamCamera&) -> GenICamCamera& = delete;

private:
    // Internal: start/stop the frame acquisition thread.
    auto AcquisitionLoop(std::stop_token st) -> void;

    // Internal: map Aravis pixel format to sai::image::PixelFormat.
    auto MapPixelFormat(int arv_format) const noexcept -> sai::image::PixelFormat;

    // Internal: accessor for the static stream callback.
    auto GetFrameCallback() noexcept -> FrameCallback&;

    Config cfg_;
    PluginManifest manifest_{};

    // Opaque Aravis handles (arv_camera*, arv_stream*) — stored as void*
    // to avoid leaking Aravis headers from this public header.
    void* arv_camera_ = nullptr;  // ArvCamera*
    void* arv_stream_ = nullptr;  // ArvStream*

    FrameCallback callback_;
    std::atomic<State> state_{State::Disconnected};
    std::atomic<bool> acquiring_{false};
    std::jthread acquisition_thread_;
    std::string serial_number_;
};

}  // namespace sai::device
