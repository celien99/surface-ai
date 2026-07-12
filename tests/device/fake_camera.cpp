#include "fake_camera.h"

#include <cstdint>
#include <vector>

#include <sai/core/error.h>
#include <sai/image/raw_image.h>

namespace sai::test {

FakeCamera::FakeCamera() : serial_("CAM-00001") {}

auto FakeCamera::OnInitialize(Context& /*context*/) -> Result<void> { return {}; }

auto FakeCamera::OnStart(Context& /*context*/) -> Result<void> { return {}; }

auto FakeCamera::OnStop(Context& /*context*/) -> Result<void> { return {}; }

auto FakeCamera::GetManifest() const noexcept -> const PluginManifest& {
    static const PluginManifest kManifest = [] {
        PluginManifest m;
        m.name = "sai.test.fake-camera";
        m.library_path = "fake_camera";
        m.version = SemVer{1, 0, 0};
        m.license_token = "fake-camera-license";
        return m;
    }();
    return kManifest;
}

auto FakeCamera::Connect() noexcept -> Result<void> {
    if (state_ != State::Disconnected) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "already connected or in error",
            std::source_location::current(),
        });
    }
    state_ = State::Connected;
    return {};
}

auto FakeCamera::Disconnect() noexcept -> Result<void> {
    if (state_ == State::Acquiring) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_AcquisitionInProgress,
            "must stop acquisition before disconnecting",
            std::source_location::current(),
        });
    }
    state_ = State::Disconnected;
    return {};
}

auto FakeCamera::IsConnected() const noexcept -> bool {
    return state_ == State::Connected || state_ == State::Acquiring;
}

auto FakeCamera::SerialNumber() const noexcept -> std::string_view { return serial_; }

auto FakeCamera::CurrentState() const noexcept -> State { return state_; }

auto FakeCamera::SetTriggerMode(TriggerMode mode) noexcept -> Result<void> {
    trigger_mode_ = mode;
    return {};
}

auto FakeCamera::StartAcquisition() noexcept -> Result<void> {
    if (state_ == State::Disconnected) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_NotConnected,
            "must connect before starting acquisition",
            std::source_location::current(),
        });
    }
    if (state_ == State::Acquiring) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_AcquisitionInProgress,
            "acquisition already in progress",
            std::source_location::current(),
        });
    }
    state_ = State::Acquiring;
    return {};
}

auto FakeCamera::StopAcquisition() noexcept -> Result<void> {
    if (state_ != State::Acquiring) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_NotConnected,
            "acquisition not in progress",
            std::source_location::current(),
        });
    }
    state_ = State::Connected;
    return {};
}

auto FakeCamera::RegisterFrameCallback(FrameCallback callback) noexcept -> Result<void> {
    if (state_ == State::Disconnected) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_NotConnected,
            "must connect before registering callback",
            std::source_location::current(),
        });
    }
    frame_callback_ = std::move(callback);
    return {};
}

auto FakeCamera::SetExposureTime(std::chrono::microseconds us) noexcept -> Result<void> {
    exposure_time_ = us;
    return {};
}

auto FakeCamera::SetGain(float db) noexcept -> Result<void> {
    gain_ = db;
    return {};
}

auto FakeCamera::SetROI(device::Rect region) noexcept -> Result<void> {
    roi_ = region;
    return {};
}

auto FakeCamera::TriggerSoftware() noexcept -> Result<void> {
    if (state_ != State::Acquiring) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_NotConnected,
            "must be acquiring to trigger software frame",
            std::source_location::current(),
        });
    }
    if (!frame_callback_) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_NotConnected,
            "no frame callback registered",
            std::source_location::current(),
        });
    }

    // Build a small BayerRG8 8×8 test pattern: pixel (r,c) = (r*8+c)*4, range [0,252].
    std::vector<std::uint8_t> bytes(64);
    for (std::size_t r = 0; r < 8; ++r) {
        for (std::size_t c = 0; c < 8; ++c) {
            bytes[r * 8 + c] = static_cast<std::uint8_t>((r * 8 + c) * 4);
        }
    }

    sai::image::ImageMeta meta;
    meta.width = 8;
    meta.height = 8;
    meta.channels = 1;
    meta.pixel_format = sai::image::PixelFormat::BayerRG8;

    frame_callback_(sai::image::RawImage::FromOwnedBuffer(std::move(bytes), meta));
    return {};
}

}  // namespace sai::test
