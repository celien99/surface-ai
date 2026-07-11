#include "fake_light_controller.h"

#include <sai/core/error.h>

namespace sai::test {

FakeLightController::FakeLightController(int channel_count)
    : channel_count_(channel_count)
    , channels_(static_cast<std::size_t>(channel_count))
    , serial_("LIGHT-00001")
{}

auto FakeLightController::OnInitialize(Context& /*context*/) -> Result<void> { return {}; }

auto FakeLightController::OnStart(Context& /*context*/) -> Result<void> { return {}; }

auto FakeLightController::OnStop(Context& /*context*/) -> Result<void> { return {}; }

auto FakeLightController::GetManifest() const noexcept -> const PluginManifest& {
    static const PluginManifest kManifest = [] {
        PluginManifest m;
        m.name = "sai.test.fake-light-controller";
        m.library_path = "fake_light_controller";
        m.version = SemVer{1, 0, 0};
        m.license_token = "fake-light-license";
        return m;
    }();
    return kManifest;
}

auto FakeLightController::Connect() noexcept -> Result<void> {
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

auto FakeLightController::Disconnect() noexcept -> Result<void> {
    state_ = State::Disconnected;
    return {};
}

auto FakeLightController::IsConnected() const noexcept -> bool {
    return state_ == State::Connected;
}

auto FakeLightController::SerialNumber() const noexcept -> std::string_view { return serial_; }

auto FakeLightController::CurrentState() const noexcept -> State { return state_; }

auto FakeLightController::ChannelCount() const noexcept -> int { return channel_count_; }

auto FakeLightController::IsValidChannel(int channel) const noexcept -> bool {
    return channel >= 0 && channel < channel_count_;
}

auto FakeLightController::SetIntensity(int channel, float intensity) noexcept -> Result<void> {
    if (!IsValidChannel(channel)) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "channel out of range",
            std::source_location::current(),
        });
    }
    channels_[static_cast<std::size_t>(channel)].intensity = intensity;
    return {};
}

auto FakeLightController::GetIntensity(int channel) const noexcept -> Result<float> {
    if (!IsValidChannel(channel)) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "channel out of range",
            std::source_location::current(),
        });
    }
    return channels_[static_cast<std::size_t>(channel)].intensity;
}

auto FakeLightController::Enable(int channel) noexcept -> Result<void> {
    if (!IsValidChannel(channel)) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "channel out of range",
            std::source_location::current(),
        });
    }
    channels_[static_cast<std::size_t>(channel)].enabled = true;
    return {};
}

auto FakeLightController::Disable(int channel) noexcept -> Result<void> {
    if (!IsValidChannel(channel)) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "channel out of range",
            std::source_location::current(),
        });
    }
    channels_[static_cast<std::size_t>(channel)].enabled = false;
    return {};
}

auto FakeLightController::SetStrobeMode(int channel, StrobeMode mode) noexcept -> Result<void> {
    if (!IsValidChannel(channel)) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "channel out of range",
            std::source_location::current(),
        });
    }
    channels_[static_cast<std::size_t>(channel)].strobe_mode = mode;
    return {};
}

}  // namespace sai::test
