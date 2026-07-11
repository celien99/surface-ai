#pragma once

#include <sai/device/light_controller.h>

#include <string>
#include <vector>

namespace sai::test {

class FakeLightController final : public device::ILightController {
public:
    SAI_DECLARE_TYPE_ID(sai.test.fake-light-controller)

    explicit FakeLightController(int channel_count = 4);
    ~FakeLightController() override = default;

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

    // --- ILightController ---
    [[nodiscard]] auto ChannelCount() const noexcept -> int override;
    [[nodiscard]] auto SetIntensity(int channel, float intensity) noexcept -> Result<void> override;
    [[nodiscard]] auto GetIntensity(int channel) const noexcept -> Result<float> override;
    [[nodiscard]] auto Enable(int channel) noexcept -> Result<void> override;
    [[nodiscard]] auto Disable(int channel) noexcept -> Result<void> override;
    [[nodiscard]] auto SetStrobeMode(int channel, StrobeMode mode) noexcept -> Result<void> override;

private:
    [[nodiscard]] auto IsValidChannel(int channel) const noexcept -> bool;

    struct Channel {
        float intensity = 0.0f;
        bool enabled = false;
        StrobeMode strobe_mode = StrobeMode::Off;
    };

    int channel_count_;
    std::vector<Channel> channels_;
    State state_{State::Disconnected};
    std::string serial_;
};

}  // namespace sai::test
