#pragma once

#include <sai/device/device.h>

namespace sai::device {

class ILightController : public IDevice {
public:
    enum class StrobeMode { Continuous, OnTrigger, Off };

    [[nodiscard]] virtual auto ChannelCount() const noexcept -> int = 0;
    [[nodiscard]] virtual auto SetIntensity(int channel, float intensity) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto GetIntensity(int channel) const noexcept -> Result<float> = 0;
    [[nodiscard]] virtual auto Enable(int channel) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto Disable(int channel) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto SetStrobeMode(int channel, StrobeMode mode) noexcept -> Result<void> = 0;
};

}  // namespace sai::device
