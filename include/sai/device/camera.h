#pragma once

#include <chrono>
#include <functional>

#include <sai/device/device.h>

namespace sai::device {

class RawImage;  // forward — defined in batch 2.2

class ICamera : public IDevice {
public:
    enum class TriggerMode { Software, Hardware, FreeRun };

    [[nodiscard]] virtual auto SetTriggerMode(TriggerMode mode) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto StartAcquisition() noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto StopAcquisition() noexcept -> Result<void> = 0;

    using FrameCallback = std::function<void(RawImage)>;
    [[nodiscard]] virtual auto RegisterFrameCallback(FrameCallback callback) noexcept -> Result<void> = 0;

    [[nodiscard]] virtual auto SetExposureTime(std::chrono::microseconds us) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto SetGain(float db) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto SetROI(Rect region) noexcept -> Result<void> = 0;
};

}  // namespace sai::device
