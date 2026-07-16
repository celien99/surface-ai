#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <string_view>

#include <sai/core/rect.h>
#include <sai/plugin/plugin.h>

namespace sai::device {

// Re-export Rect from sai::core — canonical definition lives there.
using Rect = sai::core::Rect;

class IDevice : public IPlugin {
public:
    enum class State { Disconnected, Connected, Acquiring, Error };

    [[nodiscard]] virtual auto Connect() noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto Disconnect() noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto IsConnected() const noexcept -> bool = 0;
    [[nodiscard]] virtual auto SerialNumber() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto CurrentState() const noexcept -> State = 0;
};

}  // namespace sai::device
