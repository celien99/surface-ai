#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <string_view>

#include <sai/plugin/plugin.h>

namespace sai::device {

class IDevice : public IPlugin {
public:
    enum class State { Disconnected, Connected, Acquiring, Error };

    [[nodiscard]] virtual auto Connect() noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto Disconnect() noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto IsConnected() const noexcept -> bool = 0;
    [[nodiscard]] virtual auto SerialNumber() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto CurrentState() const noexcept -> State = 0;
};

struct Rect {
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t width = 0;
    std::size_t height = 0;

    [[nodiscard]] auto Area() const noexcept -> std::size_t { return width * height; }
    [[nodiscard]] auto IsEmpty() const noexcept -> bool { return width == 0 || height == 0; }
};

}  // namespace sai::device
