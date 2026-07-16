// rect.h — 基础几何类型：轴对齐矩形
#pragma once

#include <cstddef>

namespace sai::core {

struct Rect {
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t width = 0;
    std::size_t height = 0;

    [[nodiscard]] auto Area() const noexcept -> std::size_t { return width * height; }
    [[nodiscard]] auto IsEmpty() const noexcept -> bool { return width == 0 || height == 0; }
};

}  // namespace sai::core
