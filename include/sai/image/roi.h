// roi.h — 感兴趣区域
#pragma once

#include <memory>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/rect.h>
#include <sai/image/image.h>

namespace sai::image {

using Rect = sai::core::Rect;  // 共用定义

struct ROI {
    std::vector<Rect> regions;
    [[nodiscard]] auto IsEmpty() const noexcept -> bool { return regions.empty(); }
    [[nodiscard]] auto BoundingBox() const noexcept -> Rect;
    [[nodiscard]] static auto Apply(const Image& src, const ROI& roi)
        -> Result<std::unique_ptr<Image>>;
};

}  // namespace sai::image
