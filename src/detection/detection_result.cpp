// detection_result.cpp — AnomalyMap::MaxScore 实现
#include <algorithm>
#include <sai/detection/detection_result.h>

namespace sai::detection {

auto AnomalyMap::MaxScore() const noexcept -> float {
    if (scores.empty()) return 0.0F;
    return *std::max_element(scores.begin(), scores.end());
}

}  // namespace sai::detection
