// detector.h — 批次 3.3 Detector 统一检测接口
#pragma once

#include <span>
#include <string_view>

#include <sai/core/error.h>
#include <sai/core/object.h>
#include <sai/embedding/embedding.h>
#include <sai/detection/detection_result.h>

namespace sai {
class Context;
}  // namespace sai

namespace sai::detection {

// IDetector：框架统一的异常检测接口
// 以标准化 Embedding 为输入，输出结构化 DetectionResult。
// 具体算法（PatchCore、EfficientAD 等）继承此接口并在 Initialize 中完成启动期加载。
class IDetector : public Object {
public:
    // 启动期初始化（如加载 FeatureBank / 网络权重）
    [[nodiscard]] virtual auto Initialize(sai::Context& ctx) noexcept -> Result<void> = 0;

    // 单帧检测
    [[nodiscard]] virtual auto Detect(const sai::embedding::Embedding& embedding) noexcept
        -> Result<DetectionResult> = 0;

    // 批量检测
    [[nodiscard]] virtual auto DetectBatch(
        std::span<const sai::embedding::Embedding* const> embeddings) noexcept
        -> Result<std::vector<DetectionResult>> = 0;

    // 返回 detector 的名称（如 "PatchCore"）
    [[nodiscard]] virtual auto ModelName() const noexcept -> std::string_view = 0;
};

}  // namespace sai::detection
