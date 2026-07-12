// preprocess.h — 预处理链（Compose）与内置步骤（批次 2.2）
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include <sai/core/error.h>
#include <sai/image/image.h>

namespace sai::image {

// 预处理步骤：unique_ptr<Image> → unique_ptr<Image>，失败返回错误。
// 按 unique_ptr 传递而非按值传递 Image——Image 是抽象基类且构造函数 protected，
// 按值传递会切片并丢失 RawImage/SurfaceImage/GpuImage 各自持有的 PooledPtr<uint8_t>
// （池归还所必需），unique_ptr<Image> 保留动态类型与所有权，无切片风险。
using PreprocessFn =
    std::function<auto(std::unique_ptr<Image>) -> Result<std::unique_ptr<Image>>>;

// 将 steps 顺序串联，任一步骤失败则短路返回。空链原样返回输入。
[[nodiscard]] auto Compose(std::vector<PreprocessFn> steps) -> PreprocessFn;

// 内置步骤
[[nodiscard]] auto MakeDebayer() -> PreprocessFn;

// 平场校正（FlatField）。
// 冻结签名为 `MakeFlatField(Image correction_frame)`（按值）——无法编译：Image 是抽象基类、
// 构造函数 protected 且只可移动，按值传递会切片。改为 `const Image&`（Task 11 记录的批准偏差）：
// 校正帧只读、启动时加载一次、长期存活，故在返回的 lambda 中以 const Image* 捕获。
// 调用方必须保证校正帧在整条链的生命周期内保持存活。
[[nodiscard]] auto MakeFlatField(const Image& correction_frame) -> PreprocessFn;

struct CalibrationParams {
    std::array<double, 9> camera_matrix;
    std::array<double, 5> dist_coeffs;
    double pixel_scale_mm = 1.0;
};
[[nodiscard]] auto MakeCalibration(CalibrationParams params) -> PreprocessFn;
[[nodiscard]] auto MakeWhiteBalance(float r_gain, float g_gain, float b_gain) -> PreprocessFn;
[[nodiscard]] auto MakeHDR(std::size_t num_exposures) -> PreprocessFn;
[[nodiscard]] auto MakeResize(std::size_t target_width, std::size_t target_height) -> PreprocessFn;

}  // namespace sai::image
