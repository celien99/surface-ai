#include <sai/image/preprocess.h>

#include <algorithm>
#include <cstdint>
#include <source_location>
#include <utility>
#include <vector>

#include <sai/image/surface_image.h>

namespace sai::image {

namespace {

// 递归折叠执行链：index == size 时原样返回图像，否则运行当前步骤并将结果 and_then 到下一步；
// 任一步失败短路传播。以递归而非嵌套循环串联（仓库风格要求链式遍历用递归）。
auto RunFrom(const std::vector<PreprocessFn>& steps, std::size_t index,
             std::unique_ptr<Image> img) -> Result<std::unique_ptr<Image>> {
    if (index == steps.size()) {
        return img;
    }
    return steps[index](std::move(img)).and_then([&steps, index](std::unique_ptr<Image> next) {
        return RunFrom(steps, index + 1, std::move(next));
    });
}

constexpr auto ClampU8(double v) noexcept -> std::uint8_t {
    return static_cast<std::uint8_t>(v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v));
}

}  // namespace

auto Compose(std::vector<PreprocessFn> steps) -> PreprocessFn {
    return [steps = std::move(steps)](std::unique_ptr<Image> img) -> Result<std::unique_ptr<Image>> {
        return RunFrom(steps, 0, std::move(img));
    };
}

// 平场校正：乘性逐像素校正，原地改写输入缓冲后返回同一 unique_ptr。
// 校正帧以 const Image* 捕获（只读、长期存活），调用方须保证其在链的生命周期内有效。
// 参考电平取校正帧均值 ref_mean，out = clamp(in * ref_mean / corr)，corr==0 处保留原值以防除零。
auto MakeFlatField(const Image& correction_frame) -> PreprocessFn {
    const Image* corr = &correction_frame;
    return [corr](std::unique_ptr<Image> img) -> Result<std::unique_ptr<Image>> {
        const ImageMeta& in_meta = img->Meta();
        const ImageMeta& corr_meta = corr->Meta();
        const bool mismatch = in_meta.width != corr_meta.width ||
                              in_meta.height != corr_meta.height ||
                              img->SizeBytes() != corr->SizeBytes();
        if (mismatch) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Image_DimensionMismatch,
                "FlatField correction frame dimensions do not match input",
                std::source_location::current(),
            });
        }

        const std::size_t n = img->SizeBytes();
        const std::uint8_t* c = corr->Data();
        std::uint8_t* out = img->Data();

        std::uint64_t sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += c[i];
        }
        const double ref_mean = n == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(n);

        for (std::size_t i = 0; i < n; ++i) {
            if (c[i] == 0) {
                continue;  // 除零保护：校正帧该点无有效增益，保留原像素
            }
            out[i] = ClampU8(static_cast<double>(out[i]) * ref_mean / static_cast<double>(c[i]));
        }
        return img;
    };
}

// BayerRG8 (RGGB) 去马赛克：每个 2x2 quad 内 R=左上、G=右上、B=右下，quad 四像素取同一 RGB
// （最近邻）。输出新建拥有型 RGB8 SurfaceImage，尺寸不变、通道数 3。非 Bayer 输入报错。
auto MakeDebayer() -> PreprocessFn {
    return [](std::unique_ptr<Image> img) -> Result<std::unique_ptr<Image>> {
        const ImageMeta& meta = img->Meta();
        if (meta.pixel_format != PixelFormat::BayerRG8) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Image_UnsupportedPixelFormat,
                "Debayer requires a Bayer pixel format (BayerRG8)",
                std::source_location::current(),
            });
        }

        const std::size_t w = meta.width;
        const std::size_t h = meta.height;
        const std::uint8_t* in = img->Data();
        std::vector<std::uint8_t> out(w * h * 3);

        for (std::size_t r = 0; r < h; ++r) {
            const std::size_t qr = r & ~std::size_t{1};
            const std::size_t qr1 = std::min(qr + 1, h - 1);  // 奇数高度末行钳制，防越界
            for (std::size_t c = 0; c < w; ++c) {
                const std::size_t qc = c & ~std::size_t{1};
                const std::size_t qc1 = std::min(qc + 1, w - 1);
                std::uint8_t* px = out.data() + (r * w + c) * 3;
                px[0] = in[qr * w + qc];    // R 左上
                px[1] = in[qr * w + qc1];   // G 右上
                px[2] = in[qr1 * w + qc1];  // B 右下
            }
        }

        ImageMeta rgb = meta;
        rgb.channels = 3;
        rgb.pixel_format = PixelFormat::RGB8;
        return std::make_unique<SurfaceImage>(SurfaceImage::FromOwnedBuffer(std::move(out), rgb));
    };
}

// 白平衡：对 RGB8/BGR8 逐通道乘性增益，原地改写，夹至 [0,255]。
// 非 RGB/BGR 格式返回 Image_UnsupportedPixelFormat。
auto MakeWhiteBalance(float r_gain, float g_gain, float b_gain) -> PreprocessFn {
    return [r_gain, g_gain, b_gain](std::unique_ptr<Image> img) -> Result<std::unique_ptr<Image>> {
        const PixelFormat fmt = img->Meta().pixel_format;
        if (fmt != PixelFormat::RGB8 && fmt != PixelFormat::BGR8) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Image_UnsupportedPixelFormat,
                "WhiteBalance requires RGB8 or BGR8 pixel format",
                std::source_location::current(),
            });
        }

        const bool is_rgb = (fmt == PixelFormat::RGB8);
        const std::size_t npixels = img->SizeBytes() / 3;
        std::uint8_t* data = img->Data();
        for (std::size_t i = 0; i < npixels; ++i) {
            if (is_rgb) {
                data[0] = ClampU8(static_cast<double>(data[0]) * r_gain);
                data[1] = ClampU8(static_cast<double>(data[1]) * g_gain);
                data[2] = ClampU8(static_cast<double>(data[2]) * b_gain);
            } else {
                data[0] = ClampU8(static_cast<double>(data[0]) * b_gain);
                data[1] = ClampU8(static_cast<double>(data[1]) * g_gain);
                data[2] = ClampU8(static_cast<double>(data[2]) * r_gain);
            }
            data += 3;
        }
        return img;
    };
}

// 缩放：双线性插值，输出通过 SurfaceImage::FromOwnedBuffer 分配。
// 支持 1 或 3 通道；target_width/target_height 为零返回 Image_PreprocessFailed。
auto MakeResize(std::size_t target_width, std::size_t target_height) -> PreprocessFn {
    return [target_width, target_height](
               std::unique_ptr<Image> img) -> Result<std::unique_ptr<Image>> {
        if (target_width == 0 || target_height == 0) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Image_PreprocessFailed,
                "Resize target dimensions must be > 0",
                std::source_location::current(),
            });
        }

        const ImageMeta& in_meta = img->Meta();
        const std::size_t c = in_meta.channels;
        if (c != 1 && c != 3) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Image_UnsupportedPixelFormat,
                "Resize supports 1 or 3 channel images only",
                std::source_location::current(),
            });
        }

        const std::size_t sw = in_meta.width;
        const std::size_t sh = in_meta.height;
        const std::size_t tw = target_width;
        const std::size_t th = target_height;
        const std::uint8_t* in = img->Data();

        std::vector<std::uint8_t> out_bytes(tw * th * c);

        for (std::size_t oy = 0; oy < th; ++oy) {
            for (std::size_t ox = 0; ox < tw; ++ox) {
                // 输出像素中心映射回源坐标
                double sx = (static_cast<double>(ox) + 0.5) * static_cast<double>(sw) /
                                 static_cast<double>(tw) -
                             0.5;
                double sy = (static_cast<double>(oy) + 0.5) * static_cast<double>(sh) /
                                 static_cast<double>(th) -
                             0.5;

                // 夹至源有效范围
                sx = std::max(0.0, std::min(sx, static_cast<double>(sw) - 1.0));
                sy = std::max(0.0, std::min(sy, static_cast<double>(sh) - 1.0));

                std::size_t x0 = static_cast<std::size_t>(sx);
                std::size_t y0 = static_cast<std::size_t>(sy);
                std::size_t x1 = std::min(x0 + 1, sw - 1);
                std::size_t y1 = std::min(y0 + 1, sh - 1);

                double fx = sx - static_cast<double>(x0);
                double fy = sy - static_cast<double>(y0);

                double w00 = (1.0 - fx) * (1.0 - fy);
                double w10 = fx * (1.0 - fy);
                double w01 = (1.0 - fx) * fy;
                double w11 = fx * fy;

                std::uint8_t* out_px = out_bytes.data() + (oy * tw + ox) * c;
                for (std::size_t ch = 0; ch < c; ++ch) {
                    double val = w00 * static_cast<double>(in[(y0 * sw + x0) * c + ch]) +
                                 w10 * static_cast<double>(in[(y0 * sw + x1) * c + ch]) +
                                 w01 * static_cast<double>(in[(y1 * sw + x0) * c + ch]) +
                                 w11 * static_cast<double>(in[(y1 * sw + x1) * c + ch]);
                    out_px[ch] = ClampU8(val);
                }
            }
        }

        ImageMeta out_meta = in_meta;
        out_meta.width = tw;
        out_meta.height = th;
        return std::make_unique<SurfaceImage>(
            SurfaceImage::FromOwnedBuffer(std::move(out_bytes), out_meta));
    };
}

// 标定/畸变校正：Brown-Conrady 径向+切向畸变模型，双线性逆映射。
// camera_matrix 为行主序 3x3；dist_coeffs 为 {k1,k2,p1,p2,k3}。
// identity 矩阵 + 零畸变 → 几何不变的逐像素拷贝。
// pixel_scale_mm 为元数据，ImageMeta 无专用字段，通过文档记录携带（见 task-6-report.md）。
auto MakeCalibration(CalibrationParams params) -> PreprocessFn {
    return [params](std::unique_ptr<Image> img) -> Result<std::unique_ptr<Image>> {
        const ImageMeta& in_meta = img->Meta();
        const std::size_t w = in_meta.width;
        const std::size_t h = in_meta.height;
        const std::size_t c = in_meta.channels;
        const std::uint8_t* in = img->Data();

        const double fx = params.camera_matrix[0];
        const double fy = params.camera_matrix[4];
        const double cx = params.camera_matrix[2];
        const double cy = params.camera_matrix[5];
        const double k1 = params.dist_coeffs[0];
        const double k2 = params.dist_coeffs[1];
        const double p1 = params.dist_coeffs[2];
        const double p2 = params.dist_coeffs[3];
        const double k3 = params.dist_coeffs[4];

        std::vector<std::uint8_t> out_bytes(w * h * c);

        for (std::size_t dy = 0; dy < h; ++dy) {
            for (std::size_t dx = 0; dx < w; ++dx) {
                // 归一化至相机坐标
                double xn = (static_cast<double>(dx) - cx) / fx;
                double yn = (static_cast<double>(dy) - cy) / fy;

                // Brown-Conrady 畸变模型
                double r2 = xn * xn + yn * yn;
                double r4 = r2 * r2;
                double r6 = r4 * r2;
                double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
                double x_tang = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
                double y_tang = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
                double xd = xn * radial + x_tang;
                double yd = yn * radial + y_tang;

                // 逆归一化 → 源像素坐标
                double sx = xd * fx + cx;
                double sy = yd * fy + cy;

                std::uint8_t* out_px = out_bytes.data() + (dy * w + dx) * c;

                // 越界填充 0
                if (sx < 0.0 || sy < 0.0 || sx >= static_cast<double>(w) ||
                    sy >= static_cast<double>(h)) {
                    for (std::size_t ch = 0; ch < c; ++ch) {
                        out_px[ch] = 0;
                    }
                    continue;
                }

                // 双线性插值
                std::size_t x0 = static_cast<std::size_t>(sx);
                std::size_t y0 = static_cast<std::size_t>(sy);
                std::size_t x1 = std::min(x0 + 1, w - 1);
                std::size_t y1 = std::min(y0 + 1, h - 1);

                double fx_w = sx - static_cast<double>(x0);
                double fy_w = sy - static_cast<double>(y0);

                double w00 = (1.0 - fx_w) * (1.0 - fy_w);
                double w10 = fx_w * (1.0 - fy_w);
                double w01 = (1.0 - fx_w) * fy_w;
                double w11 = fx_w * fy_w;

                for (std::size_t ch = 0; ch < c; ++ch) {
                    double val =
                        w00 * static_cast<double>(in[(y0 * w + x0) * c + ch]) +
                        w10 * static_cast<double>(in[(y0 * w + x1) * c + ch]) +
                        w01 * static_cast<double>(in[(y1 * w + x0) * c + ch]) +
                        w11 * static_cast<double>(in[(y1 * w + x1) * c + ch]);
                    out_px[ch] = ClampU8(val);
                }
            }
        }

        // pixel_scale_mm 为元数据，ImageMeta 无专用字段，通过调用方在 Meta 通道中携带。
        ImageMeta out_meta = in_meta;
        return std::make_unique<SurfaceImage>(
            SurfaceImage::FromOwnedBuffer(std::move(out_bytes), out_meta));
    };
}

// HDR — 单帧动态范围 / min-max 对比度拉伸（批准偏差 D-c）。
// MakeHDR(num_exposures) 返回单输入 PreprocessFn，无法融合多帧（类型只取一个 Image）。
// 实现为单帧 min→0、max→255 拉伸；num_exposures 效验 >0 否则 Image_PreprocessFailed。
// 真正的多曝光合成需要多输入 API，推迟至未来扩展。
auto MakeHDR(std::size_t num_exposures) -> PreprocessFn {
    return [num_exposures](std::unique_ptr<Image> img) -> Result<std::unique_ptr<Image>> {
        if (num_exposures == 0) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Image_PreprocessFailed,
                "MakeHDR requires num_exposures > 0",
                std::source_location::current(),
            });
        }

        const std::size_t n = img->SizeBytes();
        if (n == 0) {
            return img;
        }

        std::uint8_t* data = img->Data();

        std::uint8_t min_val = 255;
        std::uint8_t max_val = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }

        if (min_val == max_val) {
            return img;
        }

        const double range = static_cast<double>(max_val) - static_cast<double>(min_val);
        for (std::size_t i = 0; i < n; ++i) {
            data[i] = ClampU8(255.0 * (static_cast<double>(data[i]) - min_val) / range);
        }

        return img;
    };
}

}  // namespace sai::image
