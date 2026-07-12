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

}  // namespace sai::image
