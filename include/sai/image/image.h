// image.h — 图像类型体系基类与元数据（批次 2.2）
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include <sai/core/resource.h>

namespace sai::image {

enum class PixelFormat : std::uint16_t {
    Mono8, Mono10, Mono12, Mono16,
    BayerRG8, BayerRG10, BayerRG12, BayerRG16,
    RGB8, BGR8, RGB16,
    Undefined = 0xFFFF,
};

struct ImageMeta {
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t channels = 0;
    PixelFormat pixel_format = PixelFormat::Undefined;
    std::size_t bits_per_sample = 8;  // 8/10/12/16 — for correct clamping in preprocess
    std::chrono::nanoseconds timestamp{0};
    std::uint32_t frame_index = 0;

    // Multi-light AOI metadata — identifies which product/position/light
    // this frame belongs to.  Set by ImportDataset or CaptureStage.
    std::string surface_id;            // product barcode / serial
    std::uint16_t position_id = 0;     // camera position index
    std::uint16_t light_id = 0;        // light source index
};

// 每采样字节数：8-bit 格式 1 字节，10/12-bit 打包到 2 字节，16-bit 2 字节，Undefined 保守取 1。
[[nodiscard]] constexpr auto BytesPerChannel(PixelFormat format) noexcept -> std::size_t {
    switch (format) {
        case PixelFormat::Mono10:
        case PixelFormat::Mono12:
        case PixelFormat::BayerRG10:
        case PixelFormat::BayerRG12:
        case PixelFormat::Mono16:
        case PixelFormat::BayerRG16:
        case PixelFormat::RGB16:
            return 2;
        default:
            return 1;
    }
}

[[nodiscard]] constexpr auto BytesPerPixel(const ImageMeta& meta) noexcept -> std::size_t {
    return meta.channels * BytesPerChannel(meta.pixel_format);
}

[[nodiscard]] constexpr auto ByteSize(const ImageMeta& meta) noexcept -> std::size_t {
    return meta.width * meta.height * BytesPerPixel(meta);
}

// Image 继承 Resource（1.1）——像素缓冲独占所有权，移动不拷贝。
// 抽象基类：构造函数 protected，只能通过 RawImage/SurfaceImage/GpuImage 的具名工厂函数间接构造，
// 跨类型的通用消费路径（PreprocessFn/ROI::Apply/IImporter::ImportImage）统一以
// std::unique_ptr<Image> 传递，避免按值传递切片丢失子类私有的 owner_pool_。
class Image : public Resource {
public:
    [[nodiscard]] auto Meta() const noexcept -> const ImageMeta& { return meta_; }
    [[nodiscard]] auto Meta() noexcept -> ImageMeta& { return meta_; }
    [[nodiscard]] auto Data() const noexcept -> const std::uint8_t* { return data_; }
    [[nodiscard]] auto Data() noexcept -> std::uint8_t* { return data_; }
    [[nodiscard]] auto SizeBytes() const noexcept -> std::size_t { return size_bytes_; }

    // 查询图像是否驻留在 GPU 显存中。
    // 基类返回 false；GpuImage 子类重写返回 true。
    // 用于 Embedder::Extract 的 GPU guard——无需 dynamic_cast 或 RTTI。
    [[nodiscard]] virtual auto IsGpuImage() const noexcept -> bool { return false; }

    // Resource（1.1）纯虚契约：data_ != nullptr 即视为持有有效缓冲。
    [[nodiscard]] auto IsValid() const noexcept -> bool override { return data_ != nullptr; }
    auto Release() noexcept -> void override;

protected:
    Image(std::uint8_t* data, std::size_t size_bytes, ImageMeta meta) noexcept;

    // 显式移动：置空源 data_/size_bytes_，令被移动后的对象 IsValid()==false。默认移动只逐成员
    // 拷贝裸 data_，源仍报告有效且 Data() 悬垂——data_ 是非拥有视图，真正所有权在子类的
    // buffer_/owned_bytes_ 上，两者移动后源已释放，故必须同步置空 data_。子类保持 = default 移动
    // 即可链到此处。
    Image(Image&& other) noexcept;
    auto operator=(Image&& other) noexcept -> Image&;

    std::uint8_t* data_ = nullptr;
    std::size_t size_bytes_ = 0;
    ImageMeta meta_{};
};

}  // namespace sai::image
