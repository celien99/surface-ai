// raw_image.h — 相机原始输出
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <sai/image/image.h>
#include <sai/memory/memory_pool.h>

namespace sai::image {

using sai::memory::IMemoryPool;
using sai::memory::PooledPtr;

class RawImage final : public Image {
public:
    [[nodiscard]] static auto FromPool(IMemoryPool& pool, ImageMeta meta) noexcept
        -> Result<RawImage>;
    // 调用方自持有缓冲（例如测试夹具/内存映射文件），本实例不持有 PooledPtr，
    // 析构时不触发任何池归还——data 的生命周期完全由调用方管理。
    [[nodiscard]] static auto FromBuffer(std::uint8_t* data, std::size_t size_bytes,
                                         ImageMeta meta) noexcept -> RawImage;
    // 追加工厂（Task 11 记录的批准偏差）：以堆上 std::vector 承载像素、由本实例独占所有权，
    // 无需内存池即可返回"改变尺寸/格式"步骤（Debayer/Resize/ROI 裁剪/PPM 导入）产出的拥有型图像。
    // 冻结的 FromPool（需池）/FromBuffer（不拥有）都覆盖不到这个空档，此工厂便携地补齐。
    [[nodiscard]] static auto FromOwnedBuffer(std::vector<std::uint8_t> bytes, ImageMeta meta) noexcept
        -> RawImage;

    RawImage(RawImage&&) noexcept = default;
    auto operator=(RawImage&&) noexcept -> RawImage& = default;
    ~RawImage() override;
    RawImage(const RawImage&) = delete;
    auto operator=(const RawImage&) -> RawImage& = delete;

    auto Release() noexcept -> void override;

private:
    // 持有实际的 PooledPtr<uint8_t> 句柄（而非裸 IMemoryPool* + 裸数据指针）——
    // IMemoryPool::Release 要求传入具体句柄以定位 slab 元数据中的引用计数槽，
    // 裸指针无法归还给池；FromBuffer 路径下 buffer_ 保持默认空句柄。
    explicit RawImage(PooledPtr<std::uint8_t> buffer, ImageMeta meta) noexcept;
    RawImage(std::uint8_t* data, std::size_t size_bytes, ImageMeta meta) noexcept;  // FromBuffer 专用
    RawImage(std::vector<std::uint8_t> owned, ImageMeta meta) noexcept;             // FromOwnedBuffer 专用

    // buffer_ 与 owned_bytes_ 恰有一个被填充：池路径填 buffer_，堆拥有路径填 owned_bytes_，
    // FromBuffer 路径两者皆空（不拥有）。
    PooledPtr<std::uint8_t> buffer_{};
    std::vector<std::uint8_t> owned_bytes_{};
};

}  // namespace sai::image
