#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <sai/io/importer.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace sai::io {

auto BasicImporter::ImportMetadata(std::filesystem::path path) noexcept
    -> Result<YAML::Node> {
    // 1. Existence check
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportFileNotFound,
            .message = "Metadata file not found: " + path.string(),
        });
    }

    // 2. YAML::LoadFile in try/catch
    try {
        return YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Failed to parse metadata YAML: " + std::string(e.what()),
        });
    }
}

auto BasicImporter::ImportImage(std::filesystem::path path) noexcept
    -> Result<std::unique_ptr<Image>> {
    // 1. Existence check
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportFileNotFound,
            .message = "Image file not found: " + path.string(),
        });
    }

    // 2. Read entire file into memory for stb_image
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportFileNotFound,
            .message = "Cannot open image file: " + path.string(),
        });
    }

    auto file_size = static_cast<std::size_t>(file.tellg());
    if (file_size == 0) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Empty image file: " + path.string(),
        });
    }

    std::vector<unsigned char> file_buffer(file_size);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(file_buffer.data()),
              static_cast<std::streamsize>(file_size));
    if (!file) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Failed to read image file: " + path.string(),
        });
    }

    // 3. Decode via stb_image — force 3-channel RGB8 output.
    //    stb_image handles PPM (P6), PNG, BMP, JPEG, TGA, GIF, HDR, PIC, PNM.
    int width = 0, height = 0, channels_in_file = 0;
    auto* pixels = stbi_load_from_memory(
        file_buffer.data(),
        static_cast<int>(file_buffer.size()),
        &width, &height, &channels_in_file,
        3);  // STBI_rgb: force 3 channels

    if (pixels == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = std::string("Image decode failed: ") + stbi_failure_reason()
                       + " [" + path.string() + "]",
        });
    }

    if (width <= 0 || height <= 0) {
        stbi_image_free(pixels);
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Decoded image has invalid dimensions: " + path.string(),
        });
    }

    // 4. Copy decoded pixels into a vector owned by RawImage, then free stb buffer.
    auto total_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;
    std::vector<std::uint8_t> bytes(pixels, pixels + total_bytes);
    stbi_image_free(pixels);

    // 5. Construct RawImage from owned buffer.
    //    Use RawImage (not SurfaceImage) so the pipeline preprocess chain
    //    (debayer→white_balance→resize) receives the expected variant type.
    //    Debayer is a no-op passthrough for already-RGB8 images.
    sai::image::ImageMeta meta;
    meta.width = static_cast<std::size_t>(width);
    meta.height = static_cast<std::size_t>(height);
    meta.channels = 3;
    meta.pixel_format = sai::image::PixelFormat::RGB8;

    auto raw = sai::image::RawImage::FromOwnedBuffer(std::move(bytes), meta);
    return std::make_unique<sai::image::RawImage>(std::move(raw));
}

}  // namespace sai::io
