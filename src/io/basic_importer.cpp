#include <sai/io/importer.h>
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

    // 2. Open file for binary read
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportFileNotFound,
            .message = "Cannot open image file: " + path.string(),
        });
    }

    // 3. Parse PPM header (P6 only)
    std::string magic;
    int width = 0, height = 0, maxval = 0;
    file >> magic;
    if (magic != "P6") {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Unsupported PPM format (only P6 supported): " + path.string(),
        });
    }

    file >> width >> height >> maxval;
    if (!file || width <= 0 || height <= 0 || maxval != 255) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Invalid or unsupported PPM header in file: " + path.string(),
        });
    }

    // 4. Skip the single whitespace byte following maxval
    file.get();

    // 5. Read pixel data
    auto expected_bytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;
    std::vector<std::uint8_t> bytes(expected_bytes);
    file.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(expected_bytes));
    if (!file || static_cast<std::size_t>(file.gcount()) != expected_bytes) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Failed to read complete pixel data from PPM file: " + path.string(),
        });
    }

    // 6. Construct SurfaceImage via FromOwnedBuffer
    sai::image::ImageMeta meta;
    meta.width = static_cast<std::size_t>(width);
    meta.height = static_cast<std::size_t>(height);
    meta.channels = 3;
    meta.pixel_format = sai::image::PixelFormat::RGB8;

    auto surface = sai::image::SurfaceImage::FromOwnedBuffer(std::move(bytes), meta);
    return std::make_unique<sai::image::SurfaceImage>(std::move(surface));
}

}  // namespace sai::io
