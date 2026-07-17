#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <sai/io/importer.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace sai::io {

auto BasicImporter::ImportDataset(std::filesystem::path yaml_path) noexcept
    -> Result<std::vector<DatasetEntry>> {
    std::error_code ec;
    if (!std::filesystem::exists(yaml_path, ec) || ec) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportFileNotFound,
            .message = "Dataset manifest not found: " + yaml_path.string(),
        });
    }

    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path.string());
    } catch (const YAML::Exception& e) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = std::string("Failed to parse dataset YAML: ") + e.what(),
        });
    }

    if (!root["surface"].IsDefined() || !root["images"].IsDefined()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Dataset YAML must contain 'surface' and 'images' keys",
        });
    }

    std::string surface_id = root["surface"].as<std::string>();
    auto images_node = root["images"];
    if (!images_node.IsSequence()) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "'images' must be a sequence",
        });
    }

    auto base_dir = yaml_path.parent_path();
    std::vector<DatasetEntry> entries;
    entries.reserve(images_node.size());

    for (auto img : images_node) {
        DatasetEntry entry;
        entry.surface_id = surface_id;
        entry.path = base_dir / img["path"].as<std::string>();
        if (auto pos = img["position"]; pos.IsDefined())
            entry.position_id = pos.as<std::uint16_t>();
        if (auto lgt = img["light"]; lgt.IsDefined())
            entry.light_id = lgt.as<std::uint16_t>();
        if (auto exp = img["expected"]; exp.IsDefined())
            entry.expected_verdict = exp.as<std::string>();
        entries.push_back(std::move(entry));
    }

    return entries;
}

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

    // 3. Detect format: .tif/.tiff uses minimal TIFF parser, others use stb_image.
    auto ext = path.extension().string();
    bool is_tiff = (ext == ".tif" || ext == ".tiff");

    // 4. Try 16-bit decode first for PNG (preserves industrial camera bit depth).
    //    stb_image can load 16-bit PNG; fall back to 8-bit if not needed.
    int width = 0, height = 0, channels_in_file = 0;
    bool is_16bit = false;
    stbi_us* pixels_16 = nullptr;
    std::uint8_t* pixels = nullptr;

    if (!is_tiff) {
        // Probe: check if file is 16-bit by testing stbi_is_16_bit_from_memory
        if (stbi_is_16_bit_from_memory(file_buffer.data(),
                                        static_cast<int>(file_buffer.size()))) {
            pixels_16 = stbi_load_16_from_memory(
                file_buffer.data(), static_cast<int>(file_buffer.size()),
                &width, &height, &channels_in_file, 0);
            is_16bit = (pixels_16 != nullptr);
        }
        if (!is_16bit) {
            pixels = stbi_load_from_memory(
                file_buffer.data(), static_cast<int>(file_buffer.size()),
                &width, &height, &channels_in_file, 3);  // STBI_rgb
        }
    }

    // 5. TIFF path: minimal parser for uncompressed grayscale/RGB strips.
    if (is_tiff) {
        // Minimal TIFF parser: II (little-endian) or MM (big-endian) header,
        // IFD walk for ImageWidth(256), ImageLength(257), BitsPerSample(258),
        // StripOffsets(273), StripByteCounts(279), then raw strip data.
        // Supports 8/16-bit grayscale and RGB uncompressed only.
        if (file_buffer.size() < 8) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Io_ImportParseFailed, "TIFF file too small"});
        }
        bool little_endian = (file_buffer[0] == 'I' && file_buffer[1] == 'I');
        bool big_endian = (file_buffer[0] == 'M' && file_buffer[1] == 'M');
        if (!little_endian && !big_endian) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Io_ImportParseFailed, "Not a valid TIFF file"});
        }

        auto read16 = [&](const unsigned char* p) -> int {
            return little_endian ? (p[0] | (p[1] << 8)) : ((p[0] << 8) | p[1]);
        };
        auto read32 = [&](const unsigned char* p) -> std::size_t {
            if (little_endian)
                return p[0] | (static_cast<std::size_t>(p[1]) << 8)
                     | (static_cast<std::size_t>(p[2]) << 16)
                     | (static_cast<std::size_t>(p[3]) << 24);
            else
                return (static_cast<std::size_t>(p[0]) << 24)
                     | (static_cast<std::size_t>(p[1]) << 16)
                     | (static_cast<std::size_t>(p[2]) << 8) | p[3];
        };

        auto ifd_offset = read32(&file_buffer[4]);
        if (ifd_offset + 2 > file_buffer.size()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Io_ImportParseFailed, "TIFF: IFD out of bounds"});
        }

        auto n_entries = read16(&file_buffer[ifd_offset]);
        std::size_t img_w = 0, img_h = 0, bps = 8, spp = 1;
        std::size_t strip_offset = 0, strip_bytes = 0;

        for (int e = 0; e < n_entries; ++e) {
            auto entry_off = ifd_offset + 2 + static_cast<std::size_t>(e) * 12;
            if (entry_off + 12 > file_buffer.size()) break;
            auto tag = read16(&file_buffer[entry_off]);
            auto type = read16(&file_buffer[entry_off + 2]);
            auto count = read32(&file_buffer[entry_off + 4]);
            auto val_off = read32(&file_buffer[entry_off + 8]);

            if (tag == 256) img_w = (type == 3 && count == 1) ? read16(&file_buffer[entry_off + 8]) : val_off;
            if (tag == 257) img_h = (type == 3 && count == 1) ? read16(&file_buffer[entry_off + 8]) : val_off;
            if (tag == 258) bps = (type == 3 && count == 1) ? read16(&file_buffer[entry_off + 8]) : val_off;
            if (tag == 277) spp = (type == 3 && count == 1) ? read16(&file_buffer[entry_off + 8]) : val_off;
            if (tag == 273) strip_offset = val_off;
            if (tag == 279) strip_bytes = val_off;
        }

        if (img_w == 0 || img_h == 0 || strip_offset == 0 || strip_bytes == 0) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Io_ImportParseFailed, "TIFF: missing required tags"});
        }

        width = static_cast<int>(img_w);
        height = static_cast<int>(img_h);
        channels_in_file = static_cast<int>(spp);
        is_16bit = (bps == 16);

        auto data_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height)
                        * static_cast<std::size_t>(channels_in_file) * (is_16bit ? 2u : 1u);
        if (strip_offset + data_size > file_buffer.size()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Io_ImportParseFailed, "TIFF: strip data out of bounds"});
        }

        if (is_16bit) {
            // Allocate and copy as 16-bit
            pixels_16 = static_cast<stbi_us*>(std::malloc(data_size));
            if (pixels_16 == nullptr) {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Io_ImportParseFailed, "TIFF: malloc failed"});
            }
            std::memcpy(pixels_16, &file_buffer[strip_offset], data_size);
        } else {
            pixels = static_cast<std::uint8_t*>(std::malloc(data_size));
            if (pixels == nullptr) {
                return tl::make_unexpected(ErrorInfo{
                    ErrorCode::Io_ImportParseFailed, "TIFF: malloc failed"});
            }
            std::memcpy(pixels, &file_buffer[strip_offset], data_size);
        }
    }

    // 6. Validate decode result
    if (!is_16bit && pixels == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = std::string("Image decode failed: ") + stbi_failure_reason()
                       + " [" + path.string() + "]",
        });
    }
    if (is_16bit && pixels_16 == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "16-bit image decode failed [" + path.string() + "]",
        });
    }
    if (width <= 0 || height <= 0) {
        if (pixels) stbi_image_free(pixels);
        if (pixels_16) std::free(pixels_16);
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Io_ImportParseFailed,
            .message = "Decoded image has invalid dimensions: " + path.string(),
        });
    }

    // 7. Construct RawImage from decoded data.
    sai::image::ImageMeta meta;
    meta.width = static_cast<std::size_t>(width);
    meta.height = static_cast<std::size_t>(height);
    meta.channels = static_cast<std::size_t>(channels_in_file);

    if (is_16bit) {
        meta.bits_per_sample = 16;
        auto total = static_cast<std::size_t>(width) * static_cast<std::size_t>(height)
                    * static_cast<std::size_t>(channels_in_file) * 2;
        std::vector<std::uint8_t> bytes(
            reinterpret_cast<std::uint8_t*>(pixels_16),
            reinterpret_cast<std::uint8_t*>(pixels_16) + total);
        std::free(pixels_16);

        // Determine pixel format
        if (channels_in_file == 1)
            meta.pixel_format = sai::image::PixelFormat::Mono16;
        else if (channels_in_file == 3)
            meta.pixel_format = sai::image::PixelFormat::RGB16;
        else
            meta.pixel_format = sai::image::PixelFormat::Mono16;

        auto raw = sai::image::RawImage::FromOwnedBuffer(std::move(bytes), meta);
        return std::make_unique<sai::image::RawImage>(std::move(raw));
    }

    // 8-bit path
    meta.bits_per_sample = 8;
    auto total = static_cast<std::size_t>(width) * static_cast<std::size_t>(height)
                * static_cast<std::size_t>(channels_in_file);
    std::vector<std::uint8_t> bytes(pixels, pixels + total);
    if (!is_tiff) stbi_image_free(pixels); else std::free(pixels);

    if (channels_in_file == 1)
        meta.pixel_format = sai::image::PixelFormat::Mono8;
    else if (channels_in_file == 3)
        meta.pixel_format = sai::image::PixelFormat::RGB8;
    else
        meta.pixel_format = sai::image::PixelFormat::RGB8;

    auto raw = sai::image::RawImage::FromOwnedBuffer(std::move(bytes), meta);
    return std::make_unique<sai::image::RawImage>(std::move(raw));
}

}  // namespace sai::io
