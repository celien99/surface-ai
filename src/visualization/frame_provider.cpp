#include "sai/visualization/frame_provider.h"

#include "sai/image/image.h"
#include "sai/image/surface_image.h"
#include "sai/image/raw_image.h"

#include <QSize>

#include <mutex>
#include <shared_mutex>
#include <cstdint>

namespace sai::visualization {

namespace {

/// Convert any Image subclass pixel data to a QImage in RGBA8888 format.
/// Handles all PixelFormat variants with appropriate channel expansion.
auto CopyPixelsToQImage(const sai::image::Image& src) -> QImage {
    const auto& meta = src.Meta();
    const int w = static_cast<int>(meta.width);
    const int h = static_cast<int>(meta.height);

    if (w <= 0 || h <= 0) {
        return QImage();
    }

    const auto* src_data = src.Data();
    if (src_data == nullptr) {
        return QImage();
    }

    QImage result(w, h, QImage::Format_RGBA8888);
    auto* dst = result.bits();
    const int dst_stride = result.bytesPerLine();

    switch (meta.pixel_format) {
        case sai::image::PixelFormat::Mono8:
        case sai::image::PixelFormat::BayerRG8: {
            // Single-channel grayscale (1 Bpp) → RGBA8888
            for (int y = 0; y < h; ++y) {
                auto* row = dst + y * dst_stride;
                for (int x = 0; x < w; ++x) {
                    const std::uint8_t v = src_data[y * w + x];
                    row[x * 4 + 0] = v;
                    row[x * 4 + 1] = v;
                    row[x * 4 + 2] = v;
                    row[x * 4 + 3] = 255;
                }
            }
            break;
        }

        case sai::image::PixelFormat::Mono10:
        case sai::image::PixelFormat::Mono12:
        case sai::image::PixelFormat::BayerRG10:
        case sai::image::PixelFormat::BayerRG12: {
            // Two bytes per pixel grayscale → RGBA8888
            const int shift = (meta.pixel_format == sai::image::PixelFormat::Mono10 ||
                               meta.pixel_format == sai::image::PixelFormat::BayerRG10)
                                  ? 2
                                  : 4;
            for (int y = 0; y < h; ++y) {
                auto* row = dst + y * dst_stride;
                for (int x = 0; x < w; ++x) {
                    const std::uint16_t val =
                        static_cast<std::uint16_t>(src_data[(y * w + x) * 2]) |
                        (static_cast<std::uint16_t>(src_data[(y * w + x) * 2 + 1]) << 8);
                    const std::uint8_t v = static_cast<std::uint8_t>(val >> shift);
                    row[x * 4 + 0] = v;
                    row[x * 4 + 1] = v;
                    row[x * 4 + 2] = v;
                    row[x * 4 + 3] = 255;
                }
            }
            break;
        }

        case sai::image::PixelFormat::RGB8: {
            // 24-bit RGB → RGBA8888
            for (int y = 0; y < h; ++y) {
                auto* row = dst + y * dst_stride;
                for (int x = 0; x < w; ++x) {
                    const auto* src_px = src_data + (y * w + x) * 3;
                    row[x * 4 + 0] = src_px[0];
                    row[x * 4 + 1] = src_px[1];
                    row[x * 4 + 2] = src_px[2];
                    row[x * 4 + 3] = 255;
                }
            }
            break;
        }

        case sai::image::PixelFormat::BGR8: {
            // 24-bit BGR → RGBA8888 (swap R and B)
            for (int y = 0; y < h; ++y) {
                auto* row = dst + y * dst_stride;
                for (int x = 0; x < w; ++x) {
                    const auto* src_px = src_data + (y * w + x) * 3;
                    row[x * 4 + 0] = src_px[2];
                    row[x * 4 + 1] = src_px[1];
                    row[x * 4 + 2] = src_px[0];
                    row[x * 4 + 3] = 255;
                }
            }
            break;
        }

        case sai::image::PixelFormat::Undefined:
        default:
            return QImage();
    }

    return result;
}

}  // anonymous namespace

FrameProvider::FrameProvider()
    : QQuickImageProvider(QQuickImageProvider::Image) {
    latest_frame_id_.store(0);
}

auto FrameProvider::CopyToQImage(const sai::image::SurfaceImage& src) -> QImage {
    return CopyPixelsToQImage(src);
}

auto FrameProvider::CopyToQImage(const sai::image::RawImage& src) -> QImage {
    return CopyPixelsToQImage(src);
}

void FrameProvider::RegisterFrame(int frame_id, const sai::image::SurfaceImage& image) {
    std::unique_lock lock(cache_mutex_);
    const int slot = frame_id % kCacheSize;
    cache_[slot].frame_id = frame_id;
    cache_[slot].image = CopyToQImage(image);
    latest_frame_id_.store(frame_id);
}

void FrameProvider::RegisterRawFrame(int frame_id, const sai::image::RawImage& image) {
    std::unique_lock lock(cache_mutex_);
    const int slot = frame_id % kCacheSize;
    cache_[slot].frame_id = frame_id;
    cache_[slot].image = CopyToQImage(image);
    latest_frame_id_.store(frame_id);
}

auto FrameProvider::requestImage(const QString& id, QSize* size,
                                  const QSize& requestedSize) -> QImage {
    // Expected id format: "frame?t=<frame_id>"
    if (!id.startsWith(QStringLiteral("frame?t="))) {
        return QImage();
    }

    const QStringView num_str = QStringView(id).mid(8);
    bool ok = false;
    const int frame_id = num_str.toInt(&ok);
    if (!ok) {
        return QImage();
    }

    // ── Layer 1: in-memory ring-buffer cache (live mode) ──
    {
        std::shared_lock lock(cache_mutex_);
        const int slot = frame_id % kCacheSize;
        const auto& entry = cache_[slot];
        if (entry.frame_id == frame_id) {
            if (size) *size = entry.image.size();
            if (requestedSize.isValid() && requestedSize != entry.image.size())
                return entry.image.scaled(requestedSize);
            return entry.image;
        }
    }

    // ── Layer 2: disk-backed lazy load (review mode) ──
    {
        std::shared_lock lock(path_mutex_);
        auto it = frame_paths_.find(frame_id);
        if (it != frame_paths_.end()) {
            QImage img(it->second);
            if (!img.isNull()) {
                if (size) *size = img.size();
                // Populate cache for subsequent requests
                {
                    std::unique_lock cache_lock(cache_mutex_);
                    const int slot = frame_id % kCacheSize;
                    cache_[slot].frame_id = frame_id;
                    cache_[slot].image = img;
                    latest_frame_id_.store(frame_id);
                }
                if (requestedSize.isValid() && requestedSize != img.size())
                    return img.scaled(requestedSize);
                return img;
            }
        }
    }

    return QImage();
}

void FrameProvider::RegisterFramePath(int frame_id, const QString& image_path) {
    std::unique_lock lock(path_mutex_);
    frame_paths_[frame_id] = image_path;
}

void FrameProvider::LoadFromReviewIndex(const QString& review_dir) {
    // JSON parsing is handled by the caller (gui_runner.cpp) because
    // the visualization library does not depend on nlohmann_json.
    // The caller iterates frames and calls RegisterFramePath() directly.
    // This method is a reserved convenience entry point for future use.
    (void)review_dir;
}

}  // namespace sai::visualization
