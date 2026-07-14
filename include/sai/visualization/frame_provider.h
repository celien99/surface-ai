#pragma once

#include <QQuickImageProvider>
#include <QImage>
#include <array>
#include <shared_mutex>
#include <atomic>

namespace sai::image {
class SurfaceImage;
class RawImage;
}  // namespace sai::image

namespace sai::visualization {

/// QQuickImageProvider that serves SurfaceImage/RawImage frames to QML.
/// Ring-buffer caches the last 16 frames as QImage snapshots.
/// Thread-safe: RegisterFrame (WorkerPool write) ↔ requestImage (QML main thread read).
class FrameProvider : public QQuickImageProvider {
public:
    explicit FrameProvider();

    /// QQuickImageProvider override — called by QML Image component.
    auto requestImage(const QString& id, QSize* size,
                      const QSize& requestedSize) -> QImage override;

    /// Register a processed frame (called from Export worker thread).
    void RegisterFrame(int frame_id, const sai::image::SurfaceImage& image);

    /// Register a raw frame (called from Capture worker thread).
    void RegisterRawFrame(int frame_id, const sai::image::RawImage& image);

private:
    static constexpr int kCacheSize = 16;

    struct CachedFrame {
        int frame_id{0};
        QImage image;
    };

    /// Copy pixel data from a SurfaceImage/RawImage into a QImage (RGBA8888).
    static auto CopyToQImage(const sai::image::SurfaceImage& src) -> QImage;
    static auto CopyToQImage(const sai::image::RawImage& src) -> QImage;

    std::array<CachedFrame, kCacheSize> cache_;
    std::atomic<int> latest_frame_id_{0};
    mutable std::shared_mutex cache_mutex_;
};

}  // namespace sai::visualization
