// genicam_camera.cpp — GenICam camera via Aravis (Linux-gated, CUDA not required)
//
// Compiled only when aravis is found (CMake: find_package(aravis QUIET)).
// Uses libaravis-0.8 for GenICam device discovery, feature control, and
// GigE Vision / USB3 Vision streaming.
//
// Data flow: ArvStream callback → raw pixel buffer → RawImage → FrameCallback

#include <sai/device/genicam_camera.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <source_location>
#include <string_view>
#include <vector>

#include <arv.h>

#include <sai/core/error.h>
#include <sai/image/image.h>

namespace sai::device {

namespace {

// Convert Aravis pixel format string to sai::image::PixelFormat.
auto ArvFormatToPixelFormat(const char* format_str) noexcept -> sai::image::PixelFormat {
    std::string_view fmt(format_str);
    if (fmt == "BayerRG8")        return sai::image::PixelFormat::BayerRG8;
    if (fmt == "BayerRG10")       return sai::image::PixelFormat::BayerRG10;
    if (fmt == "BayerRG12")       return sai::image::PixelFormat::BayerRG12;
    if (fmt == "Mono8")           return sai::image::PixelFormat::Mono8;
    if (fmt == "Mono10")          return sai::image::PixelFormat::Mono10;
    if (fmt == "Mono12")          return sai::image::PixelFormat::Mono12;
    if (fmt == "RGB8")            return sai::image::PixelFormat::RGB8;
    if (fmt == "BGR8")            return sai::image::PixelFormat::BGR8;
    return sai::image::PixelFormat::Undefined;
}

// Stream callback: invoked by Aravis when a new frame buffer is ready.
void StreamCallback(void* user_data, ArvStreamCallbackType type,
                    ArvBuffer* buffer) {
    if (type != ARV_STREAM_CALLBACK_TYPE_BUFFER_DONE || buffer == nullptr) {
        return;
    }

    auto* self = static_cast<GenICamCamera*>(user_data);
    if (!arv_buffer_get_payload_type(buffer)
        || arv_buffer_get_status(buffer) != ARV_BUFFER_STATUS_SUCCESS) {
        return;
    }

    std::size_t size = 0;
    const void* data = arv_buffer_get_data(buffer, &size);
    if (data == nullptr || size == 0) return;

    auto width = static_cast<std::size_t>(arv_buffer_get_image_width(buffer));
    auto height = static_cast<std::size_t>(arv_buffer_get_image_height(buffer));
    if (width == 0 || height == 0) return;

    // Determine pixel format from buffer
    auto pf_str = arv_buffer_get_image_pixel_format(buffer);
    auto pf = ArvFormatToPixelFormat(pf_str != nullptr ? pf_str : "Mono8");

    // Construct RawImage from the frame buffer
    sai::image::ImageMeta meta;
    meta.width = width;
    meta.height = height;
    meta.channels = (pf == sai::image::PixelFormat::RGB8
                     || pf == sai::image::PixelFormat::BGR8) ? 3u : 1u;
    meta.pixel_format = pf;

    // Copy frame data into an owned buffer.
    // For zero-copy, we could pass the Aravis buffer pointer directly,
    // but RawImage::FromBuffer is non-owning and the Aravis buffer is
    // recycled after this callback returns. A future optimization could
    // use ArvBuffer ref-counting to extend buffer lifetime.
    std::vector<std::uint8_t> owned_copy(static_cast<const std::uint8_t*>(data),
                                         static_cast<const std::uint8_t*>(data) + size);
    auto frame = sai::image::RawImage::FromOwnedBuffer(std::move(owned_copy), meta);

    // Deliver frame to the registered callback.
    // Note: ICamera::FrameCallback takes RawImage by value (move).
    auto& cb = self->GetFrameCallback();
    if (cb) {
        cb(std::move(frame));
    }
}

}  // anonymous namespace

// ── GenICamCamera construction / destruction ─────────────────────────

GenICamCamera::GenICamCamera(Config cfg) : cfg_(std::move(cfg)) {
    serial_number_ = cfg_.serial_number.empty() ? cfg_.user_id : cfg_.serial_number;
}

GenICamCamera::~GenICamCamera() {
    if (acquiring_.load()) {
        (void)StopAcquisition();
    }
    if (state_.load() != State::Disconnected) {
        (void)Disconnect();
    }
}

// ── Connect / Disconnect ─────────────────────────────────────────────

auto GenICamCamera::Connect() noexcept -> Result<void> {
    if (state_.load() != State::Disconnected) {
        return {};  // Already connected
    }

    // Initialize Aravis
    arv_enable_interface("GigEVision");
    arv_enable_interface("USB3Vision");

    // Discover camera
    ArvCamera* cam = nullptr;
    GError* error = nullptr;

    if (!cfg_.serial_number.empty()) {
        cam = arv_camera_new(cfg_.serial_number.c_str(), &error);
    } else if (!cfg_.user_id.empty()) {
        // Search by user_id: iterate devices and match
        arv_update_device_list();
        auto n_devices = arv_get_n_devices();
        for (int i = 0; i < n_devices && cam == nullptr; ++i) {
            auto* device = arv_get_device(i);
            if (device == nullptr) continue;
            const char* uid = arv_get_device_string_feature_value(device, "DeviceUserID", &error);
            if (uid != nullptr && cfg_.user_id == uid) {
                cam = arv_camera_new_with_device(device, &error);
            }
            if (error != nullptr) {
                g_clear_error(&error);
                error = nullptr;
            }
        }
    }

    if (cam == nullptr) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "GenICamCamera: device not found (serial=" + serial_number_ + ")",
            std::source_location::current(),
        });
    }

    arv_camera_ = cam;

    // Configure stream
    auto* device = arv_camera_get_device(cam);
    if (device == nullptr) {
        g_object_unref(cam);
        arv_camera_ = nullptr;
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "GenICamCamera: cannot get device handle",
            std::source_location::current(),
        });
    }

    arv_device_set_string_feature_value(device, "GevSCPSPacketSize",
                                        std::to_string(cfg_.packet_size).c_str(), &error);
    if (error) { g_clear_error(&error); error = nullptr; }

    arv_device_set_integer_feature_value(device, "GevSCPD",
                                         static_cast<gint64>(cfg_.packet_delay), &error);
    if (error) { g_clear_error(&error); error = nullptr; }

    state_.store(State::Connected);
    return {};
}

auto GenICamCamera::Disconnect() noexcept -> Result<void> {
    if (state_.load() == State::Disconnected) return {};

    if (acquiring_.load()) {
        auto stop_result = StopAcquisition();
        if (!stop_result) return stop_result;
    }

    if (arv_stream_ != nullptr) {
        g_object_unref(static_cast<ArvStream*>(arv_stream_));
        arv_stream_ = nullptr;
    }

    if (arv_camera_ != nullptr) {
        g_object_unref(static_cast<ArvCamera*>(arv_camera_));
        arv_camera_ = nullptr;
    }

    state_.store(State::Disconnected);
    return {};
}

auto GenICamCamera::IsConnected() const noexcept -> bool {
    auto s = state_.load();
    return s == State::Connected || s == State::Acquiring;
}

auto GenICamCamera::SerialNumber() const noexcept -> std::string_view {
    return serial_number_;
}

auto GenICamCamera::CurrentState() const noexcept -> State {
    return state_.load();
}

// ── Acquisition control ──────────────────────────────────────────────

auto GenICamCamera::StartAcquisition() noexcept -> Result<void> {
    if (state_.load() != State::Connected) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_NotConnected,
            "GenICamCamera: not connected"});
    }
    if (acquiring_.exchange(true)) return {};  // Already acquiring

    auto* cam = static_cast<ArvCamera*>(arv_camera_);
    if (cam == nullptr) {
        acquiring_.store(false);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_NotConnected,
            "GenICamCamera: null camera handle"});
    }

    GError* error = nullptr;

    // Create stream with frame retention buffer
    auto* stream = arv_camera_create_stream(cam, StreamCallback, this, &error);
    if (stream == nullptr || error != nullptr) {
        acquiring_.store(false);
        std::string err_msg = error ? error->message : "unknown";
        if (error) g_clear_error(&error);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "GenICamCamera: failed to create stream: " + err_msg,
            std::source_location::current(),
        });
    }
    arv_stream_ = stream;

    // Configure stream buffering
    g_object_set(stream,
                 "buffer-count", static_cast<gint>(cfg_.frame_retention),
                 nullptr);

    // Push buffers and start acquisition
    arv_stream_push_buffer(stream, nullptr);  // Let Aravis allocate
    arv_camera_start_acquisition(cam, &error);
    if (error != nullptr) {
        std::string err_msg = error->message;
        g_clear_error(&error);
        g_object_unref(stream);
        arv_stream_ = nullptr;
        acquiring_.store(false);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "GenICamCamera: start acquisition failed: " + err_msg,
            std::source_location::current(),
        });
    }

    state_.store(State::Acquiring);
    return {};
}

auto GenICamCamera::StopAcquisition() noexcept -> Result<void> {
    if (!acquiring_.exchange(false)) return {};

    auto* cam = static_cast<ArvCamera*>(arv_camera_);
    if (cam != nullptr) {
        GError* error = nullptr;
        arv_camera_stop_acquisition(cam, &error);
        if (error) g_clear_error(&error);
    }

    if (state_.load() == State::Acquiring) {
        state_.store(State::Connected);
    }
    return {};
}

auto GenICamCamera::OnStop(Context&) -> Result<void> {
    return Disconnect();
}

// ── Frame callback ───────────────────────────────────────────────────

auto GenICamCamera::RegisterFrameCallback(FrameCallback callback) noexcept
    -> Result<void> {
    callback_ = std::move(callback);
    return {};
}

// GenICamCamera exposes the callback for use by the stream callback.
// This is an internal accessor used by the static StreamCallback function.
auto GenICamCamera::GetFrameCallback() noexcept -> FrameCallback& {
    return callback_;
}

// ── GenICam feature control ──────────────────────────────────────────

auto GenICamCamera::SetTriggerMode(TriggerMode mode) noexcept -> Result<void> {
    auto* cam = static_cast<ArvCamera*>(arv_camera_);
    if (cam == nullptr) return {};

    GError* error = nullptr;
    const char* mode_str = nullptr;
    switch (mode) {
        case TriggerMode::FreeRun:   mode_str = "Off";         break;
        case TriggerMode::Software:  mode_str = "Software";    break;
        case TriggerMode::Hardware:  mode_str = "Line1";       break;
    }
    if (mode_str != nullptr) {
        arv_camera_set_trigger(cam, mode_str, &error);
        if (error) { g_clear_error(&error); }
    }
    cfg_.trigger_mode = mode;
    return {};
}

auto GenICamCamera::SetExposureTime(std::chrono::microseconds us) noexcept
    -> Result<void> {
    auto* cam = static_cast<ArvCamera*>(arv_camera_);
    if (cam == nullptr) return {};

    GError* error = nullptr;
    arv_camera_set_exposure_time(cam, static_cast<double>(us.count()), &error);
    if (error) {
        std::string msg = error->message;
        g_clear_error(&error);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "GenICamCamera: set exposure failed: " + msg,
            std::source_location::current(),
        });
    }
    return {};
}

auto GenICamCamera::SetGain(float db) noexcept -> Result<void> {
    auto* cam = static_cast<ArvCamera*>(arv_camera_);
    if (cam == nullptr) return {};

    GError* error = nullptr;
    arv_camera_set_gain(cam, static_cast<double>(db), &error);
    if (error) {
        std::string msg = error->message;
        g_clear_error(&error);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Device_ConnectionFailed,
            "GenICamCamera: set gain failed: " + msg,
            std::source_location::current(),
        });
    }
    return {};
}

auto GenICamCamera::SetROI(Rect /*region*/) noexcept -> Result<void> {
    // ROI-based sub-windowing requires GenICam feature writes to
    // Width/Height/OffsetX/OffsetY. Skipped for now — full-sensor readout.
    return {};
}

}  // namespace sai::device
