// stage_node.h — Pipeline stage interface + type-erased StageData
// Batch T3 refactor: replaced std::variant<7 types> with type-erased StageData
// to break compile-time coupling.  Public headers no longer pull in
// embedding/detection/rule/reasoner modules.
#pragma once

#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/object.h>
#include <sai/core/context.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>
#include <sai/pipeline/pipeline_config.h>

// Forward declarations — StageData only needs complete types at the
// point of Make<T>() / GetIf<T>() instantiation, which happens in .cpp files.
namespace sai::embedding { class Embedding; }
namespace sai::detection { struct DetectionResult; }
namespace sai::reasoner { struct ReasoningResult; }
namespace sai::pipeline { struct RuleEvalOutput; }

namespace sai::pipeline {

// ── Type tag constants (mapped from StageType → data flow ordering) ──────
inline constexpr int kStageData_RawImage        = 0;
inline constexpr int kStageData_SurfaceImage    = 1;
inline constexpr int kStageData_Embedding       = 2;
inline constexpr int kStageData_DetectionResult = 3;
inline constexpr int kStageData_RuleEvalOutput  = 4;
inline constexpr int kStageData_ReasoningResult = 6;
inline constexpr int kStageData_PipelineFailure = 7;

using FrameImageSnapshot = std::pair<std::vector<std::uint8_t>, sai::image::ImageMeta>;

struct FrameAnomalySnapshot {
    std::vector<float> scores;
    std::size_t grid_h = 0;
    std::size_t grid_w = 0;
};

struct FrameContext {
    std::uint64_t frame_id = 0;
    std::string surface_id;
    std::uint16_t position_id = 0;
    std::optional<FrameImageSnapshot> image;
    std::optional<FrameAnomalySnapshot> anomaly;
};

// A stage failure is data, not a dropped frame.  It travels to Export so the
// frame can become RECHECK instead of silently disappearing or becoming OK.
struct PipelineFailure {
    sai::ErrorCode code = sai::ErrorCode::Core_Unknown;
    std::string stage_id;
    std::string message;
    std::string surface_id;
    std::uint16_t position_id = 0;
};

// ── StageData: type-erased pipeline message ───────────────────────────────
//
// Replaces std::variant<RawImage, SurfaceImage, Embedding, DetectionResult,
//   RuleEvalOutput, ReasoningResult>.
//
// Stores a heap-allocated concrete value + a type tag.  Move-only.
// Make<T>(val) constructs; GetIf<T>() provides type-safe access (returns
// nullptr on type mismatch).
//
// Template methods are defined inline below — they are only instantiated
// in the .cpp files that call them, where the concrete type T is complete.
class StageData {
public:
    StageData() = default;

    // Move-only
    StageData(StageData&& other) noexcept
        : ptr_(other.ptr_), deleter_(other.deleter_), type_index_(other.type_index_),
          frame_(std::move(other.frame_)) {
        other.ptr_ = nullptr;
        other.deleter_ = nullptr;
        other.type_index_ = -1;
    }

    StageData& operator=(StageData&& other) noexcept {
        if (this != &other) {
            Destroy();
            ptr_ = other.ptr_;
            deleter_ = other.deleter_;
            type_index_ = other.type_index_;
            frame_ = std::move(other.frame_);
            other.ptr_ = nullptr;
            other.deleter_ = nullptr;
            other.type_index_ = -1;
        }
        return *this;
    }

    StageData(const StageData&) = delete;
    StageData& operator=(const StageData&) = delete;

    ~StageData() { Destroy(); }

    // ── Type-safe accessors ───────────────────────────────────────────

    template<typename T>
    T* GetIf() {
        if (TypeTag<T>() != type_index_) return nullptr;
        return static_cast<T*>(ptr_);
    }

    template<typename T>
    const T* GetIf() const {
        if (TypeTag<T>() != type_index_) return nullptr;
        return static_cast<const T*>(ptr_);
    }

    int TypeIndex() const { return type_index_; }
    [[nodiscard]] auto Frame() const noexcept -> const std::shared_ptr<FrameContext>& {
        return frame_;
    }
    [[nodiscard]] auto MutableFrame() noexcept -> std::shared_ptr<FrameContext>& {
        return frame_;
    }
    auto AttachFrame(std::shared_ptr<FrameContext> frame) noexcept -> void {
        frame_ = std::move(frame);
    }

    // ── Factory ───────────────────────────────────────────────────────

    template<typename T>
    static StageData Make(T value) {
        StageData sd;
        sd.ptr_ = new T(std::move(value));
        sd.deleter_ = [](void* p) { delete static_cast<T*>(p); };
        sd.type_index_ = TypeTag<T>();
        return sd;
    }

    template<typename T>
    static StageData MakeWithContext(const StageData& source, T value) {
        auto sd = Make(std::move(value));
        sd.frame_ = source.frame_;
        return sd;
    }

    template<typename T>
    static StageData MakeWithContext(std::shared_ptr<FrameContext> frame, T value) {
        auto sd = Make(std::move(value));
        sd.frame_ = std::move(frame);
        return sd;
    }

private:
    void Destroy() {
        if (ptr_ && deleter_) {
            deleter_(ptr_);
            ptr_ = nullptr;
            deleter_ = nullptr;
            type_index_ = -1;
        }
    }

    // Compile-time type → tag mapping (specialized below for known types).
    // Primary template returns -1 (never matches any valid type_index_),
    // so GetIf<UnknownType>() safely returns nullptr at runtime.
    template<typename T> static int TypeTag() { return -1; }

    void* ptr_ = nullptr;
    void (*deleter_)(void*) = nullptr;
    int type_index_ = -1;
    std::shared_ptr<FrameContext> frame_;
};

// ── TypeTag specializations ──────────────────────────────────────────────
template<> inline int StageData::TypeTag<sai::image::RawImage>()             { return kStageData_RawImage; }
template<> inline int StageData::TypeTag<sai::image::SurfaceImage>()         { return kStageData_SurfaceImage; }
template<> inline int StageData::TypeTag<sai::embedding::Embedding>()        { return kStageData_Embedding; }
template<> inline int StageData::TypeTag<sai::detection::DetectionResult>()  { return kStageData_DetectionResult; }
template<> inline int StageData::TypeTag<sai::pipeline::RuleEvalOutput>()    { return kStageData_RuleEvalOutput; }
template<> inline int StageData::TypeTag<sai::reasoner::ReasoningResult>()   { return kStageData_ReasoningResult; }
template<> inline int StageData::TypeTag<sai::pipeline::PipelineFailure>()   { return kStageData_PipelineFailure; }

// ── Public type aliases ──────────────────────────────────────────────────
using StageInput = StageData;
using StageOutput = StageData;

// ── IStageNode ───────────────────────────────────────────────────────────
class IStageNode : public Object {
public:
    virtual auto GetType() const noexcept -> StageType = 0;
    virtual auto GetId() const -> std::string_view = 0;
    virtual auto OnInitialize(Context&) -> Result<void> = 0;
    virtual auto OnStart(Context&) -> Result<void> = 0;
    virtual auto OnStop(Context&) -> Result<void> = 0;
    virtual auto Process(StageInput) -> Result<StageOutput> = 0;

    // M7: optional hot-reload of stage parameters. Default: no-op.
    virtual auto ReloadConfig(const YAML::Node& /*config*/) -> Result<void> {
        return {};
    }
};

}  // namespace sai::pipeline
