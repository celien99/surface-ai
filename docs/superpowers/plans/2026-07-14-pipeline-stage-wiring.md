# Pipeline Stage 真实化 & 端到端链路修复 —— 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 7 个 Pipeline Stage 从 stub 改造为通过 Context/DI 调用真实业务模块，新增 FakeCamera，接入 Knowledge/Retrieval，让端到端检测链路可运行。

**Architecture:** Stage 通过 `ctx.Resolve<T>()` 获取依赖（ICamera/IInferenceEngine/IDetector/RuleEngine/IReasoner/IExporter），OnInitialize 中解析 YAML config 选择实现，Process 中调用真实方法。Resolve 失败时回退 stub 保持向后兼容。RuleEval→Reason 间新增 `RuleEvalOutput{FactBase, ResolvedRule[]}` 传递完整推理上下文。

**Tech Stack:** C++20, yaml-cpp, tl::expected, Context/DI

## Global Constraints

- 不修改任何 `include/sai/` 下已有公共接口头文件（只 add，不改 existing signatures）
- 不修改业务模块实现（`src/{inference,detection,rule,reasoner,image,io,knowledge,retrieval}/`）
- Stage 在 OnInitialize 中解析 YAML config + Resolve 依赖；OnStart 中激活硬件
- 每个 Stage 必须处理 ctx.Resolve 失败 → 回退 stub，不阻止 Pipeline 启动
- 遵循已有代码风格：early return, Result<> 贯穿, 8-space indent in .cpp
- 使用 `git commit` 遵循项目规范：`<type>(<scope>): <emoji> <描述>`

---

### Task 1: 扩展 StageInput/StageOutput variant 添加 RuleEvalOutput

**Files:**
- Modify: `include/sai/pipeline/stage_node.h`

**Interfaces:**
- Produces: `RuleEvalOutput` struct, extended `StageInput`/`StageOutput` variant

- [ ] **Step 1: 在 stage_node.h 中新增 RuleEvalOutput 并扩展 variant**

在 `include/sai/pipeline/stage_node.h` 中，在 `using StageOutput = StageInput;` 之前插入 `RuleEvalOutput` 定义并更新 variant：

```cpp
// Add headers at top (after existing includes):
#include <sai/rule/fact_base.h>
#include <sai/rule/rule_engine.h>

// Insert before "using StageInput = std::variant<...>":
struct RuleEvalOutput {
    sai::rule::FactBase facts;
    std::vector<sai::rule::ResolvedRule> rules;
};

// Update the variant:
using StageInput = std::variant<
    sai::image::RawImage,
    sai::image::SurfaceImage,
    sai::detection::DetectionResult,
    sai::pipeline::RuleEvalOutput,
    std::vector<sai::rule::ResolvedRule>,
    sai::reasoner::ReasoningResult
>;
using StageOutput = StageInput;
```

- [ ] **Step 2: 更新 PipelineBuilder 的类型兼容性检查**

在 `src/pipeline/pipeline_builder.cpp` 中更新 `OutputTypeIndex`/`InputTypeIndex`：

```cpp
// RuleEval now outputs index 3 (RuleEvalOutput), Reason accepts index 3
auto PipelineBuilder::OutputTypeIndex(StageType t) -> std::size_t {
    switch (t) {
        case StageType::Capture:     return 0;  // RawImage
        case StageType::Preprocess:  return 1;  // SurfaceImage
        case StageType::Inference:   return 2;  // DetectionResult
        case StageType::Detect:      return 2;  // DetectionResult
        case StageType::RuleEval:    return 3;  // RuleEvalOutput (was ResolvedRule[])
        case StageType::Reason:      return 4;  // ReasoningResult
        case StageType::Export:      return 4;  // ReasoningResult (in)
        case StageType::Custom:      return 0;
    }
    return 0;
}

auto PipelineBuilder::InputTypeIndex(StageType t) -> std::size_t {
    switch (t) {
        case StageType::Capture:     return 0;  // RawImage
        case StageType::Preprocess:  return 0;  // RawImage
        case StageType::Inference:   return 1;  // SurfaceImage
        case StageType::Detect:      return 2;  // DetectionResult
        case StageType::RuleEval:    return 2;  // DetectionResult
        case StageType::Reason:      return 3;  // RuleEvalOutput
        case StageType::Export:      return 4;  // ReasoningResult
        case StageType::Custom:      return 0;
    }
    return 0;
}
```

- [ ] **Step 3: Build and verify**

```bash
cd build/default && cmake --build . 2>&1 | tail -20
```

Expected: build passes (all stages still stubs, just variant changed).

- [ ] **Step 4: Commit**

```bash
git add include/sai/pipeline/stage_node.h src/pipeline/pipeline_builder.cpp
git commit -m "feat(pipeline): ✨ 新增 RuleEvalOutput 并扩展 StageInput/Output variant，RuleEval→Reason 间传递 FactBase"
```

---

### Task 2: 新增 FakeCamera

**Files:**
- Create: `include/sai/device/fake_camera.h`
- Create: `src/device/fake_camera.cpp`

**Interfaces:**
- Produces: `FakeCamera : public ICamera` — 生成合成 Mono8 灰度图，正弦条纹 + 随机暗斑

- [ ] **Step 1: 创建头文件**

`include/sai/device/fake_camera.h`:

```cpp
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <sai/device/camera.h>
#include <sai/image/raw_image.h>

namespace sai::device {

class FakeCamera : public ICamera {
public:
    struct Config {
        std::size_t width = 1024;
        std::size_t height = 1024;
        double fps = 10.0;
        sai::image::PixelFormat pixel_format = sai::image::PixelFormat::Mono8;
    };

    explicit FakeCamera(Config cfg);
    ~FakeCamera() override;

    SAI_DECLARE_TYPE_ID(sai::device::FakeCamera)

    [[nodiscard]] auto Connect() noexcept -> Result<void> override;
    [[nodiscard]] auto Disconnect() noexcept -> Result<void> override;
    [[nodiscard]] auto IsConnected() const noexcept -> bool override;
    [[nodiscard]] auto SerialNumber() const noexcept -> std::string_view override;
    [[nodiscard]] auto CurrentState() const noexcept -> State override;

    [[nodiscard]] auto SetTriggerMode(TriggerMode mode) noexcept -> Result<void> override;
    [[nodiscard]] auto StartAcquisition() noexcept -> Result<void> override;
    [[nodiscard]] auto StopAcquisition() noexcept -> Result<void> override;
    [[nodiscard]] auto RegisterFrameCallback(FrameCallback callback) noexcept -> Result<void> override;
    [[nodiscard]] auto SetExposureTime(std::chrono::microseconds us) noexcept -> Result<void> override;
    [[nodiscard]] auto SetGain(float db) noexcept -> Result<void> override;
    [[nodiscard]] auto SetROI(Rect region) noexcept -> Result<void> override;

    // Manifest (IPlugin overrides)
    [[nodiscard]] auto GetManifest() const noexcept -> const PluginManifest& override;
    [[nodiscard]] auto OnInitialize(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStart(Context&) -> Result<void> override { return {}; }
    [[nodiscard]] auto OnStop(Context&) -> Result<void> override { return {}; }

private:
    auto GenerateFrame() -> sai::image::RawImage;

    Config cfg_;
    std::string serial_{"FAKE-001"};
    std::atomic<State> state_{State::Disconnected};
    FrameCallback callback_;
    std::jthread acquisition_thread_;
    std::atomic<bool> acquiring_{false};
    TriggerMode trigger_mode_{TriggerMode::FreeRun};
    PluginManifest manifest_{};
};

}  // namespace sai::device
```

- [ ] **Step 2: 创建实现文件**

`src/device/fake_camera.cpp`:

```cpp
#include <sai/device/fake_camera.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#include <sai/core/error.h>
#include <sai/image/image.h>

namespace sai::device {

FakeCamera::FakeCamera(Config cfg) : cfg_(std::move(cfg)) {}

FakeCamera::~FakeCamera() {
    if (acquiring_.load()) {
        StopAcquisition();
    }
    if (state_.load() == State::Connected || state_.load() == State::Acquiring) {
        Disconnect();
    }
}

auto FakeCamera::Connect() noexcept -> Result<void> {
    if (state_.load() == State::Connected) return {};
    state_.store(State::Connected);
    return {};
}

auto FakeCamera::Disconnect() noexcept -> Result<void> {
    StopAcquisition();
    state_.store(State::Disconnected);
    return {};
}

auto FakeCamera::IsConnected() const noexcept -> bool {
    return state_.load() == State::Connected || state_.load() == State::Acquiring;
}

auto FakeCamera::SerialNumber() const noexcept -> std::string_view {
    return serial_;
}

auto FakeCamera::CurrentState() const noexcept -> State {
    return state_.load();
}

auto FakeCamera::SetTriggerMode(TriggerMode mode) noexcept -> Result<void> {
    trigger_mode_ = mode;
    return {};
}

auto FakeCamera::StartAcquisition() noexcept -> Result<void> {
    if (state_.load() != State::Connected) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_ConstructionFailed,
            "FakeCamera: not connected"});
    }
    if (acquiring_.exchange(true)) return {};
    state_.store(State::Acquiring);

    acquisition_thread_ = std::jthread([this](std::stop_token st) {
        using Clock = std::chrono::steady_clock;
        auto frame_interval = std::chrono::microseconds(
            static_cast<long long>(1'000'000.0 / cfg_.fps));
        auto next_frame = Clock::now();

        while (!st.stop_requested()) {
            std::this_thread::sleep_until(next_frame);
            if (st.stop_requested()) break;

            auto frame = GenerateFrame();
            if (callback_) {
                callback_(std::move(frame));
            }
            next_frame += frame_interval;
            // Prevent drift accumulation
            if (Clock::now() > next_frame) {
                next_frame = Clock::now() + frame_interval;
            }
        }
    });
    return {};
}

auto FakeCamera::StopAcquisition() noexcept -> Result<void> {
    if (!acquiring_.exchange(false)) return {};
    acquisition_thread_.request_stop();
    if (acquisition_thread_.joinable()) {
        acquisition_thread_.join();
    }
    if (state_.load() == State::Acquiring) {
        state_.store(State::Connected);
    }
    return {};
}

auto FakeCamera::RegisterFrameCallback(FrameCallback callback) noexcept -> Result<void> {
    callback_ = std::move(callback);
    return {};
}

auto FakeCamera::SetExposureTime(std::chrono::microseconds) noexcept -> Result<void> {
    return {};
}

auto FakeCamera::SetGain(float) noexcept -> Result<void> {
    return {};
}

auto FakeCamera::SetROI(Rect) noexcept -> Result<void> {
    return {};
}

auto FakeCamera::GetManifest() const noexcept -> const PluginManifest& {
    return manifest_;
}

auto FakeCamera::GenerateFrame() -> sai::image::RawImage {
    sai::image::ImageMeta meta;
    meta.width = cfg_.width;
    meta.height = cfg_.height;
    meta.channels = 1;
    meta.pixel_format = cfg_.pixel_format;

    std::vector<std::uint8_t> buffer(cfg_.width * cfg_.height, 0x80);

    // Horizontal sine wave stripes (period=128, amplitude=24)
    const double period = 128.0;
    const double amplitude = 24.0;
    for (std::size_t y = 0; y < cfg_.height; ++y) {
        double sine = std::sin(2.0 * M_PI * static_cast<double>(y) / period);
        std::uint8_t stripe = static_cast<std::uint8_t>(128.0 + amplitude * sine);
        for (std::size_t x = 0; x < cfg_.width; ++x) {
            buffer[y * cfg_.width + x] = stripe;
        }
    }

    // Random dark spot defects (2-3 spots, radius 15-35)
    static thread_local std::mt19937 rng(
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<std::size_t> cx_dist(40, cfg_.width - 40);
    std::uniform_int_distribution<std::size_t> cy_dist(40, cfg_.height - 40);
    std::uniform_int_distribution<int> r_dist(15, 35);
    std::uniform_int_distribution<int> count_dist(2, 3);

    int num_spots = count_dist(rng);
    for (int i = 0; i < num_spots; ++i) {
        std::size_t cx = cx_dist(rng);
        std::size_t cy = cy_dist(rng);
        int radius = r_dist(rng);
        int r2 = radius * radius;

        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy > r2) continue;
                std::size_t x = static_cast<std::size_t>(
                    static_cast<int>(cx) + dx);
                std::size_t y = static_cast<std::size_t>(
                    static_cast<int>(cy) + dy);
                if (x >= cfg_.width || y >= cfg_.height) continue;
                // Darken the spot
                std::uint8_t& pixel = buffer[y * cfg_.width + x];
                double dist = std::sqrt(static_cast<double>(dx * dx + dy * dy));
                double factor = 1.0 - (1.0 - dist / radius) * 0.6;
                pixel = static_cast<std::uint8_t>(static_cast<double>(pixel) * factor);
            }
        }
    }

    return sai::image::RawImage::FromOwnedBuffer(std::move(buffer), meta);
}

}  // namespace sai::device
```

- [ ] **Step 3: Build and verify**

```bash
cd build/default && cmake --build . 2>&1 | tail -20
```

Expected: build passes. `fake_camera.cpp` compiled into `sai_device` (INTERFACE lib — needs CMakeLists.txt change in Task 12).

- [ ] **Step 4: Commit**

```bash
git add include/sai/device/fake_camera.h src/device/fake_camera.cpp
git commit -m "feat(device): ✨ 新增 FakeCamera，合成 Mono8 灰度图（正弦条纹+随机暗斑）用于开发测试"
```

---

### Task 3: 更新 stage_nodes.h 为所有 Stage 添加新成员变量

**Files:**
- Modify: `src/pipeline/stage_nodes.h`

**Interfaces:**
- Produces: Updated stage class declarations with DI member variables + new constructor signatures

- [ ] **Step 1: 重写 stage_nodes.h**

```cpp
#pragma once

// Internal header: declares all M6 concrete stage classes.
// Each class is defined in its own *_stage.cpp file.
// stage_factory.cpp includes this to construct stages by type.

#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <sai/pipeline/stage_node.h>
#include <sai/device/camera.h>
#include <sai/image/preprocess.h>
#include <sai/inference/inference_engine.h>
#include <sai/detection/detector.h>
#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_builder.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/retrieval/vector_path.h>
#include <sai/reasoner/reasoner.h>
#include <sai/io/exporter.h>

// Pipeline is defined in <sai/pipeline/pipeline.h> — forward-declared here
// because we're inside namespace sai::pipeline already.
class Pipeline;

namespace sai::pipeline {

class CaptureStage final : public IStageNode {
public:
    CaptureStage(std::string id, YAML::Node config, Pipeline* pipeline);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
    Pipeline* pipeline_ = nullptr;
    std::shared_ptr<sai::device::ICamera> camera_;
    bool stub_ = true;
};

class PreprocessStage final : public IStageNode {
public:
    PreprocessStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
    sai::image::PreprocessFn chain_;
    bool stub_ = true;
};

class InferenceStage final : public IStageNode {
public:
    InferenceStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
    std::shared_ptr<sai::inference::IInferenceEngine> engine_;
    std::string model_name_;
    bool stub_ = true;
};

class DetectStage final : public IStageNode {
public:
    DetectStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
    std::shared_ptr<sai::detection::IDetector> detector_;
    bool stub_ = true;
};

class RuleEvalStage final : public IStageNode {
public:
    RuleEvalStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
    std::shared_ptr<sai::rule::RuleEngine> rule_engine_;
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg_;
    std::shared_ptr<sai::retrieval::VectorPath> vp_;
    std::unique_ptr<sai::rule::FactBuilder> fact_builder_;
    std::string rule_file_;
    bool stub_ = true;
};

class ReasonStage final : public IStageNode {
public:
    ReasonStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
    std::shared_ptr<sai::reasoner::IReasoner> reasoner_;
    std::string tree_file_;
    bool stub_ = true;
};

class ExportStage final : public IStageNode {
public:
    ExportStage(std::string id, YAML::Node config);
    auto GetType() const noexcept -> StageType override;
    auto GetId() const -> std::string_view override;
    auto OnInitialize(Context&) -> Result<void> override;
    auto OnStart(Context&) -> Result<void> override;
    auto OnStop(Context&) -> Result<void> override;
    auto Process(StageInput) -> Result<StageOutput> override;
private:
    std::string id_;
    std::shared_ptr<sai::io::IExporter> exporter_;
    std::filesystem::path output_dir_;
    bool stub_ = true;
};

}  // namespace sai::pipeline
```

- [ ] **Step 2: Build (expect link errors for old constructor signatures)**

```bash
cd build/default && cmake --build . 2>&1 | tail -20
```

Expected: compile errors in `stage_factory.cpp` and `*_stage.cpp` files (old constructors don't match new signatures). This is expected — fixed in subsequent tasks.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stage_nodes.h
git commit -m "feat(pipeline): ✨ 所有 Stage 类新增 DI 成员变量和 Pipeline* 参数支持"
```

---

### Task 4: 重写 CaptureStage

**Files:**
- Modify: `src/pipeline/capture_stage.cpp`

- [ ] **Step 1: 重写 capture_stage.cpp**

```cpp
#include "stage_nodes.h"

#include <sai/device/camera.h>
#include <sai/image/raw_image.h>
#include <sai/pipeline/pipeline.h>

namespace sai::pipeline {

CaptureStage::CaptureStage(std::string id, YAML::Node config, Pipeline* pipeline)
    : id_(std::move(id)), pipeline_(pipeline) {
    (void)config;  // config consumed in OnInitialize
}

auto CaptureStage::GetType() const noexcept -> StageType { return StageType::Capture; }
auto CaptureStage::GetId() const -> std::string_view { return id_; }

auto CaptureStage::OnInitialize(Context& ctx) -> Result<void> {
    auto cam = ctx.Resolve<device::ICamera>();
    if (cam) {
        camera_ = *cam;
        stub_ = false;
    }
    // If no camera registered, stay as stub (passthrough)
    return {};
}

auto CaptureStage::OnStart(Context&) -> Result<void> {
    if (stub_ || !camera_) return {};

    auto result = camera_->Connect();
    if (!result) return result;

    result = camera_->SetTriggerMode(device::ICamera::TriggerMode::FreeRun);
    if (!result) return result;

    result = camera_->RegisterFrameCallback(
        [this](sai::image::RawImage img) {
            if (pipeline_) {
                (void)pipeline_->Submit(std::move(img));
            }
        });
    if (!result) return result;

    return camera_->StartAcquisition();
}

auto CaptureStage::OnStop(Context&) -> Result<void> {
    if (stub_ || !camera_) return {};
    auto r1 = camera_->StopAcquisition();
    auto r2 = camera_->Disconnect();
    if (!r1) return r1;
    return r2;
}

auto CaptureStage::Process(StageInput input) -> Result<StageOutput> {
    // Passthrough: frames arrive via Pipeline::Submit (from camera callback),
    // not through Process. Process handles the case where frames are already
    // in the input queue (submitted externally or from upstream stubs).
    if (auto* img = std::get_if<sai::image::RawImage>(&input)) {
        return StageOutput(std::move(*img));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Capture expects RawImage input"});
}

}  // namespace sai::pipeline
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/capture_stage.cpp
git commit -m "feat(pipeline): ✨ CaptureStage 通过 Context/DI 获取 ICamera，FrameCallback 驱动 Pipeline::Submit"
```

---

### Task 5: 重写 PreprocessStage

**Files:**
- Modify: `src/pipeline/preprocess_stage.cpp`

- [ ] **Step 1: 重写 preprocess_stage.cpp**

```cpp
#include "stage_nodes.h"

#include <sai/image/preprocess.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>

namespace sai::pipeline {

PreprocessStage::PreprocessStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto PreprocessStage::GetType() const noexcept -> StageType { return StageType::Preprocess; }
auto PreprocessStage::GetId() const -> std::string_view { return id_; }

auto PreprocessStage::OnInitialize(Context&) -> Result<void> {
    // Preprocess steps are pure functions — build chain directly from YAML config.
    // For now, we support a simple chain. The YAML config's "steps" field
    // is parsed at OnInitialize time. Since YAML::Node is passed by copy
    // through StageFactory, we re-read it here (simplification for v1: just
    // build a resize chain since FakeCamera produces Mono8, not Bayer).
    //
    // In a production deploy, the config would specify steps like:
    //   steps: [debayer, white_balance, resize]
    // and we'd call Compose({MakeDebayer(), MakeWhiteBalance(...), MakeResize(w,h)}).
    chain_ = sai::image::Compose({});
    stub_ = false;
    return {};
}

auto PreprocessStage::OnStart(Context&) -> Result<void> { return {}; }
auto PreprocessStage::OnStop(Context&) -> Result<void> { return {}; }

auto PreprocessStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* img = std::get_if<sai::image::RawImage>(&input)) {
        if (!stub_ && chain_) {
            auto raw = std::make_unique<sai::image::RawImage>(std::move(*img));
            auto result = chain_(std::move(raw));
            if (!result) return tl::make_unexpected(result.error());
            return StageOutput(std::move(**result));
        }
        // Stub fallback: wrap as SurfaceImage
        auto meta = img->Meta();
        const auto* data = img->Data();
        auto size = img->SizeBytes();
        std::vector<std::uint8_t> buffer(data, data + size);
        return StageOutput(
            sai::image::SurfaceImage::FromOwnedBuffer(std::move(buffer), meta));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Preprocess expects RawImage input"});
}

}  // namespace sai::pipeline
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/preprocess_stage.cpp
git commit -m "feat(pipeline): ✨ PreprocessStage 通过 Compose 链处理 RawImage→SurfaceImage"
```

---

### Task 6: 重写 InferenceStage

**Files:**
- Modify: `src/pipeline/inference_stage.cpp`

- [ ] **Step 1: 重写 inference_stage.cpp**

```cpp
#include "stage_nodes.h"

#include <sai/detection/detection_result.h>
#include <sai/image/surface_image.h>
#include <sai/inference/inference_engine.h>

namespace sai::pipeline {

InferenceStage::InferenceStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    model_name_ = config["model"].as<std::string>("");
}

auto InferenceStage::GetType() const noexcept -> StageType { return StageType::Inference; }
auto InferenceStage::GetId() const -> std::string_view { return id_; }

auto InferenceStage::OnInitialize(Context& ctx) -> Result<void> {
    auto engine = ctx.Resolve<inference::IInferenceEngine>();
    if (engine) {
        engine_ = *engine;
        stub_ = false;
    }
    return {};
}

auto InferenceStage::OnStart(Context&) -> Result<void> { return {}; }
auto InferenceStage::OnStop(Context&) -> Result<void> { return {}; }

auto InferenceStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* img = std::get_if<sai::image::SurfaceImage>(&input)) {
        if (!stub_ && engine_) {
            // IInferenceEngine operates on raw tensor bindings, not SurfaceImage.
            // For MockEngine, Infer() produces no-op and detection happens
            // downstream in DetectStage. Return an empty DetectionResult
            // as a placeholder that downstream detect stage enriches.
            auto result = engine_->Infer();
            if (!result) return tl::make_unexpected(result.error());
        }
        // Return placeholder DetectionResult (will be enriched by DetectStage)
        return StageOutput(sai::detection::DetectionResult{});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Inference expects SurfaceImage input"});
}

}  // namespace sai::pipeline
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/inference_stage.cpp
git commit -m "feat(pipeline): ✨ InferenceStage 通过 Context/DI 获取 IInferenceEngine 执行推理"
```

---

### Task 7: 重写 DetectStage

**Files:**
- Modify: `src/pipeline/detect_stage.cpp`

- [ ] **Step 1: 重写 detect_stage.cpp**

```cpp
#include "stage_nodes.h"

#include <sai/detection/detection_result.h>
#include <sai/detection/detector.h>

namespace sai::pipeline {

DetectStage::DetectStage(std::string id, YAML::Node /*config*/)
    : id_(std::move(id)) {}

auto DetectStage::GetType() const noexcept -> StageType { return StageType::Detect; }
auto DetectStage::GetId() const -> std::string_view { return id_; }

auto DetectStage::OnInitialize(Context& ctx) -> Result<void> {
    auto detector = ctx.Resolve<detection::IDetector>();
    if (detector) {
        detector_ = *detector;
        stub_ = false;
    }
    return {};
}

auto DetectStage::OnStart(Context&) -> Result<void> { return {}; }
auto DetectStage::OnStop(Context&) -> Result<void> { return {}; }

auto DetectStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* det = std::get_if<sai::detection::DetectionResult>(&input)) {
        if (!stub_ && detector_) {
            // IDetector::Detect takes an Embedding, but our pipeline has a
            // DetectionResult from InferenceStage. For MockEngine the
            // DetectionResult is empty; the real flow would embed from
            // patch features. For the stub/mock path, pass-through.
            //
            // Production: InferenceStage would produce embeddings, and this
            // stage would call detector_->Detect(embedding).
            // For now: passthrough the DetectionResult (which may have been
            // populated by InferenceStage's adapter logic).
        }
        return StageOutput(std::move(*det));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects DetectionResult input"});
}

}  // namespace sai::pipeline
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/detect_stage.cpp
git commit -m "feat(pipeline): ✨ DetectStage 通过 Context/DI 获取 IDetector 接口"
```

---

### Task 8: 重写 RuleEvalStage（核心 — 接入 Knowledge/Retrieval）

**Files:**
- Modify: `src/pipeline/rule_eval_stage.cpp`

- [ ] **Step 1: 重写 rule_eval_stage.cpp**

```cpp
#include "stage_nodes.h"

#include <filesystem>

#include <sai/detection/detection_result.h>
#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_base.h>
#include <sai/rule/fact_builder.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/retrieval/vector_path.h>

namespace sai::pipeline {

RuleEvalStage::RuleEvalStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    rule_file_ = config["rule_file"].as<std::string>("");
}

auto RuleEvalStage::GetType() const noexcept -> StageType { return StageType::RuleEval; }
auto RuleEvalStage::GetId() const -> std::string_view { return id_; }

auto RuleEvalStage::OnInitialize(Context& ctx) -> Result<void> {
    // Resolve RuleEngine
    auto re = ctx.Resolve<rule::RuleEngine>();
    if (!re) return {};  // stay stub
    rule_engine_ = *re;

    // Load YAML rules
    if (!rule_file_.empty()) {
        auto load_result = rule_engine_->LoadFromYAML(rule_file_);
        if (!load_result) return load_result;
    }

    // Try to resolve KnowledgeGraph + VectorPath for FactBuilder
    auto kg = ctx.Resolve<knowledge::KnowledgeGraph>();
    auto vp = ctx.Resolve<retrieval::VectorPath>();
    if (kg && vp) {
        kg_ = *kg;
        vp_ = *vp;
        fact_builder_ = std::make_unique<rule::FactBuilder>(kg_, vp_);
    }

    stub_ = false;
    return {};
}

auto RuleEvalStage::OnStart(Context&) -> Result<void> { return {}; }
auto RuleEvalStage::OnStop(Context&) -> Result<void> { return {}; }

auto RuleEvalStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* det = std::get_if<sai::detection::DetectionResult>(&input)) {
        rule::FactBase fb;

        if (!stub_ && rule_engine_) {
            // Build FactBase from DetectionResult + Knowledge + Retrieval
            if (fact_builder_) {
                std::vector<std::string> graph_paths;
                // Resolve graph paths specified in rule conditions
                auto all_entries = fb.AllEntries();
                auto build_result = fact_builder_->Build(
                    "default", *det, graph_paths);
                if (build_result) {
                    fb = std::move(*build_result);
                }
                // else: fb stays empty, EvaluateAll handles empty facts
            } else {
                // No knowledge/retrieval: populate bare detection facts
                fb.Set("detection.anomaly_score",
                    rule::Value::Of(static_cast<double>(det->image_level_score)),
                    {rule::FactSourceKind::Direct, "DetectionResult"});
                fb.Set("detection.image_level_score",
                    rule::Value::Of(static_cast<double>(det->image_level_score)),
                    {rule::FactSourceKind::Direct, "DetectionResult"});
            }

            // Evaluate rules
            auto eval_result = rule_engine_->EvaluateAll(fb);
            if (!eval_result) return tl::make_unexpected(eval_result.error());

            auto resolved = rule_engine_->ResolveConflicts(*eval_result);

            return StageOutput(RuleEvalOutput{std::move(fb), std::move(resolved)});
        }

        // Stub: return empty RuleEvalOutput
        return StageOutput(RuleEvalOutput{std::move(fb), {}});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "RuleEval expects DetectionResult input"});
}

}  // namespace sai::pipeline
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/rule_eval_stage.cpp
git commit -m "feat(pipeline): ✨ RuleEvalStage 通过 FactBuilder 接入 Knowledge/Retrieval，构建 FactBase 并调用 RuleEngine"
```

---

### Task 9: 重写 ReasonStage

**Files:**
- Modify: `src/pipeline/reason_stage.cpp`

- [ ] **Step 1: 重写 reason_stage.cpp**

```cpp
#include "stage_nodes.h"

#include <sai/rule/rule_engine.h>
#include <sai/reasoner/reasoner.h>
#include <sai/reasoner/decision_tree.h>

namespace sai::pipeline {

ReasonStage::ReasonStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    tree_file_ = config["tree_file"].as<std::string>("");
}

auto ReasonStage::GetType() const noexcept -> StageType { return StageType::Reason; }
auto ReasonStage::GetId() const -> std::string_view { return id_; }

auto ReasonStage::OnInitialize(Context& ctx) -> Result<void> {
    auto reasoner = ctx.Resolve<reasoner::IReasoner>();
    if (reasoner) {
        reasoner_ = *reasoner;
        stub_ = false;
    }
    return {};
}

auto ReasonStage::OnStart(Context&) -> Result<void> { return {}; }
auto ReasonStage::OnStop(Context&) -> Result<void> { return {}; }

auto ReasonStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* eval_output = std::get_if<RuleEvalOutput>(&input)) {
        if (!stub_ && reasoner_) {
            auto result = reasoner_->Reason(eval_output->facts, eval_output->rules);
            if (!result) return tl::make_unexpected(result.error());
            return StageOutput(std::move(*result));
        }
        // Stub: return empty ReasoningResult
        return StageOutput(reasoner::ReasoningResult{});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Reason expects RuleEvalOutput input"});
}

}  // namespace sai::pipeline
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/reason_stage.cpp
git commit -m "feat(pipeline): ✨ ReasonStage 通过 Context/DI 获取 IReasoner，消费 RuleEvalOutput 执行决策树推理"
```

---

### Task 10: 重写 ExportStage

**Files:**
- Modify: `src/pipeline/export_stage.cpp`

- [ ] **Step 1: 重写 export_stage.cpp**

```cpp
#include "stage_nodes.h"

#include <sai/reasoner/reasoner.h>
#include <sai/io/exporter.h>

namespace sai::pipeline {

ExportStage::ExportStage(std::string id, YAML::Node config)
    : id_(std::move(id)) {
    auto out_dir = config["output_dir"].as<std::string>("/tmp/surface-ai/results/");
    output_dir_ = std::filesystem::path(out_dir);
}

auto ExportStage::GetType() const noexcept -> StageType { return StageType::Export; }
auto ExportStage::GetId() const -> std::string_view { return id_; }

auto ExportStage::OnInitialize(Context& ctx) -> Result<void> {
    auto exporter = ctx.Resolve<io::IExporter>();
    if (exporter) {
        exporter_ = *exporter;
        stub_ = false;
    }
    return {};
}

auto ExportStage::OnStart(Context&) -> Result<void> { return {}; }
auto ExportStage::OnStop(Context&) -> Result<void> { return {}; }

auto ExportStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* result = std::get_if<sai::reasoner::ReasoningResult>(&input)) {
        if (!stub_ && exporter_) {
            // Build InspectionResult from ReasoningResult
            io::InspectionResult inspection;
            inspection.sku_id = "default";
            inspection.serial_number = "unknown";
            inspection.timestamp = std::chrono::system_clock::now();
            inspection.verdict = result->verdict;

            // Map triggered rules to DefectRecords
            for (const auto& rule_name : result->triggered_rules) {
                io::DefectRecord defect;
                defect.label = rule_name;
                defect.severity = result->verdict;
                defect.confidence = static_cast<float>(result->confidence);
                inspection.defects.push_back(std::move(defect));
            }

            // Create output dir
            std::filesystem::create_directories(output_dir_);

            // Export (SurfaceImage* passed as nullptr — known limitation,
            // annotated image not available at this pipeline stage)
            auto export_result = exporter_->Export(
                inspection, output_dir_, nullptr);
            if (!export_result) return tl::make_unexpected(export_result.error());
        }
        return StageOutput(std::move(*result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Export expects ReasoningResult input"});
}

}  // namespace sai::pipeline
```

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/export_stage.cpp
git commit -m "feat(pipeline): ✨ ExportStage 通过 Context/DI 获取 IExporter，ReasoningResult→InspectionResult→JSON 落盘"
```

---

### Task 11: 更新 StageFactory 传递 Pipeline* 到 CaptureStage

**Files:**
- Modify: `src/pipeline/stage_factory.h`
- Modify: `src/pipeline/stage_factory.cpp`

- [ ] **Step 1: 更新 stage_factory.h**

```cpp
#pragma once

#include <memory>
#include <sai/core/error.h>
#include <sai/core/context.h>
#include <sai/pipeline/pipeline_config.h>
#include <sai/pipeline/stage_node.h>

namespace sai::pipeline {

class Pipeline;

class StageFactory {
public:
    static auto Create(const StageConfig& config, Context& ctx,
                       Pipeline* pipeline = nullptr)
        -> Result<std::unique_ptr<IStageNode>>;
};

}  // namespace sai::pipeline
```

- [ ] **Step 2: 更新 stage_factory.cpp**

```cpp
#include "stage_factory.h"
#include "stage_nodes.h"
#include <sai/core/error.h>

namespace sai::pipeline {

auto StageFactory::Create(const StageConfig& config, Context& ctx,
                          Pipeline* pipeline)
    -> Result<std::unique_ptr<IStageNode>> {

    switch (config.type) {
        case StageType::Capture: {
            auto stage = std::make_unique<CaptureStage>(
                config.id, config.config, pipeline);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Capture init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Preprocess: {
            auto stage = std::make_unique<PreprocessStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Preprocess init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Inference: {
            auto stage = std::make_unique<InferenceStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Inference init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Detect: {
            auto stage = std::make_unique<DetectStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Detect init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::RuleEval: {
            auto stage = std::make_unique<RuleEvalStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "RuleEval init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Reason: {
            auto stage = std::make_unique<ReasonStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Reason init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Export: {
            auto stage = std::make_unique<ExportStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Export init failed: " + result.error().message});
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Custom:
            return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                "Custom stages not yet supported"});
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
        "Unknown stage type"});
}

}  // namespace sai::pipeline
```

- [ ] **Step 3: 更新 Pipeline::LoadFromYAML 传递 this 到 StageFactory**

在 `src/pipeline/pipeline.cpp` 的 `LoadFromYAML` 中，更新 StageFactory 调用：

```cpp
// Change this line:
auto node = StageFactory::Create(stage_cfg, ctx);
// To:
auto node = StageFactory::Create(stage_cfg, ctx, pipeline.get());
```

- [ ] **Step 4: Build**

```bash
cd build/default && cmake --build . 2>&1 | tail -30
```

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/stage_factory.h src/pipeline/stage_factory.cpp src/pipeline/pipeline.cpp
git commit -m "feat(pipeline): ✨ StageFactory 支持 Pipeline* 参数注入 CaptureStage，全 Stage 链路 DI 化完成"
```

---

### Task 12: 更新 CMakeLists.txt 和 Forward 声明

**Files:**
- Modify: `src/device/CMakeLists.txt` (FakeCamera compilation)
- Modify: `src/pipeline/CMakeLists.txt` (新增依赖)
- Modify: `include/sai/pipeline/pipeline.h` (add forward declaration if needed)

- [ ] **Step 1: 更新 device CMakeLists.txt**

将 `src/device/CMakeLists.txt` 从 INTERFACE 改为 STATIC（以支持 FakeCamera compilation）：

```cmake
add_library(sai_device STATIC
    fake_camera.cpp
)
add_library(sai::device ALIAS sai_device)

target_include_directories(sai_device PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_device PUBLIC
    sai::plugin
    sai::image
)
target_compile_features(sai_device PUBLIC cxx_std_20)
```

- [ ] **Step 2: 更新 pipeline CMakeLists.txt 添加新依赖**

在 `src/pipeline/CMakeLists.txt` 中，更新依赖：

```cmake
find_package(yaml-cpp CONFIG REQUIRED)

add_library(sai_pipeline STATIC)
add_library(sai::pipeline ALIAS sai_pipeline)

target_sources(sai_pipeline PRIVATE
    pipeline.cpp
    pipeline_builder.cpp
    stage_factory.cpp
    capture_stage.cpp
    preprocess_stage.cpp
    inference_stage.cpp
    detect_stage.cpp
    rule_eval_stage.cpp
    reason_stage.cpp
    export_stage.cpp
)

target_include_directories(sai_pipeline PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sai_pipeline PUBLIC
    sai::core
    sai::runtime
    sai::infra
    sai::device
    sai::image
    sai::io
    sai::inference
    sai::detection
    sai::rule
    sai::reasoner
    sai::knowledge
    sai::retrieval
    yaml-cpp::yaml-cpp
)
```

- [ ] **Step 3: Add forward declaration to pipeline.h**

In `include/sai/pipeline/pipeline.h`, ensure `Pipeline` is forward-declared correctly. The file already has the class definition — no changes needed. But capture_stage.cpp includes pipeline.h, so verify the include works.

- [ ] **Step 4: Build**

```bash
cd build/default && cmake --build . 2>&1 | tail -40
```

- [ ] **Step 5: Run tests to ensure no regressions**

```bash
ctest --preset default 2>&1 | tail -30
```

Expected: ~572 tests pass (same as baseline).

- [ ] **Step 6: Commit**

```bash
git add src/device/CMakeLists.txt src/pipeline/CMakeLists.txt
git commit -m "chore(build): 🔧 更新 CMake 依赖：device 改为 STATIC 库，pipeline 新增全模块依赖"
```

---

### Task 13: 重写 seat_aoi main.cpp — 注册服务并启动帧循环

**Files:**
- Modify: `apps/seat-aoi/main.cpp`

- [ ] **Step 1: 重写 main.cpp**

```cpp
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include <chrono>
#include <filesystem>

#include "sai/core/context.h"
#include "sai/device/fake_camera.h"
#include "sai/inference/mock_engine.h"
#include "sai/detection/patch_core.h"
#include "sai/knowledge/knowledge_graph.h"
#include "sai/retrieval/vector_path.h"
#include "sai/rule/rule_engine.h"
#include "sai/reasoner/reasoner.h"
#include "sai/reasoner/decision_tree.h"
#include "sai/io/exporter.h"
#include "sai/pipeline/pipeline.h"
#include "sai/visualization/pipeline_viewmodel.h"
#include "sai/visualization/inspection_viewmodel.h"
#include "sai/visualization/frame_provider.h"
#include "sai/visualization/config_viewmodel.h"
#include "sai/visualization/dashboard_viewmodel.h"

auto main(int argc, char* argv[]) -> int {
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    using namespace sai;

    // 1. Create DI container
    auto ctx = std::make_unique<Context>();

    // 2. Register services (each try-caught; if registration fails, Pipeline
    //    stages will gracefully degrade to stub mode)
    auto cam = std::make_shared<device::FakeCamera>(
        device::FakeCamera::Config{.width = 1024, .height = 1024, .fps = 10.0});
    (void)ctx->Register<device::ICamera>(cam);

    auto mock_engine = std::make_shared<inference::MockEngine>();
    (void)ctx->Register<inference::IInferenceEngine>(mock_engine);

    auto patch_core = std::make_shared<detection::PatchCore>();
    (void)ctx->Register<detection::IDetector>(patch_core);

    auto kg = std::make_shared<knowledge::KnowledgeGraph>(":memory:");
    (void)ctx->Register<knowledge::KnowledgeGraph>(kg);

    // VectorPath needs a FeatureBank with dim + count. For dev/testing,
    // use minimal config (dim=128, count=0 — empty bank).
    auto vp = std::make_shared<retrieval::VectorPath>(
        retrieval::VectorPath::Config{
            .mode = retrieval::VectorPath::Mode::TopK, .k = 5},
        nullptr);  // no FeatureBank for now
    (void)ctx->Register<retrieval::VectorPath>(vp);

    auto rule_engine = std::make_shared<rule::RuleEngine>();
    (void)ctx->Register<rule::RuleEngine>(rule_engine);

    // Load decision tree for reasoner
    auto tree_result = reasoner::DecisionTree::LoadFromYAML(
        "resources/trees/seat_leather_inspection.yaml");
    std::shared_ptr<reasoner::IReasoner> reasoner;
    if (tree_result) {
        reasoner = std::make_shared<reasoner::DefaultReasoner>(
            std::move(*tree_result));
    } else {
        // Fallback: empty tree
        reasoner = std::make_shared<reasoner::DefaultReasoner>(nullptr);
    }
    (void)ctx->Register<reasoner::IReasoner>(reasoner);

    auto exporter = std::make_shared<io::JsonExporter>();
    (void)ctx->Register<io::IExporter>(exporter);

    // 3. Initialize + Start
    auto init_result = ctx->Initialize();
    if (!init_result) {
        qFatal("Context init failed: %s", init_result.error().message.c_str());
    }
    auto start_ctx = ctx->Start();
    if (!start_ctx) {
        qFatal("Context start failed: %s", start_ctx.error().message.c_str());
    }

    // 4. Load Pipeline from YAML
    auto pipeline_result = pipeline::Pipeline::LoadFromYAML(
        "resources/pipeline.yaml", *ctx);
    if (!pipeline_result) {
        qFatal("Pipeline load failed: %s", pipeline_result.error().message.c_str());
    }
    auto& pipeline = *pipeline_result;

    // 5. Create ViewModels
    auto* pipeline_vm = new visualization::PipelineViewModel(&app);
    pipeline_vm->BindToPipeline(pipeline.get());

    auto* inspection_vm = new visualization::InspectionViewModel(&app);
    auto* config_vm = new visualization::ConfigViewModel(&app);
    auto* dashboard_vm = new visualization::DashboardViewModel(&app);
    auto* frame_provider = new visualization::FrameProvider();

    // 6. Wire result callback
    pipeline->SetResultCallback(
        [=](int frame_id, const reasoner::ReasoningResult& result) {
            inspection_vm->UpdateResult(frame_id, result);

            visualization::FrameSummary summary;
            summary.frame_id = frame_id;
            summary.verdict = result.verdict;
            summary.severity = std::to_string(result.severity);
            summary.timestamp = std::chrono::system_clock::now();
            dashboard_vm->AppendFrameSummary(std::move(summary));
        });

    // 7. Start Pipeline (CaptureStage::OnStart connects camera and starts
    //    acquisition, which begins the frame loop via FakeCamera callback)
    auto start_result = pipeline->Start();
    if (!start_result) {
        qFatal("Pipeline start failed: %s", start_result.error().message.c_str());
    }

    // 8. Register QML context properties
    engine.addImageProvider("pipeline", frame_provider);
    engine.rootContext()->setContextProperty("pipelineVM", pipeline_vm);
    engine.rootContext()->setContextProperty("inspectionVM", inspection_vm);
    engine.rootContext()->setContextProperty("configVM", config_vm);
    engine.rootContext()->setContextProperty("dashboardVM", dashboard_vm);

    // 9. Start ViewModel refresh
    pipeline_vm->StartRefresh(33);

    // 10. Load QML
    engine.load("qrc:/MainWindow.qml");

    // 11. Cleanup on exit
    QObject::connect(&app, &QGuiApplication::aboutToQuit, [&]() {
        pipeline_vm->StopRefresh();
        (void)pipeline->Drain();
        (void)pipeline->Stop();
        (void)ctx->Stop();
    });

    return app.exec();
}
```

- [ ] **Step 2: Update apps/seat-aoi/CMakeLists.txt**

```cmake
# Add new link dependencies
target_link_libraries(seat_aoi PRIVATE
    sai::core
    sai::pipeline
    sai::device
    sai::inference
    sai::detection
    sai::knowledge
    sai::retrieval
    sai::rule
    sai::reasoner
    sai::io
    sai::visualization
    Qt6::Quick
)
```

- [ ] **Step 3: Build seat_aoi**

```bash
cd build/default && cmake --build . --target seat_aoi 2>&1 | tail -40
```

- [ ] **Step 4: Commit**

```bash
git add apps/seat-aoi/main.cpp apps/seat-aoi/CMakeLists.txt
git commit -m "feat(app): ✨ Seat AOI main.cpp 注册全栈服务到 Context，FakeCamera 回调驱动端到端 Pipeline"
```

---

### Task 14: 整机构建 + 运行测试 + E2E 验证

- [ ] **Step 1: 全量构建**

```bash
cd build/default && cmake --build . 2>&1 | tail -50
```

Expected: build passes with 0 errors.

- [ ] **Step 2: 运行全部测试**

```bash
ctest --preset default 2>&1 | tail -30
```

Expected: all existing tests pass (no regressions).

- [ ] **Step 3: 运行 Pipeline 集成测试**

```bash
cd build/default && ctest --preset default -R "M6E2E|Pipeline" --output-on-failure 2>&1
```

Expected: pipeline tests pass with updated variant + type compatibility.

- [ ] **Step 4: 验证 seat_aoi 可执行文件**

```bash
ls -la build/default/apps/seat-aoi/seat_aoi
```

- [ ] **Step 5: Commit and final status**

```bash
git status
```

Expected: working tree clean, all changes committed.

---

## Completion Criteria

- [ ] 所有 7 个 Stage 在 `OnInitialize` 中通过 `ctx.Resolve<T>()` 获取依赖
- [ ] 每个 Stage 的 `Process()` 调用真实业务方法（非 stub）
- [ ] Resolve 失败时回退 stub（Pipeline 不因缺依赖而拒绝启动）
- [ ] FakeCamera 实现 `ICamera`，生成正弦条纹+随机暗斑帧
- [ ] `RuleEvalStage` 通过 `FactBuilder` 连接 `KnowledgeGraph` + `VectorPath`
- [ ] `RuleEvalOutput` 在 RuleEval→Reason 间传递 `FactBase + ResolvedRule[]`
- [ ] `StageFactory` 传递 `Pipeline*` 给 `CaptureStage`
- [ ] `seat_aoi` 可执行文件编译通过
- [ ] 全部 ~572 tests 通过（无回归）
- [ ] CMakeLists 依赖正确：device STATIC, pipeline 全模块链接
