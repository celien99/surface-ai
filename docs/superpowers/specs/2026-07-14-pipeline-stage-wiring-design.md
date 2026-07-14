# Pipeline Stage 真实化 & 端到端链路修复 —— 设计文档

> Status: Draft
> Date: 2026-07-14
> Purpose: 将当前全 stub Pipeline 改造为 DI 驱动的真实管线，补齐 Device 层，接入 Knowledge/Retrieval 到推理链路，实现端到端检测链路可运行。

---

## 1. 背景与问题

当前 7 个 Pipeline Stage（Capture/Preprocess/Inference/Detect/RuleEval/Reason/Export）的 `Process()` 全部是硬编码 stub/passthrough，不调用任何业务模块。同时 Device 层只有接口没有实现，Knowledge/Retrieval 层未接入推理链路。导致：

- "从采一帧到出决策"的端到端链路无法走通
- Seat AOI 参考应用启动后闲置，无帧提交
- 各业务模块虽独立测试通过，但从未被 Pipeline 调用

## 2. 目标

- 改造 7 个 Stage，使其通过 Context/DI 解析并调用真实业务模块
- 新增 FakeCamera 作为最小可用 Device 实现，驱动帧采集
- 在 RuleEvalStage 中接入 KnowledgeGraph + VectorPath（通过 FactBuilder）
- Seat AOI main.cpp 完成服务注册和 Pipeline 启动闭环
- 所有 Stage 支持从 YAML config 解析参数选择具体实现

## 3. 设计

### 3.1 Core Principle

**Stage 通过 Context::Resolve\<T\>() 获取依赖，通过 YAML config 选择实现变体。** Stage 本身保持无状态——所有有状态对象（engine/detector/fact_builder 等）由调用方注册到 Context 中，Stage 仅在 `OnInitialize` 时解析引用。

当 Context 中未注册对应接口时，Stage 回退到 stub 行为（保持向后兼容，允许渐进式接入）。

### 3.2 新增接口

#### 3.2.1 IPreprocessChain

为让 `PreprocessStage` 通过 DI 获取预处理链，新增一个薄封装接口：

```cpp
// include/sai/pipeline/stage_node.h 或独立头文件
namespace sai::image {
using PreprocessChain = std::function<auto(std::unique_ptr<Image>)->Result<std::unique_ptr<Image>>>;
}

class IPreprocessChainProvider : public IService {
public:
    virtual auto Build(const std::vector<std::string>& step_names,
                       const YAML::Node& config) -> Result<PreprocessChain> = 0;
};
```

实际上，PreprocessStage 不需要如此复杂——它可以直接调用 `Compose({...})` 构建链。但如果后续预处理步骤需要插件化（如第三方标定库），此接口留扩展点。当前实现：PreprocessStage 直接从 Context 中获取 `PreprocessFnFactory` 或自行调用 `Compose`。

**决策**：不新增 IPreprocessChain 接口。`PreprocessStage` 直接在 `OnInitialize` 中从 YAML config 构建 `Compose` 链。预处理步骤是纯函数，不需要 DI。

### 3.3 Stage 设计详述

#### 3.3.0 StageInput/StageOutput 扩展

**问题**：`IReasoner::Reason(const FactBase&, const vector<ResolvedRule>&)` 需要 FactBase + ResolvedRules。但 RuleEvalStage 输出 `vector<ResolvedRule>`，FactBase 在传递中丢失。ReasonStage 无法获取 FactBase。

**解决方案**：新增 `RuleEvalOutput` 结构体，同时携带 FactBase 和 ResolvedRules。扩展 `StageInput/StageOutput` variant：

```cpp
// 新增结构体
struct RuleEvalOutput {
    sai::rule::FactBase facts;
    std::vector<sai::rule::ResolvedRule> rules;
};

// 扩展后的 variant
using StageInput = std::variant<
    sai::image::RawImage,
    sai::image::SurfaceImage,
    sai::detection::DetectionResult,
    sai::pipeline::RuleEvalOutput,           // ← 新增
    std::vector<sai::rule::ResolvedRule>,    // 保留：ReasonStage 内部也可能用到
    sai::reasoner::ReasoningResult
>;
using StageOutput = StageInput;
```

RuleEvalStage 输出 `RuleEvalOutput`，ReasonStage 从 `RuleEvalOutput` 中取出 `facts` 和 `rules`，传给 `IReasoner::Reason(facts, rules)`。

#### 3.3.1 CaptureStage

CaptureStage 是 Pipeline 的入口。在有真实相机时，帧由 `FakeCamera` 的独立采集线程通过 `Pipeline::Submit()` 推入；CaptureStage 的 worker thread 从 input queue pop 后直接 passthrough 给下游。

```
OnInitialize:
  1. 从 YAML config 读取 "device" 字段（默认 "mock_camera_0"）
  2. ctx.Resolve<ICamera>() 获取相机实例
  3. 若 Resolve 失败 → 回退到 stub（passthrough）

OnStart:
  1. 若 camera_ → camera_->Connect() + SetTriggerMode(FreeRun)
  2. 注册 FrameCallback: [this](RawImage img) { pipeline_->Submit(std::move(img)); }
  3. 若 camera_ → camera_->StartAcquisition()

OnStop:
  1. 若 camera_ → camera_->StopAcquisition() + Disconnect()

Process:
  1. 直接返回输入 RawImage（passthrough）
  2. 因为帧已由 FakeCamera callback → Pipeline::Submit() 注入到 CaptureStage 的 input queue
```

**循环依赖处理**：Pipeline 持有 `unique_ptr<CaptureStage>`，CaptureStage 持有 `Pipeline*`（non-owning）。`LoadFromYAML` 在构造 CaptureStage 后调用 `SetPipeline(this)` 完成注入，此时 `this` 完全构造。

#### 3.3.2 PreprocessStage

```
OnInitialize:
  1. 从 YAML config 读取 "steps": ["debayer", "white_balance", "resize"]
  2. 从 config 读取各步骤参数（resize_width, resize_height 等）
  3. 构建 PreprocessFn chain = Compose({MakeDebayer(), MakeWhiteBalance(...), MakeResize(w, h)})
  4. 存储 chain_

Process:
  1. get_if<RawImage>(input) → 调用 chain_(image)
  2. 返回 SurfaceImage
```

不需要 DI（预处理步骤是纯函数），不需要 Resolve。直接从 YAML config 构建。

#### 3.3.3 InferenceStage

```
OnInitialize:
  1. 从 YAML config 读取 "engine": "MockEngine" 或 "TensorRtEngine"
  2. 从 YAML config 读取 "model": "dino_v3_vit_base"
  3. ctx.Resolve<IInferenceEngine>() 获取 engine 实例
  4. 若失败 → 回退到 stub，存储 engine_ = nullptr
  5. 存储 model_name_

Process:
  1. get_if<SurfaceImage>(input)
  2. 若 engine_ → 调用 engine_->Infer(image, model_name_)
  3. 若 stub → 返回空 DetectionResult
  4. 返回 DetectionResult
```

#### 3.3.4 DetectStage

```
OnInitialize:
  1. 从 YAML config 读取 "detector": "PatchCore" 或 "PcaDetector"
  2. ctx.Resolve<IDetector>() 获取 detector 实例
  3. 若失败 → 回退到 stub（passthrough）

Process:
  1. get_if<DetectionResult>(input)
  2. 若 detector_ → 调用 detector_->Detect(...) 增强 DetectionResult
  3. 若 stub → passthrough
  4. 返回 DetectionResult
```

#### 3.3.5 RuleEvalStage（核心改造——接入 Knowledge/Retrieval）

```
OnInitialize:
  1. 从 YAML config 读取 "rule_file": "rules/seat_leather_defects.yaml"
  2. ctx.Resolve<RuleEngine>() → 获取 engine，调用 engine_->LoadFromYAML(rule_file)
  3. ctx.Resolve<KnowledgeGraph>() → 获取 kg_
  4. ctx.Resolve<VectorPath>() → 获取 vp_
  5. 若成功获取 kg_ + vp_ → 构造 fact_builder_ = FactBuilder(kg_, vp_)
  6. 若任一失败 → 回退 stub，fact_builder_ = nullptr

Process:
  1. get_if<DetectionResult>(input)
  2. 从 input 提取 surface_id / sku_id 等
  3. 若 fact_builder_ → fact_builder_->Build(surface_id, det_result, graph_paths) → FactBase
  4. 若 stub → 直接构造 FactBase（仅含 detection 字段，无 knowledge/retrieval）
  5. engine_->EvaluateAll(fact_base) → ResolvedRule[]
  6. engine_->ResolveConflicts(resolved) → 消解后 ResolvedRule[]
  7. 返回 RuleEvalOutput{fact_base, resolved_rules}
```

这是最关键的改动：`RuleEvalStage` 成为 **DetectionResult → KnowledgeGraph → VectorPath → FactBase → RuleEngine → RuleEvalOutput(FactBase + ResolvedRule[])** 的完整编排点。

#### 3.3.6 ReasonStage

```
OnInitialize:
  1. 从 YAML config 读取 "tree_file": "trees/seat_leather_inspection.yaml"
  2. ctx.Resolve<IReasoner>() 获取 reasoner
  3. 若成功 → reasoner_，加载 DecisionTree::LoadFromYAML(tree_file)
  4. 若失败 → 回退 stub

Process:
  1. get_if<RuleEvalOutput>(input)
  2. 若 reasoner_ → reasoner_->Reason(output.facts, output.rules) → ReasoningResult
  3. 若 stub → 返回空 ReasoningResult
  4. 返回 ReasoningResult
```

**注意**：ReasonStage 不再从 variant 中取 `vector<ResolvedRule>`，而是取 `RuleEvalOutput`。`vector<ResolvedRule>` 仍保留在 variant 中以支持跳过 RuleEvalStage 的自定义 Pipeline。

#### 3.3.7 ExportStage

```
OnInitialize:
  1. 从 YAML config 读取 "exporter": "JsonExporter"
  2. 从 YAML config 读取 "output_dir": "/tmp/surface-ai/results/"
  3. ctx.Resolve<IExporter>() 获取 exporter
  4. 若失败 → 回退 stub

Process:
  1. get_if<ReasoningResult>(input)
  2. 若 exporter_ → exporter_->Export(inspection_result, output_dir_, annotated_image)
  3. 若 stub → passthrough
  4. 返回 ReasoningResult（保持链路完整，后续可消费）
```

### 3.4 StageFactory 与 PipelineBuilder 改造

`StageFactory` 原先接受 `(std::string id, YAML::Node config)` 构造 Stage。需要扩展为接受 `(std::string id, YAML::Node config, Pipeline* pipeline)`，以便 CaptureStage 持有 Pipeline 引用：

```cpp
// stage_factory.h
class StageFactory {
public:
    static auto Create(
        const std::string& id,
        StageType type,
        const YAML::Node& config,
        Pipeline* pipeline = nullptr  // ← 新增参数
    ) -> Result<std::unique_ptr<IStageNode>>;
};
```

`PipelineBuilder` 在 `BuildStages()` 中调用 `StageFactory::Create(id, type, config, this)`，将 `this`(Pipeline*) 传入。其他 Stage 忽略此参数。

### 3.5 FakeCamera

```
class FakeCamera : public ICamera {
    // Configurable: width, height, pixel_format, fps, pattern_type
    // On StartAcquisition: 启动独立 jthread，按 fps 生成合成帧
    // 合成帧: 正弦条纹 + 模拟缺陷（圆形暗斑）
    // 使用 FrameCallback 推送帧
}
```

注册方式：`Context::Register<ICamera>(make_shared<FakeCamera>(config))`

### 3.6 Seat AOI main.cpp 改造

```
main:
  1. Context ctx
  2. Register services:
     - ICamera → FakeCamera
     - IInferenceEngine → MockEngine
     - IDetector → PatchCore (需 FeatureBank 初始化)
     - KnowledgeGraph → KnowledgeGraph(":memory:")
     - VectorPath → VectorPath(faiss_index)
     - RuleEngine → RuleEngine
     - IReasoner → DefaultReasoner(DecisionTree)
     - IExporter → JsonExporter
  3. ctx.Initialize() + ctx.Start()
  4. Pipeline::LoadFromYAML("pipeline.yaml", ctx)
  5. pipeline->Start()
  6. QML UI...
  7. FakeCamera 开始推送帧 → Pipeline::Submit() → 全链路执行
```

### 3.7 回退策略

每个 Stage 在 `OnInitialize` 中先尝试 `ctx.Resolve<T>()`，失败则设置 `stub_ = true`。`Process` 中根据 `stub_` 选择行为：

```cpp
auto InferenceStage::OnInitialize(Context& ctx) -> Result<void> {
    auto engine = ctx.Resolve<IInferenceEngine>();
    if (engine) {
        engine_ = *engine;
        return {};
    }
    // Graceful degrade: stay as stub
    return {};
}

auto InferenceStage::Process(StageInput input) -> Result<StageOutput> {
    auto* img = std::get_if<SurfaceImage>(&input);
    if (!img) return error(Pipeline_StageTypeMismatch);
    
    if (engine_) {
        return engine_->Infer(*img, model_name_);
    }
    // Stub fallback
    return StageOutput(DetectionResult{});
}
```

这样 Pipeline 在未注册对应服务的环境下仍能运行（返回空结果），便于渐进开发。

### 3.8 已知局限

1. **ExportStage 无法获取 SurfaceImage**：`JsonExporter::Export` 需要 `SurfaceImage*` 用于标注图生成。但当前 Pipeline 链路中 SurfaceImage 在 PreprocessStage 产出后经 InferenceStage → DetectStage 时被消费和丢弃，无法传递到 ExportStage。**处理方式**：ExportStage 传 `nullptr` 给 Exporter，标注图生成降级为 no-op（沿用 D2-d PPM 占位模式）。

2. **FakeCamera 仅支持 Mono8 格式**：为简化实现，FakeCamera 不生成 Bayer 格式帧。因此 `seat_aoi` Pipeline YAML 的 PreprocessStage 应配置 `steps: [resize]` 而非 `steps: [debayer, white_balance, resize]`。

3. **MockEngine 不产生有意义的 DetectionResult**：`MockEngine::Infer` 返回空 `DetectionResult`（anomaly_map 为空，image_level_score=0）。PatchCore 对空 DetectionResult 的行为是 passthrough。真正的 anomaly 检测需要训练好的 TensorRT 模型。

4. **KnowledgeGraph/VectorPath 为空时不填充知识字段**：若 main.cpp 不注册 KnowledgeGraph + VectorPath，FactBuilder 不可用，FactBase 仅含 detection 字段。决策树中依赖 `knowledge.batch_reject_rate_pct` 的分支将走 `false_branch`。

## 4. 接口

### 4.1 无新增公共接口

本次改造不引入新接口。所有 Stage 依赖通过现有接口解析：

| Stage | 依赖接口 | 所属模块 |
|-------|----------|----------|
| CaptureStage | `ICamera` | device (已有) |
| PreprocessStage | 无（直接调用 Compose） | — |
| InferenceStage | `IInferenceEngine` | inference (已有) |
| DetectStage | `IDetector` | detection (已有) |
| RuleEvalStage | `RuleEngine`, `KnowledgeGraph`, `VectorPath` | rule / knowledge / retrieval (已有) |
| ReasonStage | `IReasoner` | reasoner (已有) |
| ExportStage | `IExporter` | io (已有) |
| **所有 Stage** | `RuleEvalOutput`（新增结构体，用于 RuleEval→Reason 数据传递） | pipeline (本次新增) |

### 4.2 新增数据结构

```cpp
// include/sai/pipeline/stage_node.h
namespace sai::pipeline {

struct RuleEvalOutput {
    sai::rule::FactBase facts;
    std::vector<sai::rule::ResolvedRule> rules;
};

}  // namespace sai::pipeline
```

### 4.3 Stage 构造函数签名变更

```cpp
// 旧
CaptureStage(std::string id, YAML::Node /*config*/);

// 新
CaptureStage(std::string id, YAML::Node config, Pipeline* pipeline);
```

`Pipeline*` 用于 FrameCallback 中调用 `pipeline->Submit()`。

### 4.3 FakeCamera 接口

```cpp
class FakeCamera : public ICamera {
public:
    struct Config {
        size_t width = 1024;
        size_t height = 1024;
        double fps = 30.0;
        sai::image::PixelFormat pixel_format = PixelFormat::Mono8;
    };
    
    explicit FakeCamera(Config cfg);
    
    // ICamera overrides...
    auto Connect() noexcept -> Result<void> override;
    auto Disconnect() noexcept -> Result<void> override;
    auto IsConnected() const noexcept -> bool override;
    auto SerialNumber() const noexcept -> std::string_view override;
    auto CurrentState() const noexcept -> State override;
    auto SetTriggerMode(TriggerMode) noexcept -> Result<void> override;
    auto StartAcquisition() noexcept -> Result<void> override;
    auto StopAcquisition() noexcept -> Result<void> override;
    auto RegisterFrameCallback(FrameCallback) noexcept -> Result<void> override;
    auto SetExposureTime(std::chrono::microseconds) noexcept -> Result<void> override;
    auto SetGain(float) noexcept -> Result<void> override;
    auto SetROI(Rect) noexcept -> Result<void> override;

private:
    Config cfg_;
    FrameCallback callback_;
    std::jthread acquisition_thread_;
    std::atomic<bool> acquiring_{false};
    // ...
};
```

## 5. 工作流

### 5.1 端到端检测链路

```
1. 应用启动
2. Context::Register<ICamera>(FakeCamera)
3. Context::Register<IInferenceEngine>(MockEngine)
4. Context::Register<IDetector>(PatchCore)
5. Context::Register<KnowledgeGraph>(...)
6. Context::Register<VectorPath>(...)
7. Context::Register<RuleEngine>(...)
8. Context::Register<IReasoner>(DefaultReasoner)
9. Context::Register<IExporter>(JsonExporter)
10. Context::Initialize() + Context::Start()
11. Pipeline::LoadFromYAML("pipeline.yaml", ctx)
    → PipelineBuilder 解析 YAML → 创建 7 个 Stage → 每个 Stage.OnInitialize(ctx)
    → OnInitialize 中 ctx.Resolve<T>() 获取依赖 → 构建 StageQueue → 验证拓扑
12. pipeline->Start()
    → 每个 Stage.OnStart(ctx) → 启动 worker threads
    → CaptureStage::OnStart → camera->Connect() + StartAcquisition()
13. FakeCamera 独立线程：定时生成合成帧 → callback(raw_image)
14. callback → pipeline->Submit(raw_image)
15. Submit → push 到 CaptureStage input queue
16. CaptureStage worker: pop → Process → push 到 PreprocessStage input queue
17. PreprocessStage: Debayer → WhiteBalance → Resize → SurfaceImage
18. ... → ExportStage → JsonExporter → result.json 落盘
19. ResultCallback → ViewModels 更新 UI
```

### 5.2 Seat AOI 运行流程

```
1. QGuiApplication 启动
2. Context 注册所有服务
3. Pipeline::LoadFromYAML + Start
4. FakeCamera 开始连续采集（如 30fps）
5. 每帧走完整链路
6. Export 阶段通过 ResultCallback 通知 UI 更新
7. 用户可按 Ctrl+C 或关闭窗口停止
8. Pipeline::Drain → Stop → Context::Stop
```

## 6. 数据结构

### 6.1 Stage 内部状态

每个 Stage 新增成员变量：

```cpp
class CaptureStage : public IStageNode {
    std::string id_;
    Pipeline* pipeline_ = nullptr;   // non-owning
    std::shared_ptr<ICamera> camera_;
    bool stub_ = true;
    // ...
};

class InferenceStage : public IStageNode {
    std::string id_;
    std::shared_ptr<IInferenceEngine> engine_;
    std::string model_name_;
    bool stub_ = true;
};

// ... 其他 Stage 类似
```

### 6.2 FakeCamera 帧数据

```
Config: width=1024, height=1024, fps=10, format=Mono8
生成帧: 1024x1024 uint8_t 灰度图
  - 背景: 水平正弦条纹 (period=128, amplitude=16)
  - 模拟缺陷: 2-3 个随机圆形暗斑 (radius=20-40, center 随机)
  - 足够 PatchCore 检测出异常
```

## 7. 类图

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Pipeline   │────>│ CaptureStage │────>│ FakeCamera   │
│              │     │              │     │ (ICamera)    │
│ Submit()     │     │ Process()    │     │              │
│ Start()      │     │ OnInitialize │     │ GenerateFrame│
│ SetResultCb()│     │ OnStart()    │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
        │                    │
        ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│PreprocessStag│────>│InferenceStage│────>│ DetectStage  │
│              │     │              │     │              │
│ Compose()    │     │ IInference   │     │ IDetector    │
│ Debayer/     │     │ Engine       │     │ (PatchCore)  │
│ FlatField/   │     │              │     │              │
│ Resize       │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│RuleEvalStage │────>│ ReasonStage  │────>│ ExportStage  │
│              │     │              │     │              │
│ FactBuilder  │     │ IReasoner    │     │ IExporter    │
│ RuleEngine   │     │ DecisionTree │     │ (JsonExporter│
│ KnowledgeGrph│     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

## 8. 序列图

```
FakeCamera   CaptureStage  Preprocess  Inference  Detect  RuleEval  Reason  Export
   │              │            │           │         │        │        │       │
   │─callback────>│            │           │         │        │        │       │
   │[RawImage]    │            │           │         │        │        │       │
   │              │─Submit()──>│           │         │        │        │       │
   │              │=passthru=> │           │         │        │        │       │
   │              │            │─Debayer──>│         │        │        │       │
   │              │            │─WB+Resize │         │        │        │       │
   │              │            │=[Surface]→│         │        │        │       │
   │              │            │           │─Infer──>│         │        │       │
   │              │            │           │=[DetRes]→│        │        │       │
   │              │            │           │         │─Detect→│        │       │
   │              │            │           │         │=[DetRes]→       │       │
   │              │            │           │         │        │─Build │       │
   │              │            │           │         │        │FactBase│      │
   │              │            │           │         │        │─Eval──>│       │
   │              │            │           │         │        │=[RuleEval│      │
   │              │            │           │         │        │ Output]→│       │
   │              │            │           │         │        │        │─Reason>│
   │              │            │           │         │        │        │=[Result]→
   │              │            │           │         │        │        │       │─Export
   │              │            │           │         │        │        │       │[JSON+PPM]
```

## 9. 线程模型

```
FakeCamera Thread (1)
  │─ 生成合成帧 ─ FrameCallback ─ Pipeline::Submit()
  │─ Submit 将帧 push 到 CaptureStage input queue (lock-free SPSC)
  ▼
CaptureStage Worker Thread (1)
  │─ Pop input queue → Process(passthrough) → Push PreprocessStage queue
  ▼
PreprocessStage Worker Thread (1)
  │─ Pop → Process(Debayer+WB+Resize) → Push InferenceStage queue
  ▼
... (每个 Stage 一个 Worker Thread)
  ▼
ExportStage Worker Thread (1)
  │─ Pop → Process(JsonExporter) → Push (final)
  │─ ResultCallback(frame_id, ReasoningResult) ← UI 线程安全
```

每个 Stage 运行在独立的 `std::jthread` 上（由 Pipeline 管理），Stage 间通过 `StageQueue<T>` 通信（bounded SPSC lock-free ring buffer）。

FakeCamera 的采集线程独立于 Pipeline worker threads。

## 10. 性能

- FakeCamera 默认 10fps（可配置），生成 1024×1024 Mono8 帧约 1ms
- MockEngine::Infer 约 10ms（模拟推理延迟）
- Preprocess 链（Debayer + WB + Resize）约 5-10ms
- Pipeline 吞吐：在 FakeCamera 10fps 下无背压
- 端到端延迟（一帧从采集到导出）：约 50ms

## 11. 内存

- FakeCamera 每帧分配 1MB（1024×1024 uint8_t），帧经 Pipeline 传递后由下游 Stage 释放
- Stage 间通过 variant 传值（move-only），无额外拷贝
- 各 Stage 只持有 shared_ptr 引用（知识/检索/规则/推理引擎实例由 Context 持有）

## 12. 未来扩展

- **真实相机插件**：实现 GenICamCamera : ICamera，替换 FakeCamera 无需改动 Pipeline
- **GPU 预处理**：PreprocessStage 可接入 GpuImage 路径（CUDA gated）
- **Stage 并行度**：当前每个 Stage 1 个 worker thread，未来可配置并行度
- **多实例 Pipeline**：Context 支持注册多个 Pipeline 实例（多工位）

## 13. 最佳实践

### 13.1 Stage 生命周期

- `OnInitialize(Context&)` 中解析 YAML config **和** 调用 `ctx.Resolve<T>()` 获取依赖。此阶段允许失败（返回 Error），PipelineBuilder 会拒绝无效配置。
- `OnStart(Context&)` 中激活硬件资源（相机连接、采集启动）。此阶段不应做重的初始化。
- `OnStop(Context&)` 中释放硬件资源。必须幂等——多次调用不崩溃。
- `Process(StageInput)` 保持无副作用（除 ExportStage 写磁盘）。它是纯数据变换：拿到输入，产出输出。
- `ReloadConfig(YAML::Node)` 支持热更新参数（如阈值调整），默认 no-op。

### 13.2 DI 使用规范

- `ctx.Resolve<T>()` 返回 `Result<shared_ptr<T>>`，Stage 存储 `shared_ptr<T>`，不存储裸指针。
- 若 Resolve 失败，设置 `stub_ = true` 并以空 `shared_ptr` 回退——Stage 仍可启动但不能做实际工作。Pipeline 不应因此拒绝启动。
- 不在 Stage 中调用 `Context::Register<T>()`——注册是应用入口的职责。

### 13.3 YAML Config 解析

- 使用 `YAML::Node` 的 `operator[]` 配合默认值，避免 throw：`auto engine = config["engine"].as<std::string>("MockEngine");`
- 必需字段缺失时尽早失败：`if (!config["rule_file"]) return ErrorInfo{Pipeline_InvalidConfig, "missing rule_file"};`
- 不修改 `PipelineConfig` 结构体——YAML 解析仍在 Stage 内部完成。

### 13.4 错误处理

- 使用 `Result<T>` 的 monadic 操作（`and_then`/`or_else`）而非嵌套 if。
- Stage 的 `Process` 失败返回 `tl::unexpected`，worker loop 将失败计入 `StageMetrics::frames_failed` 并继续处理下一帧。
- 不因单帧失败停止 Pipeline。只有 `Drain()`/`Stop()` 或 `stop_token` 触发才退出 worker loop。

## 14. 反模式

### 14.1 不要在 Stage 中 new 业务对象
❌ `auto engine = std::make_shared<TensorRtEngine>(...);`  （紧耦合、绕过 DI）
✅ `auto engine = ctx.Resolve<IInferenceEngine>();`          （通过 Context 解耦）

### 14.2 不要在 Process 中持有锁或做阻塞 I/O
❌ `std::lock_guard<std::mutex> lock(mtx_); file.write(...);` （阻塞上游 Pipeline）
✅ 文件写入放在 ExportStage worker thread 的 Process 中（已在独立线程），ResultCallback 通知 UI。

### 14.3 不要用一个 stub 导致 Pipeline 启动失败
❌ `if (!engine_) return tl::unexpected(...);`                （OnInitialize 中因缺依赖而拒绝启动）
✅ 设置 `stub_ = true`，OnInitialize 返回成功。Process 中检查 stub_ 回退。

### 14.4 不要假设 Stage 的执行顺序
❌ 在 DetectStage 中依赖 InferenceStage 已设置某些全局状态
✅ Stage 间仅通过 StageInput/StageOutput variant 传递数据，不共享可变状态

### 14.5 不要硬编码路径或参数
❌ `auto rules = engine.LoadFromYAML("rules/seat_leather_defects.yaml");`
✅ `auto rule_file = config["rule_file"].as<std::string>();` 从 YAML config 读取
