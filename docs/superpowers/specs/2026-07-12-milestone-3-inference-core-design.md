### 3.12 ErrorCode

M3 引入 `Inference_*` / `Embedding_*` / `Detection_*` 错误码前缀（append-only，追加在 M2 的 `Io_ImportParseFailed` 之后）：

| 错误码 | 含义 | 触发条件 |
|--------|------|---------|
| `Inference_EngineLoadFailed` | engine 文件加载失败 | 文件不存在/损坏/版本不兼容 |
| `Inference_EngineExecutionFailed` | 推理执行失败 | TensorRT enqueue 返回非零 status |
| `Inference_InvalidBinding` | I/O binding 与 engine 不匹配 | 名称/形状/数据类型不一致 |
| `Inference_ReloadFailed` | 热重载失败 | 新 engine 反序列化或校验失败 |
| `Inference_ModelConfigMismatch` | adapter 配置与 engine 不匹配 | patch_size/embed_dim 不一致 |
| `Embedding_NotGpuImage` | 输入非 GPU 图像 | Extract 收到 SurfaceImage（CPU）而非 GpuImage |
| `Embedding_DimensionMismatch` | 降维输入 dim 与 params 不匹配 | PCA/Whitening 输入向量维度错误 |
| `Detection_FeatureBankLoadFailed` | coreset 文件加载失败 | 文件不存在/格式错误 |
| `Detection_InvalidPatchGrid` | embedding grid 与配置不匹配 | Detect 输入的 grid_h/w 与 Config 不一致 |

以上 9 个错误码按表内顺序追加。M1 的 `error.h` 附加规则同样适用——永不重排、永不碰其他批次的成员。

# Surface AI Framework —— 里程碑 3 AI 推理核心 设计文档

> Status: Draft
> Date: 2026-07-12
> Based on: `docs/superpowers/specs/2026-07-07-surface-ai-framework-phased-plan-design.md` §4 里程碑 3
> Depends on: 里程碑 1（1.1-1.6 全部冻结接口）、里程碑 2（2.1-2.3 全部冻结接口）

---

## 1. 背景与范围

里程碑 3 覆盖"AI 推理核心"：从 M2 产出的 `SurfaceImage`/`GpuImage` 出发，走 TensorRT 推理引擎，产出标准化 Embedding，最终通过 PatchCore 异常检测算法输出 `DetectionResult`。

### 1.1 架构总览

三层分离，每层定义于独立批次：

```
3.1 Foundation（推理引擎抽象）
    IInferenceEngine（薄层：GPU 显存 in → GPU 显存 out）
    TensorRtEngine（门控）/ MockEngine（可移植）
    模型 Adapter：DINOv3 / SAM2 / CLIP（工厂函数 + 具体类型）

3.2 Embedding（特征提取与标准化）
    IEmbedder（SurfaceImage/GpuImage → Embedding）
    PatchEmbedder(DINOv3) / GlobalEmbedder(CLIP)
    DimensionReducer（PCA / Whitening / Pooling）
    FeatureCache（LRU，CPU 端）

3.3 Detector（统一检测接口）
    IDetector（Embedding → DetectionResult）
    PatchCore（coreset 特征库 + k-NN + 上采样 + RegionProposals）
    SAM2 在 M5 消费，M3 只做 engine 绑定
```

### 1.2 批次划分与执行顺序

```
里程碑 3：AI 推理核心
├── 3.1 Foundation（推理引擎 + 模型 adapters）
├── 3.2 Embedding（特征提取 + 降维 + 缓存）
└── 3.3 Detector（统一接口 + PatchCore）
```

执行顺序：**3.1 设计 → 3.1 review（检查接口与 glossary-and-contracts.md 一致性）→ 3.1 代码 → 3.2 设计 → 3.2 review → 3.2 代码 → 3.3 设计 → 3.3 review → 3.3 代码 → 最终整体回顾**

每个批次的设计阶段产出冻结接口后，必须经过一次 review 检查——验证接口与 glossary-and-contracts.md 的已有契约无冲突、各层依赖方向正确、新增 ErrorCode 追加顺序正确——再进入代码阶段。此闸门避免设计缺陷流入实现阶段后返工。

### 1.3 项目锚点

延续 M1/M2 的首要落地场景：**汽车座椅 AOI 缺陷检测**。M3 的 PatchCore + DINOv3 组合直接服务于此场景：

- **DINOv3** 作为 patch-level 特征提取器，输出空间保持的 dense feature
- **PatchCore** 用 coreset 采样 + k-NN 距离做 anomaly localization
- **CLIP** 提供图像级语义 embedding，供 M4 Knowledge 做跨模态检索
- **SAM2** 的 segmentation 能力留到 M5 Reasoner 阶段做缺陷边界精修

### 1.4 明确排除项

- 训练/微调流程（DINOv3/CLIP/SAM2 全部使用预训练权重，ONNX→TensorRT 转换脚本作为独立资产，不在 C++ 构建中）
- 模型文件（.onnx/.engine）不提交到 repo——C++ 代码只实现加载和推理
- SAM2 的 pixel-level segmentation 消费（M5 Reasoner 处理）
- CLIP 的 text encoder（M3 只做 image encoder）
- 非 TensorRT 推理后端（ONNX Runtime、OpenVINO、libtorch 等）

---

## 2. 对里程碑 1/2 的依赖

| M3 批次 | 依赖接口 | 用途 |
|---------|---------|------|
| 3.1 | `IMemoryPool`、`GpuPool`（1.5） | I/O tensor 显存分配 |
| 3.1 | `GpuStreamQueue`、`CopyDirection`、`Task<T>`（1.4） | 异步推理 + DtoH 回传 |
| 3.1 | `Result<T>`、`Object`（1.1） | 错误处理 + 接口基类 |
| 3.1 | `ConfigStore`（1.6） | engine 路径 + binding 配置 |
| 3.1 | `Logger`（1.6） | 推理耗时/错误日志 |
| 3.2 | `SurfaceImage`、`GpuImage`、`Image`、`PixelFormat`（2.2） | 推理输入图像 |
| 3.2 | `PinnedPool`（1.5） | DtoH 中转缓冲 |
| 3.2 | `WorkerPool`（1.4） | 批量特征提取并行化 |
| 3.3 | `sai::device::Rect`（2.1） | RegionProposal 边界框 |
| 3.3 | `InspectionResult`、`DefectRecord`（2.3） | 检测结果 → 缺陷记录映射（M5 时触发） |

---

## 3. 批次 3.1 Foundation

### 3.1 Purpose

定义薄层推理引擎抽象：封装 TensorRT engine 的加载、动态 shape 绑定、异步推理和热重载。每个 AI 模型（DINOv3/SAM2/CLIP）通过工厂函数配置为具体 adapter 类型——无多态基类，类型安全。

### 3.2 Responsibilities

- `IInferenceEngine`：统一推理接口（Load/InferAsync/Infer/Reload）
- `TensorRtEngine`：封装 nvinfer1::IRuntime + ICudaEngine + IExecutionContext
- `MockEngine`：可移植空壳，使接口管线和错误路径在 macOS 上可编译测试
- `DinoV3Adapter`/`Sam2Adapter`/`ClipAdapter`：各模型的具体工厂+类型
- 动态 shape 支持（batch、分辨率变化）通过 optimization profile
- 多 GPU 支持通过构造参数指定 device ordinal

### 3.3 Design

**为什么用薄 engine 抽象**：TensorRT engine 的职责是且仅是从 GPU 显存读数据、做 forward pass、写回 GPU 显存。Pre/post processing、特征后处理、距离度量都是调用方的语义——放到 engine 内部会导致引擎与模型类型耦合，违反单一职责。薄层使 DINOv3（patch features）、SAM2（masks）、CLIP（embeddings）共享同一个 `IInferenceEngine` 实现，差异仅体现在 binding 配置和 adapter 层。

**为什么 adapter 是工厂函数+具体类型而非 IModelAdapter 多态接口**：四个模型输出类型不同（patch grid vs mask vs global embedding），没有共同的返回类型可以合理抽象。强制抽象为 `Infer() → vector<Tensor>` 会丢失类型信息，且在调用方要 cast/重解释。工厂函数返回具体类型（`DinoV3Adapter`、`Sam2Adapter` 等），每个类型提供其独有的类型安全 `Infer()` 签名——这是 OOP 推荐"继承有意义时才用"的正面案例。

**为什么不用 nvinfer1::IRuntime 作为 IInferenceEngine 的伪包装**：TensorRT API 在 v10 与 v8 之间有 breaking change。`IInferenceEngine` 作为框架自己的抽象层，隔离了 TensorRT 版本的变更——升级 TensorRT 只需修改 `TensorRtEngine` 的实现，不影响所有下游 adapter 和 embedder。

**为什么 I/O buffer 由调用方分配而非 engine 内部管理**：薄层定位——engine 不负责内存生命周期。调用方（adapter）通过 M1 的 `GpuPool` 分配 I/O tensor 显存，填入 `TensorBinding::device_ptr`，engine 只负责填充数据。这既保持了内存管理的一致性（全框架统一走 `IMemoryPool`），又使调用方可以复用 buffer（比如 DINOv3 的输出 tensor 的下一个消费者是 3.2 的 Embedder，可以直接原地读取，不必拷贝）。

**为什么在 Reload 失败时保持旧 context 生效**：热重载的语义是"尝试加载新 engine，失败不回退默认值"——与 M1 `ConfigStore::EnableHotReload` 的设计一致（keep-stale-never-default）。`TensorRtEngine` 在 `Reload()` 中 deserialize 新 engine → 校验 binding 一致性 → 通过后 `std::atomic<shared_ptr<IExecutionContext>>` swap——旧 context 随引用计数自然释放，校验失败旧 context 继续服务。

### 3.4 Interfaces

```cpp
// inference_engine.h
#pragma once
#include <filesystem>
#include <vector>
#include <cstdint>
#include <sai/core/error.h>
#include <sai/core/object.h>

namespace sai::inference {

struct TensorBinding {
    std::string name;
    std::vector<std::int64_t> shape;
    std::size_t size_bytes = 0;
    void* device_ptr = nullptr;
};

class IInferenceEngine : public Object {
public:
    [[nodiscard]] virtual auto Load(const std::filesystem::path& engine_path,
                                     std::vector<TensorBinding> inputs,
                                     std::vector<TensorBinding> outputs) noexcept
        -> Result<void> = 0;

    [[nodiscard]] virtual auto Infer() noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto InferAsync(cudaStream_t stream) noexcept -> Result<void> = 0;

    [[nodiscard]] virtual auto Reload(const std::filesystem::path& engine_path) noexcept
        -> Result<void> = 0;

    // 更新指定 I/O tensor 的设备端地址——adapter 每帧可能从 GpuPool 分配/复用不同 slab，
    // TensorRT 需要知道当前帧的数据位置。name 在 Load() 的 bindings 中首次注册，此处仅更新地址。
    [[nodiscard]] virtual auto SetTensorAddress(const std::string& name,
                                                  void* device_ptr) noexcept -> Result<void> = 0;


    [[nodiscard]] virtual auto InputBindings() const noexcept
        -> const std::vector<TensorBinding>& = 0;
    [[nodiscard]] virtual auto OutputBindings() const noexcept
        -> const std::vector<TensorBinding>& = 0;
};

}  // namespace sai::inference
```

```cpp
// tensorrt_engine.h（门控头文件——仅目标平台包含 <NvInfer.h>）
#pragma once
#include <atomic>
#include <memory>
#include <sai/inference/inference_engine.h>

namespace nvinfer1 { class IRuntime; class ICudaEngine; class IExecutionContext; }

namespace sai::inference {

class TensorRtEngine final : public IInferenceEngine {
public:
    explicit TensorRtEngine(std::size_t device_ordinal = 0) noexcept;
    ~TensorRtEngine() override;

    [[nodiscard]] auto Load(const std::filesystem::path& engine_path,
                             std::vector<TensorBinding> inputs,
                             std::vector<TensorBinding> outputs) noexcept
        -> Result<void> override;
    [[nodiscard]] auto Infer() noexcept -> Result<void> override;
    [[nodiscard]] auto InferAsync(cudaStream_t stream) noexcept -> Result<void> override;
    [[nodiscard]] auto Reload(const std::filesystem::path& engine_path) noexcept
        -> Result<void> override;
    [[nodiscard]] auto InputBindings() const noexcept -> const std::vector<TensorBinding>& override;
    [[nodiscard]] auto OutputBindings() const noexcept -> const std::vector<TensorBinding>& override;

    TensorRtEngine(const TensorRtEngine&) = delete;
    auto operator=(const TensorRtEngine&) -> TensorRtEngine& = delete;
    TensorRtEngine(TensorRtEngine&&) = delete;
    auto operator=(TensorRtEngine&&) -> TensorRtEngine& = delete;

private:
    struct EngineState {
        std::shared_ptr<nvinfer1::IRuntime> runtime;
        std::shared_ptr<nvinfer1::ICudaEngine> engine;
        std::shared_ptr<nvinfer1::IExecutionContext> context;
    };
    std::atomic<std::shared_ptr<EngineState>> state_{nullptr};
    std::vector<TensorBinding> inputs_;
    std::vector<TensorBinding> outputs_;
    std::size_t device_ordinal_;
};

}  // namespace sai::inference
```

```cpp
// mock_engine.h（可移植——macOS 上编译测试）
#pragma once
#include <sai/inference/inference_engine.h>

namespace sai::inference {

class MockEngine final : public IInferenceEngine {
public:
    // 测试数据注入：每次 Infer()/InferAsync() 调用后，对每个 output binding 调用此回调。
    // 回调签名为 void(std::string_view name, void* device_ptr, std::size_t size_bytes)，
    // 测试代码将预期数据 memcpy 到 device_ptr 中，模拟 TensorRT 的推理输出。
    using OutputFillCallback =
        std::function<void(std::string_view name, void* device_ptr, std::size_t size_bytes)>;

    auto SetOutputFillCallback(OutputFillCallback callback) noexcept -> void;

    [[nodiscard]] auto Load(const std::filesystem::path&,
                             std::vector<TensorBinding> inputs,
                             std::vector<TensorBinding> outputs) noexcept
        -> Result<void> override;
    [[nodiscard]] auto Infer() noexcept -> Result<void> override;
    [[nodiscard]] auto InferAsync(cudaStream_t) noexcept -> Result<void> override;
    [[nodiscard]] auto Reload(const std::filesystem::path&) noexcept -> Result<void> override;
    [[nodiscard]] auto SetTensorAddress(const std::string&, void*) noexcept -> Result<void> override;
    [[nodiscard]] auto InputBindings() const noexcept -> const std::vector<TensorBinding>& override;
    [[nodiscard]] auto OutputBindings() const noexcept -> const std::vector<TensorBinding>& override;

private:
    std::vector<TensorBinding> inputs_;
    std::vector<TensorBinding> outputs_;
    OutputFillCallback output_fill_;
};

}  // namespace sai::inference
```

```cpp
// dino_v3_adapter.h
#pragma once
#include <cstddef>
#include <sai/core/error.h>
#include <sai/inference/inference_engine.h>

namespace sai::inference {

struct DinoV3Config {
    std::filesystem::path engine_path;
    std::size_t image_size;
    std::size_t patch_size;   // DINOv3 的 patch size（如 14）
    std::size_t embed_dim;    // DINOv3 的输出维度
};

struct PatchFeatures {
    float* device_ptr;          // GPU 端特征数据（grid_h × grid_w × dim）
    std::size_t grid_h;
    std::size_t grid_w;
    std::size_t dim;
};

class DinoV3Adapter {
public:
    [[nodiscard]] static auto Create(IInferenceEngine& engine,
                                      const DinoV3Config& cfg) noexcept -> Result<DinoV3Adapter>;

    // 从 M2 的 GpuImage 提取 patch 特征——该图像必须在 GPU 显存中。
    [[nodiscard]] auto Infer(const class GpuImage& image) noexcept -> Result<PatchFeatures>;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view { return "DINOv3"; }

    DinoV3Adapter(DinoV3Adapter&&) noexcept = default;
    auto operator=(DinoV3Adapter&&) noexcept -> DinoV3Adapter& = default;
    DinoV3Adapter(const DinoV3Adapter&) = delete;
    auto operator=(const DinoV3Adapter&) -> DinoV3Adapter& = delete;

private:
    DinoV3Adapter(IInferenceEngine* engine, DinoV3Config cfg) noexcept;
    IInferenceEngine* engine_;
    DinoV3Config cfg_;
};

// sam2_adapter.h — SAM2 adapter
struct Sam2Config { std::filesystem::path engine_path; std::size_t image_size; };
struct SegmentationMask { float* device_ptr; std::size_t height; std::size_t width; };

class Sam2Adapter {
    // 当前 Infer 签名为占位——仅支持 mask prompt（const GpuImage&）。
    // M5 将扩展为 variant<PointPrompt, BoxPrompt, MaskPrompt> 以覆盖 SAM2 的三种 prompt 类型。
    // 见 §5.12 Future Extension。
public:
    [[nodiscard]] static auto Create(IInferenceEngine& engine,
                                      const Sam2Config& cfg) noexcept -> Result<Sam2Adapter>;
    [[nodiscard]] auto Infer(const GpuImage& image,
                              const GpuImage& prompt) noexcept -> Result<SegmentationMask>;
    Sam2Adapter(Sam2Adapter&&) noexcept = default;
    Sam2Adapter(const Sam2Adapter&) = delete;
private:
    Sam2Adapter(IInferenceEngine* engine, Sam2Config cfg) noexcept;
    IInferenceEngine* engine_;
    Sam2Config cfg_;
};

// clip_adapter.h — CLIP adapter（仅 image encoder）
struct ClipConfig { std::filesystem::path engine_path; std::size_t image_size; std::size_t embed_dim; };
struct GlobalFeatures { float* device_ptr; std::size_t dim; };

class ClipAdapter {
public:
    [[nodiscard]] static auto Create(IInferenceEngine& engine,
                                      const ClipConfig& cfg) noexcept -> Result<ClipAdapter>;
    [[nodiscard]] auto Infer(const GpuImage& image) noexcept -> Result<GlobalFeatures>;
    ClipAdapter(ClipAdapter&&) noexcept = default;
    ClipAdapter(const ClipAdapter&) = delete;
private:
    ClipAdapter(IInferenceEngine* engine, ClipConfig cfg) noexcept;
    IInferenceEngine* engine_;
    ClipConfig cfg_;
};

}  // namespace sai::inference
```

### 3.5 Workflow

```
1. ConfigStore(YAML) → 指定各模型 engine 路径 + binding 配置
2. TensorRtEngine::Load(dino_v3.engine, inputs, outputs)
     → deserialize engine 文件 → 创建 execution context
     → 校验 binding 名称/形状/数据类型与配置一致 → 返回错误或就绪
3. DinoV3Adapter::Create(engine, config)
     → 持有 engine 引用 + I/O tensor 绑定元信息
4. adapter.Infer(gpu_image)
     → GpuImage 数据指针填入 input binding
     → engine.InferAsync(stream) 或 Infer()
     → 从 output binding 读取结果 → 构造 PatchFeatures
5. [可选] 热重载：engine.Reload(new.engine)
     → 反序列化新 engine → 校验 binding 兼容性
     → atomic swap context → 旧 context 释放
```

### 3.6 Data Structure

见 §3.4 中的 `TensorBinding`、`DinoV3Config`/`PatchFeatures`、`Sam2Config`/`SegmentationMask`、`ClipConfig`/`GlobalFeatures`。

```yaml
# 模型配置 YAML 示例
inference:
  engines:
    dino_v3:
      engine_path: /opt/models/dino_v3_vitl14.engine
      image_size: 518
      patch_size: 14
      embed_dim: 1024
      device_ordinal: 0
    sam2:
      engine_path: /opt/models/sam2_hiera_large.engine
      image_size: 1024
      device_ordinal: 0
    clip:
      engine_path: /opt/models/clip_vit_b32.engine
      image_size: 224
      embed_dim: 512
      device_ordinal: 0
```

### 3.7 Class Diagram

```
Object (1.1)
  └── IInferenceEngine ─────────── TensorBinding / Load / Infer / InferAsync / Reload
        ├── TensorRtEngine (final) ── ICudaEngine + IExecutionContext + atomic<shared_ptr<EngineState>>
        └── MockEngine (final) ───── 空壳，记录 binding 元信息

独立类型（非多态，工厂函数创建）：
  DinoV3Adapter ── IInferenceEngine* + DinoV3Config → Infer(GpuImage) → PatchFeatures
  Sam2Adapter ──── IInferenceEngine* + Sam2Config → Infer(GpuImage, prompt) → SegmentationMask
  ClipAdapter ──── IInferenceEngine* + ClipConfig → Infer(GpuImage) → GlobalFeatures
```

### 3.8 Sequence Diagram

```
Pipeline Worker 线程（M6）
  → DinoV3Adapter::Infer(gpu_image)
    → engine.Infer() / engine.InferAsync(stream)
      → [TensorRT] enqueueV3(IExecutionContext::enqueue)
      → cudaStreamSynchronize (如果同步)
    → 从 output binding 读取 patch features
    → PatchFeatures{device_ptr, grid_h, grid_w, dim} → 传给 3.2 Embedder
```

### 3.9 Thread Model

| 线程 | 归属 | 职责 |
|------|------|------|
| Inference Worker | M1 WorkerPool | 调用 `InferAsync` 提交 GPU 工作，不阻塞 |
| CUDA Stream | GPU 硬件 | 异步执行推理，完成后触达回调 |
| GPU Callback Thread | M1 GpuStreamQueue | 转发完成事件，恢复协程 |

规则：`InferAsync` 提交后协程挂起——Worker 线程立即被释放处理下一帧。GPU 完成后由 M1 的 `GpuStreamQueue` 回调恢复协程。

### 3.10 Performance

| 指标 | 目标 |
|------|------|
| DINOv3 推理延迟（ViT-L/14, 518×518, FP16） | < 15ms @ NVIDIA A10 |
| CLIP 推理延迟（ViT-B/32, 224×224, FP16） | < 3ms @ NVIDIA A10 |
| SAM2 推理延迟（hiera_large, 1024×1024, FP16） | < 30ms @ NVIDIA A10 |
| TensorRT Engine 加载（含 context 创建） | < 2s |
| 热重载总耗时（Load→校验→swap） | < 2s（旧 context 持续服务，无停机） |

### 3.11 Memory

| 对象 | 分配来源 | 生命周期 |
|------|---------|---------|
| I/O tensor GPU buffer | GpuPool（adapter 分配） | 每次 `Infer` 前分配，推理完成后可复用 |
| IExecutionContext | CUDA 内部管理 | `Load` → `~TensorRtEngine` |
| EngineState | 堆（`shared_ptr`） | `atomic` swap 后旧 context 随引用计数释放 |
| Adapter 实例 | 栈/堆（调用方持有） | 与 engine 实例等长 |

### 3.13 Future Extension

- 更多推理后端：`OnnxRuntimeEngine` 实现同一 `IInferenceEngine` 接口（已定义 `TensorBinding` 为与后端无关的形状描述）
- FP8 精度：TensorRT v10+ 原生支持，通过 `DinoV3Config` 加 `precision` 字段即可
- 多 GPU model parallelism：多个 `TensorRtEngine` 实例，每个绑定不同 `device_ordinal`
- 更多模型 adapter：`EfficientAD`、`DeepLabv3+`、`InternImage` 等，复用同一 `IInferenceEngine` + 工厂函数模式

### 3.14 Best Practice

- ✅ 所有 I/O tensor buffer 从 `GpuPool` 分配——启动期预分配，运行期不调 `cudaMalloc`
- ✅ Engine 路径从 `ConfigStore` 的 YAML 加载——不硬编码
- ✅ 推理失败记录 `Inference_EngineExecutionFailed` 错误——不静默吞错
- ✅ adapter 的 `Infer()` 校验输入图像尺寸/格式匹配 engine 的 optimization profile
- ✅ 热重载在推理执行间隙进行——不中断正在进行的 inference

### 3.15 Anti Pattern

- ❌ 在 `InferAsync` 后立即 `cudaStreamSynchronize`——应走协程挂起+回调恢复，不阻塞 Worker
- ❌ 假设所有模型在同一 device ordinal——每个 engine 实例绑定独立 ordinal
- ❌ engine 文件路径硬编码——一律走 `ConfigStore`
- ❌ 在 adapter 内做图像预处理——M2 预处理链已负责此职责

---

## 4. 批次 3.2 Embedding

### 4.1 Purpose

定义标准化 Embedding 数据结构和特征提取接口，把 3.1 各 adapter 产出的异构 GPU tensor 统一为框架可消费的格式，并提供降维和缓存能力。

### 4.2 Responsibilities

- `Embedding`：双存储（GPU 池 / CPU vector），move-only
- `IEmbedder`：统一特征提取接口（`Extract` / `ExtractBatch`）
- `PatchEmbedder`：DINOv3 adapter → patch feature grid → `Embedding`
- `GlobalEmbedder`：CLIP adapter → [CLS] token → `Embedding`
- `DimensionReducer`：PCA / Whitening / Pooling（批量特征降维）
- `FeatureCache`：LRU 缓存（CPU 端，免重复推理）

### 4.3 Design

**为什么 Embedding 支持双存储（GPU/CPU）**：DINOv3 产出的 patch features 在 GPU 端被 PatchCore 的 k-NN 直接消费——不做 DtoH 就能搜索，零额外拷贝。CLIP 产出的 global embedding 被 M4 Knowledge 索引（FAISS）——也需要 GPU 端直接建索引。但 FeatureCache 需要 CPU 端存储（LRU 使用 `std::vector<float>`），PCA 拟合也需要 CPU 端。双存储让消费者按需选择——GPU 路径零拷贝，CPU 路径通过 `ToCpuAsync()` 按需搬移。

**为什么 IEmbedder 不从 IService/IModule 派生**：Embedder 是算法组件而非框架模块——它不在 `Context` 的 DI 体系中注册，而是由 Detector（3.3）或 Pipeline（M6）直接持有和使用。它与 `PreprocessFn`（2.2）的地位类似——生命周期由调用方管理（作为局部对象或 `unique_ptr<IEmbedder>`），不在 Context 中注册。

**为什么 DimensionReducer 的 Fit/Reduce 分离**：PCA 和 Whitening 的参数（components、mean、transform）是离线标定阶段（calibration pipeline）计算并持久化到 YAML 或 bin 文件的。运行期 `Reduce` 只需要矩阵乘法——无需每帧重新拟合。`Fit*` 是静态方法，产出的参数对象由 `ConfigStore` 或 `IImporter` 加载后构造 `DimensionReducer`。

**为什么 FeatureCache 仅 CPU 端**：GPU 显存是稀缺资源（8-16 GB），缓存应该用廉价的 CPU 内存（32-128 GB）。被缓存的 embedding 已经通过 `ToCpuAsync()` 从 GPU 搬移——LRU 命中时直接返回 CPU 端数据，调用方如需 GPU 端再做一次 HtoD（成本远小于重新推理）。

### 4.4 Interfaces

```cpp
// embedding.h
#pragma once
#include <vector>
#include <chrono>
#include <coroutine>
#include <stop_token>
#include <sai/core/error.h>
#include <sai/image/image.h>

namespace sai::memory { class PinnedPool; }
namespace sai::runtime { class GpuStreamQueue; template<typename T> struct Task; }

namespace sai::embedding {

enum class EmbeddingType : std::uint8_t { Patch, Global };

struct EmbeddingMeta {
    std::string model_name;
    EmbeddingType type = EmbeddingType::Patch;
    std::size_t dim = 0;
    std::size_t count = 0;
    std::array<std::size_t, 2> grid{0, 0};
    std::chrono::nanoseconds inference_latency{0};
};

class Embedding final {
public:
    [[nodiscard]] static auto FromGpu(sai::memory::PooledPtr<std::uint8_t> device_data,
                                       EmbeddingMeta meta) noexcept -> Embedding;
    [[nodiscard]] static auto FromCpu(std::vector<float> data,
                                       EmbeddingMeta meta) noexcept -> Embedding;

    [[nodiscard]] auto Data() const noexcept -> const float*;
    [[nodiscard]] auto Meta() const noexcept -> const EmbeddingMeta& { return meta_; }
    [[nodiscard]] auto SizeBytes() const noexcept -> std::size_t;
    [[nodiscard]] auto IsOnGpu() const noexcept -> bool { return on_gpu_; }

    Embedding(Embedding&&) noexcept = default;
    auto operator=(Embedding&&) noexcept -> Embedding& = default;
    Embedding(const Embedding&) = delete;
    auto operator=(const Embedding&) -> Embedding& = delete;
    ~Embedding() noexcept;

private:
    Embedding() noexcept = default;
    sai::memory::PooledPtr<std::uint8_t> device_buffer_{};
    std::vector<float> cpu_data_{};
    EmbeddingMeta meta_{};
    bool on_gpu_ = false;
};

}  // namespace sai::embedding
```

```cpp
// embedder.h
#pragma once
#include <span>
#include <sai/core/error.h>
#include <sai/embedding/embedding.h>

namespace sai::embedding {

class IEmbedder : public sai::Object {
public:
    // 接受基类引用——实现层负责校验是否为 GPU 端图像（检查 Image 的存储类型）。
    // 非 GPU 图像应返回 Embedding_NotGpuImage 错误。
    [[nodiscard]] virtual auto Extract(const sai::image::Image& image) noexcept
        -> Result<Embedding> = 0;

    [[nodiscard]] virtual auto ExtractBatch(std::span<const sai::image::Image* const> images) noexcept
        -> Result<std::vector<Embedding>> = 0;

    [[nodiscard]] virtual auto ModelName() const noexcept -> std::string_view = 0;
};

// patch_embedder.h — DINOv3
class PatchEmbedder final : public IEmbedder {
public:
    [[nodiscard]] static auto Create(sai::inference::DinoV3Adapter adapter) noexcept
        -> Result<PatchEmbedder>;

    [[nodiscard]] auto Extract(const sai::image::Image& image) noexcept
        -> Result<Embedding> override;
    [[nodiscard]] auto ExtractBatch(std::span<const sai::image::Image* const> images) noexcept
        -> Result<std::vector<Embedding>> override;
    [[nodiscard]] auto ModelName() const noexcept -> std::string_view override
    { return "DINOv3"; }

    PatchEmbedder(PatchEmbedder&&) noexcept = default;
    PatchEmbedder(const PatchEmbedder&) = delete;

private:
    explicit PatchEmbedder(sai::inference::DinoV3Adapter adapter) noexcept;
    sai::inference::DinoV3Adapter adapter_;
};

// global_embedder.h — CLIP
class GlobalEmbedder final : public IEmbedder {
public:
    [[nodiscard]] static auto Create(sai::inference::ClipAdapter adapter) noexcept
        -> Result<GlobalEmbedder>;

    [[nodiscard]] auto Extract(const sai::image::Image& image) noexcept
        -> Result<Embedding> override;
    [[nodiscard]] auto ExtractBatch(std::span<const sai::image::Image* const> images) noexcept
        -> Result<std::vector<Embedding>> override;
    [[nodiscard]] auto ModelName() const noexcept -> std::string_view override { return "CLIP"; }

    GlobalEmbedder(GlobalEmbedder&&) noexcept = default;
    GlobalEmbedder(const GlobalEmbedder&) = delete;

private:
    explicit GlobalEmbedder(sai::inference::ClipAdapter adapter) noexcept;
    sai::inference::ClipAdapter adapter_;
};

}  // namespace sai::embedding
```

```cpp
// dimension_reducer.h
#pragma once
#include <vector>
#include <sai/embedding/embedding.h>

namespace sai::embedding {

enum class PoolingStrategy : std::uint8_t { Average, Max };

class DimensionReducer final {
public:
    struct PcaParams {
        std::vector<float> components;
        std::size_t target_dim;
        std::vector<float> mean;
    };
    struct WhiteningParams {
        std::vector<float> transform;
        std::size_t target_dim;
    };

    [[nodiscard]] static auto FitPca(const std::vector<Embedding>& samples,
                                      std::size_t target_dim) noexcept -> Result<PcaParams>;
    [[nodiscard]] static auto FitWhitening(const std::vector<Embedding>& samples,
                                            std::size_t target_dim) noexcept -> Result<WhiteningParams>;


    // 用已拟合的 PCA 或 Whitening 参数构造——参数通常从标定文件（calibration.yaml）或
    // IImporter 加载，运行期不重新拟合。
    explicit DimensionReducer(PcaParams params) noexcept;
    explicit DimensionReducer(WhiteningParams params) noexcept;

    [[nodiscard]] auto Reduce(const Embedding& input) noexcept -> Result<Embedding>;
    [[nodiscard]] auto ReduceBatch(const std::vector<Embedding>& inputs) noexcept
        -> Result<std::vector<Embedding>>;

    [[nodiscard]] static auto Pool(const Embedding& input,
                                    PoolingStrategy strategy) noexcept -> Result<Embedding>;
};

}  // namespace sai::embedding
```

```cpp
// feature_cache.h
#pragma once
#include <cstdint>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <list>

namespace sai::embedding {

class FeatureCache final {
public:
    explicit FeatureCache(std::size_t max_entries) noexcept : max_entries_(max_entries) {}

    // 返回指向缓存中 Embedding 的只读指针——不 move 出缓存，保留 LRU 条目。
    // 指针仅在下次 Put（可能淘汰该条目）之前有效；若需延长生命周期，调用方自行拷贝。
    [[nodiscard]] auto Get(std::uint64_t key) noexcept -> const Embedding*;
    auto Put(std::uint64_t key, Embedding value) noexcept -> void;
    [[nodiscard]] auto HitRate() const noexcept -> float;
    [[nodiscard]] auto Size() const noexcept -> std::size_t;

    FeatureCache(const FeatureCache&) = delete;
    auto operator=(const FeatureCache&) -> FeatureCache& = delete;

private:
    using LruItem = std::pair<std::uint64_t, Embedding>;
    std::size_t max_entries_;
    std::list<LruItem> lru_;
    std::unordered_map<std::uint64_t, decltype(lru_)::iterator> index_;
    std::atomic<std::size_t> hits_{0};
    std::atomic<std::size_t> misses_{0};
    mutable std::mutex mutex_;
};

}  // namespace sai::embedding
```

### 4.5 Workflow

```
SurfaceImage (M2, CPU)
  → [可选] HtoD → GpuImage（若用 DINOv3/CLIP adapter，需 GPU 端）
  → PatchEmbedder::Extract(gpu_image)
    → DinoV3Adapter::Infer(gpu_image)
      → TensorRtEngine::Infer()
    → PatchFeatures{device_ptr, grid, dim}
    → Embedding::FromGpu(device_data, meta) → GPU 驻留 Embedding
  → [可选] DimensionReducer::Reduce(embedding) → 降维
  → [可选] Embedding::ToCpuAsync(queue, pinned, token) → CPU 端
  → FeatureCache::Put(key, embedding)
  → 传给 3.3 Detector / M4 Retrieval
```

### 4.6 Data Structure

见 §4.4 中的 `Embedding`、`EmbeddingMeta`、`EmbeddingType`。

### 4.7 Class Diagram

```
Object (1.1)
  └── IEmbedder ─────────── Extract(Image) → Embedding, ExtractBatch, ModelName
        ├── PatchEmbedder (final) ── DinoV3Adapter → patch Embedding (GPU)
        └── GlobalEmbedder (final) ─ CLIP Adapter → global Embedding (GPU)

独立类型：
  Embedding ──────── GPU (PooledPtr) | CPU (vector<float>), EmbeddingMeta
  DimensionReducer ─ PCA / Whitening / Pooling
  FeatureCache ───── LRU（CPU 端）
```

### 4.8 Sequence Diagram

```
Inference Worker 线程
  → PatchEmbedder::Extract(surface_image)
    → HtoD (GpuStreamQueue, 协程)
    → adapter.Infer(gpu_image)
      → engine.Infer() → PatchFeatures
    → Embedding::FromGpu(patch_features) → 返回 GPU Embedding
  → [若需 CPU 端]
    → embedding.ToCpuAsync(queue, pinned) → vector<float>
    → cache.Put(key, embedding)
```

### 4.9 Thread Model

| 场景 | 线程 | 说明 |
|------|------|------|
| 单帧提取 | WorkerPool 线程 | 同步走完 Extract → 返回 Embedding（GPU 端） |
| 批量提取 | WorkerPool 线程 × N | 多帧并行提取（每帧独立 adapter 调用，GPU 串行化） |
| DtoH 搬移 | Worker + GPU Callback | 协程挂起/恢复，不阻塞 Worker |
| FeatureCache 访问 | 任意线程 | mutex 保护 LRU |

### 4.10 Performance

| 指标 | 目标 |
|------|------|
| DINOv3 单帧 Extract（不含推理） | < 0.1ms（纯 GPU 指针包装） |
| CLIP 单帧 Extract（不含推理） | < 0.1ms |
| DtoH 搬移（256×256×1024 float32, PCIe 3.0 x16, 含 pinned memory + CUDA API overhead） | < 50ms |
| PCA Reduce（256×256 → 256×128, CPU） | < 5ms |
| Pooling（Average, 256×256 → 1×1024） | < 1ms |
| FeatureCache Get（LRU 命中） | < 1μs（hash + splice） |

### 4.11 Memory

| 对象 | 分配来源 | 生命周期 |
|------|---------|---------|
| Embedding（GPU 路径） | GpuPool | 由消费者（Detector/Cache）释放 |
| Embedding（CPU 路径） | 堆（vector<float>） | move 传递，RAII 析构 |
| FeatureCache 条目 | 堆（vector<float>） | LRU 淘汰或析构释放 |
| PcaParams/WhiteningParams | 堆 | 启动期加载，与 DimensionReducer 等长 |

### 4.12 Future Extension

- SAM2 adapter 嵌入：`SegmentationMask` 可包装为 `EmbeddingType::Segmentation`
- 多模态 embedding：CLIP text encoder → text embedding，供 M4 做跨模态检索
- 在线 PCA 更新：增量算法（如 IPCA）替代批量拟合
- GPU FeatureCache：`GpuPool` 分配 + CUDA LRU，避免 HtoD/DtoH 往返

### 4.13 Best Practice

- ✅ Embedding 不拷贝——move-only，传 `unique_ptr` 或移动语义
- ✅ Batch 提取优先——一次 `ExtractBatch` 比 N 次 `Extract` 高效（复用 engine context）
- ✅ FeatureCache key 用 image hash 而非 frame_index——同一帧重传不应 cache miss
- ✅ PCA/Whitening 参数从 `IImporter` 或 `ConfigStore` 加载——不硬编码

### 4.14 Anti Pattern

- ❌ 在 Extract 中做 DtoH——消费者可能只需要 GPU 端 embedding（如 PatchCore），不必要的搬移浪费 PCIe 带宽
- ❌ FeatureCache 存储 GPU 端 Embedding——GPU 显存是稀缺资源，缓存用 CPU 内存
- ❌ 每帧重新分配 GpuPool buffer——复用上帧的 I/O tensor buffer
- ❌ 假设所有 embedding dim 相同——DINOv3 1024D vs CLIP 512D，DimensionReducer 必须校验输入 dim

---

## 5. 批次 3.3 Detector

### 5.1 Purpose

定义统一的异常检测接口，以 PatchCore 作为第一个落地实现——接受标准化 Embedding，输出结构化检测结果（异常热力图 + 候选区域）。

### 5.2 Responsibilities

- `IDetector`：统一检测接口（`Initialize` / `Detect` / `DetectBatch`）
- `PatchCore`：coreset 特征库 + k-NN 搜索 + 上采样 + 区域提取
- `DetectionResult` / `AnomalyMap` / `RegionProposal`：结构化检测产出
- `FeatureBank`：GPU 驻留 coreset 特征矩阵，k-NN 搜索后端

### 5.3 Design

**为什么 PatchCore 离线/在线分离**：coreset 采样（greedy furthest-point）在正常样本的特征库上运行一次，结果持久化为 `.bin` 文件。推理时 `PatchCore::Initialize` 加载该文件到 GPU 显存（`FeatureBank`），`Detect` 每帧只做 k-NN 搜索 + 后处理。这避免了每帧重新做 coreset（O(N²) 复杂度）。

**为什么用 FAISS 做 k-NN 搜索**：FAISS 是框架技术栈已锁定的向量检索引擎（CLAUDE.md 技术栈表）。`faiss::IndexFlatL2` 提供精确 L2 距离，在 coreset ~10K-50K 规模下延迟可接受。faiss-gpu 可选——目标平台有 NVIDIA GPU 时，同一 `IndexFlatL2` 自动放在 GPU 上搜索。

**为什么 AnomalyMap 上游采样 + Gaussian 平滑**：PatchCore 的输出是 patch-level（步长 14 → H/14 × W/14 的网格），需要上采样到原始输入分辨率做像素级缺陷定位。Bilinear upscale → Gaussian blur 是标准的 anomaly map 后处理管线——平滑消除分辨率跳跃产生的块状伪影，阈值后连通分量标记提取候选区域。

**为什么 IDetector 包含 Initialize**：PatchCore 的 `Initialize` 加载 FeatureBank 到 GPU——这是启动期一次性的重操作。其他 detector 实现（如 EfficientAD）可能有不同的初始化需求（加载 teacher/student 网络权重）。`Initialize` 统一各 detector 的启动点。

**为什么 RegionProposal 使用 M2 的 Rect 类型**：`sai::device::Rect`（M2）是框架统一的边界框类型——2.3 的 `DefectRecord::location` 也用它。M5 Reasoner 在填充 `InspectionResult.defects` 时直接 copy `RegionProposal::bounding_box` 到 `DefectRecord::location`，零转换。

### 5.4 Interfaces

```cpp
// detection_result.h
#pragma once
#include <vector>
#include <chrono>
#include <sai/device/device.h>  // Rect

namespace sai::detection {

struct AnomalyMap {
    std::vector<float> scores;
    std::size_t grid_h = 0;
    std::size_t grid_w = 0;

    [[nodiscard]] auto At(std::size_t y, std::size_t x) const noexcept -> float {
        return scores[y * grid_w + x];
    }
    [[nodiscard]] auto MaxScore() const noexcept -> float;
    [[nodiscard]] auto IsDefective(float threshold) const noexcept -> bool {
        return MaxScore() > threshold;
    }
};

struct RegionProposal {
    sai::device::Rect bounding_box;
    float max_anomaly_score = 0.0F;
    float mean_anomaly_score = 0.0F;
    std::size_t area_pixels = 0;
};

struct DetectionResult {
    AnomalyMap anomaly_map;
    std::vector<RegionProposal> regions;
    float image_level_score = 0.0F;
    std::chrono::nanoseconds inference_latency{0};

    [[nodiscard]] auto IsDefective(float threshold) const noexcept -> bool {
        return image_level_score > threshold;
    }
};

}  // namespace sai::detection
```

```cpp
// detector.h
#pragma once
#include <span>
#include <sai/core/error.h>
#include <sai/core/object.h>
#include <sai/embedding/embedding.h>
#include <sai/detection/detection_result.h>

namespace sai {
class Context;
}  // namespace sai

namespace sai::detection {

class IDetector : public Object {
public:
    [[nodiscard]] virtual auto Initialize(sai::Context& ctx) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto Detect(const sai::embedding::Embedding& embedding) noexcept
        -> Result<DetectionResult> = 0;
    [[nodiscard]] virtual auto DetectBatch(
        std::span<const sai::embedding::Embedding* const> embeddings) noexcept
        -> Result<std::vector<DetectionResult>> = 0;
    [[nodiscard]] virtual auto ModelName() const noexcept -> std::string_view = 0;
};

}  // namespace sai::detection
```

```cpp
// patch_core.h
#pragma once
#include <filesystem>
#include <sai/detection/detector.h>

namespace sai::detection {

// GPU 驻留 coreset 特征矩阵
class FeatureBank final {
public:
    [[nodiscard]] static auto LoadFromFile(const std::filesystem::path& path,
                                            std::size_t dim) noexcept -> Result<FeatureBank>;

    [[nodiscard]] auto Search(const float* query, std::size_t query_count,
                               std::size_t k) noexcept -> Result<std::vector<float>>; // distances

    [[nodiscard]] auto NumSamples() const noexcept -> std::size_t;
    [[nodiscard]] auto Dim() const noexcept -> std::size_t;

    FeatureBank(FeatureBank&&) noexcept = default;
    FeatureBank(const FeatureBank&) = delete;

private:
    FeatureBank() noexcept = default;
    sai::memory::PooledPtr<std::uint8_t> data_;  // GpuPool 分配
    std::size_t num_samples_ = 0;
    std::size_t dim_ = 0;
};

class PatchCore final : public IDetector {
public:
    struct Config {
        std::filesystem::path feature_bank_path;
        float anomaly_threshold = 0.8F;
        std::size_t k_nearest = 1;
        std::size_t gaussian_sigma = 4;
        std::size_t image_width = 518;
        std::size_t image_height = 518;
        std::size_t patch_size = 14;
        std::size_t embed_dim = 1024;
    };

    explicit PatchCore(Config cfg) noexcept;

    [[nodiscard]] auto Initialize(sai::Context& ctx) noexcept -> Result<void> override;
    [[nodiscard]] auto Detect(const sai::embedding::Embedding& embedding) noexcept
        -> Result<DetectionResult> override;
    [[nodiscard]] auto DetectBatch(
        std::span<const sai::embedding::Embedding* const> embeddings) noexcept
        -> Result<std::vector<DetectionResult>> override;
    [[nodiscard]] auto ModelName() const noexcept -> std::string_view override
    { return "PatchCore"; }

    PatchCore(PatchCore&&) noexcept = default;
    PatchCore(const PatchCore&) = delete;

private:
    Config cfg_;
    FeatureBank feature_bank_;
};

}  // namespace sai::detection
```

### 5.5 Workflow

```
离线标定（Calibration Pipeline，非实时）:
  normal 样本集 (SurfaceImage × N)
    → DINOv3 PatchEmbedder::Extract() × N   → N×H×W 个 D 维 patch 特征
    → Coreset 采样（greedy furthest-point） → N_coreset × D 矩阵
    → 序列化到 feature_bank.bin

在线推理（每帧）:
  test SurfaceImage
    → DINOv3 PatchEmbedder::Extract()  → Embedding (H_p×W_p, D, GPU)
    → PatchCore::Detect(embedding)
      → FeatureBank::Search(query=H_p*W_p×D, k=1) → 每个 patch 的最近距离
      → DtoH → AnomalyMap(H_p, W_p)
      → Bilinear Upsample → (H, W)
      → GaussianBlur(sigma)
      → threshold → binary mask
      → connectedComponents → vector<RegionProposal>
    → DetectionResult → 传给 M5 Reasoner
```

### 5.6 Data Structure

见 §5.4 中的 `AnomalyMap`、`RegionProposal`、`DetectionResult`、`FeatureBank`、`PatchCore::Config`。

### 5.7 Class Diagram

```
Object (1.1)
  └── IDetector ─────────── Initialize(Context) / Detect(Embedding) / DetectBatch / ModelName
        └── PatchCore (final) ── FeatureBank + Config + Detect pipeline

独立类型：
  DetectionResult ── AnomalyMap + vector<RegionProposal> + image_level_score + latency
  AnomalyMap ─────── scores(grid) + grid_h/w + At/MaxScore/IsDefective
  RegionProposal ─── Rect + max_score + mean_score + area_pixels
  FeatureBank ────── GpuPool[float32: N_samples × dim] + Search(query,k) → distances
```

### 5.8 Sequence Diagram

```
Inference Worker 线程
  → PatchCore::Detect(embedding)
    → FeatureBank::Search(query, 1)  // GPU k-NN (FAISS)
      → 返回 distances[H_p × W_p]   // GPU → DtoH 同步/异步
    → BuildAnomalyMap(distances, grid_h, grid_w)
    → Upsample(grid → image_size)   // CPU bilinear
    → GaussianBlur(sigma)           // CPU
    → Threshold + connectedComponents → RegionProposals
    → DetectionResult{anomaly_map, regions, image_level_score}
    → 返回
```

### 5.9 Thread Model

| 操作 | 线程 | 说明 |
|------|------|------|
| FeatureBank::Search | CUDA worker（FAISS 内部） | GPU 端 k-NN，同步或异步 |
| Upsample/Blur/CCL | WorkerPool 线程 | CPU 端后处理，<5ms |
| DetectBatch | 多个 WorkerPool 线程 | 每帧一个 Worker，多帧并行（GPU 串行化） |

### 5.10 Performance

| 指标 | 目标 |
|------|------|
| FeatureBank::Search（10K coreset, 256×256, D=1024, k=1, L2） | < 5ms @ GPU |
| FAISS IndexFlatL2 build（10K-50K coreset） | < 50ms @ GPU |
| Bilinear Upsample（14×14 → 224×224） | < 1ms @ CPU |
| GaussianBlur（224×224, σ=4） | < 1ms @ CPU |
| connectedComponents（224×224 binary） | < 1ms @ CPU |
| PatchCore 总延迟（不含 DINOv3 推理） | < 10ms |

### 5.11 Memory

| 对象 | 分配来源 | 生命周期 |
|------|---------|---------|
| FeatureBank | GpuPool | `Initialize` 加载 → `~PatchCore` 释放 |
| AnomalyMap::scores | 堆（vector<float>） | 每帧分配，`DetectionResult` move 传递 |
| RegionProposals | 堆（vector） | 与 `DetectionResult` 等长 |
| FAISS Index（GPU） | CUDA 内部管理 | FeatureBank::Load → 析构 |

### 5.12 Future Extension

- 更多检测算法：`EfficientADDetector`、`CFADetector` 实现同一 `IDetector` 接口
- 多 backbone 支持：PatchCore 不强制 DINOv3——`PatchCore::Config` 可接受任意 `IEmbedder`
- 多类别检测：`Detect(embedding, defect_type)` 按缺陷类别切换不同的 FeatureBank
- SAM2 精细化：`RegionProposal` → `SAM2::Segment(image, bbox)` → 精确轮廓，在 M5 Reasoner 触发
- 时序异常检测：多帧 `DetectionResult` 做时序平滑/累积

### 5.13 Best Practice

- ✅ FeatureBank 启动期一次性加载到 GPU——运行期不读文件
- ✅ k-NN 距离用 L2（欧氏距离），embedding 必须先 L2-normalize（FAISS IndexFlatIP 等价）
- ✅ AnomalyMap 阈值从 `ConfigStore` 加载——不同 SKU/产线可能不同
- ✅ Gaussian sigma 锚定 patch_size——sigma ≥ patch_size / 4 避免块状伪影
- ✅ RegionProposal 按 max_anomaly_score 降序排列——M5 Reasoner 优先处理最可疑区域

### 5.14 Anti Pattern

- ❌ 在 Detect 中重新加载 FeatureBank——每帧都 I/O 不可接受
- ❌ 假设所有输入图像分辨率固定——PatchCore 需校验 `embedding.grid` 与 `cfg_.image_size/patch_size` 一致
- ❌ 忽略 k=1 的边界情况——coreset 为空或 embedding 为零向量时距离计算可能产生 NaN，必须回退为 PASS 判定
- ❌ 在 Detect 中分配 GPU buffer（`cudaMalloc`）——复用预分配的 FeatureBank buffer
- ❌ RegionProposal 没有最小面积过滤——噪声级别的 1-pixel 连通分量应直接丢弃

---

## 6. 验证点

> 一次 TensorRT 推理跑通（MockEngine 模拟），拿到 embedding 和检测得分

验证步骤：

1. 构造 `MockEngine` → `Load("test.engine", inputs, outputs)` → 返回成功
2. 构造 `DinoV3Adapter::Create(mock_engine, config)` → 成功
3. `PatchEmbedder::Create(std::move(adapter))` → 成功
4. 构造一个假 `SurfaceImage`（M2 的 `FromOwnedBuffer`，小尺寸测试图案）
5. `embedder.Extract(image)` → 返回 Embedding（CPU 端，测试数据）
6. `FeatureBank::LoadFromFile(test_coreset.bin, dim)` → 加载假 coreset
7. `PatchCore(Config{...}).Detect(embedding)` → DetectionResult 含 AnomalyMap.scores
8. 断言 `image_level_score > 0`（有异常分）或 `IsDefective(threshold)` 返回合理布尔值

---

## 7. 对外接口存量

M3 提供以下接口给后续里程碑消费：

| 接口 | 消费方 | 用途 |
|------|--------|------|
| `IInferenceEngine` | M3 内部 adapter, M6 Pipeline | 推理引擎生命周期 |
| `DinoV3Adapter` / `Sam2Adapter` / `ClipAdapter` | 3.2 Embedder, M5 Reasoner (SAM2) | 模型推理 |
| `IEmbedder` | 3.3 Detector, M4 Retrieval | 特征提取 |
| `Embedding` | 3.3 Detector, M4 FAISS Index, M5 Reasoner | 标准化特征向量 |
| `IDetector` | M6 Pipeline, M5 Reasoner | 异常检测 |
| `DetectionResult` / `AnomalyMap` / `RegionProposal` | M5 Reasoner, 2.3 IExporter | 缺陷判定+导出 |

---

## 8. 与前置/后续里程碑的关系

- M1：M3 依赖 M1 的 `GpuPool`/`PinnedPool`（显存/锁页内存分配）、`GpuStreamQueue`（异步推理回调）、`WorkerPool`（批量并行）、`ConfigStore`（模型路径/binding 配置）
- M2：M3 消费 M2 的 `SurfaceImage`/`GpuImage`（推理输入）、`sai::device::Rect`（RegionProposal 边界框）、`InspectionResult`（M5 填充时引用结构定义）
- M4（知识与检索）：使用 3.2 的 `Embedding`（global, CLIP）建 FAISS 索引，使用 `FeatureCache` 做 reference 特征缓存
- M5（推理决策）：消费 3.3 的 `DetectionResult` → 填充 M2 的 `InspectionResult.defects`；消费 SAM2 adapter 做缺陷边界精修
- M6（编排调度）：M3 的 `IInferenceEngine` / `IEmbedder` / `IDetector` 作为 Pipeline 图中的可替换节点
