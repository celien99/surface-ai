# 多机位 Coreset 检测支持

> 里程碑：填补 M3（Detection）与 M7（Application）之间的设计缺口
> 依赖批次：M3 Detection（PatchCore/FeatureBank）、M7 Application（seat_aoi）
> 本批次修改涉及检测层、pipeline 路由层、应用装配层，不引入新模块。

## 1. Purpose

当前 PatchCore 持有单个 `FeatureBank`，所有相机机位的特征向量混在同一个 FAISS 索引中做 k-NN 检索。在真实工业产线中，不同机位拍摄的物体表面角度、光照、背景完全不同——机位 A 的正常纹理在机位 B 的索引中找不到合理近邻，导致误判。本批次为检测链路补齐多机位支持：

- 每个 `(surface_id, position_id)` 拥有独立的 `FeatureBank` + `PatchCore` + `CoresetEvolution` 实例
- `position_id` 从 `ImageMeta` 经 `Embedding` → `DetectionResult` 全程透传，作为检测路由 key
- coreset 文件通过 YAML manifest 注册，与现有 dataset YAML 风格一致
- 单机位场景完全兼容（position=0 为默认值）

## 2. Responsibilities

本批次负责：

- 扩展 `Embedding`：添加 `position_id_` 成员 + getter/setter
- 扩展 `DetectionResult`：添加 `position_id` 字段用于透传
- 改造 `InferenceStage::Process()`：从 `ImageMeta` 拷贝 `position_id` 到 `Embedding`
- 改造 `DetectStage`：维护 `map<(surface_id, position_id), shared_ptr<IDetector>>`，根据 embedding 的 position 路由到正确的检测器实例
- 扩展 `RuleEvalStage::Process()` 和 `ReasonStage`：透传 `position_id`（不参与推理逻辑，仅保持数据完整）
- 定义 coreset manifest YAML 格式（`resources/coresets/{surface}.yaml`）
- 改造 `BuildCoreset`：按 `position_id` 分组构建多个 coreset，自动生成 manifest
- 改造 `AppBuilder::AssembleApplication`：解析 coreset manifest → 为每个 `(sid, pid)` 创建 PatchCore + FeatureBank + Evolution 实例 → 注入 DetectStage
- CLI 新增 `--coreset-manifest` 参数；`--coreset` 保持单机位兼容（position=0）

本批次不负责：

- 多光源融合逻辑（`light_id` 的多光源取并集策略由 headless runner 的统计层处理，不在本批次范围）
- SAM2 集成（仍为 M5 placeholder）
- OPC UA / MES 集成

## 3. Design

**采用方案 B：多个 PatchCore 实例，每个机位独立检测器。** 拒绝方案 A（PatchCore 内部 `map<position, FeatureBank>`），因为不同机位的检测阈值（角度、光照差异导致异常分数分布不同）、k 值可能需要独立配置，独立实例天然支持这一点且 PatchCore 职责不变。拒绝方案 C（BankProvider 回调注入），因为多一层间接调用对热路径（每帧 k-NN 检索）增加不必要的函数指针开销，且配置管理反而不如显式 map 直观。

**Coreset 文件组织采用 YAML manifest，拒绝目录约定和复用 dataset manifest。** 目录约定要求机位编号从 0 连续（生产线上机位 ID 可能是任意 uint16），且无法显式声明路径映射——排查问题时需要靠文件名猜。复用 dataset manifest 将 coreset 路径与图像路径混在一起，职责不清，同一机位的多张图片会冗余声明相同 coreset。YAML manifest 显式、灵活、可读，且与 dataset YAML 风格一致：

```yaml
surface: "driver_seat"
banks:
  - position: 0
    path: "pos_0.bin"
  - position: 1
    path: "pos_1.bin"
```

**自进化采用方案 A：每个机位独立 CoresetEvolution 实例，独立后台线程。** 拒绝方案 B（先做静态再补进化），因为静态多机位交付后仍需补齐进化，且届时进化接口可能已与多实例 PatchCore 产生耦合，改动范围不会更小。每个机位独立进化带来的线程数增长是可接受的——产线上典型机位数 ≤ 8，每个进化线程是低频率后台任务（分钟级间隔），不会与 GPU 推理线程争抢资源。

## 4. Interfaces

### 4.1 Embedding 扩展

```cpp
// embedding.h — 新增成员
class Embedding {
public:
    // existing...
    [[nodiscard]] auto SurfaceId() const noexcept -> const std::string&;
    auto SetSurfaceId(std::string id) noexcept -> void;
    
    // NEW
    [[nodiscard]] auto PositionId() const noexcept -> std::uint16_t;
    auto SetPositionId(std::uint16_t id) noexcept -> void;
    
private:
    std::string surface_id_{};
    std::uint16_t position_id_ = 0;  // NEW
};
```

### 4.2 DetectionResult 扩展

```cpp
// detection_result.h — 新增字段
struct DetectionResult {
    // existing...
    std::string surface_id;
    std::uint16_t position_id = 0;  // NEW
};
```

### 4.3 DetectStage 接口变更

```cpp
// stage_nodes.h — DetectStage
class DetectStage final : public IStageNode {
public:
    // OLD: SetDetector(shared_ptr<IDetector>)
    // NEW: register detector for a specific (surface_id, position_id)
    auto AddDetector(std::string surface_id, std::uint16_t position_id,
                     std::shared_ptr<sai::detection::IDetector> det) -> void;
    
    // Convenience: register for single-position (position=0, surface_id="")
    auto SetDetector(std::shared_ptr<sai::detection::IDetector> det) -> void {
        AddDetector("", 0, std::move(det));
    }
    
    // Access for CoresetEvolution wiring
    auto GetDetector(std::string_view surface_id, std::uint16_t position_id) const
        -> std::shared_ptr<sai::detection::IDetector>;
    
private:
    std::string id_;
    // OLD: shared_ptr<IDetector> detector_;
    // NEW:
    using BankKey = std::pair<std::string, std::uint16_t>;
    std::map<BankKey, std::shared_ptr<sai::detection::IDetector>> detectors_;
    std::shared_ptr<sai::detection::IDetector> default_detector_;  // fallback
    bool stub_ = true;
};
```

### 4.4 Coreset Manifest

```cpp
// 新文件: include/sai/io/coreset_manifest.h
namespace sai::io {

struct CoresetBankEntry {
    std::uint16_t position_id = 0;
    std::filesystem::path path;
};

struct CoresetManifest {
    std::string surface_id;
    std::vector<CoresetBankEntry> banks;
};

// Parse from YAML file
auto LoadCoresetManifest(const std::filesystem::path& yaml_path) noexcept
    -> Result<CoresetManifest>;

// Generate manifest from built coresets directory
auto SaveCoresetManifest(const std::filesystem::path& yaml_path,
                          const CoresetManifest& manifest) noexcept -> Result<void>;

}  // namespace sai::io
```

### 4.5 CliArgs 扩展

```cpp
struct CliArgs {
    // existing...
    std::string coreset_path;           // --coreset: single-position compat (position=0)
    std::string coreset_manifest_path;  // --coreset-manifest: multi-position YAML
};
```

## 5. Workflow

### 5.1 Coreset 构建流程

```
BuildCoreset --dataset dataset.yaml --coreset-output resources/coresets/
  │
  ├─ 1. ImportDataset → vector<DatasetEntry>
  │
  ├─ 2. Group by position_id:
  │      map<position_id, vector<DatasetEntry>>
  │
  ├─ 3. For each position:
  │      Extract embeddings → FeatureBank::BuildWithGreedyCoreset
  │      → SaveToFile("pos_{pid}.bin")
  │
  ├─ 4. Generate manifest:
  │      CoresetManifest{surface_id, [{0, "pos_0.bin"}, {1, "pos_1.bin"}, ...]}
  │      → SaveCoresetManifest("resources/coresets/{surface}.yaml")
  │
  └─ 5. Print summary per position
```

### 5.2 检测流程（运行时）

```
ImageMeta.(surface_id="driver_seat", position_id=1)
  │
  ▼ InferenceStage::Process
  │   emb.SetSurfaceId(meta.surface_id)
  │   emb.SetPositionId(meta.position_id)    // NEW
  │
  ▼ DetectStage::Process
  │   key = (emb.SurfaceId(), emb.PositionId())
  │   detector = detectors_[key] ?? default_detector_
  │   result = detector->Detect(emb)
  │   result.surface_id = emb.SurfaceId()
  │   result.position_id = emb.PositionId()  // NEW
  │
  ▼ RuleEvalStage::Process
  │   透传 (surface_id, position_id)
  │
  ▼ ReasonStage::Process
  │   透传
  │
  ▼ ExportStage → ResultCallback
```

### 5.3 自进化流程（每机位独立）

```
ResultCallback(frame_id, result)
  │
  │  key = (result.surface_id, result.position_id)
  │  evolution = evolutions_[key]
  │
  ▼ evolution.AssessAndOffer(...)
  │
  │  (后台线程，分钟级间隔)
  ▼
  │  new_bank = BuildCoreset(candidates)
  │  patch_core = detectors_[key]
  │  old_bank = patch_core.SwapFeatureBank(std::move(new_bank))
  │  old_bank → backup file
```

## 6. Data Structure

### 6.1 DetectStage 内部路由表

```
detectors_: map<BankKey, shared_ptr<IDetector>>
  BankKey = (surface_id: string, position_id: uint16)

  Example:
    ("driver_seat", 0)  → PatchCore[0] (pos_0.bin coreset)
    ("driver_seat", 1)  → PatchCore[1] (pos_1.bin coreset)
    ("driver_seat", 2)  → PatchCore[2] (pos_2.bin coreset)
    ("passenger_seat", 0) → PatchCore[3] (pos_0.bin coreset)

  default_detector_: fallback when key not found
```

### 6.2 AppBuilder 装配产物

```cpp
struct AssembledApp {
    // ... existing fields ...

    // ── 多机位检测器 ──
    // Ownership: PatchCore owns FeatureBank.
    // Declared BEFORE evolutions so destruction order is evolutions → patch_cores
    std::map<std::pair<std::string, std::uint16_t>,
             std::shared_ptr<sai::detection::PatchCore>> patch_cores;

    // ── 每机位独立自进化 ──
    // Each evolution holds a PatchCore& (non-owning). Declared AFTER patch_cores
    // so C++ reverse destruction order destroys evolutions first, then patch_cores.
    std::map<std::pair<std::string, std::uint16_t>,
             sai::detection::CoresetEvolution> evolutions;
    std::map<std::pair<std::string, std::uint16_t>,
             std::stop_source> evolution_stop_sources;

    // ── 向后兼容：单机位模式 ──
    std::shared_ptr<sai::detection::PatchCore> patch_core;  // position=0 default
    std::optional<sai::detection::CoresetEvolution> evolution;
};
```

## 7. Class Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          DetectStage                                 │
├──────────────────────────────────────────────────────────────────────┤
│ - detectors_: map<BankKey, shared_ptr<IDetector>>                   │
│ - default_detector_: shared_ptr<IDetector>                           │
├──────────────────────────────────────────────────────────────────────┤
│ + AddDetector(sid, pid, det): void                                  │
│ + GetDetector(sid, pid): shared_ptr<IDetector>                       │
│ + Process(StageInput) → Result<StageOutput>                         │
└──────┬───────────────────────────────────────────────────────────────┘
       │ 1:N
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          PatchCore                                   │
├──────────────────────────────────────────────────────────────────────┤
│ - feature_bank_: unique_ptr<FeatureBank>                            │
│ - last_ctx_: DetectionContext                                       │
├──────────────────────────────────────────────────────────────────────┤
│ + Detect(Embedding&) → Result<DetectionResult>                      │
│ + SwapFeatureBank(unique_ptr<FeatureBank>) → unique_ptr<FeatureBank> │
│ + LastContext() → const DetectionContext&                            │
└──────┬───────────────────────────────────────────────────────────────┘
       │ owns
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          FeatureBank                                 │
├──────────────────────────────────────────────────────────────────────┤
│ - index_: unique_ptr<faiss::Index>                                  │
│ - dim_: size_t                                                       │
│ - num_samples_: size_t                                              │
├──────────────────────────────────────────────────────────────────────┤
│ + Search(query, count, k) → vector<float>                           │
│ + SaveToFile(path) → Result<void>                                    │
│ + LoadFromFile(path, dim) → Result<FeatureBank>                     │
└──────────────────────────────────────────────────────────────────────┘

Per-position evolution:

┌──────────────────────────────────────────────────────────────────────┐
│                       CoresetEvolution                              │
├──────────────────────────────────────────────────────────────────────┤
│ - patch_core_: PatchCore& (reference to corresponding instance)     │
│ - profile_: NormalityProfile                                        │
│ - buffer_: CandidateBuffer                                          │
├──────────────────────────────────────────────────────────────────────┤
│ + AssessAndOffer(distances, ..., result, verdict, ...) → void       │
│ + Start(stop_token) → void                                          │
│ + Stop() → void                                                     │
└──────────────────────────────────────────────────────────────────────┘
```

## 8. Sequence Diagram

### 8.1 多机位检测一帧

```
Camera  →  ImageMeta(sid="S1", pid=2, lid=0)
  │
  │  CaptureStage::Process
  │
  ▼  InferenceStage::Process
  │     emb = patch_embedder.Extract(image)
  │     emb.SetSurfaceId("S1")
  │     emb.SetPositionId(2)
  │     StageOutput(move(emb))
  │
  ▼  DetectStage::Process
  │     emb = get<Embedding>(input)
  │     key = (emb.SurfaceId(), emb.PositionId())  = ("S1", 2)
  │     det = detectors_[key] ?? default_detector_
  │     result = det->Detect(emb)
  │     result.surface_id = "S1"
  │     result.position_id = 2
  │     StageOutput(move(result))
  │
  ▼  RuleEvalStage::Process
  │     fact_builder.Build(result)  // (sid, pid) carried in facts
  │     rule_engine.Evaluate(facts)
  │
  ▼  ReasonStage::Process
  │     reasoner.Reason(rule_output)
  │     → verdict
  │
  ▼  ExportStage → JSON + ResultCallback
```

### 8.2 Coreset 构建

```
BuildCoreset --dataset seat.yaml
  │
  │  ImportDataset → [{sid="S1", pid=0, ...}, {sid="S1", pid=1, ...}, ...]
  │
  │  Group by position_id:
  │    pid=0 → [entries...]
  │    pid=1 → [entries...]
  │
  ├─ pid=0:
  │    DINOv3 → embeddings → FeatureBank::BuildWithGreedyCoreset
  │    → SaveToFile("pos_0.bin")
  │
  ├─ pid=1:
  │    DINOv3 → embeddings → FeatureBank::BuildWithGreedyCoreset
  │    → SaveToFile("pos_1.bin")
  │
  └─ SaveCoresetManifest("S1.yaml", {
       surface_id: "S1",
       banks: [{position: 0, path: "pos_0.bin"},
               {position: 1, path: "pos_1.bin"}]
     })
```

## 9. Thread Model

- **检测路径**（多 reader）：每个 `PatchCore::Detect()` 只读 `FeatureBank` 的 FAISS 索引，多个 pipeline worker 线程并发调用不同机位的 Detect。`FeatureBank::Search()` 是 const 方法，FAISS 读操作线程安全。
- **进化路径**（单 writer per bank）：每个 `CoresetEvolution` 独占一个后台 `std::jthread`，通过 `stop_source` 控制生命周期。`SwapFeatureBank` 使用 `std::unique_ptr` 的 move 语义完成原子替换——旧 bank 由返回的 `unique_ptr` 保持存活直到调用方回收，新 bank 写入后立即对新的 reader 可见。
- **bank 映射读写**：`DetectStage::detectors_` 和 `evolutions_` 在 `Pipeline::Start()` 之前完成装配，运行时只读——无需锁。

## 10. Performance

- **路由开销**：`map<pair<string, uint16>, shared_ptr<IDetector>>` 查找为 O(log N)，机位数 N ≤ 8，一次查找 < 100ns，相比 k-NN 检索（O(N·D)，N 为 coreset 大小，通常 1000~10000，D=1024，耗时 ~1ms）可忽略。
- **内存**：每个机位独立 FAISS 索引。coreset 典型大小 10000 样本 × 1024 维 × 4 bytes = ~40 MB per position。8 机位 ≈ 320 MB，在现代 GPU 服务器上可行。
- **进化线程**：每机位一个后台线程，分钟级唤醒间隔，CPU 占用 < 0.1%。

## 11. Memory

- `DetectStage::detectors_` 持有 `shared_ptr<IDetector>`，实际指向 `PatchCore` 实例，后者持有 `unique_ptr<FeatureBank>`（FAISS 索引）。生命周期由 `AssembledApp` 中的 `patch_cores` map 保证。
- `CoresetEvolution` 持有 `PatchCore&` 引用（不拥有），`evolutions` 在 `AssembledApp` 中声明于 `patch_cores` 之后——C++ 逆序析构保证 evolution 实例先于 PatchCore 释放。
- `CandidateBuffer` 每个进化实例持有最大 50000 patch × 1024 dim × 4 bytes ≈ 200 MB 的候选缓冲区。

## 12. Future Extension

- **bank 热加载/热替换**：当前 `SwapFeatureBank` 已支持运行时替换，未来可与 OPC UA 指令集成（MES 下发"切换到新 coreset"）。
- **跨 surface bank 共享**：如果多个 surface 共用同一机位配置，`AddDetector` 的 key 支持任意 `(sid, pid)` 映射，可注册同一 PatchCore 实例到多个 key。
- **多光源 coreset**：`light_id` 暂不参与路由，但 `BankKey` 可扩展为 `(surface_id, position_id, light_id)` 三元组（当前保留 `light_id` 的 metadata 传递能力，路由 key 不包含它）。

## 13. Best Practice

- **单机位场景**：用 `SetDetector(det)` 或 `AddDetector("", 0, det)`，行为与重构前完全一致。`--coreset` CLI 参数保持兼容。
- **多机位场景**：始终用 `--coreset-manifest` 指定 manifest YAML，不要手动管理文件命名。
- **构建 coreset**：用 `--dataset` + `--coreset-output <dir>` 自动分组构建并生成 manifest，不要手动调 `BuildCoreset` 多次。
- **position_id 分配**：在同一 surface 内 position_id 应唯一且连续从 0 开始（不强制，但建议——便于 manifest 可读性）。

## 14. Anti Pattern

- **不要**在 `DetectStage::Process()` 里用 `SurfaceId()` / `PositionId()` 做字符串拼接生成 key——用 `pair<string, uint16>` 直接比较，避免每次帧都分配临时字符串。
- **不要**在 coreset manifest 里用绝对路径——所有路径相对于 manifest 文件所在目录解析。
- **不要**假设 position_id 从 0 开始连续——`AddDetector` 接受任意 uint16 position_id，manifest 里显式声明。
- **不要**在 headless runner 里直接访问 `detectors_` map 做自进化——通过 `ResultCallback` 的 `result.position_id` 查找对应 evolution 实例。
