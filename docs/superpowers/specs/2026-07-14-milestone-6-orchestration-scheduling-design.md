# Surface AI Framework —— 里程碑 6 编排调度 设计文档

> Status: Draft
> Date: 2026-07-14
> Based on: `docs/superpowers/specs/2026-07-07-surface-ai-framework-phased-plan-design.md` §4 里程碑 6
> Depends on: 里程碑 1（1.1-1.6 全部冻结接口）、里程碑 2（2.1-2.3 全部冻结接口）、里程碑 3（3.1-3.3 全部冻结接口）、里程碑 4（4.1-4.2 全部冻结接口）、里程碑 5（5.1-5.2 全部冻结接口）

---

## 1. 背景与范围

里程碑 6 覆盖"编排调度"：在 M1-M5 各独立模块之上，构建配置驱动的 Pipeline 编排层和阶段间 Scheduler 调度层，将采集→预处理→推理→检测→规则求值→推理决策→导出的全链路串联为异步流水线。

### 1.1 架构总览

两层分离，每层定义于独立批次：

```
6.1 Pipeline（编排层）
    PipelineConfig（YAML 反序列化目标）
    PipelineBuilder（YAML → M1 TaskGraph 拓扑）
    IStageNode（业务语义化阶段节点接口）
    Pipeline（生命周期宿主 + 监控入口，内部持有 M1 TaskGraph + PipelineExecutor）
    StageQueue<T>（阶段间 lock-free SPSC/MPSC 队列 + BackpressurePolicy）

6.2 Scheduler（调度层）
    Scheduler（StageType → WorkerPool 映射 + 队列分配）
    StageMetrics（per-stage 原子计数器）
```

### 1.2 与 M1 TaskGraph 的关系

M6 Pipeline **包装** M1 `TaskGraph` 作为底层 DAG 执行引擎：

- M1 `TaskGraph`：泛型 DAG 节点（`TaskNode`）+ 依赖边 + `PipelineExecutor` 分派到 `WorkerPool`
- M6 `Pipeline`：持有 `TaskGraph` 实例，对外暴露业务语义接口（`Submit(RawImage)` / `Drain()` / `Metrics()`）
- M6 `PipelineBuilder`：把 YAML 配置翻译为 `TaskGraph::AddNode()` 调用，节点包裹的是 `IStageNode::Process()`
- M6 `Scheduler`：为每种 `StageType` 从 `Registry<WorkerPool>` 查找对应的物理线程池，创建阶段间队列

### 1.3 批次划分与执行顺序

```
里程碑 6：编排调度
├── 6.1 Pipeline（PipelineConfig → PipelineBuilder → IStageNode → Pipeline → StageQueue）
└── 6.2 Scheduler（Scheduler → StageMetrics）
```

执行顺序：**6.1 设计 → 6.1 代码 → 6.2 设计 → 6.2 代码 → E2E 集成 + 契约更新**

### 1.4 项目锚点

延续 M1-M5 的汽车座椅 AOI 场景：

- **单帧完整链路**：相机采帧 → Debayer+FlatField+白平衡+Resize → DINOv3 推理 → PatchCore 检测 → 规则引擎求值 → Reasoner 决策 → JSON 报告导出
- **异步流水线**：7 个阶段并行执行，`Submit()` 异步返回，帧在阶段间通过 lock-free 队列传递
- **背压可配置**：Capture 阶段 `drop_oldest`（旧帧时效性差），Export 阶段 `block`（判定结果不能丢），中间阶段默认 `block`
- **产线切换**：换产品 SKU 只需修改 Pipeline YAML（换 detector、换 rule_set、换 decision_tree），不重编译

### 1.5 明确排除项

- 多 GPU 调度（v1 不做，Future Extension）
- 分布式 Pipeline（跨进程阶段——v1 不做）
- Pipeline 热重载（运行时原子替换拓扑——M6 做 YAML 加载期校验 + 启动期构建，热重载延后至 Future Extension）
- 条件分支（`depends_on` 运行时动态选路径——v1 不做，DAG 拓扑启动期固定）
- GPU → CPU fallback（推理排队超时降级——v1 不做）
- 跨帧状态共享（每帧独立处理，不维护帧间状态）

---

## 2. 对里程碑 1/2/3/4/5 的依赖

| M6 批次 | 依赖接口 | 用途 |
|---------|---------|------|
| 6.1 | `TaskGraph`、`PipelineExecutor`、`TaskNode`（1.4） | Pipeline 内部的 DAG 拓扑 + 执行引擎 |
| 6.1 | `WorkerPool`、`TaskScheduler`（1.4） | 阶段线程池 + 任务提交 |
| 6.1 | `Result<T>`、`Object`（1.1） | 错误处理 + IStageNode 基类 |
| 6.1 | `ConfigStore`、`Logger`（1.6） | Pipeline YAML 加载 + 日志 |
| 6.1 | `ICamera`（2.1） | Capture 阶段设备 |
| 6.1 | `PreprocessFn`、`RawImage`、`SurfaceImage`（2.2） | Preprocess 阶段 |
| 6.1 | `IInferenceEngine`（3.1） | Inference 阶段 |
| 6.1 | `IDetector`、`DetectionResult`（3.3） | Detect 阶段 |
| 6.1 | `RuleEngine`、`FactBuilder`（5.1） | RuleEval 阶段 |
| 6.1 | `IReasoner`、`ReasoningResult`（5.2） | Reason 阶段 |
| 6.1 | `IExporter`（2.3） | Export 阶段 |
| 6.2 | `Registry<WorkerPool>`（1.2/1.4） | StageType → WorkerPool 查找 |
| 6.2 | `GpuStreamQueue`（1.4） | Lock-free MPSC 算法参考（Vyukov 有界 MPSC） |

---

## 3. 架构设计

### 3.1 核心设计原则

**Push 模型 + 阶段间队列**。每个阶段完成后主动将结果 Push 到下游队列，下游 WorkerPool 线程从队列 Pop 并处理。队列是阶段间的解耦边界——上游不关心下游的处理速率，队列深度自然形成缓冲。

```
Pipeline::Submit(raw_image)
        │
        ▼
  ┌───────────┐    Q0    ┌───────────┐    Q1    ┌───────────┐
  │  Capture  │─────────►│Preprocess │─────────►│ Inference │
  │  Pool(2)  │          │  Pool(2)  │          │  Pool(1)  │
  └───────────┘          └───────────┘          └─────┬─────┘
                                                       │ Q2
                                                       ▼
                                                  ┌───────────┐
                                                  │  Detect   │
                                                  │  Pool(1)  │
                                                  └─────┬─────┘
                                                        │ Q3
  ┌───────────┐    Q5    ┌───────────┐    Q4    ┌───────────┐
  │  Export   │◄─────────│  Reason   │◄─────────│  RuleEval │
  │  Pool(1)  │          │  Pool(1)  │          │  Pool(2)  │
  └───────────┘          └───────────┘          └───────────┘
```

### 3.2 架构取舍

| 决策 | 采用 | 拒绝 | 理由 |
|------|------|------|------|
| 编排方式 | YAML + C++ 混合 | 纯 YAML / 纯代码 | 拓扑在 YAML（产线切换只改配置），节点行为在 C++ 工厂/插件（复用 M1-M5 能力）；纯 YAML 无法表达复杂节点行为，纯代码违背"配置化"意图 |
| 执行引擎 | 包装 M1 TaskGraph | 全新 Pipeline 引擎 / 直接复用 | M1 `TaskGraph` 的 DAG 拓扑 + `PipelineExecutor` 分派已成熟；M6 只加业务语义层（StageType、背压、监控），不重做执行引擎 |
| 驱动模型 | Push（生产者驱动） | Pull（消费者驱动） | 相机硬件触发源不适合 Pull——帧到了必须接；Push 队列自然形成背压缓冲 |
| 背压策略 | 可配置（block/drop_oldest/degrade），默认 block | 单一策略 | 不同阶段对丢帧的容忍度不同：Capture 时效性优先（drop_oldest）、Export 完整性优先（block）；默认 block 遵循"工业产线不漏检"的安全原则 |
| 阶段间队列 | bounded SPSC/MPSC lock-free ring buffer | `std::queue` + mutex / 无界队列 | 有界队列提供背压信号；lock-free 避免 Pipeline 热路径上的锁竞争；SPSC 针对默认一进一出的拓扑优化 |
| 节点可替换性 | 启动期工厂装配 | 运行期热替换 | M6 不做热重载（见 1.5 排除项）；启动期通过 YAML `config` 段选择具体实现（detector: PatchCore / rule_set: xxx.yaml） |
| StageInput/Output | `std::variant<...>` + 加载期类型校验 | 模板化 `Pipeline<Input, Output>` / `std::any` | variant 在加载期可穷举检查类型兼容性；模板化导致类型组合爆炸（7 个阶段的排列）；`std::any` 失败时无编译期/加载期类型信息 |

---

## 4. 模块结构

### 4.1 文件布局

```
include/sai/pipeline/
    pipeline_config.h        # PipelineConfig, StageConfig, BackpressureConfig, StageType, BackpressurePolicy
    stage_node.h             # IStageNode, StageInput, StageOutput
    pipeline.h               # Pipeline
    stage_queue.h            # StageQueue<T>（模板，header-only）

src/pipeline/
    CMakeLists.txt
    pipeline_builder.h       # 内部：PipelineBuilder（YAML → TaskGraph）
    stage_factory.h          # 内部：StageFactory（StageType → IStageNode）
    pipeline_builder.cpp
    stage_factory.cpp
    pipeline.cpp
    capture_stage.cpp        # Capture 阶段实现
    preprocess_stage.cpp     # Preprocess 阶段实现
    inference_stage.cpp      # Inference 阶段实现
    detect_stage.cpp         # Detect 阶段实现
    rule_eval_stage.cpp      # RuleEval 阶段实现
    reason_stage.cpp         # Reason 阶段实现
    export_stage.cpp         # Export 阶段实现

src/scheduler/
    CMakeLists.txt
    scheduler.h              # 内部：Scheduler
    stage_metrics.h          # StageMetrics
    scheduler.cpp

tests/pipeline/
    pipeline_config_test.cpp
    stage_queue_test.cpp
    pipeline_builder_test.cpp
    pipeline_test.cpp
    integration_test.cpp

tests/scheduler/
    scheduler_test.cpp
    integration_test.cpp
```

### 4.2 命名空间

- `sai::pipeline`：6.1 Pipeline + 6.2 Scheduler 的所有公共类型

两个批次的类型统一在 `sai::pipeline` 命名空间下——Scheduler 是 Pipeline 的调度子系统，不作为独立命名空间暴露。

---

## 5. 接口设计

### 5.1 PipelineConfig

```cpp
namespace sai::pipeline {

enum class StageType {
    Capture,
    Preprocess,
    Inference,
    Detect,
    RuleEval,
    Reason,
    Export,
    Custom
};

enum class BackpressurePolicy {
    Block,          // 上游阻塞等待
    DropOldest,     // 丢弃队列最旧元素
    Degrade         // 跳过非关键阶段（v1 预留枚举值，实现仅限 Block/DropOldest）
};

struct StageConfig {
    std::string id;
    StageType type;
    std::vector<std::string> depends_on;
    YAML::Node config;                      // 阶段特定参数，透传给工厂
    BackpressurePolicy backpressure;        // 覆盖 pipeline 默认（可选）
    std::optional<size_t> queue_capacity;   // 覆盖默认队列容量（可选）
};

struct BackpressureConfig {
    BackpressurePolicy default_policy = BackpressurePolicy::Block;
    std::map<std::string, BackpressurePolicy> stage_overrides;
};

struct PipelineConfig {
    std::string name;
    std::string version;
    BackpressureConfig backpressure;
    std::vector<StageConfig> stages;
};

} // namespace sai::pipeline
```

### 5.2 IStageNode

```cpp
namespace sai::pipeline {

// StageInput / StageOutput：阶段间传递的数据 variant
// 注：FeatureMap 是 M3 inference 的内部类型，此处前向声明
using StageInput = std::variant<
    RawImage,
    SurfaceImage,
    DetectionResult,
    std::vector<rule::ResolvedRule>,
    rule::ReasoningResult
>;

using StageOutput = StageInput;

class IStageNode : public Object {
public:
    virtual auto GetType() const noexcept -> StageType = 0;
    virtual auto GetId() const -> std::string_view = 0;
    virtual auto OnInitialize(Context&) -> Result<void> = 0;
    virtual auto OnStart(Context&) -> Result<void> = 0;
    virtual auto OnStop(Context&) -> Result<void> = 0;
    virtual auto Process(StageInput) -> Result<StageOutput> = 0;
};

} // namespace sai::pipeline
```

**生命周期约定**：
- `OnInitialize`：加载期调用，阶段在此获取依赖资源（设备连接、模型加载、规则文件解析）
- `OnStart`：Pipeline 启动时调用，阶段在此开始接收帧
- `OnStop`：Pipeline 停止时调用，阶段在此释放资源
- `Process`：每帧调用一次，输入来自上游队列，输出推入下游队列

**失败语义**：
- `Process()` 返回 `Result<StageOutput>` 的失败 → 本帧标记为 ERROR + 日志记录，不影响后续帧
- `OnInitialize()`/`OnStart()` 失败 → `Pipeline_StageInitFailed`，Pipeline 整体拒绝启动

### 5.3 StageQueue

```cpp
namespace sai::pipeline {

template<typename T>
class StageQueue {
public:
    static auto Create(size_t capacity, BackpressurePolicy policy)
        -> Result<std::unique_ptr<StageQueue>>;

    // 非阻塞 Push，成功返回 true，队列满返回 false
    auto TryPush(std::unique_ptr<T> item) -> bool;

    // 阻塞 Push，等待到有空位
    auto PushBlocking(std::unique_ptr<T> item) -> void;

    // 非阻塞 Pop，队列空返回 nullptr
    auto TryPop() -> std::unique_ptr<T>;

    // 阻塞 Pop，等待到有元素
    auto PopBlocking() -> std::unique_ptr<T>;

    auto Depth() const noexcept -> size_t;
    auto Capacity() const noexcept -> size_t;

private:
    // 内部实现：
    // - 单生产者单消费者（SPSC）→ 无锁 ring buffer + cache line padding
    // - 默认拓扑是一进一出，SPSC 覆盖绝大多数场景
    // - fan-in（多上游→一下游）自动升级为 MPSC（复用 Vyukov 有界 MPSC 算法）
};

} // namespace sai::pipeline
```

### 5.4 Pipeline

```cpp
namespace sai::pipeline {

struct StageMetrics {
    std::string stage_id;
    StageType type;
    std::atomic<size_t> frames_processed{0};
    std::atomic<size_t> frames_failed{0};
    std::atomic<size_t> frames_dropped{0};
    std::atomic<size_t> queue_depth{0};
    std::chrono::microseconds avg_latency{0};   // 非原子，由 Metrics() 计算
    std::chrono::microseconds p99_latency{0};   // 非原子，由 Metrics() 计算
};

class Pipeline {
public:
    static auto LoadFromYAML(std::filesystem::path, Context&)
        -> Result<std::unique_ptr<Pipeline>>;

    auto Start() -> Result<void>;
    auto Submit(RawImage) -> Result<void>;
    auto Drain() -> Result<void>;
    auto Stop() -> Result<void>;
    auto Metrics() const -> std::vector<StageMetrics>;

private:
    std::unique_ptr<TaskGraph> graph_;
    std::unique_ptr<PipelineExecutor> executor_;
    std::map<std::string, std::unique_ptr<IStageNode>> nodes_;
    // queues_ 按边索引：(from_stage_id, to_stage_id) -> StageQueue
    std::map<std::pair<std::string, std::string>, std::unique_ptr<detail::ErasedStageQueue>> queues_;
    std::stop_source stop_source_;              // 广播给所有 StageLoop
    std::atomic<bool> running_{false};
    std::atomic<bool> draining_{false};
};

} // namespace sai::pipeline
```

### 5.5 Scheduler

```cpp
namespace sai::pipeline {

class Scheduler {
public:
    explicit Scheduler(Registry<WorkerPool>& pools, const BackpressureConfig& bp_config);

    // 为 stages 分配 WorkerPool + 创建 StageQueue
    auto Allocate(const std::vector<StageConfig>& stages) -> Result<void>;
    auto Deallocate() -> Result<void>;

    auto PoolFor(StageType) const -> Result<WorkerPool&>;

private:
    Registry<WorkerPool>& pools_;
    BackpressureConfig bp_config_;
    std::map<StageType, TypeId> stage_pool_map_;     // StageType → WorkerPool TypeId
    std::map<std::string, std::unique_ptr<detail::ErasedStageQueue>> queues_;
};

} // namespace sai::pipeline
```

### 5.6 StageType → WorkerPool 映射表

| StageType | M1 stage_id | 默认线程数 | 默认队列容量 | 说明 |
|-----------|-------------|-----------|-------------|------|
| `Capture` | Capture | 2 | 4 | 相机回调线程，延迟敏感 |
| `Preprocess` | Capture | 复用 Capture 池 | 复用 Capture 池 | CPU 图像处理，与 Capture 共享池 |
| `Inference` | Inference | 1 | 4 | GPU 推理，排队等 CUDA stream |
| `Detect` | Inference | 复用 Inference 池 | 复用 Inference 池 | 共享 GPU 上下文 |
| `RuleEval` | Reason | 2 | 16 | CPU 密集型 AST 解释执行 |
| `Reason` | Reason | 复用 Reason 池 | 复用 Reason 池 | 决策树遍历 |
| `Export` | IO | 1 | 32 | 磁盘 I/O，队列深缓冲 |
| `Custom` | Background | 1 | 16 | 用户插件隔离 |

**设计决策**：`Inference` 和 `Detect` 共享 `Inference` 池——两个阶段都依赖 CUDA stream，分开反而造成 GPU 上下文切换开销。`RuleEval` 和 `Reason` 同理共享 `Reason` 池。

线程数默认值偏保守（产线节拍通常 > 1s/帧），可在 YAML `stages[].config.thread_count` 覆盖。

---

## 6. YAML Schema

### 6.1 Pipeline 文件格式

```yaml
pipeline:
  name: "seat_aoi_inspection"
  version: "1.0"
  backpressure:
    default: block
    stage_overrides:
      capture: drop_oldest

  stages:
    - id: capture
      type: Capture
      depends_on: []
      config:
        device: "genicam_camera_0"
        trigger: hardware

    - id: preprocess
      type: Preprocess
      depends_on: [capture]
      config:
        steps: [debayer, flat_field, white_balance, resize]
        resize_width: 1024
        resize_height: 1024

    - id: inference
      type: Inference
      depends_on: [preprocess]
      config:
        engine: TensorRtEngine
        model: dino_v3_vit_base
        batch_size: 1
        precision: fp16

    - id: detect
      type: Detect
      depends_on: [inference]
      config:
        detector: PatchCore
        feature_bank: faiss_gpu
        k_nearest: 5

    - id: rule_eval
      type: RuleEval
      depends_on: [detect]
      config:
        rule_file: "rules/seat_leather_defects.yaml"
        graph_paths:
          - material->supplier->batch.reject_rate
          - material->supplier->batch.quality_score
        vector_top_k: 5

    - id: reason
      type: Reason
      depends_on: [rule_eval]
      config:
        tree_file: "trees/seat_leather_inspection.yaml"

    - id: export
      type: Export
      depends_on: [reason]
      config:
        exporter: JsonExporter
        output_dir: "/var/log/surface-ai/results/"
        save_annotated_image: true
```

### 6.2 Schema 校验规则

- `name`：必填，非空字符串
- `version`：必填，`\d+\.\d+` 格式
- `stages`：至少 1 个阶段
- `stages[].id`：必填，唯一，非空
- `stages[].type`：必填，必须是 `StageType` 枚举值之一
- `stages[].depends_on`：引用的 `id` 必须存在，拓扑必须无环
- 所有 `depends_on: []` 的阶段中必须至少一个是 `Capture` 类型——Pipeline 的入口必须是采帧
- 所有不被任何阶段依赖的阶段中必须至少一个是 `Export` 类型——Pipeline 的出口必须是导出

---

## 7. 工作流

### 7.1 Pipeline 加载与构建

```
Pipeline::LoadFromYAML(path, context)
  ├── Step 1: YAML::Load(path) → PipelineConfig
  ├── Step 2: PipelineBuilder::Validate(config)
  │   ├── name/version 非空，version 格式校验
  │   ├── stages[].id 唯一性检查
  │   ├── stages[].depends_on 引用存在性检查
  │   ├── 拓扑无环检查（Kahn's algorithm）
  │   ├── 入口检查：至少一个 stage 的 depends_on 为空且 type == Capture
  │   ├── 出口检查：至少一个 stage 不被任何其他 stage 依赖且 type == Export
  │   └── 阶段类型兼容性检查：
  │       Capture → Preprocess: RawImage → RawImage ✓
  │       Preprocess → Inference: SurfaceImage（Preprocess 内部产出）✓
  │       Inference → Detect: DetectionResult（经 M3 adapter）✓
  │       Detect → RuleEval: DetectionResult ✓
  │       RuleEval → Reason: ResolvedRule[] ✓
  │       Reason → Export: ReasoningResult ✓
  │       不兼容 → Pipeline_StageTypeMismatch
  ├── Step 3: Scheduler::Allocate(config.stages)
  │   ├── 为每种 StageType 的 thread_count 总和创建或查找 WorkerPool
  │   ├── 为每条有向边创建 StageQueue<T>（T = 上游输出类型）
  │   └── 失败 → Scheduler_PoolNotFound / Scheduler_QueueCreateFailed
  ├── Step 4: StageFactory::Create(config)
  │   ├── Capture → CaptureStage(device_config) → Wrap M2 ICamera
  │   ├── Preprocess → PreprocessStage(steps_config) → Wrap M2 PreprocessFn
  │   ├── Inference → InferenceStage(model_config) → Wrap M3 IInferenceEngine
  │   ├── Detect → DetectStage(detector_config) → Wrap M3 IDetector
  │   ├── RuleEval → RuleEvalStage(rule_config) → Wrap M5 RuleEngine + FactBuilder
  │   ├── Reason → ReasonStage(tree_config) → Wrap M5 IReasoner
  │   ├── Export → ExportStage(exporter_config) → Wrap M2 IExporter
  │   └── Custom → CustomStage(plugin_config) → Wrap 用户插件
  ├── Step 5: IStageNode::OnInitialize(context) × N
  │   └── 任一失败 → Pipeline_StageInitFailed
  ├── Step 6: 构建 M1 TaskGraph
  │   ├── 每个 stage → TaskNode { id = stage.id, stage_id = Scheduler::PoolFor(type),
  │   │                         dependencies = stage.depends_on }
  │   │                          work = lambda 包装:
  │   │                           1. 从上游 StageQueue 读取 StageInput
  │   │                           2. stage->Process(input)
  │   │                           3. Push 结果到下游 StageQueue
  │   └── graph_->AddNode(task_node) × N
  └── Step 7: 返回 Pipeline(graph, executor, nodes, queues)
```

### 7.2 运行期（单帧）

```
Pipeline::Submit(raw_image)
  ├── running_ == false → 拒绝，返回 Pipeline_PipelineRunning（注：语义为"未启动"）
  ├── 构造 StageInput(raw_image)
  └── Push 到 capture→preprocess 队列（入口队列）
      └── 返回（异步，不等待帧完成）

Pipeline::Drain()
  ├── draining_ = true
  └── 等待所有 StageQueue 为空（即所有在途帧处理完毕）+ 所有 WorkerPool 空闲
      └── 超时 30s → 强制清空队列 + 丢弃未完成帧

Pipeline::Stop()
  ├── stop_source_.request_stop()  // 信号所有 StageLoop 退出 while 循环
  ├── 标记 running_ = false，拒绝新的 Submit()
  ├── Drain()：等待所有 TaskNode 完成（TaskGraph 拓扑执行结束）
  ├── 逆序遍历 IStageNode[]：node->OnStop(context)
  ├── Scheduler::Deallocate()      // 销毁所有 StageQueue
  └── 标记为已停止
```

### 7.3 阶段 Process 包装（TaskNode 的 work lambda）

每个阶段在 TaskGraph 中注册为一个持久运行的协程任务（而非每帧一个 TaskNode）。TaskNode 内部循环：从上游队列 Pop → Process → Push 到下游队列。

```
Task<void> StageLoop(input_queue, output_queues[], stage_node, metrics, stop_token):
  while (!stop_token.stop_requested()):
    input = input_queue.PopBlocking()  // 阻塞等上游帧
    if (!input) continue;              // 空唤醒（stop_token 触发）

    auto start = steady_clock::now()
    result = stage_node->Process(std::move(*input))

    if result.has_value():
      metrics.frames_processed.fetch_add(1, memory_order::relaxed)
      for each downstream_q in output_queues:
        bool pushed = downstream_q->TryPush(
            std::make_unique<StageOutput>(std::move(result.value())))
        if (!pushed):
          switch backpressure_policy:
            case Block:
              downstream_q->PushBlocking(...)  // 阻塞等待
            case DropOldest:
              downstream_q->TryPop()           // 丢弃最旧
              downstream_q->PushBlocking(...)  // 推入最新
            case Degrade:
              // v1 预留，实现标记为 unimplemented
              downstream_q->PushBlocking(...)  // fallback to block
    else:
      metrics.frames_failed.fetch_add(1, memory_order::relaxed)
      logger.Error("Stage {}: {}", stage_id, result.error().message)
      // 失败的帧不传播到下游

    auto elapsed = steady_clock::now() - start
    // 滑动窗口更新 avg/p99 latency（非原子，由 Metrics() 快照时计算）
```

---

## 8. 数据流与类型兼容性

### 8.1 阶段间数据类型契约

| 上游阶段 | 下游阶段 | 上游输出类型 | 下游期望输入类型 | 校验方式 |
|---------|---------|-------------|-----------------|---------|
| Capture | Preprocess | `RawImage` | `RawImage` | variant index 匹配 |
| Preprocess | Inference | `SurfaceImage` | `SurfaceImage` | variant index 匹配 |
| Inference | Detect | `DetectionResult` | `DetectionResult` | variant index 匹配 |
| Detect | RuleEval | `DetectionResult` | `DetectionResult` | variant index 匹配 |
| RuleEval | Reason | `ResolvedRule[]` | `ResolvedRule[]` | variant index 匹配 |
| Reason | Export | `ReasoningResult` | `ReasoningResult` | variant index 匹配 |

**注**：`Inference` 阶段的内部 `FeatureMap` 不暴露给 Pipeline——`InferenceStage::Process()` 内部完成 `FeatureMap → DetectStage` 的衔接，将 `FeatureMap` 暂存在 `Context` 中供 `DetectStage` 读取。这是 M3 内部的数据流约定，不在 Pipeline 层序列化。

### 8.2 类型兼容性校验（加载期）

`PipelineBuilder::Validate()` 持有 StageType→输入输出签名表：

```cpp
struct StageSignature {
    StageType type;
    std::type_index input_type;   // variant 中的期望输入 index
    std::type_index output_type;  // variant 中的期望输出 index
};
```

校验逻辑：遍历每条有向边 `(A→B)`，检查 `signature[A].output_type == signature[B].input_type`。不匹配 → `Pipeline_StageTypeMismatch`。

---

## 9. 线程模型

### 9.1 线程池分配

```
┌──────────────────────────────────────────────┐
│ Pipeline::Submit() 调用线程（通常为相机回调）  │
│ → Push 到入口队列 → 立即返回（异步）           │
└──────────────────────────────────────────────┘
                    │
    ┌───────────────┼──────────────────────────┐
    │               ▼                          │
    │  Capture Pool (2 threads)                │
    │  ├── CaptureStage.Process()              │
    │  │   └── ICamera 触发 → RawImage          │
    │  └── PreprocessStage.Process()           │
    │      └── PreprocessFn compose → SurfaceImage
    │           │                              │
    │           ▼ Q1                           │
    │  Inference Pool (1 thread)               │
    │  ├── InferenceStage.Process()            │
    │  │   └── IInferenceEngine → FeatureMap    │
    │  └── DetectStage.Process()               │
    │      └── IDetector → DetectionResult      │
    │           │                              │
    │           ▼ Q3                           │
    │  Reason Pool (2 threads)                 │
    │  ├── RuleEvalStage.Process()             │
    │  │   └── RuleEngine + FactBuilder         │
    │  └── ReasonStage.Process()               │
    │      └── IReasoner → ReasoningResult      │
    │           │                              │
    │           ▼ Q5                           │
    │  IO Pool (1 thread)                      │
    │  └── ExportStage.Process()               │
    │      └── IExporter → filesystem           │
    └──────────────────────────────────────────┘
```

### 9.2 线程安全约定

| 对象 | 线程模型 | 说明 |
|------|---------|------|
| `Pipeline` | `Submit()`/`Drain()`/`Stop()` 可被外部多线程调用 | 内部 `std::atomic<bool> running_/draining_` 保护状态转换 |
| `StageQueue<T>` | SPSC/MPSC lock-free | Push/Pop 各自单线程，无锁。内部 ring buffer 用 `std::atomic<size_t>` head/tail + cache line padding |
| `IStageNode::Process()` | 各自 WorkerPool 线程内串行 | 同一 stage 的多帧按队列顺序处理，无并发 |
| `StageMetrics` | `std::atomic<size_t>` 计数器 | 各阶段独立更新，无全局锁 |
| `PipelineConfig` | 只读 | 加载后不可变 |

### 9.3 错误恢复

- 单个 `Process()` 失败 → 本帧标记为 ERROR，`StageMetrics.frames_failed++`，Logger::Error 记录 stage_id + surface_id + error message，帧不传播到下游
- WorkerPool 线程 panic（未捕获异常）→ `std::terminate`（与 M1 的线程模型一致——WorkerPool 不捕获异常）
- 背压 block 超时（30s）→ `Pipeline_QueueFull`，Pipeline 进入 degraded 状态，`Drain()` 强制清空队列

---

## 10. 性能目标

| 指标 | 目标 | 条件 |
|------|------|------|
| Pipeline 空载启动延迟 | < 500ms | 7 个阶段，WorkerPool 创建 + StageQueue 分配 |
| `Submit()` 返回延迟 | < 1μs | 纯 Queue push，不做业务处理 |
| 单帧端到端延迟 | < 200ms | 复用 M2-M5 各自性能预算：Preprocess < 10ms + Inference < 50ms + Detect < 10ms + RuleEval < 50ms + Reason < 1ms + Export < 10ms |
| Pipeline 稳态吞吐 | ≥ 5 fps | 单 GPU，DINOv3 ViT-Base，7 阶段流水线并行 |
| StageQueue Push/Pop | < 100ns | SPSC，单元素，无竞争 |
| 背压 block 恢复 | < 队列容量 × 单帧时间 | 例：Q=4, 帧=200ms → < 800ms |

### 10.1 性能策略

- **Lock-free SPSC queue**：cache line padding 防止 false sharing，原子操作 `memory_order_acquire/release` 而非 `seq_cst`
- **Pipeline 并行**：阶段间流水线并行——当帧 N 在 Inference 阶段时，帧 N+1 可同时在 Preprocess 阶段
- **零数据拷贝**：`SurfaceImage` 的缓冲区通过 `unique_ptr` 传递所有权，不在阶段间拷贝像素数据
- **队列容量调优**：生产环境通过 `stages[].queue_capacity` 逐阶段调优，而非一刀切

---

## 11. 内存策略

| 对象 | 生命周期 | 分配策略 |
|------|---------|---------|
| `PipelineConfig` | 加载期→下次重载 | 栈分配，移动给 `PipelineBuilder` |
| `IStageNode[]` | `Start()` → `Stop()` | `unique_ptr`，每个持有各自 M1-M5 对象的 `shared_ptr`（`ICamera`、`IInferenceEngine`、`IDetector`、`RuleEngine`、`IReasoner`、`IExporter`） |
| `StageQueue<T>[]` | `Start()` → `Stop()` | 固定大小 ring buffer（`unique_ptr<T>[]` + `atomic<size_t>` head/tail），元素 `unique_ptr<StageOutput>` 无额外堆分配 |
| `StageOutput` | 单帧 | `unique_ptr` 沿队列传递所有权，下游 Pop 后自动析构 |
| `StageMetrics` | `Start()` → 终身 | `atomic<size_t>` 内嵌在 Pipeline 对象，无额外堆分配 |
| `TaskGraph` + `PipelineExecutor` | `Start()` → `Stop()` | M1 对象，Pipeline 以 `unique_ptr` 持有 |

**关键零拷贝路径**：
- 所有 `StageOutput` 以 `unique_ptr` 传递所有权——无引用计数开销，无共享所有权复杂性
- `SurfaceImage` 的像素缓冲区随 `unique_ptr<SurfaceImage>` 传递，不拷贝
- `DetectionResult` / `ResolvedRule[]` / `ReasoningResult` 同理——Move 语义沿 Pipeline 单向流动

---

## 12. ErrorCode

M6 新增 `Pipeline_*` / `Scheduler_*` 错误码，append-only 追加在 M5 错误码 `Reasoner_ScoreComputationFailed` 之后：

| 错误码 | 含义 | 触发条件 |
|--------|------|---------|
| `Pipeline_InvalidConfig` | Pipeline YAML schema 校验失败 | 缺少必填字段 / 循环依赖 / `depends_on` 引用不存在的阶段 / 入口非 Capture / 出口非 Export |
| `Pipeline_StageTypeMismatch` | 相邻阶段数据类型不兼容 | 上游输出类型 ≠ 下游期望输入类型 |
| `Pipeline_StageInitFailed` | 阶段初始化失败 | `OnInitialize()` 或 `OnStart()` 返回失败（设备连接失败、模型加载失败等） |
| `Pipeline_PipelineRunning` | Pipeline 状态错误 | 未启动时调用 `Submit()` / 已启动时调用 `Start()` |
| `Pipeline_QueueFull` | 背压 block 超时 | 下游队列满 + block 等待超过 deadline（30s） |
| `Scheduler_PoolNotFound` | StageType 无对应 WorkerPool | `Registry<WorkerPool>` 中缺少所需阶段的线程池 |
| `Scheduler_QueueCreateFailed` | Lock-free 队列分配失败 | 内存不足 / capacity 为 0 |

7 个错误码，按表内顺序追加。M1 `error.h` 附加规则同样适用——永不重排、永不碰其他批次的成员。

---

## 13. 未来扩展

| 扩展点 | 方向 | 触发条件 |
|--------|------|---------|
| 多 GPU 调度 | `StageType::Inference` 按 GPU ID 分配不同 `WorkerPool`，Pipeline YAML 中 `config.gpu_id` 选卡 | 多 GPU 部署 |
| 分布式 Pipeline | 阶段拆分到不同进程/节点，`StageQueue` → 网络传输（gRPC/shared memory） | 超大规模产线（多工位→中心分析） |
| Pipeline 热重载 | `Pipeline::Reload(path)` 原子替换拓扑而不断流 | 产线不停机切换产品 SKU |
| 条件分支 | `depends_on` 支持运行时表达式——基于上一阶段 `StageOutput` 的字段值选择下游路径 | 不同缺陷类型走不同推理分支 |
| GPU → CPU fallback | 推理队列超时后自动降级到 ONNX Runtime CPU | GPU 故障或过载 |
| Degrade 策略实现 | 队列水位超过阈值时跳过非关键阶段（`Export` 的 annotated image），水位恢复后恢复 | 高吞吐压力场景 |
| Pipeline 版本管理 | `PipelineConfig.version` SemVer + 变更 diff + `ReasoningResult` 记录 `pipeline_version` | 多产线不同 Pipeline 版本共存 |

---

## 14. 最佳实践

1. **队列容量 = 下游处理速率 × 可容忍的最大延迟**。不要盲目设大队列——队列越深，`drop_oldest` 时被丢弃的帧越旧
2. **`Capture` 阶段始终 `drop_oldest`**——相机帧具有时效性，处理不完的旧帧应丢弃而非阻塞产线节拍
3. **`Export` 阶段始终 `block`**——判定结果和 Evidence 不能丢，宁可写盘慢也不能跳过
4. **Pipeline YAML 版本化**——每次修改拓扑或参数后递增 `version`，`ReasoningResult` 中记录 `pipeline_version` 以便问题回溯
5. **阶段间只传 `unique_ptr`**——避免共享所有权的复杂性，所有权随帧在 Pipeline 中单向流动
6. **`Process()` 失败不阻塞 Pipeline**——单帧失败记录到 `StageMetrics.frames_failed` 和日志，继续处理下一帧
7. **加载期校验优先**——`Validate()` 发现的配置错误在 `Pipeline_InvalidConfig` 中给出精确诊断（循环依赖的具体路径、类型不兼容的上下游 stage id），而非运行时崩溃

## 15. 反模式

1. **在 `Process()` 中做阻塞 I/O**——数据库查询/文件读写应在 M5 `FactBuilder` 物化阶段批量完成，不在 Pipeline 主路径上
2. **队列容量设为 0**——`Scheduler_QueueCreateFailed`，拒绝创建
3. **跨阶段共享可变状态**——`StageOutput` 的所有权随队列传递，下游是唯一消费者。不通过 `Context` 或全局变量在阶段间传递可变数据
4. **Pipeline YAML 中硬编码设备 ID**——应使用逻辑设备名，由 M2 Device 层的设备发现机制解析到物理设备
5. **跳过 `Validate()` 直接启动**——拓扑错误（循环依赖、类型不兼容）在生产中表现为运行时 crash 或静默丢帧，而非优雅的 `Pipeline_InvalidConfig`
6. **假设阶段顺序固定**——`depends_on` 允许非线性的 DAG 拓扑（如 `Inference` → `Detect` + `Inference` → `RuleEval` 分叉），不要在代码中做"第 N 个阶段一定是 X"的隐式假设

---

## 16. 验证点

**主验证点**：完整异步流水线跑通一帧从采集到决策输出。

具体场景：模拟一帧 `RawImage` 经过 7 个阶段全链路处理，产出 `ReasoningResult` + JSON 报告。

验证链：
1. `Pipeline::LoadFromYAML("seat_aoi_pipeline.yaml")` → 校验通过，TaskGraph 构建成功
2. `Pipeline::Start()` → 创建 WorkerPool × 4（Capture/Inference/Reason/IO）、StageQueue × 6
3. `Pipeline::Submit(mock_raw_image)` → 立即返回
4. Capture 阶段消费 → Preprocess 阶段预处理 → Inference 阶段（MockEngine）推理 → Detect 阶段（MockDetector）检测 → RuleEval 阶段求值 → Reason 阶段决策 → Export 阶段写 JSON
5. `Pipeline::Drain()` → 等待帧完成
6. `Pipeline::Metrics()` → 每阶段 `frames_processed == 1`
7. `Pipeline::Stop()` → 阶段逆序停止，资源释放

**子验证点**：
- 背压 drop_oldest：Capture 队列满时，Submit 的新帧替换队列中最旧帧
- 阶段失败隔离：中间阶段 Process() 失败，本帧标记失败但不影响后续帧
- 拓扑校验：循环依赖、类型不兼容的 YAML 在加载时被拒绝

---

## 17. 契约增量

M6 新增以下概念与接口到 `glossary-and-contracts.md`：

### 17.1 概念归属表新增

| 概念名称 | 归属批次 | 定义所在文档 | 说明 |
|---------|---------|-------------|------|
| `Pipeline` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | 配置驱动的编排宿主，持有 M1 TaskGraph + PipelineExecutor，对外暴露 `Submit()`/`Drain()`/`Stop()`/`Metrics()` |
| `PipelineConfig` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | YAML 反序列化的 Pipeline 配置：name + version + backpressure + stages[] |
| `IStageNode` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | 业务语义化阶段节点接口：`OnInitialize`/`OnStart`/`OnStop`/`Process` |
| `StageQueue<T>` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | 阶段间 bounded SPSC/MPSC lock-free ring buffer，内建 BackpressurePolicy |
| `Scheduler` | 6.2 | design/milestone-06-orchestration-scheduling/6.2-scheduler.md | StageType → WorkerPool 映射 + 阶段间队列分配 |
| `StageMetrics` | 6.2 | design/milestone-06-orchestration-scheduling/6.2-scheduler.md | per-stage 原子计数器：frames_processed / failed / dropped / queue_depth / latency |

### 17.2 接口签名表新增

| 接口名称 | 归属批次 | 定义所在文档 | 签名摘要 |
|---------|---------|-------------|---------|
| `IStageNode` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | `class IStageNode : public Object { virtual auto GetType() const noexcept -> StageType = 0; virtual auto GetId() const -> string_view = 0; virtual auto OnInitialize(Context&) -> Result<void> = 0; virtual auto OnStart(Context&) -> Result<void> = 0; virtual auto OnStop(Context&) -> Result<void> = 0; virtual auto Process(StageInput) -> Result<StageOutput> = 0; }` |
| `Pipeline` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | `class Pipeline { static auto LoadFromYAML(path, Context&) -> Result<unique_ptr<Pipeline>>; auto Start() -> Result<void>; auto Submit(RawImage) -> Result<void>; auto Drain() -> Result<void>; auto Stop() -> Result<void>; auto Metrics() const -> vector<StageMetrics>; }` |
| `StageQueue<T>` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | `template<typename T> class StageQueue { static auto Create(size_t, BackpressurePolicy) -> Result<unique_ptr<StageQueue>>; auto TryPush(unique_ptr<T>) -> bool; auto PushBlocking(unique_ptr<T>) -> void; auto TryPop() -> unique_ptr<T>; auto PopBlocking() -> unique_ptr<T>; auto Depth() const noexcept -> size_t; }` |
| `Scheduler` | 6.2 | design/milestone-06-orchestration-scheduling/6.2-scheduler.md | `class Scheduler { explicit Scheduler(Registry<WorkerPool>&, const BackpressureConfig&); auto Allocate(const vector<StageConfig>&) -> Result<void>; auto Deallocate() -> Result<void>; auto PoolFor(StageType) const -> Result<WorkerPool&>; }` |
