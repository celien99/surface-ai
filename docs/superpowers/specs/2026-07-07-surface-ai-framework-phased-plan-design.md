# Surface AI Framework —— 分阶段执行计划设计文档

> Status: Approved
> Date: 2026-07-07
> Source: `prompt.md`（原始 master prompt，1180 行）
> Purpose: 把原始 master prompt 中笼统的"支持 A/B/C/D"式技术选型收敛为确定技术栈，并把巨型单次生成任务拆分为可执行、可验证的批次计划。

---

## 1. 背景与问题

原始 `prompt.md` 定义了一个 17 层的工业级 C++20 Surface AI Framework 设计任务，存在两个结构性问题：

1. **产出量级与单次生成能力不匹配**：17 层 × 每层数个子模块 × 每模块 20+ 项必需内容（架构图/类图/序列图/线程模型/性能策略等），叠加"Never skip any section"、"Never generate simplified content"的强约束，实际产出量远超单次生成上限，直接执行会导致被截断或注水，违反文档自身的质量要求。
2. **技术选型笼统，无法收敛为可执行设计**：原文档大量使用"支持 X/Y/Z/W"的清单式写法（例如推理后端同时列 TensorRT/ONNX Runtime/OpenVINO/DirectML，向量检索同时列 FAISS/Milvus/Qdrant/SQLite），没有做取舍。这种写法在设计阶段看似灵活，实际会导致后续生成时对同一模块给出前后矛盾的接口定义。

此外还发现两个次要问题：模块概念跨层重叠（如 Reflectance 同时出现在 Imaging 和 Knowledge 模块）、机制堆叠但无边界说明（如 Runtime 层同时要求 Fiber + Coroutine + 硬实时调度器，这些机制在工程上通常互斥）。

本文档的目标是解决以上问题，产出一份可直接执行的分阶段计划。

---

## 2. 已确定技术栈

以下技术选型已与用户逐项确认，全部收敛为唯一确定值，后续所有批次的 Design 章节必须基于此表，禁止再使用"支持 A/B/C"的清单式写法。

| 决策点 | 选定技术 | 备注 |
|---|---|---|
| 主力部署平台 | Ubuntu 22.04 x64 + NVIDIA GPU | Windows/ARM64 作为长期兼容目标，非本轮优化对象 |
| 推理后端 | TensorRT | FP16/INT8/动态 shape/多 GPU，与 CUDA Stream/Pinned Memory/Zero-Copy 集成 |
| 向量检索 | FAISS（可选 faiss-gpu） | 进程内嵌入，无需独立部署服务 |
| 规则引擎 | 自研表达式引擎（AST）+ YAML 声明式存储 | 不引入 Lua，避免任意代码执行风险 |
| 配置格式 | YAML（yaml-cpp） | 与规则引擎存储格式统一，降低认知负担 |
| 并发模型 | C++20 协程（co_await）+ 固定阶段 Worker Pool | 不做硬实时保证；GPU 任务通过 CUDA Stream + callback 回调到协程 |
| 日志 | spdlog + 异步 sink | 配合 fmt 库 |
| 依赖管理 | vcpkg | TensorRT/CUDA 等 vendor SDK 仍需手动安装 |
| 错误处理 | tl::expected 为主，异常仅用于真正例外场景（构造失败、资源初始化失败） | 避免热路径异常带来不可预测性能开销 |
| 测试框架 | GoogleTest + gmock | 用于 mock 相机/PLC 设备接口 |
| GUI 框架 | Qt 6 | 需确认商业/GPL 授权合规性 |
| PLC 通信协议 | OPC UA（open62541） | 跨主流 PLC 厂商中立标准 |
| 相机采集标准 | GenICam / GigE Vision | 通过统一 GenTL 传输层接入不同厂商相机，避免为每个厂商写专用插件 |
| 知识库元数据存储 | SQLite | 字段式数据（Material/Supplier/Batch），与 FAISS 同样走进程内嵌入路径 |
| 部署方式 | Docker 容器 + systemd | 配合 nvidia-container-toolkit 调用 GPU |

---

## 3. 拆分策略：方案对比与选择

评估了三种批次拆分策略：

- **方案 1（按原文档 17 层顺序）**：改动最小，但批次粒度严重不均衡（Core 层 13 个子模块 vs IO 层内容单薄），且没有阶段性验证点。
- **方案 2（按依赖关系聚类为里程碑，已选择）**：把 17 层按真实依赖关系压缩为 7 个里程碑，每个里程碑对应一个可验证的能力节点，而非单纯的文档产出。同时引入跨批次一致性机制解决概念重叠问题。
- **方案 3（MVP 垂直切片优先）**：能最快验证架构假设，但与"10 年可维护、跨行业不重新设计架构"的框架定位冲突，容易导致核心接口后续返工。

**选定方案 2。**

---

## 4. 里程碑与批次拆分

7 个里程碑，18 个批次：

### 里程碑 1：基座设施
- 1.1 Core 基础（Object / Result(tl::expected) / Resource / Type System / Reflection / Interface System）
- 1.2 Core 生命周期与装配（Module / Service / Context / Lifecycle / DI / Factory / Registry）
- 1.3 Core 插件体系（Plugin / PluginManager / ModuleManager / VersionManager / LicenseManager / CapabilityManager —— 合并原文档独立的"PLUGIN SYSTEM"章节）
- 1.4 Runtime（协程 Task Scheduler / 固定 Worker Pool / CUDA Stream GPU Queue / Task Graph / Pipeline Executor / Cancellation / Back Pressure / Retry / Metrics —— 不含 Fiber、不含硬实时调度器）
- 1.5 Memory（Memory/Object/Image/Tensor/GPU Pool / Pinned Memory / Arena Allocator / Zero Copy / Reference Counting / SIMD Alignment / Leak Detection —— NUMA 列为未来扩展，v1 不做）
- 1.6 横切关注点（Error Handling=tl::expected / Logging=spdlog async / Configuration=YAML）

**验证点**：能启动一个空 task graph，走完 Runtime 调度、Memory 池分配、Logging 输出、Config 加载的最小闭环。

### 里程碑 2：采集与影像
- 2.1 Device（Camera=GenICam/GigE Vision 插件、PLC=OPC UA 插件、统一 Device 接口与插件生命周期）
- 2.2 Imaging（RawImage→SurfaceImage 类型体系、Calibration/HDR/FlatField/白平衡/去畸变等预处理链、GPU Image、ROI）
- 2.3 IO（导入导出接口、Exporter 插件）

**验证点**：GenICam 相机采一帧，走完预处理链，产出 SurfaceImage。

### 里程碑 3：AI 推理核心
- 3.1 Foundation（TensorRT engine 抽象、动态 shape/FP16/INT8/多 GPU/热重载，DINOv3/SAM2/CLIP 作为具体 adapter 实现）
- 3.2 Embedding（patch/global embedding、PCA/白化/池化、距离度量、feature cache）
- 3.3 Detector（统一 Detector 接口，PatchCore 作为第一个落地实现，其余算法留扩展点）

**验证点**：一次 TensorRT 推理跑通，拿到 embedding 和检测得分。

### 里程碑 4：知识与检索
- 4.1 Knowledge（Material/Supplier/Batch 等字段落 SQLite，Knowledge Graph/Evolution/Snapshot）
- 4.2 Retrieval（FAISS TopK/Range/Hybrid，SQLite 做 metadata filter join，Score Fusion）

**验证点**：写入一条知识记录，检索命中并返回融合分数。

### 里程碑 5：推理决策
- 5.1 Rule（自研表达式 AST + YAML 规则文件、动态重载、优先级与冲突消解）
- 5.2 Reasoner（汇总 Detection+Knowledge+Rule，输出 Severity/Recommendation/Confidence/Trace/Evidence）

**验证点**：一条 YAML 规则触发，Reasoner 输出带 Trace 的结论。

### 里程碑 6：编排调度
- 6.1 Pipeline（DAG 配置化节点图，串联里程碑 1-5 的所有能力，节点可替换、异步执行）
- 6.2 Scheduler（把 Pipeline 节点落到 Runtime 的线程池模型：Capture/Inference/Retrieval/Reason/IO/Logging/GUI/Background/GPU 线程，Lock-Free Queue/Ring Buffer 落地）

**验证点**：完整异步流水线跑通一帧从采集到决策输出。

### 里程碑 7：呈现与首个应用
- 7.1 Visualization（Qt 6 实时预览、图表、参数面板）
- 7.2 Application（参考实现，如 Seat AOI，只做模块组合、零业务逻辑，验证"Product 只是 metadata"的核心设计原则）

---

## 5. 跨批次一致性机制

维护一份贯穿所有批次的活文档：`docs/surface-ai/glossary-and-contracts.md`（《术语与接口契约表》），包含两部分：

1. **概念归属表**：每个跨层出现的名词（Reflectance/Material/Texture/Geometry 等）只在唯一一层定义，其他层通过引用而非重新定义。例如 Reflectance 的原始测量值归 Imaging 层，Knowledge 层只存指向它的统计摘要。
2. **核心接口签名表**：`ISurface`、`IPlugin`、`IDetector`、`IDevice` 等跨层接口一旦在某批次中定稿，后续批次必须复用签名，不能各自发明。

每个批次的生成 prompt 开头强制注入这份契约表的当前内容。批次产出后如果新增了共享概念或接口，必须先更新契约表再产出章节正文，并输出本批次的契约表增量 diff，供合并进主契约文档。

---

## 6. 批次产出物与 Prompt 模板

**批次产出物**：纯设计文档，不产出可编译代码。代码实现留给后续独立的开发阶段（届时可用 `superpowers:executing-plans` 按里程碑逐个实现）。

**批次 Prompt 模板结构**：

```
# 上下文注入（每批次开头强制携带）
- 已确定技术栈表（第 2 节，原样贴入）
- 术语与接口契约表当前内容（截至本批次之前所有已定稿的概念归属 + 接口签名）
- 本批次所属里程碑 + 本批次在里程碑中的依赖关系（依赖哪些已完成批次的接口）

# 任务声明
- 本批次范围：仅限本批次清单内的子模块，不越界到其他批次职责
- 明确排除项：原文档中列出但本轮不做的机制（例如涉及 Runtime 时注明"不含 Fiber/硬实时调度"）

# 产出章节（遵循原文档 Document Style，固定顺序）
1. Purpose
2. Responsibilities
3. Design（基于已确定技术栈给出选定方案的具体设计，禁止"支持 A/B/C"清单式写法）
4. Interfaces（C++ 头文件级别的接口声明，非实现）
5. Workflow
6. Data Structure
7. Class Diagram（PlantUML 或 Mermaid 文本）
8. Sequence Diagram
9. Thread Model（仅当涉及跨线程交互时，复用里程碑 1 定稿的线程模型）
10. Performance（给出可验证的数字目标，非口号式表述）
11. Memory（生命周期、归属、池化策略，复用里程碑 1 的 Memory 子系统）
12. Future Extension
13. Best Practice
14. Anti Pattern

# 收尾动作
- 若本批次引入新的跨层共享概念或接口，先写入契约表增量，再产出本批次正文
- 输出契约表本批次的增量 diff，供合并进主契约文档
```

**核心约束**：Design 章节禁止出现"支持 X/Y/Z"的清单式写法，必须是"采用 X，因为...，拒绝 Y 因为..."的确定性表述。

---

## 7. 执行顺序

批次按里程碑编号顺序执行（1.1 → 1.2 → ... → 7.2），同一里程碑内的批次存在依赖顺序（如 1.1 早于 1.2，因为生命周期管理依赖基础类型系统）。跨里程碑之间严格顺序执行，因为里程碑 2-7 均直接或间接依赖里程碑 1 的接口。

---

## 8. 后续步骤

本 spec 批准后，下一步调用 `superpowers:writing-plans` 生成里程碑 1（批次 1.1-1.6）的详细实施计划。里程碑 1 完成并通过验证点后，再规划里程碑 2 的实施计划。不在此 spec 阶段一次性规划全部 7 个里程碑的实施细节。
