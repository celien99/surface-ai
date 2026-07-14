# Surface AI Framework —— 术语与接口契约表

> 本文档是跨批次的活文档。每个设计批次完成后，必须在此追加自己的契约增量，
> 不得修改其他批次已提交的行（若确需修改，必须在 PR 描述中说明原因并经评审）。

## 1. 概念归属表

| 概念名称 | 归属批次 | 定义所在文档 | 说明 |
|---|---|---|---|
| `Object` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | 框架内所有一等对象的公共基类语义，禁止拷贝与移动 |
| `Result<T>` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `tl::expected<T, ErrorInfo>` 的框架别名，框架统一错误返回通道 |
| `Resource` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | RAII 资源句柄公共基类语义，允许移动、禁止拷贝 |
| `TypeId` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | 编译期 `constexpr` 字符串哈希生成的运行时类型标识，不依赖 RTTI |
| `Module` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | 编译期/部署期粒度的功能单元，装配阶段的注册单位，非运行期可替换对象 |
| `Service` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | 运行期可替换的能力提供者，`Context::Resolve<T>()` 解析的目标类型 |
| `Context` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | 依赖注入容器 + 生命周期宿主，内部持有 `Registry<IService>` |
| `LifecycleState` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | 进程级生命周期状态机，`Created→Initialized→Running→Stopped→Destroyed` 严格线性迁移 |
| `Factory` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | 无状态工厂，`Create()` 每次产出全新实例，与 `Registry` 的已注册单例查找语义互补 |
| `Registry` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | 通用的按 `TypeId` 存储已注册 `TInterface` 实例的表，与 `Context` 正交，供 1.3 插件体系实例化为 `Registry<IPlugin>` 复用 |
| `Plugin` | 1.3 | design/milestone-01-foundation/1.3-core-plugin-system.md | 动态库（`.so`）粒度的可插拔单元，随附 `plugin.yaml` 清单，区别于 1.2 批次编译期粒度的 `Module` |
| `PluginManifest` | 1.3 | design/milestone-01-foundation/1.3-core-plugin-system.md | `plugin.yaml` 反序列化后的内存表示，含版本（`SemVer`）、能力标签、依赖声明、许可凭证 |
| `MemoryPool` | 1.5 | design/milestone-01-foundation/1.5-memory.md | `IMemoryPool` 及其实现共同遵循的抽象概念，本身不是 C++ 类型；固定大小 slab + 空闲链表的池化分配策略统称 |
| `ImagePool` | 1.5 | design/milestone-01-foundation/1.5-memory.md | 按图像业务用途命名的 `GpuPool`/`PinnedPool` 实例（非独立类型），用特定 `slab_size`/`slab_count` 构造参数区分于其他命名池 |
| `TensorPool` | 1.5 | design/milestone-01-foundation/1.5-memory.md | 按张量业务用途命名的 `GpuPool`/`PinnedPool` 实例（非独立类型），与 `ImagePool` 同理，仅构造参数不同 |
| `GpuPool` | 1.5 | design/milestone-01-foundation/1.5-memory.md | `IMemoryPool` 在设备显存上的具体实现，内部封装 `cudaMalloc`/`cudaFree`，启动期一次性预分配 |
| `PinnedPool` | 1.5 | design/milestone-01-foundation/1.5-memory.md | `IMemoryPool` 在锁页主机内存上的具体实现，内部封装 `cudaHostAlloc`/`cudaFreeHost`，启动期一次性预分配 |
| `TaskGraph` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | DAG 任务图，节点为 `TaskNode`，边隐含在节点的 `dependencies` 字段中，递归拓扑遍历驱动执行 |
| `PipelineExecutor` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | 把 `TaskGraph` 节点分派到其所属阶段 `WorkerPool` 的执行引擎，不做业务语义判断 |
| `WorkerPool` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | 固定线程数的工作线程池，一个 Pipeline 阶段（Capture/Inference/Retrieval/Reason/IO）绑定一个独立实例，启动时创建、运行期不动态伸缩 |
| `Logger` | 1.6 | design/milestone-01-foundation/1.6-cross-cutting.md | 按类别（category）持有的日志入口，封装 `spdlog::async_logger` + 独立后台 IO 线程，`Trace`/`Debug` 与 `Warning`+ 分两级队列采用不同溢出策略 |
| `ConfigStore` | 1.6 | design/milestone-01-foundation/1.6-cross-cutting.md | 封装 yaml-cpp 解析 + `ConfigSchema` 校验的配置加载/查询/热重载入口，启动期一次性加载校验，热重载失败保留旧配置生效 |
| `ErrorCode 分类表` | 1.6 | design/milestone-01-foundation/1.6-cross-cutting.md | 汇总 1.1-1.5 已提交的 `Core_*`/`Lifecycle_*`/`Plugin_*`/`Runtime_*`/`Memory_*` 前缀并新增 `Infra_*`，规定模块前缀命名规则与追加纪律，不重新定义 `ErrorInfo`/`Result<T>` 本身 |
| `FactBase` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | 扁平键值表，求值前批量物化外部数据（Detection + Knowledge + Retrieval），每条记录带 `FactSource` 溯源元数据 |
| `FactSource` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | 事实来源元数据（Direct/GraphPath/VectorSearch/Computed/Default），含 SQL 语句/FAISS 参数/耗时 |
| `Rule` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | YAML 定义的工业判定规则（name + condition AST + action + overrides/overridden_by + rule_set） |
| `RuleEngine` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | 规则求值引擎，批量加载 YAML、并行求值、冲突消解、热重载 |
| `DecisionTree` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | YAML 配置的分层决策树（BranchNode + LeafNode + ScoreFormula），叶子用加权 sigmoid 评分 |
| `ReasoningResult` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | 推理产出：verdict（OK/NG/WARN/UNCERTAIN）+ severity + recommendation + confidence + trace[] + evidence[] |
| `TraceStep` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | 算子级溯源步骤（Expression/Rule/TreeBranch/Scoring 四级），含描述、源码位置、父节点引用。定义于 `sai::rule` 命名空间，5.2 Reasoner 复用 |
| `EvidenceItem` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | 全链路证据项：FactBase 键值对 + FactSource 溯源 + TraceStep 关联 |
| `Pipeline` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | 配置驱动的编排宿主，持有 M1 TaskGraph + PipelineExecutor，对外暴露 `Submit()`/`Drain()`/`Stop()`/`Metrics()` |
| `PipelineConfig` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | YAML 反序列化的 Pipeline 配置：name + version + backpressure + stages[] |
| `IStageNode` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | 业务语义化阶段节点接口：`OnInitialize`/`OnStart`/`OnStop`/`Process` |
| `StageQueue<T>` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | 阶段间 bounded SPSC/MPSC lock-free ring buffer，内建 BackpressurePolicy |
| `Scheduler` | 6.2 | design/milestone-06-orchestration-scheduling/6.2-scheduler.md | StageType → WorkerPool 映射 + 阶段间队列分配 |
| `StageMetrics` | 6.2 | design/milestone-06-orchestration-scheduling/6.2-scheduler.md | per-stage 原子计数器：frames_processed / failed / dropped / queue_depth |

## 2. 核心接口签名表

| 接口名称 | 归属批次 | 定义所在文档 | 签名摘要 |
|---|---|---|---|
| `Object` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `class Object`（虚析构，拷贝/移动均禁用，构造函数受保护） |
| `IReflectable` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `class IReflectable { virtual auto TypeId() const noexcept -> sai::TypeId = 0; }` |
| `TypeRegistry` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `class TypeRegistry { static auto Instance() noexcept -> TypeRegistry&; template<Reflectable T> auto Register() -> Result<void>; auto Resolve(TypeId) const -> Result<TypeInfo>; }` |
| `Result<T>` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `template<typename T> using Result = tl::expected<T, ErrorInfo>;`（`ErrorInfo` 含 `code`/`message`/`source_location`） |
| `IModule` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | `class IModule : public Object { virtual auto OnInitialize(Context&) -> Result<void> = 0; virtual auto OnStart(Context&) -> Result<void> = 0; virtual auto OnStop(Context&) -> Result<void> = 0; }` |
| `IService` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | `class IService : public Object, public IReflectable {}`（无额外方法，用作 `Context::Register<T>`/`Resolve<T>` 的约束基类） |
| `Context` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | `class Context { auto RegisterModule(unique_ptr<IModule>) -> Result<void>; template<Reflectable T> requires derived_from<T, IService> auto Register(shared_ptr<T>) -> Result<void>; template<Reflectable T> requires derived_from<T, IService> auto Resolve() const -> Result<shared_ptr<T>>; auto Initialize() -> Result<void>; auto Start() -> Result<void>; auto Stop() -> Result<void>; auto CurrentState() const noexcept -> LifecycleState; }` |
| `Factory<TInterface>` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | `template<typename TInterface> class Factory { virtual auto Create() -> Result<unique_ptr<TInterface>> = 0; }` |
| `Registry<TInterface>` | 1.2 | design/milestone-01-foundation/1.2-core-lifecycle.md | `template<typename TInterface> class Registry { auto Register(TypeId, shared_ptr<TInterface>) -> Result<void>; auto Resolve(TypeId) const -> Result<shared_ptr<TInterface>>; }`（本批次定稿签名，1.3 批次实例化 `Registry<IPlugin>` 直接复用，不重新定义） |
| `IPlugin` | 1.3 | design/milestone-01-foundation/1.3-core-plugin-system.md | `class IPlugin : public IModule, public IReflectable { virtual auto GetManifest() const noexcept -> const PluginManifest& = 0; }` |
| `PluginManager` | 1.3 | design/milestone-01-foundation/1.3-core-plugin-system.md | `class PluginManager { auto DiscoverManifests(path) -> Result<void>; auto Load(const std::string&) -> Result<void>; auto Resolve(TypeId) const -> Result<shared_ptr<IPlugin>>; auto Shutdown() -> Result<void>; }`（内部持有 `Registry<IPlugin>` 实例，实例化复用 1.2 批次模板，不重新定义；`Shutdown()` 按加载顺序的逆序驱动已加载插件的 `OnStop`，应用入口必须在调用 `Context::Stop()` 之前调用本方法） |
| `ModuleManager` | 1.3 | design/milestone-01-foundation/1.3-core-plugin-system.md | `class ModuleManager { explicit ModuleManager(Context&); auto RegisterBuiltin(unique_ptr<IModule>) -> Result<void>; }`（管理编译期静态链接内建模块，与运行期动态加载的 `PluginManager` 边界见 1.3 批次 3. Design） |
| `VersionManager` | 1.3 | design/milestone-01-foundation/1.3-core-plugin-system.md | `class VersionManager { static auto CheckCompatible(const VersionRange&, const SemVer&) noexcept -> Result<void>; }` |
| `LicenseManager` | 1.3 | design/milestone-01-foundation/1.3-core-plugin-system.md | `class LicenseManager { auto Validate(std::string_view license_token) const -> Result<void>; }` |
| `CapabilityManager` | 1.3 | design/milestone-01-foundation/1.3-core-plugin-system.md | `class CapabilityManager { auto RegisterKnownCapability(std::string) -> Result<void>; auto Validate(const std::vector<std::string>&) const -> Result<void>; }` |
| `IMemoryPool` | 1.5 | design/milestone-01-foundation/1.5-memory.md | `class IMemoryPool : public Object { virtual auto Acquire(size_t bytes) noexcept -> Result<PooledPtr<uint8_t>> = 0; virtual void Release(PooledPtr<uint8_t>&) noexcept = 0; virtual auto SlabSize() const noexcept -> size_t = 0; virtual auto SlabCount() const noexcept -> size_t = 0; virtual auto AvailableSlabCount() const noexcept -> size_t = 0; }` |
| `GpuPool` | 1.5 | design/milestone-01-foundation/1.5-memory.md | `class GpuPool final : public IMemoryPool { static auto Create(MemoryPoolConfig, ArenaAllocator&) noexcept -> Result<unique_ptr<GpuPool>>; }`（封装 `cudaMalloc`/`cudaFree`，启动期一次性预分配，运行期不再调用） |
| `PinnedPool` | 1.5 | design/milestone-01-foundation/1.5-memory.md | `class PinnedPool final : public IMemoryPool { static auto Create(MemoryPoolConfig, ArenaAllocator&) noexcept -> Result<unique_ptr<PinnedPool>>; }`（封装 `cudaHostAlloc`/`cudaFreeHost`，启动期一次性预分配，运行期不再调用） |
| `ArenaAllocator` | 1.5 | design/milestone-01-foundation/1.5-memory.md | `class ArenaAllocator final { explicit ArenaAllocator(size_t capacity_bytes) noexcept; template<typename T, typename... Args> auto Construct(Args&&...) noexcept -> Result<T*>; }`（池元数据独立分配来源，与业务数据 slab 区域不重叠） |
| `PooledPtr<T>` | 1.5 | design/milestone-01-foundation/1.5-memory.md | `template<typename T> class PooledPtr final { auto Get() const noexcept -> T*; auto UseCount() const noexcept -> int; }`（内部 `std::atomic<int>` 引用计数，非 `std::shared_ptr`，析构时归还池而非 `delete`；仅可由 `IMemoryPool::Acquire` 构造） |
| `TaskScheduler` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | `class TaskScheduler final { explicit TaskScheduler(Registry<WorkerPool>&) noexcept; auto Submit(TypeId stage_id, coroutine_handle<>) noexcept -> Result<void>; }`（有界队列 + 拒绝新任务背压策略，队列满返回 `Runtime_QueueFull`） |
| `Task<T>` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | `template<typename T> using Task = std::coroutine_handle<TaskPromise<T>>;`（本框架所有异步工作函数的统一返回类型，`TaskPromise<T>` 持有 `Result<T>` 结果与 `std::stop_token`） |
| `WorkerPool` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | `class WorkerPool final { WorkerPool(size_t thread_count, size_t queue_capacity) noexcept; auto TryEnqueue(coroutine_handle<>) noexcept -> Result<void>; auto ThreadCount() const noexcept -> size_t; }`（固定线程数，构造后不支持动态调整） |
| `GpuStreamQueue` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | `class GpuStreamQueue final { static auto Create(size_t stream_count, PinnedPool&) noexcept -> Result<unique_ptr<GpuStreamQueue>>; auto EnqueueAsyncCopy(PooledPtr<uint8_t>, size_t, CopyDirection, stop_token) noexcept -> Task<void>; }`（封装 `cudaStream_t` 池，host-device 拷贝中转缓冲复用 1.5 批次 `PinnedPool`；GPU 完成回调经由专用 GPU Callback Thread 转发 `resume()`，不在驱动线程上直接恢复协程） |
| `CopyDirection` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | `enum class CopyDirection { HostToDevice, DeviceToHost };`（`GpuStreamQueue::EnqueueAsyncCopy` 的拷贝方向参数，只覆盖 host-device 双向拷贝，不含 `DeviceToDevice`） |
| `TaskGraph` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | `class TaskGraph final { auto AddNode(TaskNode) noexcept -> Result<void>; auto NodeAt(TaskId) const noexcept -> Result<const TaskNode*>; auto RunToCompletion(PipelineExecutor&, stop_token) noexcept -> Task<void>; }`（节点 `struct TaskNode { TaskId id; TypeId stage_id; std::function<Task<void>()> work; std::vector<TaskId> dependencies; }`，拓扑执行递归驱动） |
| `PipelineExecutor` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | `class PipelineExecutor final { explicit PipelineExecutor(TaskScheduler&) noexcept; auto Dispatch(TypeId stage_id, Task<void>, stop_token) noexcept -> Task<void>; }` |
| `Logger` | 1.6 | design/milestone-01-foundation/1.6-cross-cutting.md | `class Logger final { static auto Get(std::string_view category) -> Logger&; static auto InitializeGlobalSinks(std::filesystem::path) -> Result<void>; void SetLevel(LogLevel) noexcept; template<typename... Args> void Log(LogLevel, fmt::format_string<Args...>, Args&&...) noexcept; auto DroppedCount() const noexcept -> uint64_t; }`（8192 容量异步队列，`Trace`/`Debug` 满时丢弃并计数，`Warning`+ 绝不丢弃） |
| `ConfigStore` | 1.6 | design/milestone-01-foundation/1.6-cross-cutting.md | `class ConfigStore final { explicit ConfigStore(ConfigSchema) noexcept; auto Load(std::filesystem::path) -> Result<void>; template<typename T> auto Get(std::string_view key) const -> Result<T>; auto EnableHotReload(stop_token) -> Result<void>; }`（热重载校验失败保留旧配置，不回退默认值） |
| `ConfigSchema` | 1.6 | design/milestone-01-foundation/1.6-cross-cutting.md | `class ConfigSchema final { auto RequireField(std::string field_path, FieldValidator) -> ConfigSchema&; auto Validate(const YAML::Node&) const -> Result<void>; }`（手写字段校验函数集合，JSON-Schema 等价校验，不引入独立 schema 库） |
| `DailyAndSizeRotatingFileSink` | 2.0 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §7 | `template<typename Mutex> class DailyAndSizeRotatingFileSink final : public spdlog::sinks::base_sink<Mutex>` — 按天+100MB 双条件轮转 sink，偿还 M1 债务 D3 |
| `IDevice` | 2.1 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §3.4 | `class IDevice : public IPlugin` — 设备统一抽象，`enum class State {Disconnected,Connected,Acquiring,Error}`，`Connect/Disconnect/IsConnected/SerialNumber/CurrentState` |
| `ICamera` | 2.1 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §3.4 | `class ICamera : public IDevice` — `enum class TriggerMode {Software,Hardware,FreeRun}`，`using FrameCallback = std::function<void(RawImage)>`，`SetTriggerMode/StartAcquisition/StopAcquisition/RegisterFrameCallback/SetExposureTime/SetGain/SetROI` |
| `ILightController` | 2.1 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §3.4 | `class ILightController : public IDevice` — `enum class StrobeMode {Continuous,OnTrigger,Off}`，`ChannelCount/SetIntensity/GetIntensity/Enable/Disable/SetStrobeMode` |
| `Rect` | 2.1 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §3.6 | `struct Rect {std::size_t x,y,width,height; Area()->size_t; IsEmpty()->bool;}` — 定义于 `sai::device`，`sai::image::Rect` 为 using 别名 |
| `RingBuffer<T>` | 2.1 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §3.4 | `template<typename T> class RingBuffer final` — 固定容量环形缓冲，满时覆盖最旧元素，互斥锁守护 |
| `Image` | 2.2 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §4.4 | `class Image : public Resource` — 图像抽象基类，`Meta/Data(const+mutable)/SizeBytes`，move-only，`PixelFormat` 枚举 |
| `RawImage` | 2.2 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §4.4 | `class RawImage final : public Image` — 相机原始帧，`FromPool/FromBuffer/FromOwnedBuffer`（FromOwnedBuffer 为追加的 owning-without-pool 工厂） |
| `SurfaceImage` | 2.2 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §4.4 | `class SurfaceImage final : public Image` — 预处理完毕帧，`FromPool/FromPinned/FromOwnedBuffer` |
| `GpuImage` | 2.2 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §4.4 | `class GpuImage final : public Image` — 显存驻留帧，`FromPool(GpuPool&)`，CUDA 门控 |
| `PreprocessFn` | 2.2 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §4.4 | `using PreprocessFn = std::function<auto(std::unique_ptr<Image>)->Result<std::unique_ptr<Image>>>;` — 预处理步骤自由函数，`Compose` 递归串联 |
| `ROI` | 2.2 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §4.6 | `struct ROI {vector<Rect> regions; IsEmpty/BoundingBox/Apply;}` — Apply 裁剪首个 region（多 region 合成延后） |
| `IExporter` | 2.3 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §5.4 | `class IExporter : public IPlugin` — 导出插件接口，`Export(InspectionResult&,path,SurfaceImage*)->Result<void>`，`FormatName()` |
| `IImporter` | 2.3 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §5.4 | `class IImporter : public IPlugin` — 导入插件接口，`ImportImage(path)->Result<unique_ptr<Image>>`，`ImportMetadata(path)->Result<YAML::Node>`，`FormatName()` |
| `JsonExporter` | 2.3 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §5.4 | `class JsonExporter final : public IExporter` — 内置 JSON 报告导出，nlohmann-json，标注图暂用 PPM 占位 |
| `BasicImporter` | 2.3 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §5.4 | `class BasicImporter final : public IImporter` — 内置 YAML 元数据 + PPM 图像导入 |
| `DefectRecord` | 2.3 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §5.4 | `struct DefectRecord {string label,severity; float confidence; Rect location; string evidence_path;}` |
| `InspectionResult` | 2.3 | specs/2026-07-10-milestone-2-acquisition-imaging-design.md §5.4 | `struct InspectionResult {string sku_id,serial_number; system_clock::time_point timestamp; vector<DefectRecord> defects; string verdict;}` |
| `IExpression` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | `class IExpression : public Object { virtual auto Evaluate(FactBase&) const -> Result<Value> = 0; virtual auto CollectFieldRefs() const -> vector<string> = 0; virtual auto SourceText() const -> string_view = 0; }` |
| `FactBase` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | `class FactBase { auto Set(string_view, Value, FactSource) -> void; auto Get(string_view) const -> optional<Value>; auto Has(string_view) const -> bool; auto SourceOf(string_view) const -> const FactSource&; auto AllEntries() const -> vector<pair<string,Value>>; auto AllSources() const -> vector<pair<string,FactSource>>; }` |
| `FactBuilder` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | `class FactBuilder { explicit FactBuilder(shared_ptr<KnowledgeGraph>, shared_ptr<VectorPath>); auto Build(string_view surface_id, const DetectionResult&, const vector<string>& graph_paths) -> Result<FactBase>; }` |
| `RuleEngine` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | `class RuleEngine { auto LoadFromYAML(path) -> Result<void>; auto EvaluateAll(FactBase&) -> Result<vector<ResolvedRule>>; auto ResolveConflicts(const vector<ResolvedRule>&) -> vector<ResolvedRule>; auto EnableHotReload(path, stop_token) -> Result<void>; auto DetectOverlaps() const -> vector<OverlapWarning>; }` |
| `DecisionTree` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | `class DecisionTree { static auto LoadFromYAML(path) -> Result<unique_ptr<DecisionTree>>; auto Root() const -> const IDecisionNode&; }` |
| `IReasoner` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | `class IReasoner : public IService { virtual auto Reason(const FactBase&, const vector<ResolvedRule>&) -> Result<ReasoningResult> = 0; }` |
| `DefaultReasoner` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | `class DefaultReasoner final : public IReasoner { explicit DefaultReasoner(unique_ptr<DecisionTree>); auto Reason(const FactBase&, const vector<ResolvedRule>&) -> Result<ReasoningResult> override; }` |
| `IStageNode` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | `class IStageNode : public Object { virtual auto GetType() const noexcept -> StageType = 0; virtual auto GetId() const -> string_view = 0; virtual auto OnInitialize(Context&) -> Result<void> = 0; virtual auto OnStart(Context&) -> Result<void> = 0; virtual auto OnStop(Context&) -> Result<void> = 0; virtual auto Process(StageInput) -> Result<StageOutput> = 0; }` |
| `Pipeline` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | `class Pipeline { static auto LoadFromYAML(path, Context&) -> Result<unique_ptr<Pipeline>>; auto Start() -> Result<void>; auto Submit(RawImage) -> Result<void>; auto Drain() -> Result<void>; auto Stop() -> Result<void>; auto Metrics() const -> vector<StageMetrics>; }` |
| `StageQueue<T>` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | `template<typename T> class StageQueue { static auto Create(size_t, BackpressurePolicy) -> Result<unique_ptr<StageQueue>>; auto TryPush(unique_ptr<T>) -> bool; auto PushBlocking(unique_ptr<T>) -> void; auto TryPop() -> unique_ptr<T>; auto PopBlocking() -> unique_ptr<T>; auto Depth() const noexcept -> size_t; }` |
| `Scheduler` | 6.2 | design/milestone-06-orchestration-scheduling/6.2-scheduler.md | `class Scheduler { explicit Scheduler(Registry<WorkerPool>&, const BackpressureConfig&); auto Allocate(const vector<StageConfig>&) -> Result<void>; auto Deallocate() -> Result<void>; auto PoolFor(StageType) const -> Result<WorkerPool&>; }` |

## 里程碑 2 偏差记录（2026-07-12）

| 偏差编号 | 类型 | 描述 | 处置 |
|----------|------|------|------|
| D2-a | FromOwnedBuffer 追加 | `RawImage`/`SurfaceImage` 追加 `FromOwnedBuffer(vector<uint8_t>, ImageMeta)` 工厂 — 预处理/ROI/Importer 需要无池的 owning 分配（冻结签名无 pool 参数），纯粹追加修改 | 计划内，投入代码 |
| D2-b | MakeFlatField 参数类型 | spec §4.4 声明的 `Image correction_frame` 按值传参无法编译（Image 抽象、move-only 且 ctor protected）→ 改为 `const Image&`，捕获为 `const Image*`，调用方保活 | 编译必需，投入代码 |
| D2-c | MakeHDR 单帧语义 | `MakeHDR(num_exposures)` 返回仅接收一个 `Image` 的 `PreprocessFn`——无法融合多帧。实现为单帧 min-max 对比拉伸；真正多曝光融合需多输入 API，延后至 Future Extension | 类型限制，投入代码 |
| D2-d | 标注图 PPM 占位 | `JsonExporter::Export` 在 `annotated_image != nullptr` 时写 PPM 而非 PNG——本里程碑不引入 PNG 库，PPM 可移植且可测试 | 依赖延后，投入代码 |
| D2-e | D2 修复——per-category drop pool | 将 1.6 §9 的"单共享 thread_pool"改为每类别独立 drop-tier pool——M2 spec §7 显式授权修复 | 债务清偿，投入代码 |
| D2-f | Image 基类显式 move | 冻结 defaulted move 会将原始 `data_` 指针复制到移动后目标——移动源报告 `IsValid()==true` 但持有悬空指针。显式 move ctor/assign 将源 `data_` 置 nullptr | 正确性修复，投入代码 |
| D2-g | Release() 重写于子类 | `Release()` 在 `RawImage`/`SurfaceImage` 中重写以同时归还池 slab 并清空 `owned_bytes_` | 编译必需，投入代码 |
| D2-h | surface_image.h 追加 `#include <memory_pool.h>` | 该头文件需要 `PooledPtr` 完整定义而不仅仅是前向声明——冻结设计未提及此 include，但缺少则无法编译 | 编译必需，投入代码 |
| D2-i | ROI Apply 单 region 裁剪 | `ROI::Apply` 仅裁剪首个 region——多 region 合成延后 | 计划内，投入代码 |
| D2-j | RingBuffer 零容量保护 | `RingBuffer::Push` 在容量为 0 时未持锁自增 `dropped_count_`——在生产中不可能出现，属过度防御 | 待定（最终审查时判定） |
| D2-k | `NowTm()` 在构造函数中虚调用 | `DailyAndSizeRotatingFileSink` 在构造函数中调用 `NowTm()` 以设定初始日期——总是解析为基类版本，对子类不可见 | Minor，记录供参考 |
| D2-l | `BoundingBox` 对 `front()` 进行冗余比较 | 循环中包含首个 region，无害 | Minor |
| D2-m | `Release()` 中 `shrink_to_fit()` | `Release()` 在 clear 之外还调用了 `shrink_to_fit()`——非正确性所需 | Minor |
| D2-n | 时间戳序列化依赖平台 | `system_clock::time_point` 序列化为 epoch 计数——macOS 单位为 μs，Linux 单位为 ns，跨平台 JSON 数值不可移植 | Minor，M3 考察 |

## 里程碑 1 复核记录

- 复核日期：2026-07-08
- 覆盖批次：1.1, 1.2, 1.3, 1.5, 1.4, 1.6（按实际执行顺序）
- 结构校验：6 份设计文档（1.1-1.6）均为 14 节结构，编号与顺序一致。
- 交叉引用校验：1.4 引用的 `GpuPool`/`PinnedPool`（来自 1.5）、`Registry<TInterface>`（来自 1.2）签名逐字核对一致；1.6 汇总的 `Core_*`/`Lifecycle_*`/`Plugin_*`/`Runtime_*`/`Memory_*` 错误码前缀与 1.1-1.5 各自文档中实际使用的具体成员名逐一核对一致，无改名、无冲突；1.1 中 `ErrorCode` 片段的截断注释（"完整分类表由1.6批次补完"）与 1.6 的汇总表内容一致，非遗留 bug。
- 概念归属唯一性校验：概念归属表与接口签名表各自内部均无重复归属的概念/接口名称（同一名称同时出现在两张表中属于设计上允许的"概念+接口"配对，非重复）。
- 结论：6 份设计文档满足里程碑 1 的全部结构与一致性要求，可作为后续里程碑（2-7）设计文档的接口基线。
- 遗留事项：里程碑 1 spec 验证点（"启动空 task graph 走通调度/内存/日志/配置闭环"）需在后续代码实现阶段验证，本阶段仅完成设计文档；1.6 任务报告中提到的 1.1 `ErrorCode` 截断片段已在本次复核中确认为一致，非遗留问题。
