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

## 里程碑 1 复核记录

- 复核日期：2026-07-08
- 覆盖批次：1.1, 1.2, 1.3, 1.5, 1.4, 1.6（按实际执行顺序）
- 结构校验：6 份设计文档（1.1-1.6）均为 14 节结构，编号与顺序一致。
- 交叉引用校验：1.4 引用的 `GpuPool`/`PinnedPool`（来自 1.5）、`Registry<TInterface>`（来自 1.2）签名逐字核对一致；1.6 汇总的 `Core_*`/`Lifecycle_*`/`Plugin_*`/`Runtime_*`/`Memory_*` 错误码前缀与 1.1-1.5 各自文档中实际使用的具体成员名逐一核对一致，无改名、无冲突；1.1 中 `ErrorCode` 片段的截断注释（"完整分类表由1.6批次补完"）与 1.6 的汇总表内容一致，非遗留 bug。
- 概念归属唯一性校验：概念归属表与接口签名表各自内部均无重复归属的概念/接口名称（同一名称同时出现在两张表中属于设计上允许的"概念+接口"配对，非重复）。
- 结论：6 份设计文档满足里程碑 1 的全部结构与一致性要求，可作为后续里程碑（2-7）设计文档的接口基线。
- 遗留事项：里程碑 1 spec 验证点（"启动空 task graph 走通调度/内存/日志/配置闭环"）需在后续代码实现阶段验证，本阶段仅完成设计文档；1.6 任务报告中提到的 1.1 `ErrorCode` 截断片段已在本次复核中确认为一致，非遗留问题。
