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
| `PluginManager` | 1.3 | design/milestone-01-foundation/1.3-core-plugin-system.md | `class PluginManager { auto DiscoverManifests(path) -> Result<void>; auto Load(const std::string&) -> Result<void>; auto Resolve(TypeId) const -> Result<shared_ptr<IPlugin>>; }`（内部持有 `Registry<IPlugin>` 实例，实例化复用 1.2 批次模板，不重新定义） |
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
| `TaskGraph` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | `class TaskGraph final { auto AddNode(TaskNode) noexcept -> Result<void>; auto RunToCompletion(PipelineExecutor&, stop_token) noexcept -> Task<void>; }`（节点 `struct TaskNode { TaskId id; std::function<Task<void>()> work; std::vector<TaskId> dependencies; }`，拓扑执行递归驱动） |
| `PipelineExecutor` | 1.4 | design/milestone-01-foundation/1.4-runtime.md | `class PipelineExecutor final { explicit PipelineExecutor(TaskScheduler&) noexcept; auto Dispatch(TypeId stage_id, Task<void>, stop_token) noexcept -> Task<void>; }` |
