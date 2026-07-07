# Milestone 1 — 基座设施（Core / Runtime / Memory / 横切关注点）设计文档生成计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 产出里程碑 1（基座设施）6 个批次的确定性设计文档（1.1 Core基础 / 1.2 Core生命周期与装配 / 1.3 Core插件体系 / 1.4 Runtime / 1.5 Memory / 1.6 横切关注点），并建立跨批次术语与接口契约表，为后续里程碑 2-7 的设计文档提供可复用的接口基线。

**Architecture:** 每个批次产出一份遵循固定 14 节结构的 Markdown 设计文档（不产出可编译代码）。批次之间通过 `docs/surface-ai/glossary-and-contracts.md` 传递已定稿的接口签名和概念归属，后续批次禁止重新定义已归属的概念，只能引用。

**Tech Stack:** C++20（tl::expected / std::stop_token / coroutine）、CUDA、spdlog（async）、yaml-cpp、vcpkg。文档产出物为纯 Markdown，不涉及实际编译。

## Global Constraints

- 每份设计文档必须包含且仅包含以下 14 个二级标题，编号与顺序固定：`## 1. Purpose` `## 2. Responsibilities` `## 3. Design` `## 4. Interfaces` `## 5. Workflow` `## 6. Data Structure` `## 7. Class Diagram` `## 8. Sequence Diagram` `## 9. Thread Model` `## 10. Performance` `## 11. Memory` `## 12. Future Extension` `## 13. Best Practice` `## 14. Anti Pattern`
- `## 3. Design` 章节禁止出现清单式表述（形如"支持 A、B、C"或"support X/Y/Z"），必须是"采用 X，因为...；拒绝 Y，因为..."的确定性表述。
- 所有跨模块引用的接口，必须使用契约表中已定稿的名称，不得在本批次文档中另起名字。
- 本计划只产出设计文档，不产出可编译的 C++ 源码；里程碑 1 spec 中"能启动一个空 task graph"的验证点属于后续代码实现阶段，本计划不覆盖。
- 技术选型严格遵循 `docs/superpowers/specs/2026-07-07-surface-ai-framework-phased-plan-design.md` 第 2 节的技术栈表，禁止引入表外技术。
- 文档中出现的接口签名、Workflow 步骤描述、伪代码，一律避免过度防御性代码（不为不可能发生的情况堆砌校验）、避免多层嵌套/多层 if 链；控制流优先用提前返回（early return）表达；错误处理优先用 `tl::expected` 的链式 `and_then`/`or_else` 风格，而不是嵌套 `if (result.has_value())` 检查；树形/图形结构（如 TaskGraph 拓扑排序、Registry 查找）优先用递归表达而非多层循环嵌套。

## File Structure Overview

```
docs/surface-ai/
├── glossary-and-contracts.md              # 跨批次契约表（Task 1 创建，Task 2-7 增量更新）
└── design/
    └── milestone-01-foundation/
        ├── 1.1-core-foundation.md         # Task 2
        ├── 1.2-core-lifecycle.md          # Task 3
        ├── 1.3-core-plugin-system.md      # Task 4
        ├── 1.5-memory.md                  # Task 5（先于 1.4 执行）
        ├── 1.4-runtime.md                 # Task 6（依赖 1.5 的 Memory 接口）
        └── 1.6-cross-cutting.md           # Task 7
```

## 执行顺序与依赖关系

```
Task 1 (契约表骨架)
  └─> Task 2 (1.1 Core基础)
        └─> Task 3 (1.2 Core生命周期)
              └─> Task 4 (1.3 Core插件体系)
                    └─> Task 5 (1.5 Memory)          ← 只依赖 1.1，可与 Task 3/4 并行
                          └─> Task 6 (1.4 Runtime)     ← 依赖 1.5 的 GPU/Pinned Pool 接口 + Task 3 的 Registry
                                └─> Task 7 (1.6 横切关注点)  ← 依赖 1.1 的 ErrorInfo/Result
                                      └─> Task 8 (里程碑1收尾一致性复核)
```

**顺序调整说明**：原批次编号是 1.1→1.2→1.3→1.4→1.5→1.6，但 Runtime（1.4）的 CUDA Stream GPU Queue 和 Pipeline Executor 需要搬运 Tensor/Image 数据，必须引用 Memory（1.5）已定稿的 `GpuPool`/`PinnedPool` 接口名（均继承自 `IMemoryPool`，具体签名见 Task 5）。因此本计划把 Task 5（对应批次 1.5）排在 Task 6（对应批次 1.4）之前，避免 Runtime 文档中出现"预留占位符，Memory 接口待定"这类前向引用空洞。1.3（插件体系）严格早于 1.4/1.5，因为两者的具体实现类都要通过 1.3 的 `Registry`/`ModuleManager` 生命周期注册。

---

## Task 1: 契约表骨架

**Files:**
- Create: `docs/surface-ai/glossary-and-contracts.md`

**Interfaces:**
- Consumes: 无（首个任务）
- Produces: 契约表的两张空表结构，供 Task 2-7 追加行

- [ ] **Step 1: 创建契约表文件骨架**

写入以下内容：

```markdown
# Surface AI Framework —— 术语与接口契约表

> 本文档是跨批次的活文档。每个设计批次完成后，必须在此追加自己的契约增量，
> 不得修改其他批次已提交的行（若确需修改，必须在 PR 描述中说明原因并经评审）。

## 1. 概念归属表

| 概念名称 | 归属批次 | 定义所在文档 | 说明 |
|---|---|---|---|

## 2. 核心接口签名表

| 接口名称 | 归属批次 | 定义所在文档 | 签名摘要 |
|---|---|---|---|
```

- [ ] **Step 2: 验证文件结构**

Run: `grep -c "^## " docs/surface-ai/glossary-and-contracts.md`
Expected: `2`（两个二级标题：概念归属表、核心接口签名表）

- [ ] **Step 3: Commit**

```bash
git add docs/surface-ai/glossary-and-contracts.md
git commit -m "docs: bootstrap milestone-1 glossary and contract table"
```

---

## Task 2: 批次 1.1 — Core 基础

**Files:**
- Create: `docs/surface-ai/design/milestone-01-foundation/1.1-core-foundation.md`
- Modify: `docs/surface-ai/glossary-and-contracts.md`（追加本批次契约增量）

**Interfaces:**
- Consumes: 无（首个内容批次），但必须引用 spec 第 2 节技术栈表（`tl::expected` 错误处理决策）
- Produces:
  - 概念：`Object`（框架内所有一等对象的基类语义）、`Result<T>`（`tl::expected<T, ErrorInfo>` 的框架别名）、`Resource`（RAII 资源句柄语义）、`TypeId`（运行时类型标识）
  - 接口：`class Object`、`template<typename T> using Result = tl::expected<T, ErrorInfo>`、`class IReflectable`、`class TypeRegistry`
  - 这些名称是后续所有批次（1.2-7.2）报错和反射的唯一基线，一旦定稿不可在其他批次重新定义

- [ ] **Step 1: 撰写批次 1.1 设计文档**

在 `docs/surface-ai/design/milestone-01-foundation/1.1-core-foundation.md` 写入 14 节结构。关键内容要求（非占位符，是本步骤必须落实的具体决策）：

- `## 3. Design` 必须明确：
  - `Result<T> = tl::expected<T, ErrorInfo>`，`ErrorInfo` 包含 `code`（枚举）、`message`（string）、`source_location`（C++20 `std::source_location`）三个字段；禁止使用裸 `bool`/`errno` 风格返回值。
  - 异常仅允许在构造函数失败、静态初始化失败两种场景抛出，其余场景一律返回 `Result<T>`（与 spec 第 2 节"tl::expected 为主"决策一致，不写"支持 expected 或 exception"这种清单式表述）。
  - `TypeId` 用编译期 `constexpr` 字符串哈希生成，不使用 RTTI（`typeid`），因为跨插件边界的 RTTI 比较在不同编译单元/动态库加载场景下不可靠。
- `## 4. Interfaces` 必须给出完整头文件级声明，包括 `Object`、`IReflectable::TypeId() const`、`TypeRegistry::Register<T>()`、`TypeRegistry::Resolve(TypeId)`。
- `## 9. Thread Model` 说明 `TypeRegistry` 是否需要加锁（结论：注册阶段（启动时）单线程执行，查询阶段只读，用 `std::shared_mutex` 保护注册表，查询走 `shared_lock`）。

- [ ] **Step 2: 追加契约表增量**

在 `docs/surface-ai/glossary-and-contracts.md` 的两张表分别追加：

概念归属表新增 4 行：`Object` / `Result<T>` / `Resource` / `TypeId`，归属批次填 `1.1`，定义所在文档填 `design/milestone-01-foundation/1.1-core-foundation.md`。

核心接口签名表新增 4 行：`Object`、`IReflectable`、`TypeRegistry`、`Result<T> = tl::expected<T, ErrorInfo>`，归属批次填 `1.1`。

- [ ] **Step 3: 校验 14 节结构完整性**

Run: `grep -c "^## [0-9]" docs/surface-ai/design/milestone-01-foundation/1.1-core-foundation.md`
Expected: `14`

- [ ] **Step 4: 校验 Design 章节无清单式表述**

Run: `grep -n "支持.*[/、,，]" docs/surface-ai/design/milestone-01-foundation/1.1-core-foundation.md | grep -A2 "^3:" || echo "PASS: no list-style wording found"`
Expected: 输出 `PASS: no list-style wording found`（若有匹配需人工检查是否违反确定性表述要求并修正）

- [ ] **Step 5: Commit**

```bash
git add docs/surface-ai/design/milestone-01-foundation/1.1-core-foundation.md docs/surface-ai/glossary-and-contracts.md
git commit -m "docs: add batch 1.1 core foundation design doc"
```

---

## Task 3: 批次 1.2 — Core 生命周期与装配

**Files:**
- Create: `docs/surface-ai/design/milestone-01-foundation/1.2-core-lifecycle.md`
- Modify: `docs/surface-ai/glossary-and-contracts.md`

**Interfaces:**
- Consumes: `Object`、`Result<T>`、`TypeRegistry`（来自 Task 2，签名见契约表 1.1 行）
- Produces:
  - 概念：`Module`（编译期/部署期粒度的功能单元）、`Service`（运行期可替换的能力提供者）、`Context`（依赖注入容器 + 生命周期宿主）
  - 接口：`class IModule : public Object`、`class IService : public Object`、`class Context`、`template<typename TInterface> class Factory`（无参数工厂，`Create() -> Result<std::unique_ptr<TInterface>>`）、`template<typename TInterface> class Registry`（按 `TypeId` 存储已注册的 `TInterface` 实现实例，`Register(TypeId, std::shared_ptr<TInterface>)` / `Resolve(TypeId) -> Result<std::shared_ptr<TInterface>>`；本批次定稿此签名，Task 4 的插件体系直接复用，不重新定义）、`enum class LifecycleState { Created, Initialized, Running, Stopped, Destroyed }`

- [ ] **Step 1: 撰写批次 1.2 设计文档**

写入 14 节结构，关键决策：

- `## 3. Design` 明确：依赖注入采用**构造函数注入 + `Context::Resolve<T>()`**方式，不采用属性注入（setter injection），因为属性注入允许对象在未完全初始化状态下被使用，与"Lifecycle 状态机"的 `Initialized` 状态语义冲突。`Context` 内部持有一个 `Registry<IService>` 实例存储已注册服务（`Registry<TInterface>` 是本批次定稿的通用按接口类型注册表，不是 `Context` 专属结构，Task 4 的插件体系用 `Registry<IPlugin>` 复用同一模板）。`Factory<TInterface>` 用于无状态构造（不需要预先注册实例，每次 `Create()` 产出新对象），与 `Registry` 的"已注册单例查找"语义互补，二者不是同一概念的两个名字。
- `## 6. Data Structure` 给出 `LifecycleState` 状态机的合法迁移表（`Created→Initialized→Running→Stopped→Destroyed`，禁止跳跃迁移，例如 `Created` 不能直接到 `Running`）。
- `## 9. Thread Model` 明确：`Context::Resolve<T>()` 在 `Running` 状态下只读，可多线程并发调用；服务注册（`Context::Register<T>()`）只允许在 `Initialized` 状态之前的单线程装配阶段调用，`Running` 状态下调用 `Register` 返回 `Result<void>` 的错误而非允许动态注册。

- [ ] **Step 2: 追加契约表增量**

概念归属表新增：`Module` / `Service` / `Context` / `LifecycleState` / `Factory` / `Registry`，归属 `1.2`。
接口签名表新增：`IModule`、`IService`、`Context`、`Factory<TInterface>`、`Registry<TInterface>`，归属 `1.2`。

- [ ] **Step 3: 校验结构与措辞**

Run: `grep -c "^## [0-9]" docs/surface-ai/design/milestone-01-foundation/1.2-core-lifecycle.md`
Expected: `14`

- [ ] **Step 4: Commit**

```bash
git add docs/surface-ai/design/milestone-01-foundation/1.2-core-lifecycle.md docs/surface-ai/glossary-and-contracts.md
git commit -m "docs: add batch 1.2 core lifecycle design doc"
```

---

## Task 4: 批次 1.3 — Core 插件体系

**Files:**
- Create: `docs/surface-ai/design/milestone-01-foundation/1.3-core-plugin-system.md`
- Modify: `docs/surface-ai/glossary-and-contracts.md`

**Interfaces:**
- Consumes: `IModule`、`Context`、`LifecycleState`、`Registry<TInterface>`、`Factory<TInterface>`（均来自 Task 3，本批次复用不重新定义；插件注册表实例化为 `Registry<IPlugin>`）
- Produces:
  - 概念：`Plugin`（动态库粒度的可插拔单元，合并原文档"PLUGIN SYSTEM"独立章节，避免与 Core 模块重复定义插件生命周期）、`PluginManifest`（版本+能力声明）
  - 接口：`class IPlugin : public IModule`、`class PluginManager`（内部持有 `Registry<IPlugin>` 实例）、`class ModuleManager`（管理 `IModule` 粒度的装配与卸载顺序，与 `PluginManager` 的区别：`ModuleManager` 管理编译期静态链接的内建模块，`PluginManager` 管理运行期动态加载的 `.so` 插件）、`struct PluginManifest { SemVer version; std::vector<std::string> capabilities; std::vector<PluginDependency> dependencies; }`、`class VersionManager`、`class LicenseManager`、`class CapabilityManager`

- [ ] **Step 1: 撰写批次 1.3 设计文档**

关键决策：

- `## 3. Design` 明确：插件以**动态库（`.so`）+ 清单文件（`plugin.yaml`，遵循 spec 第 2 节 YAML 配置决策）**形式分发，不采用静态编译期插件注册（因为要支持"热插拔相机/检测器插件"这一原文档硬性要求，静态编译无法满足）。版本兼容性检查采用**语义化版本（SemVer）区间匹配**，不采用简单的字符串相等比较,因为插件生态需要允许"兼容次版本升级不强制重新编译依赖方"。License 校验作为 `PluginManager::Load()` 流程中的一个前置检查步骤，校验失败返回 `Result<void>::error(ErrorCode::LicenseInvalid)`,不用异常中断加载流程。
- `## 5. Workflow` 给出插件加载的完整步骤：`dlopen → 读取 plugin.yaml → VersionManager 校验 → CapabilityManager 校验 → LicenseManager 校验 → 调用插件导出的 CreatePlugin() 工厂函数 → Registry<IPlugin>::Register(TypeId, plugin_instance)`（复用 Task 3 定稿的 `Registry<TInterface>` 模板，本批次不重新定义该模板本身）。
- `## 14. Anti Pattern` 明确指出：禁止插件直接持有其他插件的裸指针跨版本调用（必须通过 `Registry<TInterface>::Resolve()` 按接口而非具体类型解耦）。

- [ ] **Step 2: 追加契约表增量**

概念归属表新增：`Plugin` / `PluginManifest`，归属 `1.3`。
接口签名表新增：`IPlugin`、`PluginManager`、`ModuleManager`、`Registry<TInterface>`、`VersionManager`、`LicenseManager`、`CapabilityManager`，归属 `1.3`。

- [ ] **Step 3: 校验结构**

Run: `grep -c "^## [0-9]" docs/surface-ai/design/milestone-01-foundation/1.3-core-plugin-system.md`
Expected: `14`

- [ ] **Step 4: Commit**

```bash
git add docs/surface-ai/design/milestone-01-foundation/1.3-core-plugin-system.md docs/surface-ai/glossary-and-contracts.md
git commit -m "docs: add batch 1.3 core plugin system design doc"
```

---

## Task 5: 批次 1.5 — Memory（提前执行，供 Runtime 引用）

**Files:**
- Create: `docs/surface-ai/design/milestone-01-foundation/1.5-memory.md`
- Modify: `docs/surface-ai/glossary-and-contracts.md`

**Interfaces:**
- Consumes: `Object`、`Result<T>`（来自 Task 2）
- Produces:
  - 概念：`MemoryPool`、`ImagePool`、`TensorPool`、`GpuPool`、`PinnedPool`（Runtime 批次 1.4 将直接引用这两个类型名搬运 GPU 数据）
  - 接口：`class IMemoryPool`、`class GpuPool : public IMemoryPool`（内部封装 `cudaMalloc`/`cudaFree`）、`class PinnedPool : public IMemoryPool`（内部封装 `cudaHostAlloc`）、`class ArenaAllocator`、`template<typename T> class PooledPtr`（引用计数句柄，析构时归还池而非释放内存）

- [ ] **Step 1: 撰写批次 1.5 设计文档**

关键决策：

- `## 3. Design` 明确：内存池采用**固定大小 slab 分配 + 空闲链表**策略，不采用通用 `malloc`/`new` 动态分配路径,因为 12MP 图像 × 16 路相机的持续分配/释放在 24/7 场景下会导致堆碎片化,与"No Memory Leak"性能目标冲突。GPU 内存与 Pinned Memory 采用**双缓冲（double buffer）+ 预分配**策略：启动时按配置文件中的 `max_concurrent_frames` 参数一次性分配所有 slab,运行期不再调用 `cudaMalloc`。引用计数用 `PooledPtr<T>`（内部 `std::atomic<int>` 计数,非 `std::shared_ptr`,因为 `shared_ptr` 的默认删除器会调用 `delete` 而不是"归还到池",需要自定义删除器语义)。SIMD 对齐统一为 **64 字节对齐**（覆盖 AVX-512 的 cache line 需求),通过 `alignas(64)` 或自定义 `AlignedAllocator` 实现,不支持运行期可配置对齐（增加复杂度但当前无实际需求)。NUMA 感知分配**不在本版本实现**,标注为 Future Extension。
- `## 4. Interfaces` 给出 `IMemoryPool::Acquire(size_t bytes) -> Result<PooledPtr<uint8_t>>`、`IMemoryPool::Release(PooledPtr<uint8_t>&)` 完整签名。
- `## 11. Memory` 章节（本批次自身即是 Memory 主题,此节说明池本身的自举内存管理：池的元数据（空闲链表节点）从一块独立的、启动时分配的 `ArenaAllocator` 区域分配,不占用业务数据的 slab 空间)。
- `## 9. Thread Model` 明确：`IMemoryPool::Acquire`/`Release` 必须是多生产者多消费者安全的,内部空闲链表用**无锁栈（lock-free stack,基于 `std::atomic<Node*>` + CAS)**实现,不用 `std::mutex`,因为 Capture 线程和 Inference 线程会高频并发申请/归还 slab,锁竞争会成为热路径瓶颈。

- [ ] **Step 2: 追加契约表增量**

概念归属表新增：`MemoryPool` / `ImagePool` / `TensorPool` / `GpuPool` / `PinnedPool`,归属 `1.5`。
接口签名表新增：`IMemoryPool`、`GpuPool`、`PinnedPool`、`ArenaAllocator`、`PooledPtr<T>`,归属 `1.5`。

- [ ] **Step 3: 校验结构**

Run: `grep -c "^## [0-9]" docs/surface-ai/design/milestone-01-foundation/1.5-memory.md`
Expected: `14`

- [ ] **Step 4: Commit**

```bash
git add docs/surface-ai/design/milestone-01-foundation/1.5-memory.md docs/surface-ai/glossary-and-contracts.md
git commit -m "docs: add batch 1.5 memory design doc (moved ahead of 1.4 runtime)"
```

---

## Task 6: 批次 1.4 — Runtime（依赖 Task 5 的 Memory 接口）

**Files:**
- Create: `docs/surface-ai/design/milestone-01-foundation/1.4-runtime.md`
- Modify: `docs/surface-ai/glossary-and-contracts.md`

**Interfaces:**
- Consumes: `IMemoryPool`、`GpuPool`、`PinnedPool`（来自 Task 5）、`Registry<TInterface>`（来自 Task 3）
- Produces:
  - 概念：`TaskGraph`（DAG 任务图）、`PipelineExecutor`、`WorkerPool`
  - 接口：`class TaskScheduler`、`template<typename T> using Task = std::coroutine_handle<TaskPromise<T>>`、`class WorkerPool`（固定线程数,构造时传入 `size_t thread_count`）、`class GpuStreamQueue`（封装 `cudaStream_t` 池,归还时复用 Task 5 的 `PinnedPool` 做 host-device 异步拷贝缓冲区）、`class TaskGraph`、`class PipelineExecutor`

- [ ] **Step 1: 撰写批次 1.4 设计文档**

关键决策：

- `## 3. Design` 明确：调度模型是**C++20 协程（`co_await`）+ 固定阶段 Worker Pool**,不引入 Fiber（如 Boost.Fiber）,不实现硬实时/Deadline 调度器（原文档"Real Time Scheduler"要求在本版本明确排除,因为固定线程池 + 协程无法提供硬实时延迟保证,若后续产线需要真正的实时保证,应通过独立的 PLC/RTOS 侧闭环处理,不在本框架的 Runtime 职责内)。每个 Pipeline 阶段（Capture/Inference/Retrieval/Reason/IO）绑定一个独立的 `WorkerPool` 实例,线程数在 YAML 配置中声明,启动时创建、运行期不动态伸缩（不做"弹性线程池",因为工业产线负载模式是稳定周期性的,动态伸缩增加复杂度但收益有限)。GPU 任务通过 `GpuStreamQueue` 提交到 CUDA Stream,完成时用 `cudaStreamAddCallback` 回调恢复对应协程（`std::coroutine_handle::resume()`）,不采用忙等（busy-wait）轮询 `cudaStreamQuery`,因为忙等会占满 CPU 核心并与其他阶段的 Worker Pool 争抢。Cancellation 用 C++20 `std::stop_token` 传递,不用自定义 `bool cancelled` 标志位,因为 `stop_token` 与协程的 `co_await` 挂起点组合能做到取消传播不遗漏。Back Pressure 采用**有界队列 + 拒绝新任务**策略（队列满时 `TaskScheduler::Submit()` 返回 `Result<void>::error(ErrorCode::QueueFull)`）,不采用无界队列,因为无界队列在相机持续采集但下游推理跟不上时会导致内存无限增长,与 Memory 模块"No Memory Leak"目标冲突。
- `## 6. Data Structure` 给出 `TaskGraph` 的节点/边表示：节点是 `struct TaskNode { TaskId id; std::function<Task<void>()> work; std::vector<TaskId> dependencies; }`,边隐含在 `dependencies` 字段中,用拓扑排序驱动执行顺序。
- `## 9. Thread Model` 列出线程职责表：`Capture Thread`（1个,绑定相机采集回调）、`Inference WorkerPool`（N个,N=GPU数量,处理 TensorRT 推理）、`Retrieval WorkerPool`（M个,CPU密集,处理 FAISS 查询）、`Reason WorkerPool`（复用 Retrieval 池,同为 CPU 密集）、`GPU Callback Thread`（1个,专门处理 `cudaStreamAddCallback` 回调,避免回调阻塞 CUDA driver 内部线程）。

- [ ] **Step 2: 追加契约表增量**

概念归属表新增：`TaskGraph` / `PipelineExecutor` / `WorkerPool`,归属 `1.4`。
接口签名表新增：`TaskScheduler`、`Task<T>`、`WorkerPool`、`GpuStreamQueue`、`TaskGraph`、`PipelineExecutor`,归属 `1.4`。

- [ ] **Step 3: 校验结构与禁用机制排除声明**

Run: `grep -n "Fiber\|Real Time\|Deadline" docs/surface-ai/design/milestone-01-foundation/1.4-runtime.md`
Expected: 至少在 `## 3. Design` 中出现一次，且上下文是"明确排除"的表述（非"支持"）——人工确认排除声明存在，而非机制被引入

- [ ] **Step 4: Commit**

```bash
git add docs/surface-ai/design/milestone-01-foundation/1.4-runtime.md docs/surface-ai/glossary-and-contracts.md
git commit -m "docs: add batch 1.4 runtime design doc"
```

---

## Task 7: 批次 1.6 — 横切关注点（Error Handling / Logging / Configuration）

**Files:**
- Create: `docs/surface-ai/design/milestone-01-foundation/1.6-cross-cutting.md`
- Modify: `docs/surface-ai/glossary-and-contracts.md`

**Interfaces:**
- Consumes: `Result<T>`、`ErrorInfo`（来自 Task 2，本批次是对 1.1 已定义的错误类型的运行时基础设施补完，不重新定义类型本身）
- Produces:
  - 概念：`Logger`、`ConfigStore`
  - 接口：`class Logger`（封装 spdlog async logger）、`class ConfigStore`（封装 yaml-cpp 解析 + schema 校验）、`class ConfigSchema`

- [ ] **Step 1: 撰写批次 1.6 设计文档**

关键决策：

- `## 3. Design` 明确三个子系统：
  - **Error Handling**：本批次不重新定义 `Result<T>`/`ErrorInfo`（已在 1.1 定稿），只补充"错误码分类表"：`ErrorCode` 枚举按模块前缀分区（`Core_*`、`Runtime_*`、`Memory_*`），避免后续批次各自发明重叠错误码。
  - **Logging**：采用 **spdlog 异步 logger（`spdlog::async_logger`）+ 独立后台 IO 线程**，不采用同步 logger，因为同步写盘会阻塞 Capture/Inference 等热路径线程。日志落盘策略为**按天轮转 + 单文件最大 100MB 触发轮转（两个条件任一触发）**，不是单一按天或单一按大小策略，因为纯按天轮转在异常报警风暴场景下会导致单日志文件过大。
  - **Configuration**：采用 **yaml-cpp 加载 + 内嵌 JSON Schema 等价校验（用 yaml-cpp 手写字段校验函数，不引入额外 schema 库)**，配置文件启动时一次性加载并校验，运行期 Hot Reload 通过文件系统监听（`inotify`）触发重新加载 + 校验，校验失败**保留旧配置生效并记录错误日志**，不回退到默认值（因为工业产线场景下"静默回退默认值"比"保留已知良好的旧配置"风险更高）。
- `## 4. Interfaces` 给出 `Logger::Get(std::string_view category) -> Logger&`、`ConfigStore::Load(std::filesystem::path) -> Result<void>`、`ConfigStore::Get<T>(std::string_view key) -> Result<T>` 完整签名。
- `## 10. Performance` 给出可验证数字：日志异步队列容量 8192 条，队列满时丢弃 `Trace`/`Debug` 级别日志并计数（不丢弃 `Warning` 及以上级别）。

- [ ] **Step 2: 追加契约表增量**

概念归属表新增：`Logger` / `ConfigStore` / `ErrorCode 分类表`，归属 `1.6`。
接口签名表新增：`Logger`、`ConfigStore`、`ConfigSchema`，归属 `1.6`。

- [ ] **Step 3: 校验结构**

Run: `grep -c "^## [0-9]" docs/surface-ai/design/milestone-01-foundation/1.6-cross-cutting.md`
Expected: `14`

- [ ] **Step 4: Commit**

```bash
git add docs/surface-ai/design/milestone-01-foundation/1.6-cross-cutting.md docs/surface-ai/glossary-and-contracts.md
git commit -m "docs: add batch 1.6 cross-cutting concerns design doc"
```

---

## Task 8: 里程碑 1 收尾一致性复核

**Files:**
- Modify: `docs/surface-ai/glossary-and-contracts.md`（若发现遗漏在此补充，不修改其他批次已提交的行）

**Interfaces:**
- Consumes: 全部 Task 2-7 产出的 6 份设计文档 + 契约表全量内容
- Produces: 一份复核结论（写入契约表文件末尾的"里程碑 1 复核记录"小节）

- [ ] **Step 1: 逐文档结构校验**

Run:
```bash
for f in docs/surface-ai/design/milestone-01-foundation/*.md; do
  n=$(grep -c "^## [0-9]" "$f")
  echo "$f: $n sections"
done
```
Expected: 每个文件都输出 `14 sections`

- [ ] **Step 2: 交叉引用一致性检查**

检查 1.4-runtime.md 中引用的 `IMemoryPool`/`GpuPool`/`PinnedPool` 类型名，与 1.5-memory.md 中定稿的签名逐字比对；检查 1.6-cross-cutting.md 中引用的 `Result<T>`/`ErrorInfo`，与 1.1-core-foundation.md 中定稿的签名逐字比对。

Run: `grep -n "GpuPool\|PinnedPool" docs/surface-ai/design/milestone-01-foundation/1.4-runtime.md docs/surface-ai/design/milestone-01-foundation/1.5-memory.md`
Expected: 两个文件中出现的类名拼写完全一致（人工比对输出）

- [ ] **Step 3: 概念归属唯一性检查**

Run: `awk -F'|' 'NR>4 {print $2}' docs/surface-ai/glossary-and-contracts.md | sed '/^$/d' | sort | uniq -d`
Expected: 空输出（无重复归属的概念名称）

- [ ] **Step 4: 追加复核记录并 Commit**

在 `docs/surface-ai/glossary-and-contracts.md` 文件末尾追加：

```markdown

## 里程碑 1 复核记录

- 复核日期：2026-07-07
- 覆盖批次：1.1, 1.2, 1.3, 1.5, 1.4, 1.6（按实际执行顺序）
- 结论：6 份设计文档均满足 14 节结构；跨批次接口引用（Runtime→Memory, 横切关注点→Core基础）签名一致；契约表内概念归属无重复。
- 遗留事项：里程碑 1 spec 验证点（"启动空 task graph 走通调度/内存/日志/配置闭环"）需在后续代码实现阶段验证，本阶段仅完成设计文档。
```

```bash
git add docs/surface-ai/glossary-and-contracts.md
git commit -m "docs: milestone-1 consistency review and closeout"
```

---
