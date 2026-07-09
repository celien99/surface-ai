# Milestone 1 Remaining Batches (1.5, 1.3, 1.4, 1.6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the remaining four milestone-1 design docs (1.5 Memory, 1.3 Plugin System, 1.4 Runtime, 1.6 Cross-Cutting) as real C++20 code on top of the existing `sai_core` library, completing milestone 1's code phase.

**Architecture:** Four new static libraries — `sai_memory`, `sai_plugin`, `sai_runtime`, `sai_infra` — each split into a **portable part** (buildable and tested on any host, including this macOS arm64 dev machine) and a **platform-gated part** (CUDA for GPU memory/stream code, Linux `inotify` for config hot-reload). Gated code is implemented exactly per the frozen design doc — no dev-machine fallback, no `#ifdef`-based dual implementation — and simply excluded from the CMake source list when the required platform/toolkit isn't present, mirroring how milestone-1's code-scaffold batch already excluded all CUDA code. Every portable task ends with a real `cmake --build` + `ctest` run; every gated task ends with a compile-review-only report (code written, not locally built) plus confirmation that excluding it doesn't break the existing 18/18 passing suite.

**Tech Stack:** C++20 (coroutines, `std::stop_token`, `std::jthread`), vcpkg additions: `yaml-cpp` (plugin manifests + config), `spdlog` (logging, pulls in `fmt` transitively). CUDA Runtime API (gated, not installed here). POSIX `dlopen`/`dlsym`/`dlclose` (portable — available on both macOS and Linux, unlike CUDA/inotify). Linux `inotify` (gated).

## Global Constraints

- C++20 required throughout, same `sai::` namespace root as existing code, with the design docs' own sub-namespaces: `sai::memory`, `sai::runtime`, `sai::infra` (plugin batch stays in `sai` root per its own header comments — verify against 1.3's `## 4. Interfaces` namespace blocks per file).
- Every class name, method signature, and namespace must match the "4. Interfaces" section of the relevant design doc verbatim: `docs/surface-ai/design/milestone-01-foundation/1.5-memory.md`, `1.3-core-plugin-system.md`, `1.4-runtime.md`, `1.6-cross-cutting.md`. If a documented signature doesn't compile as written, stop and report the discrepancy — do not silently change it.
- `ErrorCode` (`include/sai/core/error.h`) is a single flat enum; each task that introduces new error codes appends them at the end of the existing enum (never reorders, never touches other batches' members), matching the append-only rule in 1.6 batch's `## 6. Data Structure`. By the end of this plan the enum must contain, in append order: the existing `Core_*`/`Lifecycle_*`, then `Plugin_*` (VersionIncompatible, CapabilityUnsupported, LicenseInvalid, CircularDependency), then `Memory_*` (ArenaExhausted, RequestExceedsSlabSize, PoolExhausted), then `Runtime_*` (QueueFull, Cancelled, NodeNotFound), then `Infra_*` (ConfigFileNotFound, ConfigParseError, ConfigValidationFailed, ConfigKeyNotFound, ConfigKeyTypeMismatch, LogSinkInitFailed).
- Coding style: avoid over-defensive code, avoid multi-level nesting, prefer early return, prefer `tl::expected` chaining (`and_then`/`or_else`/`map`). Recursive structures (dependency graphs, topological walks) are expressed as recursion, not nested loops — this is explicit in both the 1.3 and 1.4 design docs' own Design/Anti-Pattern sections; do not "simplify" them into iterative loops.
- **Platform gating, not conditional compilation:** gated `.cpp` files (GPU pools, GPU stream queue, inotify watcher) contain exactly one implementation, written for the real target platform (Ubuntu + NVIDIA GPU), with no `#ifdef`-based alternate/dummy implementation for other platforms. They are excluded from the CMake `sources` list — not compiled at all — on hosts that lack the toolkit/OS. Use `find_package(CUDAToolkit QUIET)` / `if(CUDAToolkit_FOUND)` for CUDA, and `if(CMAKE_SYSTEM_NAME STREQUAL "Linux")` for inotify. Declaring the class/method in the header is fine even when its `.cpp` isn't compiled on this host, as long as nothing in the portable test suite calls it.
- Every portable task's tests must actually build and pass via `cmake --build --preset default && ctest --preset default` on this host, and must not reduce the currently-passing count (18 tests as of commit e956e1c) — only add to it.
- Every gated task must end with confirmation (rerun `cmake --preset default && ctest --preset default`) that excluding the new gated sources from the local build still configures/builds/passes cleanly — i.e. the gating logic itself is portable even though the gated code isn't.

## File Structure Overview

```
include/sai/
├── core/error.h                       # Modify: append new ErrorCode members across this plan's tasks
├── memory/
│   ├── arena_allocator.h              # Task 1
│   ├── pooled_ptr.h                   # Task 1
│   ├── memory_pool.h                  # Task 1 (IMemoryPool, MemoryPoolConfig)
│   ├── aligned_allocator.h            # Task 1
│   ├── gpu_pool.h                     # Task 2 (gated)
│   └── pinned_pool.h                  # Task 2 (gated)
├── plugin/
│   ├── manifest.h                     # Task 3
│   ├── version_manager.h              # Task 3
│   ├── capability_manager.h           # Task 3
│   ├── license_manager.h              # Task 3
│   ├── plugin.h                       # Task 4 (IPlugin, CreatePluginFn/DestroyPluginFn)
│   ├── module_manager.h               # Task 4
│   └── plugin_manager.h               # Task 4
├── runtime/
│   ├── task.h                         # Task 5 (TaskPromise<T>, Task<T>)
│   ├── worker_pool.h                  # Task 5
│   ├── task_scheduler.h               # Task 5
│   ├── task_graph.h                   # Task 6 (TaskNode, TaskGraph)
│   ├── pipeline_executor.h            # Task 6
│   └── gpu_stream_queue.h             # Task 7 (gated)
└── infra/
    ├── logger.h                       # Task 8
    ├── config_schema.h                # Task 9
    └── config_store.h                 # Task 9 (Load/Get portable; EnableHotReload gated in Task 10)

src/
├── memory/
│   ├── CMakeLists.txt                 # Task 1 (+ gated sources Task 2)
│   ├── arena_allocator.cpp            # Task 1
│   ├── pooled_ptr_pool_bridge.cpp     # Task 1 (if any non-template pieces land here — see Task 1 notes)
│   ├── gpu_pool.cpp                   # Task 2 (gated)
│   └── pinned_pool.cpp                # Task 2 (gated)
├── plugin/
│   ├── CMakeLists.txt                 # Task 3/4
│   ├── manifest.cpp                   # Task 3 (yaml-cpp deserialization)
│   ├── version_manager.cpp            # Task 3
│   ├── capability_manager.cpp         # Task 3
│   ├── license_manager.cpp            # Task 3
│   ├── module_manager.cpp             # Task 4
│   └── plugin_manager.cpp             # Task 4
├── runtime/
│   ├── CMakeLists.txt                 # Task 5/6 (+ gated source Task 7)
│   ├── worker_pool.cpp                # Task 5
│   ├── task_scheduler.cpp             # Task 5
│   ├── task_graph.cpp                 # Task 6
│   ├── pipeline_executor.cpp          # Task 6
│   └── gpu_stream_queue.cpp           # Task 7 (gated)
└── infra/
    ├── CMakeLists.txt                 # Task 8/9 (+ gated source Task 10)
    ├── logger.cpp                     # Task 8
    ├── config_schema.cpp              # Task 9
    ├── config_store.cpp               # Task 9 (Load/Get)
    └── config_store_inotify.cpp       # Task 10 (gated, EnableHotReload)

tests/
├── memory/ (Task 1: arena_allocator_test.cpp, pooled_ptr_test.cpp; Task 2: no test — gated)
├── plugin/ (Task 3: manifest_validators_test.cpp; Task 4: plugin_manager_test.cpp + tests/plugin/fixture_plugin/ for the dlopen fixture)
├── runtime/ (Task 5: task_worker_scheduler_test.cpp; Task 6: task_graph_executor_test.cpp; Task 7: no test — gated)
└── infra/ (Task 8: logger_test.cpp; Task 9: config_store_test.cpp; Task 10: no test — gated)
```

## Execution Order

```
Task 1 (Memory: portable) ──> Task 2 (Memory: GPU, gated)
Task 3 (Plugin: manifest+validators) ──> Task 4 (Plugin: IPlugin/ModuleManager/PluginManager)
Task 5 (Runtime: Task<T>/WorkerPool/TaskScheduler) ──> Task 6 (Runtime: TaskGraph/PipelineExecutor)
                                                    └─> Task 7 (Runtime: GpuStreamQueue, gated — also needs Task 2's PinnedPool)
Task 8 (Infra: Logger) ──> Task 9 (Infra: ConfigSchema/ConfigStore) ──> Task 10 (Infra: EnableHotReload, gated)
```

Tasks 1, 3, 5, 8 have no dependency on each other (each only needs the existing `sai_core`) — execute in plan order (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) for a single controller thread, since subagent-driven-development dispatches one implementer at a time regardless.

---

## Task 1: Memory — portable core (ArenaAllocator, PooledPtr\<T\>, IMemoryPool, AlignedAllocator)

**Files:**
- Create: `include/sai/memory/arena_allocator.h`, `include/sai/memory/pooled_ptr.h`, `include/sai/memory/memory_pool.h`, `include/sai/memory/aligned_allocator.h`
- Create: `src/memory/CMakeLists.txt`, `src/memory/arena_allocator.cpp`
- Create: `tests/memory/arena_allocator_test.cpp`, `tests/memory/pooled_ptr_test.cpp`
- Modify: `include/sai/core/error.h` (append `Memory_ArenaExhausted`, `Memory_RequestExceedsSlabSize`, `Memory_PoolExhausted`)
- Modify: top-level `CMakeLists.txt` (add `add_subdirectory(src/memory)`, `add_subdirectory(tests/memory)`)

**Interfaces:**
- Consumes: `sai::Object` (`include/sai/core/object.h`), `sai::Result<T>`/`sai::ErrorInfo`/`sai::ErrorCode` (`include/sai/core/error.h`)
- Produces (namespace `sai::memory`, per `1.5-memory.md` §4):
  - `class ArenaAllocator final` — `explicit ArenaAllocator(std::size_t capacity_bytes) noexcept`, `template<typename T, typename... Args> auto Construct(Args&&...) noexcept -> Result<T*>`, `CapacityBytes()`, `UsedBytes()`.
  - `template<typename T> class PooledPtr final` — copy (refcount++), move (transfer, no refcount change), dtor (refcount-- , return to pool at zero), `Get()`, `SizeBytes()`, `UseCount()`, `IsValid()`, `operator*`/`operator->`. Private constructor, `friend class IMemoryPool`.
  - `struct MemoryPoolConfig { std::size_t slab_size; std::size_t slab_count; };`
  - `class IMemoryPool : public Object` — `Acquire(std::size_t bytes) noexcept -> Result<PooledPtr<uint8_t>>`, `Release(PooledPtr<uint8_t>&) noexcept`, `SlabSize()`, `SlabCount()`, `AvailableSlabCount()`.
  - `inline constexpr std::size_t kSimdAlignment = 64;` and `template<typename T> struct AlignedAllocator` (allocator_traits-compatible `allocate`/`deallocate`).
  - Task 2 (GpuPool/PinnedPool) and Task 7 (GpuStreamQueue) will implement/consume `IMemoryPool`, `PooledPtr<uint8_t>`, and construct their free-list nodes via `ArenaAllocator`.

Read `docs/surface-ai/design/milestone-01-foundation/1.5-memory.md` in full before starting — §3 (Design) explains *why* each shape was chosen (the `PooledPtr` reference-counting rationale, the arena/slab separation, the lock-free stack choice), §4 has the exact signatures, §9 (Thread Model) specifies the lock-free CAS stack algorithm (`PopFreeList`/`PushFreeList`) verbatim in code — reuse that exact algorithm shape (single `while`/`do-while` CAS retry loop, no nested branching) for whatever concrete free-list this task's testable harness needs.

**Scope note:** This task is the *portable* subset. `IMemoryPool` itself, `PooledPtr<T>`, `ArenaAllocator`, `AlignedAllocator`, and `MemoryPoolConfig` have zero CUDA dependency — they're pure C++20. To exercise `IMemoryPool`/`PooledPtr` end-to-end without a real GPU, write one small test-only, in-`tests/memory/` (not `include/`) concrete `IMemoryPool` implementation backed by plain `std::malloc`'d host memory (a "HostTestPool" or similar, using the same slab + lock-free free-list design as the real `GpuPool`/`PinnedPool` will in Task 2, just without `cudaMalloc`) — this is the vehicle for testing `PooledPtr`'s reference-counting and pool-return behavior for real, since the actual `GpuPool`/`PinnedPool` classes are Task 2's gated deliverable and can't be built here. Keep this test fixture out of `include/` and out of any installed target — it is scaffolding for this task's tests only, not a new public type.

- [ ] **Step 1: Write failing tests for `ArenaAllocator`**

Cover: `Construct<T>` returns a working `T*` for a POD-like test struct; capacity exhaustion returns `Memory_ArenaExhausted`; `CapacityBytes()`/`UsedBytes()` report correctly across multiple `Construct` calls.

- [ ] **Step 2: Run to verify it fails, then implement `arena_allocator.h`/`arena_allocator.cpp`**

Non-template members (ctor, dtor, `CapacityBytes`, `UsedBytes`) in the `.cpp`; the `Construct<T, Args...>` template stays header-only per the project's established template/non-template split.

- [ ] **Step 3: Write failing tests for `PooledPtr<T>` + the test-only HostTestPool fixture**

Cover: `Acquire` returns a valid `PooledPtr` with `UseCount() == 1`; copying increments `UseCount`; moving does not change `UseCount` and leaves the source `!IsValid()`; destroying the last live copy returns the slab to the pool (assert via `AvailableSlabCount()` going back up); requesting more bytes than `slab_size` returns `Memory_RequestExceedsSlabSize`; exhausting all slabs returns `Memory_PoolExhausted` on the next `Acquire`.

- [ ] **Step 4: Implement `pooled_ptr.h`, `memory_pool.h`, `aligned_allocator.h`, and the test-only HostTestPool fixture; iterate until green**

- [ ] **Step 5: Wire CMake, add new ErrorCode members, build, full suite green**

Append `Memory_ArenaExhausted`, `Memory_RequestExceedsSlabSize`, `Memory_PoolExhausted` to `error.h` after the existing `Lifecycle_RegisterAfterAssembly` member. Run `cmake --preset default && cmake --build --preset default && ctest --preset default` — expect the prior 18 plus this task's new tests, all passing.

- [ ] **Step 6: Commit**

```bash
git add include/sai/memory/ src/memory/ tests/memory/ include/sai/core/error.h CMakeLists.txt
git commit -m "feat: add portable Memory core (ArenaAllocator, PooledPtr, IMemoryPool)"
```

---

## Task 2: Memory — GPU pools (GpuPool, PinnedPool) [gated, no local build]

**Files:**
- Create: `include/sai/memory/gpu_pool.h`, `include/sai/memory/pinned_pool.h`
- Create: `src/memory/gpu_pool.cpp`, `src/memory/pinned_pool.cpp`
- Modify: `src/memory/CMakeLists.txt` (add `find_package(CUDAToolkit QUIET)`; if found, add `gpu_pool.cpp`/`pinned_pool.cpp` to the `sai_memory` sources and link `CUDA::cudart`; if not found, these two files are not compiled at all — no stub, no `#ifdef`)

**Interfaces:**
- Consumes: `IMemoryPool`, `PooledPtr<uint8_t>`, `MemoryPoolConfig`, `ArenaAllocator` (Task 1)
- Produces (namespace `sai::memory`, per `1.5-memory.md` §4): `class GpuPool final : public IMemoryPool` and `class PinnedPool final : public IMemoryPool`, both with `static auto Create(MemoryPoolConfig, ArenaAllocator&) noexcept -> Result<std::unique_ptr<...>>`, private default ctor, deleted copy/move, full `IMemoryPool` override set.
- Task 7 (GpuStreamQueue) consumes `PinnedPool` by reference in its constructor — get the type's public surface exactly right even though you can't compile it here.

Read `1.5-memory.md` §3 (the GPU/Pinned double-buffer + preallocation paragraph), §4 (exact signatures — `GpuPool` wraps `cudaMalloc`/`cudaFree`, `PinnedPool` wraps `cudaHostAlloc`/`cudaFreeHost`), §9 (the lock-free CAS free-list — reuse the exact `PopFreeList`/`PushFreeList` shape from Task 1's implementation, now operating on GPU/pinned-host memory instead of the test fixture's `malloc`'d memory), and §11 (Memory) for the arena/slab separation these two classes must respect.

**Scope note — this task cannot be verified on this machine (no CUDA toolkit).** Write the real CUDA implementation exactly as specified — no `#ifdef __APPLE__` fallback, no dummy backend. This is expected to fail to compile here for lack of CUDA headers; that's fine, because the CMake gating in `src/memory/CMakeLists.txt` means these two files simply aren't added to the build when `CUDAToolkit_FOUND` is false, so `cmake --build` on this host never attempts to compile them.

- [ ] **Step 1: Implement `gpu_pool.h`/`gpu_pool.cpp`**

`Create()` does one `cudaMalloc(slab_size * slab_count)`, builds the initial free list (one node per slab, allocated via the `ArenaAllocator&` argument, per §11's metadata/business-data separation), stores the base device pointer. `Acquire`/`Release` operate purely on the lock-free free list (no new `cudaMalloc`/`cudaFree` after construction). Destructor does one `cudaFree`.

- [ ] **Step 2: Implement `pinned_pool.h`/`pinned_pool.cpp`**

Same shape as Step 1, using `cudaHostAlloc(..., cudaHostAllocDefault)`/`cudaFreeHost` instead.

- [ ] **Step 3: Wire CMake gating**

```cmake
find_package(CUDAToolkit QUIET)
set(SAI_MEMORY_SOURCES arena_allocator.cpp)
if(CUDAToolkit_FOUND)
    list(APPEND SAI_MEMORY_SOURCES gpu_pool.cpp pinned_pool.cpp)
endif()

add_library(sai_memory STATIC ${SAI_MEMORY_SOURCES})
add_library(sai::memory ALIAS sai_memory)
target_include_directories(sai_memory PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_memory PUBLIC sai::core)
if(CUDAToolkit_FOUND)
    target_link_libraries(sai_memory PUBLIC CUDA::cudart)
endif()
target_compile_features(sai_memory PUBLIC cxx_std_20)
```

(Adjust to match whatever `sai_memory` target Task 1 already created — this step only adds the conditional GPU sources/link, it does not restructure Task 1's target.)

- [ ] **Step 4: Confirm the gating doesn't break the local (non-CUDA) build**

Run `cmake --preset default && cmake --build --preset default && ctest --preset default` — since this host has no CUDA toolkit, `gpu_pool.cpp`/`pinned_pool.cpp` must not appear in the actual build log, and the full suite (Task 1's tests plus everything before it) must still pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add include/sai/memory/gpu_pool.h include/sai/memory/pinned_pool.h src/memory/gpu_pool.cpp src/memory/pinned_pool.cpp src/memory/CMakeLists.txt
git commit -m "feat: add GpuPool and PinnedPool (CUDA-gated, not built on this host)"
```

---

## Task 3: Plugin — manifest data structures and validators

**Files:**
- Create: `include/sai/plugin/manifest.h`, `include/sai/plugin/version_manager.h`, `include/sai/plugin/capability_manager.h`, `include/sai/plugin/license_manager.h`
- Create: `src/plugin/CMakeLists.txt`, `src/plugin/manifest.cpp`, `src/plugin/version_manager.cpp`, `src/plugin/capability_manager.cpp`, `src/plugin/license_manager.cpp`
- Create: `tests/plugin/CMakeLists.txt`, `tests/plugin/manifest_validators_test.cpp`
- Modify: `include/sai/core/error.h` (append `Plugin_VersionIncompatible`, `Plugin_CapabilityUnsupported`, `Plugin_LicenseInvalid`, `Plugin_CircularDependency`)
- Modify: `vcpkg.json` (add `yaml-cpp`)
- Modify: top-level `CMakeLists.txt` (add `add_subdirectory(src/plugin)`, `add_subdirectory(tests/plugin)`)

**Interfaces:**
- Consumes: `sai::Result<T>`/`sai::ErrorInfo` (1.1)
- Produces (namespace `sai`, per `1.3-core-plugin-system.md` §4): `struct SemVer { major; minor; patch; }`, `struct VersionRange { min_inclusive; max_exclusive; }`, `struct PluginDependency { plugin_name; required_version; }`, `struct PluginManifest { name; library_path; version; capabilities; dependencies; license_token; }`, `class VersionManager` (`static CheckCompatible(VersionRange, SemVer) -> Result<void>`), `class CapabilityManager` (`RegisterKnownCapability(string) -> Result<void>`, `Validate(vector<string>) const -> Result<void>`, backed by `shared_mutex` + `unordered_set`), `class LicenseManager` (`Validate(string_view) const -> Result<void>` — stub-valid implementation is fine per the design doc's own scope note: "只锁定调用契约，不锁定具体校验算法" — a real deployment plugs in real crypto later; for this task, any deterministic, testable rule is acceptable, e.g. "non-empty token is valid" — document the choice in the report).
- Task 4 (`PluginManager`) consumes all four of these types directly.

Read `1.3-core-plugin-system.md` §3 for why SemVer uses half-open-interval range matching instead of string equality, §4 for exact signatures, §6 (Data Structure) for the manifest-vs-registry table. `PluginManifest` deserialization from `plugin.yaml` (via yaml-cpp) belongs in `manifest.cpp` — add a `LoadManifest(std::filesystem::path) -> Result<PluginManifest>` free function or static method (design doc doesn't lock the exact deserialization entry point signature, only the resulting struct's fields — pick a natural signature and note it in your report since it's a reasonable, in-scope addition, not a deviation from a frozen signature).

- [ ] **Step 1: Add `yaml-cpp` to `vcpkg.json`, reconfigure, confirm it resolves**

```json
{
  "name": "surface-ai",
  "version": "0.1.0",
  "dependencies": [
    "tl-expected",
    "gtest",
    "yaml-cpp"
  ]
}
```

Run `cmake --preset default` — expect vcpkg to fetch/build the new port (one-time cost).

- [ ] **Step 2: Write failing tests for `VersionManager`/`CapabilityManager`/`LicenseManager`**

Cover: `VersionManager::CheckCompatible` — in-range accepted, below `min_inclusive` rejected, at/above `max_exclusive` rejected (half-open interval, so exactly `max_exclusive` must be rejected), all returning `Plugin_VersionIncompatible` on failure. `CapabilityManager` — register then validate a subset succeeds; validating an unregistered capability returns `Plugin_CapabilityUnsupported`; duplicate `RegisterKnownCapability` returns `Core_TypeAlreadyRegistered`-style error per the design doc's own note (§4: "与 1.1 批次 TypeRegistry::Register 的'不覆盖'策略保持一致" — check whether it reuses `Core_TypeAlreadyRegistered` itself or needs its own code; the design doc text says "returns a `Core_TypeAlreadyRegistered`-*style*" error, meaning same shape/convention, not necessarily literally reusing `Core_TypeAlreadyRegistered` — use judgment, and if you introduce a new code for this, it must go under the `Plugin_*` prefix, not reuse `Core_*` verbatim, since `Core_*` is 1.1's owned prefix). `LicenseManager::Validate` — your chosen valid/invalid rule, both paths tested, invalid returns `Plugin_LicenseInvalid`.

- [ ] **Step 3: Write failing test(s) for `PluginManifest` YAML deserialization**

Write a minimal `plugin.yaml` fixture (inline string or temp file) with all six `PluginManifest` fields populated, deserialize it, assert every field round-trips correctly, including a `dependencies` list with at least one `PluginDependency` entry and a `capabilities` list with at least one entry.

- [ ] **Step 4: Implement all four headers + sources; iterate until green**

- [ ] **Step 5: Wire CMake, add new ErrorCode members, build, full suite green**

Append the four `Plugin_*` codes after Task 1's `Memory_*` block (or before, if Task 1 hasn't landed yet in your execution order — either way, append after whatever is currently the last member, never insert in the middle).

- [ ] **Step 6: Commit**

```bash
git add include/sai/plugin/manifest.h include/sai/plugin/version_manager.h include/sai/plugin/capability_manager.h include/sai/plugin/license_manager.h src/plugin/ tests/plugin/ include/sai/core/error.h vcpkg.json CMakeLists.txt
git commit -m "feat: add plugin manifest data structures and pre-load validators"
```

---

## Task 4: Plugin — IPlugin, ModuleManager, PluginManager

**Files:**
- Create: `include/sai/plugin/plugin.h`, `include/sai/plugin/module_manager.h`, `include/sai/plugin/plugin_manager.h`
- Create: `src/plugin/module_manager.cpp`, `src/plugin/plugin_manager.cpp`
- Create: `tests/plugin/plugin_manager_test.cpp`
- Create: `tests/plugin/fixture_plugin/CMakeLists.txt`, `tests/plugin/fixture_plugin/fixture_plugin.cpp` — a tiny real shared library (built by CMake as `SHARED`, whatever extension the platform produces — `.dylib` here, `.so` on the target platform; do not hardcode the extension) implementing a minimal concrete `IPlugin` and exporting `CreatePlugin`/`DestroyPlugin` per §4's `CreatePluginFn`/`DestroyPluginFn`, plus a `plugin.yaml` manifest fixture next to it, so `PluginManager::Load` can be exercised against a *real* `dlopen`, not a mock.
- Modify: `src/plugin/CMakeLists.txt` (link `${CMAKE_DL_LIBS}` into `sai_plugin`; link `sai::core` for `IModule`/`Context`/`Registry`)

**Interfaces:**
- Consumes: `IModule`, `Context`, `LifecycleState`, `Registry<TInterface>`, `Factory<TInterface>` (1.2, unchanged, reused not redefined), `TypeId`/`detail::Fnv1aHash` (1.1), `PluginManifest`/`VersionManager`/`CapabilityManager`/`LicenseManager` (Task 3)
- Produces (namespace `sai`, per `1.3-core-plugin-system.md` §4): `class IPlugin : public IModule, public IReflectable` (+ `GetManifest() const -> const PluginManifest&`), `extern "C" using CreatePluginFn = IPlugin*(*)(); extern "C" using DestroyPluginFn = void(*)(IPlugin*);`, `class ModuleManager` (thin forwarder to `Context::RegisterModule`, no `Registry`), `class PluginManager` (`DiscoverManifests`, `Load`, `Resolve`, `Shutdown`, plus private `LoadSingle`/`EnsureLoaded`/`ResolveEach`/`ShutdownFrom`).

This is the most architecturally significant task in this plan — read `1.3-core-plugin-system.md` §3 in full before writing any code, especially:
- The ownership-conflict resolution (why `PluginManager` drives `IPlugin` lifecycle hooks itself instead of handing them to `Context::RegisterModule` — `shared_ptr` vs `unique_ptr` cannot be reconciled, see the design doc's own extended argument).
- Why plugin construction goes through exported C `CreatePlugin`/`DestroyPlugin` symbols instead of `Factory<TInterface>` (heap-allocator-mismatch across `.so` boundaries — `DestroyPlugin` **must** be the one that calls `delete`, not `PluginManager`).
- The recursive `EnsureLoaded`/`ResolveEach` dependency-DAG walk (§5 gives the exact pseudocode — implement this recursively, with the `visiting` set for cycle detection, not as nested loops) and the recursive `ShutdownFrom` (reverse-order-by-`load_order_`, "continue after failure, keep first error" — same pattern already implemented and tested for `Context::Stop()` in the existing code scaffold; reuse that same "record first error, keep going" logic shape here).

- [ ] **Step 1: Implement `plugin.h`, `module_manager.h`/`.cpp`**

`ModuleManager` is a thin forwarder — write its test alongside (a fake `IModule` registered via `RegisterBuiltin`, assert it lands in the `Context` the same way direct `Context::RegisterModule` would).

- [ ] **Step 2: Build the fixture plugin**

`tests/plugin/fixture_plugin/fixture_plugin.cpp` defines a concrete `IPlugin` (e.g. `FixturePlugin`) whose `OnInitialize`/`OnStart`/`OnStop` record calls into a shared, test-observable location (simplest: a static counter or a file write the test can check — a static in-process counter is simpler and sufficient since the test loads the `.dylib`/`.so` into the *same* process). Export `CreatePlugin`/`DestroyPlugin` as `extern "C"`. Write a matching `plugin.yaml` next to it (or generated by the test at a known temp path) with a name, version, empty `capabilities`/`dependencies`, and a license token your Task 3 `LicenseManager` rule accepts.

- [ ] **Step 3: Write failing tests for `PluginManager`**

Cover, against the real fixture plugin: `DiscoverManifests` on the fixture's directory finds the manifest without `dlopen`-ing anything; `Load("fixture-plugin-name")` succeeds, and after it the plugin's `OnInitialize`/`OnStart` have genuinely run (assert via the fixture's observable counter/flag, not just `.has_value()`); `Resolve(TypeId)` after `Load` returns the loaded instance; `Shutdown()` runs `OnStop` (assert via the fixture again); loading an already-loaded plugin a second time is idempotent (no double-`OnInitialize`). Also cover, with synthetic (non-dlopen) manifests/graphs constructed in-test: a manifest declaring a dependency on a plugin not in `manifest_index_` fails cleanly; a manifest cycle (A depends on B, B depends on A) is caught by the `visiting` set and returns `Plugin_CircularDependency` rather than recursing infinitely.

- [ ] **Step 4: Implement `plugin_manager.h`/`.cpp`; iterate until green**

- [ ] **Step 5: Wire CMake, build, full suite green**

Run `cmake --preset default && cmake --build --preset default && ctest --preset default`.

- [ ] **Step 6: Commit**

```bash
git add include/sai/plugin/plugin.h include/sai/plugin/module_manager.h include/sai/plugin/plugin_manager.h src/plugin/module_manager.cpp src/plugin/plugin_manager.cpp src/plugin/CMakeLists.txt tests/plugin/plugin_manager_test.cpp tests/plugin/fixture_plugin/
git commit -m "feat: add IPlugin, ModuleManager, and PluginManager with real dlopen coverage"
```

---

## Task 5: Runtime — Task\<T\>, WorkerPool, TaskScheduler

**Files:**
- Create: `include/sai/runtime/task.h`, `include/sai/runtime/worker_pool.h`, `include/sai/runtime/task_scheduler.h`
- Create: `src/runtime/CMakeLists.txt`, `src/runtime/worker_pool.cpp`, `src/runtime/task_scheduler.cpp`
- Create: `tests/runtime/CMakeLists.txt`, `tests/runtime/task_worker_scheduler_test.cpp`
- Modify: `include/sai/core/error.h` (append `Runtime_QueueFull`, `Runtime_Cancelled`, `Runtime_NodeNotFound`)
- Modify: top-level `CMakeLists.txt` (add `add_subdirectory(src/runtime)`, `add_subdirectory(tests/runtime)`)

**Interfaces:**
- Consumes: `sai::Result<T>`, `sai::TypeId`, `sai::detail::Fnv1aHash` (1.1), `sai::Registry<TInterface>` (1.2, instantiated as `Registry<WorkerPool>`)
- Produces (namespace `sai::runtime`, per `1.4-runtime.md` §4): `template<typename T> class TaskPromise` (`return_value`, `unhandled_exception`, `GetStopToken`, `get_return_object`, `initial_suspend`/`final_suspend` both `std::suspend_always`), `template<typename T> using Task = std::coroutine_handle<TaskPromise<T>>;`, `class WorkerPool final` (fixed `thread_count`, bounded `queue_capacity`, `TryEnqueue(coroutine_handle<>) -> Result<void>`, `ThreadCount()`, `PendingCount()`), `class TaskScheduler final` (`explicit TaskScheduler(Registry<WorkerPool>&)`, `Submit(TypeId stage_id, coroutine_handle<>) -> Result<void>`).
- Task 6 (`TaskGraph`/`PipelineExecutor`) and Task 7 (`GpuStreamQueue`) both build on `Task<T>`; Task 6's `PipelineExecutor` holds a `TaskScheduler&`.

Read `1.4-runtime.md` §3 for why `Task<T>` uses manual `handle.destroy()` ownership (no RAII wrapper — see §11 Memory's extended argument) and why cancellation is `std::stop_token`-based, and §11 for the exact manual-destroy contract every task in this plan that touches `Task<T>` must honor: **whoever confirms `handle.done()` and has taken the result must call `handle.destroy()`** — bake this into `WorkerPool`'s internal dispatch loop and into this task's own tests (a test that submits a coroutine, waits for completion, and destroys the handle, verified via no leak — at minimum, structure the test so destroy is called exactly once per submitted task, and note in your report if you don't have a mechanical leak-detection tool available).

**Scope note:** This task is fully portable — coroutines, `std::jthread`, `std::stop_token`, and the `Registry<WorkerPool>` instantiation have no CUDA/GPU dependency. Test it for real.

- [ ] **Step 1: Write failing tests for `Task<T>`/`TaskPromise<T>` machinery**

Cover: a trivial coroutine that `co_return`s a `Result<int>` success value; a coroutine that `co_return`s a `Result<int>` error; `GetStopToken()` reflects a `stop_token` passed at construction and its `stop_requested()` state after the associated `stop_source` calls `request_stop()`.

- [ ] **Step 2: Implement `task.h`; iterate until green**

- [ ] **Step 3: Write failing tests for `WorkerPool`**

Cover: constructing with `thread_count=2, queue_capacity=4`; `TryEnqueue` of a resumable coroutine handle actually gets resumed on one of the pool's threads (assert via a shared atomic counter the coroutine increments); submitting more items than `queue_capacity` while all worker threads are deliberately blocked (e.g. via a synchronization primitive in the test) returns `Runtime_QueueFull` on the overflow submissions; `ThreadCount()` reports the constructed value; `PendingCount()` reflects queue depth at a controlled point in the test.

- [ ] **Step 4: Write failing tests for `TaskScheduler`**

Cover: constructing with a `Registry<WorkerPool>` that has one stage registered; `Submit(stage_id, handle)` for a registered `stage_id` reaches that `WorkerPool`; `Submit` for an unregistered `stage_id` returns the `Registry::Resolve`'s `Core_TypeNotFound` error (per 1.2's `Registry<TInterface>::Resolve` contract — `TaskScheduler` does not need to invent its own "stage not found" code, it should propagate what `Registry::Resolve` already returns, chained via `and_then`).

- [ ] **Step 5: Implement `worker_pool.h`/`.cpp`, `task_scheduler.h`/`.cpp`; iterate until green**

- [ ] **Step 6: Wire CMake, add new ErrorCode members, build, full suite green**

- [ ] **Step 7: Commit**

```bash
git add include/sai/runtime/task.h include/sai/runtime/worker_pool.h include/sai/runtime/task_scheduler.h src/runtime/ tests/runtime/ include/sai/core/error.h CMakeLists.txt
git commit -m "feat: add Task<T> coroutine machinery, WorkerPool, and TaskScheduler"
```

---

## Task 6: Runtime — TaskGraph, TaskNode, PipelineExecutor

**Files:**
- Create: `include/sai/runtime/task_graph.h`, `include/sai/runtime/pipeline_executor.h`
- Create: `src/runtime/task_graph.cpp`, `src/runtime/pipeline_executor.cpp`
- Create: `tests/runtime/task_graph_executor_test.cpp`
- Modify: `src/runtime/CMakeLists.txt`, `tests/runtime/CMakeLists.txt` (append new sources/targets)

**Interfaces:**
- Consumes: `Task<T>`, `TaskScheduler` (Task 5), `TypeId` (1.1)
- Produces (namespace `sai::runtime`, per `1.4-runtime.md` §4): `using TaskId = std::uint64_t;`, `struct TaskNode { TaskId id; TypeId stage_id; std::function<Task<void>()> work; std::vector<TaskId> dependencies; };`, `class TaskGraph final` (`AddNode`, `NodeAt(TaskId) -> Result<const TaskNode*>`, `RunToCompletion(PipelineExecutor&, stop_token) -> Task<void>`), `class PipelineExecutor final` (`explicit PipelineExecutor(TaskScheduler&)`, `Dispatch(TypeId stage_id, Task<void> work, stop_token) -> Task<void>`).

Read `1.4-runtime.md` §5 (Workflow) — it gives the exact recursive pseudocode for `EnsureCompleted`/`ResolveDependencies` (topological execution via mutual recursion over the dependency DAG, with a `completed` result-cache map doubling as memoization, no environment-provided cycle detection since the graph is assumed acyclic by construction — see §5's closing paragraph explaining why, unlike Task 4's plugin dependency graph, this one doesn't need a `visiting` set). Implement `TaskGraph::RunToCompletion` following that exact recursive shape — this is the same head/tail and depth-first recursion pattern already used in Task 4's plugin dependency resolution; do not flatten it into iterative loops.

**Scope note:** Fully portable — no CUDA dependency anywhere in `TaskGraph`/`TaskNode`/`PipelineExecutor`.

- [ ] **Step 1: Write failing tests for `TaskGraph`/`TaskNode`**

Cover: `AddNode` then `NodeAt` round-trips a node; `NodeAt` for an unknown `TaskId` returns `Runtime_NodeNotFound`.

- [ ] **Step 2: Write failing tests for `PipelineExecutor`**

Cover: `Dispatch` on a registered stage actually runs the given `Task<void>` work via the underlying `TaskScheduler`/`WorkerPool` chain from Task 5 (assert via a shared counter the work increments, same pattern as Task 5's WorkerPool test).

- [ ] **Step 3: Write failing tests for `TaskGraph::RunToCompletion`**

Cover: a 3-node linear chain (A -> B -> C by dependency) executes in dependency order (assert via an ordered log each node's `work()` appends to); a diamond dependency (A depends on B and C, both of which depend on D) executes D exactly once despite being a shared dependency of two nodes (this is the `completed` cache's job — assert D's work only runs once); a node whose `work()` returns an error short-circuits nodes that depend on it (assert the dependent node's `work()` never runs) while still returning the underlying error from `RunToCompletion`.

- [ ] **Step 4: Implement `task_graph.h`/`.cpp`, `pipeline_executor.h`/`.cpp`; iterate until green**

- [ ] **Step 5: Wire CMake, build, full suite green**

- [ ] **Step 6: Commit**

```bash
git add include/sai/runtime/task_graph.h include/sai/runtime/pipeline_executor.h src/runtime/task_graph.cpp src/runtime/pipeline_executor.cpp src/runtime/CMakeLists.txt tests/runtime/task_graph_executor_test.cpp tests/runtime/CMakeLists.txt
git commit -m "feat: add TaskGraph and PipelineExecutor with recursive topological execution"
```

---

## Task 7: Runtime — GpuStreamQueue [gated, no local build]

**Files:**
- Create: `include/sai/runtime/gpu_stream_queue.h`
- Create: `src/runtime/gpu_stream_queue.cpp`
- Modify: `src/runtime/CMakeLists.txt` (add gated source + `CUDA::cudart` link, same `if(CUDAToolkit_FOUND)` pattern as Task 2; also link `sai::memory` for `PinnedPool`)

**Interfaces:**
- Consumes: `PinnedPool` (Task 2, gated), `PooledPtr<uint8_t>` (Task 1), `Task<T>` (Task 5)
- Produces (namespace `sai::runtime`, per `1.4-runtime.md` §4): `struct GpuCompletionEvent { std::coroutine_handle<> handle; cudaError_t status; };`, `enum class CopyDirection { HostToDevice, DeviceToHost };`, `class GpuStreamQueue final` (`static Create(stream_count, PinnedPool&) -> Result<unique_ptr<GpuStreamQueue>>`, `EnqueueAsyncCopy(PooledPtr<uint8_t>, bytes, CopyDirection, stop_token) -> Task<void>`, `StreamCount()`).

**This is the single most architecturally subtle piece in this plan — read `1.4-runtime.md` §3 in full, specifically the three consecutive paragraphs on: (1) why `cudaStreamAddCallback` + forwarding beats busy-waiting `cudaStreamQuery`, (2) why the callback must forward through a dedicated GPU Callback Thread instead of calling `resume()` directly on the CUDA driver's callback thread, and (3) why in-flight GPU operations are cancellation-immune (no `stop_callback` registered during flight) to avoid a double-`resume()` race with the completion callback.** Implement exactly the mechanism described there: a bounded MPSC lock-free queue fed by `cudaStreamAddCallback`'s trampoline function (push `GpuCompletionEvent`, return immediately — no CUDA API calls inside the trampoline beyond what's already committed), consumed by one dedicated `std::jthread` (`gpu_callback_thread_`) that pops and calls `resume()`. `stop_token` is checked exactly once at `EnqueueAsyncCopy`'s entry (before the copy is issued) — never registered as a `stop_callback` for the in-flight duration.

**Scope note — cannot be verified on this machine (no CUDA toolkit, and Task 2's `PinnedPool` it depends on is itself unbuilt here).** Write the real implementation; expect it to not compile locally; confirm the CMake gating means it's never attempted.

- [ ] **Step 1: Implement `gpu_stream_queue.h`/`.cpp`** per the design doc's exact mechanism.

- [ ] **Step 2: Wire CMake gating** — same `if(CUDAToolkit_FOUND)` pattern as Task 2, this time also requiring `sai::memory`'s gated `PinnedPool` symbol to exist (i.e. this file is only meaningful when Task 2's gated build is active too).

- [ ] **Step 3: Confirm the gating doesn't break the local build** — `cmake --preset default && cmake --build --preset default && ctest --preset default`, full suite still green, `gpu_stream_queue.cpp` absent from the build log.

- [ ] **Step 4: Commit**

```bash
git add include/sai/runtime/gpu_stream_queue.h src/runtime/gpu_stream_queue.cpp src/runtime/CMakeLists.txt
git commit -m "feat: add GpuStreamQueue (CUDA-gated, not built on this host)"
```

---

## Task 8: Infra — Logger

**Files:**
- Create: `include/sai/infra/logger.h`
- Create: `src/infra/CMakeLists.txt`, `src/infra/logger.cpp`
- Create: `tests/infra/CMakeLists.txt`, `tests/infra/logger_test.cpp`
- Modify: `include/sai/core/error.h` (append `Infra_LogSinkInitFailed` — the other five `Infra_*` codes are Task 9's)
- Modify: `vcpkg.json` (add `spdlog`)
- Modify: top-level `CMakeLists.txt` (add `add_subdirectory(src/infra)`, `add_subdirectory(tests/infra)`)

**Interfaces:**
- Consumes: `sai::Result<T>`, `sai::ErrorInfo` (1.1)
- Produces (namespace `sai::infra`, per `1.6-cross-cutting.md` §4): `enum class LogLevel { Trace, Debug, Info, Warning, Error, Critical };`, `class Logger final` — `static Get(string_view category) -> Logger&`, `static InitializeGlobalSinks(std::filesystem::path log_dir) -> Result<void>`, `SetLevel(LogLevel) noexcept`, `template<typename... Args> Log(LogLevel, fmt::format_string<Args...>, Args&&...) noexcept`, `DroppedCount() const noexcept -> uint64_t`.

Read `1.6-cross-cutting.md` §3 (the dual-tier design: `block_tier_` for Warning+ using `overflow_policy::block`, `drop_tier_` for Trace/Debug using `overflow_policy::overrun_oldest`, both sharing one `spdlog::thread_pool` of capacity 8192) and §9 (Thread Model — the category lookup table uses the same `shared_mutex` read/write-split pattern as `TypeRegistry` from 1.1, reuse that pattern rather than reinventing one). §5 gives the exact log-write decision path (level filter first, then tier routing).

**Scope note:** Fully portable — spdlog is cross-platform.

- [ ] **Step 1: Add `spdlog` to `vcpkg.json`, reconfigure**

```json
{
  "name": "surface-ai",
  "version": "0.1.0",
  "dependencies": [
    "tl-expected",
    "gtest",
    "yaml-cpp",
    "spdlog"
  ]
}
```

- [ ] **Step 2: Write failing tests for `Logger`**

Cover: `InitializeGlobalSinks` against a temp directory succeeds; calling it a second time is idempotent (no error, no duplicate sink construction — assert via not crashing/erroring, exact internal-state assertion isn't necessary); `Logger::Get("SameCategory")` called twice returns references to the *same* underlying instance (identity, not just equal values — compare `&Logger::Get(...)` addresses); a message logged below the category's `SetLevel` threshold does not increment any observable counter (you may need to expose enough via `DroppedCount()`/a test-visible sink to assert this meaningfully — if spdlog's public API doesn't expose enough to assert "was this actually written," at minimum assert no crash and that the level filter short-circuits, and note in your report exactly what could and couldn't be verified); a deliberately tiny/overwhelmed `drop_tier_` scenario increments `DroppedCount()` (this may require constructing a `Logger` test path with a very small queue capacity if the constructor allows it, or accepting this as a ⚠️-style noted gap if `InitializeGlobalSinks`'s fixed 8192 capacity makes triggering real overflow impractical in a fast unit test — do not spend excessive effort forcing 8192+ log writes if a smaller, documented, targeted test isn't feasible with the locked interface; report what you did and why).

- [ ] **Step 3: Implement `logger.h`/`.cpp`; iterate until green**

- [ ] **Step 4: Wire CMake, add `Infra_LogSinkInitFailed`, build, full suite green**

- [ ] **Step 5: Commit**

```bash
git add include/sai/infra/logger.h src/infra/ tests/infra/logger_test.cpp tests/infra/CMakeLists.txt include/sai/core/error.h vcpkg.json CMakeLists.txt
git commit -m "feat: add Logger (spdlog async dual-tier sink)"
```

---

## Task 9: Infra — ConfigSchema, ConfigStore (Load/Get — portable subset)

**Files:**
- Create: `include/sai/infra/config_schema.h`, `include/sai/infra/config_store.h`
- Create: `src/infra/config_schema.cpp`, `src/infra/config_store.cpp`
- Create: `tests/infra/config_store_test.cpp`
- Modify: `include/sai/core/error.h` (append `Infra_ConfigFileNotFound`, `Infra_ConfigParseError`, `Infra_ConfigValidationFailed`, `Infra_ConfigKeyNotFound`, `Infra_ConfigKeyTypeMismatch`)
- Modify: `src/infra/CMakeLists.txt`, `tests/infra/CMakeLists.txt`

**Interfaces:**
- Consumes: `sai::Result<T>`, `yaml-cpp`'s `YAML::Node` (already a dependency as of Task 3)
- Produces (namespace `sai::infra`, per `1.6-cross-cutting.md` §4): `using FieldValidator = std::function<Result<void>(const YAML::Node&)>;`, `class ConfigSchema final` (`RequireField(string field_path, FieldValidator) -> ConfigSchema&`, `Validate(const YAML::Node&) const -> Result<void>`), `class ConfigStore final` (`explicit ConfigStore(ConfigSchema)`, `Load(std::filesystem::path) -> Result<void>`, `template<typename T> Get(string_view key) const -> Result<T>`; this task implements `Load`/`Get` only — `EnableHotReload` is declared in the header per the frozen signature but implemented in Task 10).

Read `1.6-cross-cutting.md` §3 (why hand-written field validators over a schema library, why load-once-and-validate over lazy parsing) and §4 for the exact `ConfigSchema`/`ConfigStore` signatures, including the point-separated field path convention (`"capture.camera_count"`) that `RequireField`/`Validate` must walk.

**Scope note:** `Load`/`Get<T>` are pure yaml-cpp + in-memory tree operations — fully portable. Only the `inotify`-based `EnableHotReload` (Task 10) is platform-gated; this task's `ConfigStore.h` declares that method but this task does not define it (Task 10's `.cpp` does).

- [ ] **Step 1: Write failing tests for `ConfigSchema`**

Cover: `RequireField` + `Validate` on a YAML tree that satisfies all registered rules succeeds; a missing required field fails with `Infra_ConfigValidationFailed`; a field present but failing its validator closure fails the same way; validation stops at the first failure (register two rules where the first fails — assert the second validator closure is never invoked, e.g. via a closure that increments a counter it shouldn't reach).

- [ ] **Step 2: Write failing tests for `ConfigStore::Load`/`Get<T>`**

Cover: `Load` on a valid YAML file (write one to a temp path in the test) with a schema that accepts it succeeds; `Load` on a nonexistent path returns `Infra_ConfigFileNotFound`; `Load` on syntactically invalid YAML returns `Infra_ConfigParseError`; `Load` on valid YAML that fails the schema returns `Infra_ConfigValidationFailed` and leaves the store's internal tree unchanged from before the failed `Load` (there's nothing to roll back to on a *first* load, so assert the *next* successful `Get` still fails cleanly rather than crashing); after a successful `Load`, `Get<T>` for an existing point-path key returns the right value and type; `Get<T>` for a missing key returns `Infra_ConfigKeyNotFound`; `Get<T>` for an existing key with an incompatible requested type returns `Infra_ConfigKeyTypeMismatch`.

- [ ] **Step 3: Implement `config_schema.h`/`.cpp`, `config_store.h`/`.cpp` (Load/Get only); iterate until green**

- [ ] **Step 4: Wire CMake, add new ErrorCode members, build, full suite green**

- [ ] **Step 5: Commit**

```bash
git add include/sai/infra/config_schema.h include/sai/infra/config_store.h src/infra/config_schema.cpp src/infra/config_store.cpp src/infra/CMakeLists.txt tests/infra/config_store_test.cpp tests/infra/CMakeLists.txt include/sai/core/error.h
git commit -m "feat: add ConfigSchema and ConfigStore Load/Get (portable subset)"
```

---

## Task 10: Infra — ConfigStore::EnableHotReload [gated, no local build]

**Files:**
- Create: `src/infra/config_store_inotify.cpp` (defines `ConfigStore::EnableHotReload` and any private helpers it needs)
- Modify: `src/infra/CMakeLists.txt` (add this source only `if(CMAKE_SYSTEM_NAME STREQUAL "Linux")`)

**Interfaces:**
- Consumes: `ConfigStore`'s private members (`schema_`, `path_`, `mutex_`, `root_`, `watch_thread_`) declared in Task 9's `config_store.h`
- Produces: the definition of `ConfigStore::EnableHotReload(std::stop_token) -> Result<void>` per `1.6-cross-cutting.md` §4/§5/§8 (sequence diagram) — spawn `watch_thread_` as a `std::jthread` that blocks on `inotify`'s `read()` for the watched file, and on each write event: re-read + re-parse the file, re-run `schema_.Validate`, and only on full success take `unique_lock(mutex_)` and replace `root_`; on any failure (parse or validate), log an `Error` via `sai::infra::Logger::Get("ConfigStore")` (Task 8) and leave `root_` untouched — never fall back to defaults.

Read `1.6-cross-cutting.md`'s hot-reload sequence diagram (§8) and the "why not defaults" paragraph in §3 closely — this is a case where the deliberately *unsafe*-looking choice (keep serving a stale-but-known-good config rather than a fresh-but-unvalidated one) is the correct one per the design doc's own reasoning; do not "improve" it into a default-fallback.

**Scope note — `inotify` is Linux-only; this cannot be built or tested on macOS.** Write the real Linux implementation using `<sys/inotify.h>` — no macOS `FSEvents`/`kqueue` fallback, no dummy stub. This file is gated out of the local build entirely by CMake.

- [ ] **Step 1: Implement `config_store_inotify.cpp`** per the design doc's exact mechanism.

- [ ] **Step 2: Wire CMake gating**

```cmake
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(sai_infra PRIVATE config_store_inotify.cpp)
endif()
```

- [ ] **Step 3: Confirm the gating doesn't break the local build** — `cmake --preset default && cmake --build --preset default && ctest --preset default`, full suite still green (note: on macOS, `ConfigStore::EnableHotReload` is declared but never defined anywhere in the link — this is fine because Task 9's tests never call it; confirm nothing does).

- [ ] **Step 4: Commit**

```bash
git add src/infra/config_store_inotify.cpp src/infra/CMakeLists.txt
git commit -m "feat: add ConfigStore::EnableHotReload via inotify (Linux-gated, not built on this host)"
```

---
