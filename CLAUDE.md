# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Surface AI Framework: a from-scratch design for an industrial-grade C++20 framework for surface understanding / anomaly detection (AOI-style: PCB, glass, fabric, steel, automotive, etc.). The core principle is "everything is Surface" — the framework is never coupled to a specific product; product is metadata.

As of 2026-07-15: All 7 milestones + Batch T1 (Bayesian auto-tuning) complete. 621 tests pass.

- **Milestone 1** (foundation): Core / Runtime / Memory / Plugin / Infra — 6 design docs frozen, 84 tests pass
- **Milestone 2** (acquisition + imaging + I/O): Device interfaces, image type system, preprocessing chains, import/export
- **Milestone 3** (AI inference core): `inference` (IInferenceEngine, MockEngine, TensorRtEngine CUDA-gated, CLIP/DINOv3/SAM2 adapters, multi-layer feature aggregation), `embedding` (Embedding, PatchEmbedder, GlobalEmbedder, DimensionReducer/PCA, FeatureCache), `detection` (DetectionResult, PatchCore, FeatureBank/FAISS, PcaDetector, SpecularFilter, CoresetEvolution, CoresetUpdater, MultiSignalConsensus, NoveltyFilter, NormalityScorer, post_process_utils)
- **Milestone 4** (knowledge & retrieval): `knowledge` (KnowledgeRecord, KnowledgeGraph SQLite property graph, KnowledgeEvolution changelog, KnowledgeSnapshot SAVEPOINT-based, KnowledgeStore unified facade), `retrieval` (VectorPath FAISS TopK/Range/Hybrid, MetadataPath SQLite filtering, IScoreFusion/WeightedFusion/RRFFusion, HybridRetriever dual-path orchestration)
- **Milestone 5** (inference decision): `rule` (RuleEngine, AST expression engine, YAML rule storage, FactBase/ConflictResolver), `reasoner` (IReasoner/DefaultReasoner, DecisionTree traversal, ScoreCalculator, TraceRecorder, EvidenceCollector)
- **Milestone 6** (orchestration & scheduling): `pipeline` (Pipeline LoadFromYAML/Start/Submit/Drain/Stop, PipelineBuilder, StageFactory, StageQueue\<T\> bounded SPSC lock-free, IStageNode), `scheduler` (StageType → WorkerPool mapping, queue allocation)
- **Milestone 7** (visualization & application): `visualization` (PipelineViewModel, InspectionViewModel, FrameProvider, ConfigViewModel, DashboardViewModel, QML 4-screen industrial dark UI), `seat_aoi` reference app
- **Batch T1** (Bayesian auto-tuning): `tuning` (TuningSpace, BayesianOptimizer GP+EI, KnowledgeGraphObjective, TuningScheduler with circuit-breaker rollback)

**All milestones + T1 complete.**

Check `.superpowers/sdd/` for per-task briefs and reports before assuming what stage the project is in.

## Build & test commands

Two CMake presets are available:

```bash
# ── Linux (WSL / Ubuntu 22.04 x64 + NVIDIA GPU) ──
cmake --preset linux          # Configure
cmake --build --preset linux  # Build
ctest --preset linux          # Run all 621 tests

# ── macOS arm64 (Apple Silicon + Homebrew) ──
cmake --preset default          # Configure (requires libomp from Homebrew)
cmake --build --preset default  # Build
ctest --preset default          # Run all tests

# Run a single test suite by name pattern
ctest --preset linux -R "logger"

# Run a specific test case
cd build/linux && ctest -R "LoggerTest.SetLevelRoundTripsFilterDecision" --output-on-failure

# Run a single test binary directly (faster iteration, supports --gtest_filter)
cd build/linux && ./tests/detection/sai_detection_test --gtest_filter="PatchCore*"

# Build and run the Seat AOI reference app (Linux + GPU required)
cmake --build --preset linux --target seat_aoi
./build/linux/apps/seat-aoi/seat_aoi                                       # GUI mode (FakeCamera + QML)
./build/linux/apps/seat-aoi/seat_aoi --image-dir ./samples --output-dir ./results  # Headless batch mode
./build/linux/apps/seat-aoi/seat_aoi --dataset-path ./samples/normal                     # Coreset build mode (no --coreset-output)
./build/linux/apps/seat-aoi/seat_aoi --dataset-path ./samples/normal --coreset-output ./coreset.bin  # Coreset build + save
```

The `default` CMake preset targets macOS arm64 (Homebrew libomp, Apple Clang). The `linux` preset targets Ubuntu 22.04 x64 (g++-12, system OpenMP, CUDA-gated code enabled). CUDA-gated code and Linux-gated code are excluded from macOS builds — they are written per frozen design but compile-verified only on the target platform. Each module's CMakeLists.txt gates compilation at the target level (no `#ifdef` shims).

**Linux dev prerequisites (Ubuntu 22.04 x64 / WSL):**
- `vcpkg` (manifest mode) + `VCPKG_ROOT` env var
- System packages: `build-essential cmake g++-12 libspdlog-dev libyaml-cpp-dev libsqlite3-dev libopen62541-dev libaravis-dev libfaiss-dev qt6-base-dev libgl1-mesa-dev libomp-dev`
- CUDA Toolkit 12.x (for TensorRT-gated code: `embedding`, `detection`, `seat_aoi`)
- FAISS is built from a local overlay port at `vcpkg-overlays/faiss/`

**macOS dev prerequisites (Apple Silicon):**
- `vcpkg` (manifest mode) + `VCPKG_ROOT` env var
- OpenMP via Homebrew: `brew install libomp`
- FAISS from local overlay port at `vcpkg-overlays/faiss/`

## Repo structure

```
docs/superpowers/specs/                            # phased-plan spec + per-milestone design specs (Approved)
docs/superpowers/plans/                            # superpowers execution plans (task-by-task, checkbox tracked)
docs/surface-ai/design/milestone-01-foundation/    # 6 design docs (1.1-1.6), 14-section structure each
docs/surface-ai/glossary-and-contracts.md          # LIVE cross-batch contract doc — concept ownership + frozen interface signatures
.superpowers/sdd/                                  # per-task briefs, reports, and review diffs for the SDD workflow
include/sai/                                       # public headers, mirrored from src/ (sai::core, sai::runtime, etc.)
src/                                               # implementation (.cpp) + per-module CMakeLists.txt (19 modules)
tests/                                             # GoogleTest suites, one dir per module + tests/integration/ (end-to-end pipelines)
```

### Source modules (under `src/` and `include/sai/`)

| Module | Namespace | Role |
|--------|-----------|------|
| `core` | `sai::core` | Object/Resource, TypeRegistry, Factory, Context (DI), lifecycle, error codes |
| `memory` | `sai::memory` | ArenaAllocator, GpuPool (CUDA-gated), PinnedPool (CUDA-gated), PooledPtr |
| `plugin` | `sai::plugin` | PluginManager, Manifest, ModuleManager, Capability/License/Version managers |
| `runtime` | `sai::runtime` | Task\<T\> (C++20 coroutines), WorkerPool, TaskGraph, PipelineExecutor, GpuStreamQueue (CUDA-gated) |
| `infra` | `sai::infra` | Logger (spdlog), ConfigSchema/ConfigStore (yaml-cpp), inotify hot-reload (Linux-gated) |
| `device` | `sai::device` | IDevice/ICamera/ILightController interfaces, RingBuffer |
| `image` | `sai::image` | Image/RawImage/SurfaceImage/GpuImage types, ROI, preprocessing chains (Debayer, FlatField, WhiteBalance, Resize, Calibration, HDR, Compose) |
| `io` | `sai::io` | IImporter/BasicImporter (YAML metadata + PPM), IExporter/JsonExporter (JSON reports) |
| `inference` | `sai::inference` | IInferenceEngine, MockEngine, TensorRtEngine (CUDA-gated), model adapters (CLIP, DINOv3, SAM2) |
| `embedding` | `sai::embedding` | Embedding (double storage), IEmbedder/PatchEmbedder/GlobalEmbedder, DimensionReducer, FeatureCache |
| `detection` | `sai::detection` | DetectionResult, IDetector/PatchCore, FeatureBank (FAISS), PcaDetector, SpecularFilter, CoresetEvolution/CoresetUpdater, MultiSignalConsensus, NoveltyFilter/CandidateBuffer, NormalityProfile/NormalityScorer |
| `knowledge` | `sai::knowledge` | KnowledgeRecord/FieldValue, KnowledgeGraph (SQLite property graph), KnowledgeEvolution (changelog), KnowledgeSnapshot (SAVEPOINT-based), KnowledgeStore (unified facade) |
| `retrieval` | `sai::retrieval` | VectorPath (FAISS TopK/Range/Hybrid), MetadataPath (SQLite filtering), IScoreFusion/WeightedFusion/RRFFusion, HybridRetriever (dual-path orchestration) |
| `rule` | `sai::rule` | RuleEngine (AST expression engine, YAML rule storage), FactBase/ConflictResolver, Lexer/Parser |
| `reasoner` | `sai::reasoner` | IReasoner/DefaultReasoner (decision tree traversal + scoring + trace), ScoreCalculator, TraceRecorder, EvidenceCollector |
| `tuning` | `sai::tuning` | TuningSpace (parameter search space), BayesianOptimizer (GP + EI), KnowledgeGraphObjective (feedback cost), TuningScheduler (background tuning + circuit-breaker rollback) |
| `pipeline` | `sai::pipeline` | Pipeline (LoadFromYAML/Start/Submit/Drain/Stop), PipelineBuilder (YAML parsing + topology validation), StageFactory, StageQueue\<T\> (bounded SPSC lock-free), IStageNode |
| `scheduler` | `sai::scheduler` | StageType → WorkerPool mapping, queue allocation (internal headers only) |
| `visualization` | `sai::visualization` | PipelineViewModel, InspectionViewModel, FrameProvider (QQuickImageProvider), ConfigViewModel, DashboardViewModel, QML industrial dark UI |

### Namespace conventions
- All public types live under `sai::<module>` — never `sai::` directly.
- Headers use `#pragma once`. Include paths mirror the module structure: `#include "sai/core/context.h"`.
- Implementation files in `src/<module>/` compile into `sai_<module>` static library targets (e.g. `sai_core`, `sai_runtime`). CMake aliases are `sai::core`, `sai::runtime`, etc.

## Workflow: this project uses superpowers spec-driven-development

Work proceeds as: spec (approved) -> plan (checkbox tasks) -> per-task brief/report in `.superpowers/sdd/` -> review diff -> commit. `.superpowers/sdd/` contains the per-task artifacts — check it first when resuming work on an in-progress task.

Design docs follow a fixed, non-negotiable 14-section structure (see any file in `design/milestone-01-foundation/` for the pattern): Purpose, Responsibilities, Design, Interfaces, Workflow, Data Structure, Class Diagram, Sequence Diagram, Thread Model, Performance, Memory, Future Extension, Best Practice, Anti Pattern. Verify with `grep -c "^## [0-9]" <file>` — expect `14`.

`docs/surface-ai/glossary-and-contracts.md` is the cross-batch source of truth: each concept/interface is owned by exactly one batch and defined in exactly one doc. Other batches reference it, never redefine it. Before writing any new design doc or code that touches a cross-cutting type (`Object`, `Result<T>`, `IModule`, `Registry<TInterface>`, `IMemoryPool`, etc.), check this file for the frozen signature and reuse it verbatim.

**Design chapters ("## 3. Design") must never use list-style hedging** ("supports A/B/C/D"). They must commit: "uses X because ..., rejects Y because ...". This is enforced by grep checks in the plan.

## Locked technology stack

Decided in `docs/superpowers/specs/2026-07-07-surface-ai-framework-phased-plan-design.md` section 2 — do not introduce alternatives outside this table without updating that spec first:

| Concern                  | Choice                                                                                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Target platform          | Ubuntu 22.04 x64 + NVIDIA GPU (Windows/ARM64 = future compat target, not optimized for now)                                                                      |
| Inference backend        | TensorRT (FP16/INT8, dynamic shape, multi-GPU)                                                                                                                   |
| Vector search            | FAISS (in-process, optional faiss-gpu)                                                                                                                           |
| Rule engine              | self-built AST expression engine + YAML storage (no Lua — avoids arbitrary code execution)                                                                       |
| Config format            | YAML (yaml-cpp)                                                                                                                                                  |
| Concurrency              | C++20 coroutines (`co_await`) + fixed per-stage worker pools; GPU work via CUDA Stream + callback resuming the coroutine. No Fiber, no hard-real-time scheduler. |
| Logging                  | spdlog, async sinks                                                                                                                                              |
| Dependency management    | vcpkg (manifest mode); CUDA/TensorRT vendor SDKs installed manually                                                                                              |
| Error handling           | `tl::expected` (aliased as `Result<T>`) is the default; exceptions only for construction/static-init failure                                                     |
| Testing                  | GoogleTest + gmock                                                                                                                                               |
| GUI                      | Qt 6                                                                                                                                                             |
| PLC comms                | OPC UA (open62541)                                                                                                                                               |
| Camera acquisition       | GenICam / GigE Vision                                                                                                                                            |
| Knowledge metadata store | SQLite (in-process)                                                                                                                                              |
| Deployment               | Docker + systemd, nvidia-container-toolkit                                                                                                                       |

## Milestone structure

7 milestones + Batch T1 (Bayesian auto-tuning), 19 batches total, strictly sequential (each milestone depends on milestone 1's frozen interfaces).

- **Milestone 1** (foundation: Core / Runtime / Memory / cross-cutting): COMPLETE. All 6 design docs frozen, all code + 84 tests pass.
- **Milestone 2** (acquisition + imaging + I/O): COMPLETE. Device interfaces, image type system, preprocessing chains, import/export.
- **Milestone 3** (AI inference core): COMPLETE. Inference engines (MockEngine, TensorRtEngine), model adapters (CLIP, DINOv3, SAM2), multi-layer feature aggregation, embedding (PatchEmbedder, GlobalEmbedder, DimensionReducer/PCA, FeatureCache), detection (PatchCore, PcaDetector, SpecularFilter, FeatureBank/FAISS).
- **Milestone 4** (knowledge & retrieval): COMPLETE. Knowledge graph (SQLite property graph), knowledge evolution, vector-path retrieval (FAISS), metadata-path retrieval (SQLite), hybrid retrieval.
- **Milestone 5** (inference decision): COMPLETE. Rule engine (AST expression, YAML rules, FactBase/ConflictResolver), reasoner (DecisionTree, ScoreCalculator, TraceRecorder, EvidenceCollector).
- **Milestone 6** (orchestration & scheduling): COMPLETE. Pipeline (LoadFromYAML/Start/Submit/Drain/Stop), PipelineBuilder, StageFactory, StageQueue, Scheduler.
- **Milestone 7** (visualization & application): COMPLETE. ViewModels (PipelineViewModel, InspectionViewModel, FrameProvider, ConfigViewModel, DashboardViewModel), QML 4-screen industrial dark UI, Seat AOI reference app. 621 tests pass.
- **Batch T1** (Bayesian auto-tuning): COMPLETE. TuningSpace, BayesianOptimizer (GP+EI), KnowledgeGraphObjective, TuningScheduler with circuit-breaker rollback.

Within milestone 1, batch execution order deviates from the numeric order: 1.1 -> 1.2 -> 1.3 -> **1.5 -> 1.4** -> 1.6, because Runtime (1.4)'s GPU stream queue depends on Memory (1.5)'s `GpuPool`/`PinnedPool`, so Memory was pulled forward. Don't "fix" this back to numeric order.

## Coding style

- Avoid over-defensive code — don't guard against states that can't occur.
- Avoid multi-level nesting / chained `if`. Prefer early return.
- Prefer recursion over nested loops for tree/graph structures (e.g. `TaskGraph` topological walk, `Registry` lookup).
- Error handling chains through `tl::expected`'s `and_then`/`or_else`, not nested `if (result.has_value())` checks.
- Template methods (`TypeRegistry::Register<T>`, `Context::Register<T>`/`Resolve<T>`) stay header-only; non-template methods move to `.cpp`.
- CUDA-gated and Linux-gated code follows the same patterns as portable code — it is written as-is against the target platform, not wrapped in `#ifdef` shims. The CMakeLists.txt in each module gates compilation at the target level.

## Language

Design docs and specs in this repo are written in Chinese (see existing docs under `docs/`). Match that when writing or editing design documents. Code identifiers, comments, and commit messages are in English (see existing commit history and interface signatures).

## Git 提交规范

约定式提交 + Gitmoji。格式：`<type>(<scope>): <emoji> <中文描述>`，emoji 与描述间**必须有一个空格**，直接用字符不用 `:code:`。允许的 type/emoji（**严禁自造**）：

- `feat`: ✨ 新功能
- `fix`: 🐛 修复 Bug
- `chore`: 🔧 构建/工具/依赖/日常
- `refactor`: ♻️ 重构（无新功能无修复）
- `docs`: 📝 仅文档/注释
- `style`: 💄 不影响含义的格式
- `perf`: ⚡ 性能优化
- `test`: ✅ 测试
- `ci`: 💚 CI/CD 配置

示例：`fix(cpp): 🐛 修复 frame_by_index 使用 std::map 字母序导致取错帧`
