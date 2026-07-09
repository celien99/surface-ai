# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Surface AI Framework: a from-scratch design for an industrial-grade C++20 framework for surface understanding / anomaly detection (AOI-style: PCB, glass, fabric, steel, automotive, etc.). The core principle is "everything is Surface" — the framework is never coupled to a specific product; product is metadata.

The original unconstrained spec is `prompt.md` (17 layers, exhaustive "support A/B/C/D" style requirements). That spec is intentionally too broad to execute directly — it has been converted into a concrete, sequenced plan. **Always work from the derived planning docs below, not from `prompt.md` directly.**

As of now, milestone 1's design docs are complete and frozen as the interface baseline. No compilable source code exists yet — the repo is currently documentation/spec-only. Check `.superpowers/sdd/progress.md` for the current ledger before assuming what stage the project is in.

## Repo structure

```
prompt.md                                          # original unconstrained master prompt (do not execute directly)
docs/superpowers/specs/                            # phased-plan spec + code-scaffold spec (both Approved)
docs/superpowers/plans/                            # superpowers execution plans (task-by-task, checkbox tracked)
docs/surface-ai/glossary-and-contracts.md          # LIVE cross-batch contract doc — concept ownership + frozen interface signatures
docs/surface-ai/design/milestone-01-foundation/    # 6 design docs (1.1-1.6), 14-section structure each
.superpowers/sdd/                                  # per-task briefs, reports, and review diffs for the SDD workflow
```

## Workflow: this project uses superpowers spec-driven-development

Work proceeds as: spec (approved) -> plan (checkbox tasks) -> per-task brief/report in `.superpowers/sdd/` -> review diff -> commit. `.superpowers/sdd/progress.md` is the ledger of which tasks are done and what review rounds found. Read it first when resuming work.

Design docs follow a fixed, non-negotiable 14-section structure (see any file in `design/milestone-01-foundation/` for the pattern): Purpose, Responsibilities, Design, Interfaces, Workflow, Data Structure, Class Diagram, Sequence Diagram, Thread Model, Performance, Memory, Future Extension, Best Practice, Anti Pattern. Verify with `grep -c "^## [0-9]" <file>` — expect `14`.

`docs/surface-ai/glossary-and-contracts.md` is the cross-batch source of truth: each concept/interface is owned by exactly one batch and defined in exactly one doc. Other batches reference it, never redefine it. Before writing any new design doc or code that touches a cross-cutting type (`Object`, `Result<T>`, `IModule`, `Registry<TInterface>`, `IMemoryPool`, etc.), check this file for the frozen signature and reuse it verbatim.

**Design chapters ("## 3. Design") must never use list-style hedging** ("supports A/B/C/D"). They must commit: "uses X because ..., rejects Y because ...". This is enforced by grep checks in the plan and is the single biggest structural difference from the original `prompt.md` style.

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

7 milestones, 18 batches total, strictly sequential (each milestone depends on milestone 1's frozen interfaces). Milestone 1 (foundation: Core / Runtime / Memory / cross-cutting) is documented and closed. Do not start milestone 2+ design docs until explicitly asked, and do not plan more than one milestone's implementation detail at a time.

Within milestone 1, batch execution order deviates from the numeric order: 1.1 -> 1.2 -> 1.3 -> **1.5 -> 1.4** -> 1.6, because Runtime (1.4)'s GPU stream queue depends on Memory (1.5)'s `GpuPool`/`PinnedPool`, so Memory was pulled forward. Don't "fix" this back to numeric order.

## Code scaffold (planned, not yet built)

`docs/superpowers/specs/2026-07-08-milestone-1-code-scaffold-design.md` defines the first real, compilable CMake project — scoped to batches 1.1 + 1.2 only (Core foundation + Core lifecycle), explicitly excluding anything touching CUDA (the dev machine is macOS arm64, not the target platform, and no conditional-compilation shim is planned — target-platform code is written as-is and simply not exercised locally). When this scaffold exists, expect:

- vcpkg manifest mode (`vcpkg.json`) providing `tl-expected` and `gtest` only
- `CMakePresets.json` with a `default` preset wired to the vcpkg toolchain
- Verification is a real `cmake --preset default` -> `cmake --build --preset default` -> `ctest --preset default` run, not static review
- `sai_core` static library target aliased as `sai::core`
- Header/source split: template methods (`TypeRegistry::Register<T>`, `Context::Register<T>`/`Resolve<T>`) stay header-only; non-template methods move to `.cpp`

If asked to implement this scaffold, follow that design doc's class names, method signatures, and namespaces verbatim from the 1.1/1.2 design docs' "4. Interfaces" sections — do not "improve" them silently. If a signature in the docs turns out not to compile, record the deviation explicitly (e.g. in a task report) rather than silently changing it.

## Coding style (applies once C++ code exists)

- Avoid over-defensive code — don't guard against states that can't occur.
- Avoid multi-level nesting / chained `if`. Prefer early return.
- Prefer recursion over nested loops for tree/graph structures (e.g. `TaskGraph` topological walk, `Registry` lookup).
- Error handling chains through `tl::expected`'s `and_then`/`or_else`, not nested `if (result.has_value())` checks.

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
