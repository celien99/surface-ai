# Task 10 Report: Rewrite ExportStage

## Summary

Rewrote `export_stage.cpp` from passthrough stub to full implementation: resolves `IExporter` from `Context` (DI), converts `ReasoningResult` to `InspectionResult`, maps triggered rules to `DefectRecord`s, and exports JSON via `IExporter::Export`.

Also modified `exporter.h` — `IExporter` now inherits both `IPlugin` and `IService` (with `SAI_DECLARE_TYPE_ID`) so `Context::Resolve<io::IExporter>()` compiles.

## Files Changed

1. **Modified**: `src/pipeline/export_stage.cpp` — Full rewrite from brief's spec:
   - Constructor reads `output_dir` from YAML config (default `/tmp/surface-ai/results/`)
   - `OnInitialize` resolves `IExporter` from Context (sets `stub_ = false` if found)
   - `Process` builds `InspectionResult` from `ReasoningResult`, maps `triggered_rules` → `DefectRecord`s, creates output dir, calls `IExporter::Export`
   - Passthrough still returns the original `ReasoningResult` for pipeline continuity

2. **Modified**: `include/sai/io/exporter.h` — `IExporter` now inherits from both `IPlugin` and `IService` (with `SAI_DECLARE_TYPE_ID`):
   - Added `#include <sai/core/service.h>`
   - Changed `class IExporter : public IPlugin` to `class IExporter : public IPlugin, public IService`
   - Added `SAI_DECLARE_TYPE_ID(sai.io.exporter)` to satisfy `Reflectable` concept

## Compilation

Both files compile cleanly (verified via `g++` on macOS arm64). The full `cmake --build` is blocked by a pre-existing unrelated error in `preprocess_stage.cpp`.

## Concerns

### C1: Brief specified single-file change, but `exporter.h` needed modification
The brief lists only `src/pipeline/export_stage.cpp`. However, `ctx.Resolve<io::IExporter>()` requires `IExporter` to satisfy `std::derived_from<T, IService>`. Since `IExporter` inherited solely from `IPlugin` (which is `IModule` + `IReflectable`, NOT `IService`), the call would not compile without adding `IService` as a base of `IExporter`.

### C2: Diamond inheritance on `Object` and `IReflectable`
`IExporter` now derives from both `IPlugin` (→ `IModule` → `Object` + `IReflectable`) and `IService` (→ `Object` + `IReflectable`). This creates non-virtual diamond inheritance on `Object` and `IReflectable`. This is acceptable because:
- `Object::~Object()` and `IReflectable::~IReflectable()` are trivial default destructors — calling them on separate sub-objects is harmless
- All pointer conversions required by `Context::Register`/`Resolve` (`IExporter*` ↔ `IService*`) are unambiguous (direct base relationship)
- `SAI_DECLARE_TYPE_ID` provides a single `TypeId()` override covering both `IReflectable` paths

### C3: `IImporter` has the same pattern but is not yet affected
`IImporter` in `importer.h` also inherits solely from `IPlugin`. If it later needs `Context::Resolve`, it will require the same treatment. Not changed in this task since the brief only covers export.

## Verification

```bash
# Compile-check export_stage.cpp (passes)
g++ -std=c++20 -c -I include -I <vcpkg>/include -o /dev/null src/pipeline/export_stage.cpp

# Compile-check json_exporter.cpp with changed exporter.h (passes)
g++ -std=c++20 -c -I include -I <vcpkg>/include -o /dev/null src/io/json_exporter.cpp

# Compile-check test file (passes — only gtest char8_t warning, unrelated)
g++ -std=c++20 -c -I include -I <vcpkg>/include -I <gtest>/include -o /dev/null tests/io/json_exporter_test.cpp

# Full build blocked by pre-existing preprocess_stage.cpp error
```

Commit: `5264095` (2 files, +47/−6 lines)
