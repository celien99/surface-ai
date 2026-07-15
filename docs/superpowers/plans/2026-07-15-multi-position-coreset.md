# Multi-Position Coreset Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add `position_id` routing through the detection pipeline so each `(surface_id, position_id)` gets its own `PatchCore` + `FeatureBank` + `CoresetEvolution`.

**Architecture:** Embedding and DetectionResult get `position_id` fields. DetectStage maintains `map<(sid,pid), shared_ptr<IDetector>>` instead of a single detector. CoresetManifest YAML maps positions to bank files. BuildCoreset groups by position. AppBuilder creates per-position PatchCore+Evolution instances.

**Tech Stack:** C++20, YAML-cpp, FAISS (existing deps — no new libraries).

## Global Constraints

- All existing tests must continue to pass
- `--coreset` single-position path must work unchanged (backward compat)
- position_id default = 0 for all existing code paths
- DetectStage::SetDetector() must still work (delegates to AddDetector("", 0, det))
- On macOS: code written but not compile-verified (seat_aoi is Linux+CUDA gated)
- Follow existing coding style: early returns, `Result<T>`, English identifiers

---

### Task 1: Embedding + position_id field

**Files:**
- Modify: `include/sai/embedding/embedding.h`

**Interfaces:**
- Produces: `Embedding::PositionId()`, `Embedding::SetPositionId()`, `position_id_` member

- [ ] **Step 1: Add position_id_ member and accessors**

In `include/sai/embedding/embedding.h`, after the existing `surface_id_` block (around line 82-85), add:

```cpp
    [[nodiscard]] auto PositionId() const noexcept -> std::uint16_t
    { return position_id_; }
    auto SetPositionId(std::uint16_t id) noexcept -> void
    { position_id_ = std::move(id); }
```

In the private section, after `surface_id_` (around line 116), add:

```cpp
    std::uint16_t position_id_ = 0;
```

- [ ] **Step 2: Verify existing tests build**

Run: `ctest --preset default -R "embedding"` (or check that no existing tests break)

- [ ] **Step 3: Commit**

```bash
git add include/sai/embedding/embedding.h
git commit -m "feat(embedding): ✨ Embedding 添加 position_id_ 字段与 getter/setter"
```

---

### Task 2: DetectionResult + position_id field

**Files:**
- Modify: `include/sai/detection/detection_result.h`

**Interfaces:**
- Produces: `DetectionResult::position_id` field

- [ ] **Step 1: Add position_id to DetectionResult**

In `include/sai/detection/detection_result.h`, after the existing `surface_id` field, add:

```cpp
    std::uint16_t position_id = 0;  // camera position index for multi-bank routing
```

Verify the file already has `surface_id` (should be there from commit a53f24c).

- [ ] **Step 2: Commit**

```bash
git add include/sai/detection/detection_result.h
git commit -m "feat(detection): ✨ DetectionResult 添加 position_id 字段"
```

---

### Task 3: CoresetManifest header + implementation

**Files:**
- Create: `include/sai/io/coreset_manifest.h`
- Create: `src/io/coreset_manifest.cpp`

**Interfaces:**
- Produces: `sai::io::CoresetBankEntry`, `sai::io::CoresetManifest`, `LoadCoresetManifest()`, `SaveCoresetManifest()`

- [ ] **Step 1: Write coreset_manifest.h**

```cpp
// coreset_manifest.h — Coreset file registry for multi-position detection
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <sai/core/error.h>

namespace sai::io {

struct CoresetBankEntry {
    std::uint16_t position_id = 0;
    std::filesystem::path path;  // relative to manifest file's directory
};

struct CoresetManifest {
    std::string surface_id;
    std::vector<CoresetBankEntry> banks;
};

// Load from YAML file. All bank paths are resolved relative to the
// manifest file's parent directory.
[[nodiscard]] auto LoadCoresetManifest(const std::filesystem::path& yaml_path) noexcept
    -> Result<CoresetManifest>;

// Write manifest to YAML file.
[[nodiscard]] auto SaveCoresetManifest(const std::filesystem::path& yaml_path,
                                        const CoresetManifest& manifest) noexcept
    -> Result<void>;

}  // namespace sai::io
```

- [ ] **Step 2: Write coreset_manifest.cpp**

Implement `LoadCoresetManifest`: parse YAML with `surface` (string) + `banks` (sequence of `{position, path}`). Resolve paths relative to `yaml_path.parent_path()`.

Implement `SaveCoresetManifest`: emit YAML with the same structure.

Error handling: return `Io_ImportFileNotFound` for missing file, `Io_ImportParseFailed` for malformed YAML.

- [ ] **Step 3: Update CMakeLists.txt**

In `src/io/CMakeLists.txt`, add `coreset_manifest.cpp` to the `sai_io` library sources.

- [ ] **Step 4: Commit**

```bash
git add include/sai/io/coreset_manifest.h src/io/coreset_manifest.cpp src/io/CMakeLists.txt
git commit -m "feat(io): ✨ CoresetManifest YAML 加载/保存"
```

---

### Task 4: CLI --coreset-manifest support

**Files:**
- Modify: `apps/seat-aoi/cli_args.h`
- Modify: `apps/seat-aoi/cli_args.cpp`

**Interfaces:**
- Produces: `CliArgs::coreset_manifest_path` field, `--coreset-manifest` CLI flag

- [ ] **Step 1: Add field to CliArgs**

In `cli_args.h`, after `coreset_path`, add:

```cpp
    std::string coreset_manifest_path;  // --coreset-manifest: multi-position YAML registry
```

- [ ] **Step 2: Add parsing to ParseArgs**

In `cli_args.cpp`, in the argument parsing loop, add:

```cpp
    } else if (arg == "--coreset-manifest" && i + 1 < argc) {
        args.coreset_manifest_path = argv[++i];
```

- [ ] **Step 3: Commit**

```bash
git add apps/seat-aoi/cli_args.h apps/seat-aoi/cli_args.cpp
git commit -m "feat(seat_aoi): ✨ CLI 添加 --coreset-manifest 参数"
```

---

### Task 5: InferenceStage copies position_id to Embedding

**Files:**
- Modify: `src/pipeline/inference_stage.cpp:52-56`

**Interfaces:**
- Consumes: `ImageMeta::position_id` (existing), `Embedding::SetPositionId()` (from Task 1)
- Produces: position_id flows from ImageMeta → Embedding

- [ ] **Step 1: Add position_id copy in InferenceStage::Process**

In `src/pipeline/inference_stage.cpp`, immediately after line 56 (`embedding.SetSurfaceId(img_meta.surface_id);`), add:

```cpp
        // Carry position identity from image metadata through the pipeline.
        if (img_meta.position_id != 0) {
            embedding.SetPositionId(img_meta.position_id);
        }
```

This mirrors the existing `surface_id` copy pattern (L52-56) exactly. The `!= 0` check avoids setting position_id when it's the default (0), keeping single-position behavior unchanged.

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/inference_stage.cpp
git commit -m "feat(pipeline): ✨ InferenceStage 传递 position_id 到 Embedding"
```

```bash
git add src/pipeline/inference_stage.cpp
git commit -m "feat(pipeline): ✨ InferenceStage 传递 position_id 到 Embedding"
```

---

### Task 6: DetectStage multi-bank routing

**Files:**
- Modify: `src/pipeline/stage_nodes.h` (DetectStage class)
- Modify: `src/pipeline/detect_stage.cpp`

**Interfaces:**
- Consumes: `Embedding::PositionId()` (Task 1), `DetectionResult::position_id` (Task 2)
- Produces: `DetectStage::AddDetector(sid, pid, det)`, `DetectStage::GetDetector(sid, pid)`
- Compat: `SetDetector(det)` delegates to `AddDetector("", 0, det)`

- [ ] **Step 1: Update stage_nodes.h DetectStage**

Replace the single detector member with a map. In `stage_nodes.h`, change DetectStage:

```cpp
class DetectStage final : public IStageNode {
public:
    // ... existing methods ...

    // NEW: register a detector for a specific (surface_id, position_id) pair
    auto AddDetector(std::string surface_id, std::uint16_t position_id,
                     std::shared_ptr<sai::detection::IDetector> det) -> void;

    // COMPAT: single-detector convenience (position=0, any surface)
    auto SetDetector(std::shared_ptr<sai::detection::IDetector> det) -> void {
        AddDetector("", 0, std::move(det));
    }

    // Access for evolution wiring
    [[nodiscard]] auto GetDetector(std::string_view surface_id,
                                    std::uint16_t position_id) const
        -> std::shared_ptr<sai::detection::IDetector>;

private:
    std::string id_;
    using BankKey = std::pair<std::string, std::uint16_t>;
    struct BankKeyCompare {
        // Use transparent comparator for string_view lookup
        using is_transparent = std::true_type;
        bool operator()(const BankKey& a, const BankKey& b) const { return a < b; }
    };
    std::map<BankKey, std::shared_ptr<sai::detection::IDetector>, BankKeyCompare> detectors_;
    std::shared_ptr<sai::detection::IDetector> default_detector_;  // fallback
    bool stub_ = true;
};
```

- [ ] **Step 2: Update detect_stage.cpp Process()**

In `detect_stage.cpp`, rewrite `Process()` to look up the detector by `(surface_id, position_id)`:

```cpp
auto DetectStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* emb = std::get_if<sai::embedding::Embedding>(&input)) {
        sai::detection::DetectionResult result;
        
        // Select detector by (surface_id, position_id)
        BankKey key{emb->SurfaceId(), emb->PositionId()};
        auto it = detectors_.find(key);
        auto* detector = (it != detectors_.end()) ? it->second.get() : default_detector_.get();
        
        if (!stub_ && detector) {
            auto det_result = detector->Detect(*emb);
            if (det_result) result = std::move(*det_result);
        }
        // Carry forward CLIP global features.
        if (emb->HasGlobalFeatures()) {
            result.global_features = emb->GlobalFeatures();
        }
        // Carry forward surface and position identity.
        if (!emb->SurfaceId().empty()) {
            result.surface_id = emb->SurfaceId();
        }
        result.position_id = emb->PositionId();
        return StageOutput(std::move(result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects Embedding input"});
}
```

Add the `AddDetector` and `GetDetector` method implementations to the .cpp file.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stage_nodes.h src/pipeline/detect_stage.cpp
git commit -m "feat(pipeline): ✨ DetectStage 多机位 bank 路由 (AddDetector/map)"
```

---

### Task 7: BuildCoreset group-by-position

**Files:**
- Modify: `apps/seat-aoi/coreset_builder.cpp` (add `#include "app_config.h"` if not already, plus `#include <sai/io/coreset_manifest.h>`)

**Interfaces:**
- Consumes: `CoresetManifest`, `SaveCoresetManifest` (Task 3), `CliArgs` (Task 4)
- Produces: multiple `pos_N.bin` files + one manifest YAML per surface

- [ ] **Step 1: Rewrite BuildCoreset to group by position**

In `coreset_builder.cpp`, after loading `entries` via `ImportDataset`:

1. Group entries by `position_id` into `map<uint16_t, vector<DatasetEntry>>`
2. For each position group:
   - Extract embeddings (reuse existing DINOv3 pipeline)
   - Build FeatureBank with `BuildWithGreedyCoreset`
   - Save to `{output_dir}/pos_{pid}.bin`
3. Build `CoresetManifest` with all positions
4. Save manifest to `{output_dir}/{surface_id}.yaml`
5. Print per-position summary

Keep the GPU upload path, preprocess chain, and error handling unchanged. Single-position datasets (all entries have position_id=0) still produce `pos_0.bin` + manifest with one entry.

- [ ] **Step 2: Commit**

```bash
git add apps/seat-aoi/coreset_builder.cpp
git commit -m "feat(seat_aoi): ✨ BuildCoreset 按 position 分组构建多 coreset + manifest"
```

---

### Task 8: AppBuilder multi-bank assembly + multi-evolution

**Files:**
- Modify: `apps/seat-aoi/app_builder.h` (AssembledApp struct)
- Modify: `apps/seat-aoi/app_builder.cpp` (AssembleApplication)

**Interfaces:**
- Consumes: `CoresetManifest` (Task 3), `CliArgs::coreset_manifest_path` (Task 4), `DetectStage::AddDetector` (Task 6)
- Produces: `AssembledApp::patch_cores`, `AssembledApp::evolutions`, `AssembledApp::evolution_stop_sources`

- [ ] **Step 1: Update AssembledApp struct**

In `app_builder.h`, add multi-position fields. After the existing single-position fields:

```cpp
    // ── Multi-position detectors ──
    std::map<std::pair<std::string, std::uint16_t>,
             std::shared_ptr<sai::detection::PatchCore>> patch_cores;

    // ── Per-position evolutions (declared AFTER patch_cores for correct destruction order)
    std::map<std::pair<std::string, std::uint16_t>,
             sai::detection::CoresetEvolution> evolutions;
    std::map<std::pair<std::string, std::uint16_t>,
             std::stop_source> evolution_stop_sources;
```

Also add `#include <map>` to the header.

- [ ] **Step 2: Update AssembleApplication for multi-bank loading**

In `app_builder.cpp`, after loading `feature_bank` / `vp` for single-position (keep existing compat path):

Add a new block when `!cli.coreset_manifest_path.empty()`:
1. `LoadCoresetManifest(cli.coreset_manifest_path)` → `CoresetManifest`
2. For each bank entry:
   - `FeatureBank::LoadFromFile(resolved_path, kEmbedDim)` → `FeatureBank`
   - Create `PatchCore` with config, `SetFeatureBank`
   - Optionally create `CoresetEvolution` if self_evolution enabled
   - Store in `patch_cores`, `evolutions`, `evolution_stop_sources`
3. Wire: for each `(sid, pid)` → `detect_stage->AddDetector(sid, pid, patch_core)`

- [ ] **Step 3: Wire result callback for multi-evolution**

Update the result callback set in `AssembleApplication` (before `Pipeline::Start()`) to handle both single-position and multi-position cases:

```cpp
if (!evolutions.empty()) {
    pipeline->SetResultCallback(
        [this, patch_cores = &patch_cores, evolutions = &evolutions]
        (int fid, const reasoner::ReasoningResult& result) {
            // Caller (gui_runner) will override this; headless uses it.
            // Look up per-position evolution and assess.
        });
}
```

- [ ] **Step 4: Commit**

```bash
git add apps/seat-aoi/app_builder.h apps/seat-aoi/app_builder.cpp
git commit -m "feat(seat_aoi): ✨ AppBuilder 多机位 bank 装配 + 多进化实例"
```

---

### Task 9: HeadlessRunner + GuiRunner per-position evolution

**Files:**
- Modify: `apps/seat-aoi/headless_runner.cpp`
- Modify: `apps/seat-aoi/gui_runner.cpp`

**Interfaces:**
- Consumes: `AssembledApp::evolutions`, `AssembledApp::evolution_stop_sources` (Task 8)

- [ ] **Step 1: Update headless_runner.cpp cleanup**

In `RunHeadless`, update the cleanup section. Replace:
```cpp
if (app.evolution.has_value()) app.evolution->Stop();
```
with:
```cpp
for (auto& [key, evo] : app.evolutions) {
    evo.Stop();
}
```

Same for the `RunGui` `aboutToQuit` cleanup.

- [ ] **Step 2: Update gui_runner.cpp result callback**

In `RunGui`, the evolution-aware result callback branch. Replace the single `app.evolution->AssessAndOffer(...)` with a lookup:
```cpp
BankKey key{result.surface_id, result.position_id};
auto it = app.evolutions.find(key);
if (it != app.evolutions.end() && it->second.IsRunning()) {
    it->second.AssessAndOffer(...);
}
```

- [ ] **Step 3: Commit**

```bash
git add apps/seat-aoi/headless_runner.cpp apps/seat-aoi/gui_runner.cpp
git commit -m "feat(seat_aoi): ✨ Headless/Gui 多机位进化 cleanup + callback 路由"
```

---

### Task 10: Final integration verification

- [ ] **Step 1: Review all changed files for consistency**

Check each modified file in order:
- `embedding.h` — PositionId/SetPositionId present ✅
- `detection_result.h` — position_id field present ✅
- `coreset_manifest.h/cpp` — Load/Save correct ✅
- `cli_args.h/cpp` — coreset_manifest_path present ✅
- `inference_stage.cpp` — position copy present ✅
- `stage_nodes.h` — AddDetector/GetDetector present ✅
- `detect_stage.cpp` — routing logic present ✅
- `coreset_builder.cpp` — group-by-position present ✅
- `app_builder.h/cpp` — multi-bank assembly present ✅
- `headless_runner.cpp` — multi-evolution cleanup present ✅
- `gui_runner.cpp` — multi-evolution callback present ✅

- [ ] **Step 2: Verify backward compat**

Check that `SetDetector(det)` still works (delegates to `AddDetector("", 0, det)`), `--coreset` CLI still works, and all existing tests pass.

- [ ] **Step 3: Commit any final fixes**

---
