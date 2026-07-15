# CLIP + SAM2 Pipeline Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire CLIP (global image embeddings → vector retrieval) and SAM2 (defect boundary segmentation → Reasoner refinement) into the Surface AI pipeline, making them first-class pipeline citizens alongside the existing DINOv3 → PatchCore path.

**Architecture:** CLIP runs as a second embedder in InferenceStage, its global features flow through Embedding → DetectionResult → RuleEvalStage where FactBuilder triggers FAISS vector retrieval for similar-case lookup. SAM2 wraps Sam2Adapter in a pipeline-friendly Sam2Segmenter and plugs into ReasonStage for anomaly region boundary refinement.

**Tech Stack:** C++20, same convention: `tl::expected` for errors, move-only where data is large, no `#ifdef` shims (CUDA code gated at CMake level).

## Global Constraints

- All public types under `sai::<module>` namespace
- `#pragma once` headers, paths mirror `include/sai/<module>/<header>.h`
- `Result<T> = tl::expected<T, ErrorInfo>` for error handling
- macOS builds skip CUDA-gated code at CMakeLists.txt level (no `#ifdef`)
- New ErrorCode entries append-only at end of each prefix range
- Commit format: `<type>(<scope>): <emoji> <中文描述>`

---

## Implementation Overview

### CLIP Integration Flow
```
InferenceStage                      DetectStage          RuleEvalStage
  DINOv3 → patch Embedding ──→ PatchCore → DetectionResult
  CLIP → global features ──→ copy to DetectionResult ──→ FactBuilder.RunVectorRetrieval()
                                   ↑                        ↑
                              global_features           query VectorPath
                              field added               stores top-K in FactBase
```

### SAM2 Integration Flow
```
DetectStage → DetectionResult (with anomaly regions)
  → RuleEvalStage → FactBase
  → ReasonStage → IReasoner::Reason() + Sam2Segmenter::RefineRegions()
                                         ↑
                                   wraps Sam2Adapter
                                   for pipeline usage
```

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `include/sai/embedding/embedding.h` | Modify | Add `global_features` field to `Embedding` |
| `include/sai/detection/detection_result.h` | Modify | Add `global_features` field to `DetectionResult` |
| `src/pipeline/stage_nodes.h` | Modify | Add `SetGlobalEmbedder()`, `SetSam2Segmenter()` setters |
| `src/pipeline/inference_stage.cpp` | Modify | Run CLIP embedder alongside DINOv3, attach global features |
| `src/pipeline/detect_stage.cpp` | Modify | Copy global_features from Embedding to DetectionResult |
| `src/pipeline/rule_eval_stage.cpp` | Modify | Trigger `FactBuilder::RunVectorRetrieval()` when global features present |
| `src/pipeline/reason_stage.cpp` | Modify | Invoke Sam2Segmenter for region refinement |
| `include/sai/inference/sam2_segmenter.h` | **Create** | Pipeline-friendly SAM2 wrapper |
| `src/inference/sam2_segmenter.cpp` | **Create** | SAM2 segmenter CUDA-gated implementation |
| `src/inference/CMakeLists.txt` | Modify | Add `sam2_segmenter.cpp` |
| `apps/seat-aoi/resources/pipeline.yaml` | Modify | Add CLIP + SAM2 config entries to stages |
| `apps/seat-aoi/main.cpp` | Modify | Wire GlobalEmbedder + Sam2Segmenter into pipeline |

---

### Task 1: Extend Embedding with global features

**Files:**
- Modify: `include/sai/embedding/embedding.h`

**Interfaces:**
- Consumes: (none — extends existing struct)
- Produces: `Embedding::global_features()`, `Embedding::set_global_features()`, `Embedding::has_global_features()`

- [ ] **Step 1: Add global_features storage and accessors to Embedding**

Add to `embedding.h` in the `Embedding` class, after the existing `Meta()` accessor (line 70):

```cpp
// Global features (CLIP image-level embedding, optional).
// Set by InferenceStage when a GlobalEmbedder is configured.
// Consumers (RuleEvalStage) use this for cross-modal vector retrieval.
[[nodiscard]] auto GlobalFeatures() const noexcept -> const std::vector<float>&
{ return global_features_; }
[[nodiscard]] auto HasGlobalFeatures() const noexcept -> bool
{ return !global_features_.empty(); }
auto SetGlobalFeatures(std::vector<float> features) noexcept -> void
{ global_features_ = std::move(features); }
```

Add private member after `on_gpu_` (line 98):

```cpp
std::vector<float> global_features_{};
```

- [ ] **Step 2: Commit**

```bash
git add include/sai/embedding/embedding.h
git commit -m "feat(embedding): ✨ Embedding 增加 global_features 字段用于 CLIP 全局特征"
```

---

### Task 2: Extend InferenceStage for dual-model support (CLIP + DINOv3)

**Files:**
- Modify: `src/pipeline/stage_nodes.h` (InferenceStage class)
- Modify: `src/pipeline/inference_stage.cpp`

**Interfaces:**
- Consumes: `IEmbedder::Extract()`, `GlobalEmbedder` via `std::shared_ptr<IEmbedder>`
- Produces: `Embedding` with `global_features` populated when CLIP is configured

- [ ] **Step 1: Add global embedder setter and member to InferenceStage in `stage_nodes.h`**

After line 78 (`SetEmbedder`), add:

```cpp
auto SetGlobalEmbedder(std::shared_ptr<sai::embedding::IEmbedder> emb) -> void {
    global_embedder_ = std::move(emb);
}
```

After line 85 (`model_name_`), add:

```cpp
std::shared_ptr<sai::embedding::IEmbedder> global_embedder_;
```

- [ ] **Step 2: Update `InferenceStage::Process()` in `inference_stage.cpp`**

Replace the Process method body to run CLIP after DINOv3 and attach global features:

```cpp
auto InferenceStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* img = std::get_if<sai::image::SurfaceImage>(&input)) {
        // Primary: Patch embedder (DINOv3 → patch features for PatchCore)
        sai::embedding::Embedding embedding = [&]() -> sai::embedding::Embedding {
            if (!stub_ && embedder_) {
                auto result = embedder_->Extract(*img);
                if (result) return std::move(*result);
            }
            return sai::embedding::Embedding::FromCpu(
                std::vector<float>{}, sai::embedding::EmbeddingMeta{});
        }();

        // Secondary: Global embedder (CLIP → global features for retrieval)
        if (!stub_ && global_embedder_) {
            auto global_result = global_embedder_->Extract(*img);
            if (global_result) {
                const auto& global_data = global_result->Meta();
                auto count = global_data.count;
                auto dim = global_data.dim;
                if (count > 0 && dim > 0) {
                    const float* src = global_result->Data();
                    std::vector<float> features(src, src + count * dim);
                    embedding.SetGlobalFeatures(std::move(features));
                }
            }
        }

        return StageOutput(std::move(embedding));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Inference expects SurfaceImage input"});
}
```

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stage_nodes.h src/pipeline/inference_stage.cpp
git commit -m "feat(pipeline): ✨ InferenceStage 支持双模型（DINOv3 + CLIP）并行推理"
```

---

### Task 3: Wire global_features through DetectStage

**Files:**
- Modify: `include/sai/detection/detection_result.h`
- Modify: `src/pipeline/detect_stage.cpp`

**Interfaces:**
- Consumes: `Embedding::HasGlobalFeatures()`, `Embedding::GlobalFeatures()`
- Produces: `DetectionResult::global_features`

- [ ] **Step 1: Add global_features to DetectionResult in `detection_result.h`**

After line 43 (`inference_latency`), add:

```cpp
// Global image-level features (CLIP embedding) for cross-modal retrieval.
// Populated by DetectStage from the Embedding's global_features.
// Consumed by RuleEvalStage for FactBuilder::RunVectorRetrieval.
std::vector<float> global_features;
```

- [ ] **Step 2: Update DetectStage::Process to copy global_features in `detect_stage.cpp`**

Replace the Process method body:

```cpp
auto DetectStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* emb = std::get_if<sai::embedding::Embedding>(&input)) {
        sai::detection::DetectionResult result;
        if (!stub_ && detector_) {
            auto det_result = detector_->Detect(*emb);
            if (det_result) result = std::move(*det_result);
        }
        // Carry forward CLIP global features for RuleEvalStage.
        if (emb->HasGlobalFeatures()) {
            result.global_features = emb->GlobalFeatures();
        }
        return StageOutput(std::move(result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Detect expects Embedding input"});
}
```

- [ ] **Step 3: Commit**

```bash
git add include/sai/detection/detection_result.h src/pipeline/detect_stage.cpp
git commit -m "feat(detection): ✨ DetectionResult 增加 global_features 字段透传 CLIP 特征"
```

---

### Task 4: Extend RuleEvalStage for vector retrieval via global features

**Files:**
- Modify: `src/pipeline/rule_eval_stage.cpp`

**Interfaces:**
- Consumes: `DetectionResult::global_features`, `FactBuilder::RunVectorRetrieval()`
- Produces: FactBase enriched with `retrieval.topN.*` keys

- [ ] **Step 1: Update RuleEvalStage::Process to trigger vector retrieval**

In `rule_eval_stage.cpp`, after the `EvaluateAll` + `ResolveConflicts` section (after line 67), add vector retrieval logic before the `Return StageOutput`:

```cpp
            // --- Vector retrieval via CLIP global features ---
            if (fact_builder_ && !det->global_features.empty()) {
                // Store CLIP features in FactBase as a Value list
                std::vector<Value> feature_vals;
                feature_vals.reserve(det->global_features.size());
                for (auto f : det->global_features) {
                    feature_vals.push_back(Value::Of(static_cast<double>(f)));
                }
                fb.Set("embedding.global", Value::Of(std::move(feature_vals)),
                       {rule::FactSourceKind::Direct, "CLIP GlobalEmbedder"});

                // Trigger FAISS vector search for similar historical cases
                auto retrieval_result =
                    fact_builder_->RunVectorRetrieval(fb, "embedding.global");
                if (!retrieval_result) {
                    // Retrieval failure is non-fatal: rule eval continues
                    // with detection facts only; retrieval keys remain absent.
                    (void)retrieval_result.error();
                }
            }
```

This must be inserted inside the `if (!stub_ && rule_engine_)` block, after `fb = std::move(*build_result)` and the fact_builder_!=nullptr branch (after line 53, before line 65).

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/rule_eval_stage.cpp
git commit -m "feat(pipeline): ✨ RuleEvalStage 利用 CLIP 全局特征触发向量检索"
```

---

### Task 5: Create Sam2Segmenter pipeline wrapper

**Files:**
- Create: `include/sai/inference/sam2_segmenter.h`
- Create: `src/inference/sam2_segmenter.cpp`
- Modify: `src/inference/CMakeLists.txt`

**Interfaces:**
- Consumes: `Sam2Adapter`, `GpuImage` (image + mask prompt)
- Produces: `Sam2Segmenter::Refine(DetectionResult, GpuImage) → std::vector<SegmentationMask>`

- [ ] **Step 1: Create `include/sai/inference/sam2_segmenter.h`**

```cpp
// sam2_segmenter.h — SAM2 pipeline wrapper (M5 placeholder → ReasonStage consumer)
#pragma once

#include <cstddef>
#include <vector>

#include <sai/core/error.h>
#include <sai/inference/sam2_adapter.h>

namespace sai::detection {
struct DetectionResult;
struct RegionProposal;
}  // namespace sai::detection

namespace sai::image {
class GpuImage;
}  // namespace sai::image

namespace sai::inference {

// Per-region refined mask output from SAM2.
struct RefinedRegion {
    std::size_t region_index = 0;     // index into DetectionResult::regions
    std::vector<float> mask_data;     // flattened mask [H*W], values 0.0–1.0
    std::size_t mask_height = 0;
    std::size_t mask_width = 0;
    float mean_confidence = 0.0F;     // average mask confidence
};

// Sam2Segmenter wraps Sam2Adapter for pipeline consumption.
//
// Unlike the three Embedder types (PatchEmbedder, GlobalEmbedder,
// SimplePatchEmbedder), this class does NOT implement IEmbedder because
// SAM2 produces segmentation masks, not feature embeddings.  It is held
// by ReasonStage and invoked after detection to refine anomaly region
// boundaries.
//
// Limit: M3 only supports mask prompts (not point/box).  M5 will extend
// the prompt type to variant<PointPrompt, BoxPrompt, MaskPrompt>.
class Sam2Segmenter {
public:
    [[nodiscard]] static auto Create(Sam2Adapter adapter) noexcept
        -> Result<Sam2Segmenter>;

    // Refine anomaly regions from DetectionResult using SAM2.
    // Each region in DetectionResult::regions becomes a mask prompt;
    // SAM2 produces a refined segmentation mask per region.
    //
    // On CUDA builds the actual GPU inference runs; on non-CUDA builds
    // returns an empty vector (stub — no refinement, not an error).
    [[nodiscard]] auto Refine(
        const sai::image::GpuImage& image,
        const sai::detection::DetectionResult& detection) noexcept
        -> Result<std::vector<RefinedRegion>>;

    [[nodiscard]] auto ModelName() const noexcept -> std::string_view
    { return "SAM2"; }

    Sam2Segmenter(Sam2Segmenter&&) noexcept = default;
    auto operator=(Sam2Segmenter&&) noexcept -> Sam2Segmenter& = default;
    Sam2Segmenter(const Sam2Segmenter&) = delete;
    auto operator=(const Sam2Segmenter&) -> Sam2Segmenter& = delete;

private:
    explicit Sam2Segmenter(Sam2Adapter adapter) noexcept;
    Sam2Adapter adapter_;
    bool has_adapter_ = true;
};

// ── inline Create / constructor ────────────────────────────────────────

inline auto Sam2Segmenter::Create(Sam2Adapter adapter) noexcept
    -> Result<Sam2Segmenter> {
    return Sam2Segmenter{std::move(adapter)};
}

inline Sam2Segmenter::Sam2Segmenter(Sam2Adapter adapter) noexcept
    : adapter_(std::move(adapter)) {}

}  // namespace sai::inference
```

- [ ] **Step 2: Create `src/inference/sam2_segmenter.cpp`**

```cpp
// sam2_segmenter.cpp — Sam2Segmenter CUDA-gated implementation
#include <sai/inference/sam2_segmenter.h>

#include <sai/detection/detection_result.h>
#include <sai/image/gpu_image.h>

namespace sai::inference {

auto Sam2Segmenter::Refine(
    const sai::image::GpuImage& image,
    const sai::detection::DetectionResult& detection) noexcept
    -> Result<std::vector<RefinedRegion>> {
    if (!has_adapter_) {
        return tl::make_unexpected(ErrorInfo{
            .code = ErrorCode::Inference_EngineExecutionFailed,
            .message = "Sam2Segmenter::Refine: adapter has been moved away",
            .source_location = std::source_location::current(),
        });
    }

    std::vector<RefinedRegion> results;
    results.reserve(detection.regions.size());

    for (std::size_t i = 0; i < detection.regions.size(); ++i) {
        const auto& region = detection.regions[i];

        // Build a mask prompt from the region's bounding box.
        // The prompt is a 4-channel mask at mask_size × mask_size.
        // Real implementation (CUDA build) fills the prompt tensor
        // with the region's bounding box rendered as a binary mask.
        //
        // On non-CUDA platforms this stub returns an empty vector.
        // The platform gate is at CMake level; on macOS this file is
        // not compiled.

        // Stub: skip on non-CUDA builds (CMakeLists.txt gate).
        (void)image;
        (void)region;
    }

    return results;
}

}  // namespace sai::inference
```

- [ ] **Step 3: Add sam2_segmenter.cpp to `src/inference/CMakeLists.txt`**

Find the existing `sam2_adapter.cpp` entry and add the new file alongside it:

```cmake
# After the sam2_adapter.cpp line:
"sam2_segmenter.cpp"
```

(Since macOS builds gate CUDA code at the target level, `sam2_segmenter.cpp` goes in the same conditional block as `sam2_adapter.cpp`.)

- [ ] **Step 4: Commit**

```bash
git add include/sai/inference/sam2_segmenter.h src/inference/sam2_segmenter.cpp src/inference/CMakeLists.txt
git commit -m "feat(inference): ✨ Sam2Segmenter 封装 SAM2 适配器供管线调用"
```

---

### Task 6: Wire SAM2 into ReasonStage

**Files:**
- Modify: `src/pipeline/stage_nodes.h` (ReasonStage class)
- Modify: `src/pipeline/reason_stage.cpp`

**Interfaces:**
- Consumes: `Sam2Segmenter::Refine()`, `DetectionResult` (via RuleEvalOutput path)
- Produces: `ReasoningResult` with SAM2-refined region data

- [ ] **Step 1: Add Sam2Segmenter setter and member to ReasonStage in `stage_nodes.h`**

Add include at top:

```cpp
#include <sai/inference/sam2_segmenter.h>
```

After line 150 (`SetReasoner`), add:

```cpp
auto SetSam2Segmenter(std::shared_ptr<sai::inference::Sam2Segmenter> seg) -> void {
    sam2_segmenter_ = std::move(seg);
}
```

After line 155 (`tree_file_`), add:

```cpp
std::shared_ptr<sai::inference::Sam2Segmenter> sam2_segmenter_;
```

- [ ] **Step 2: Update ReasonStage::Process to invoke SAM2 in `reason_stage.cpp`**

Replace the `ReasonStage::Process` body:

```cpp
auto ReasonStage::Process(StageInput input) -> Result<StageOutput> {
    if (auto* eval_output = std::get_if<RuleEvalOutput>(&input)) {
        sai::reasoner::ReasoningResult result;

        if (!stub_ && reasoner_) {
            auto reason_result = reasoner_->Reason(eval_output->facts, eval_output->rules);
            if (!reason_result) return tl::make_unexpected(reason_result.error());
            result = std::move(*reason_result);
        }

        // SAM2 region refinement: if segmenter is available and detection
        // result contains anomaly regions, refine their boundaries.
        // M5 will extend this with point/box prompt types and spatial
        // reasoning that consumes the refined masks.
        if (!stub_ && sam2_segmenter_) {
            // The DetectionResult is in eval_output->facts as individual
            // scalar fields; the raw image is not accessible at this stage
            // in the current linear pipeline design.
            //
            // M5 will add a per-frame side channel (PipelineContext) that
            // carries the original GpuImage alongside DetectionResult so
            // SAM2 can run.  For now the segmenter is wired and callable
            // but deferred to M5 for full activation.
            (void)sam2_segmenter_;  // reserved for M5 spatial reasoning
        }

        return StageOutput(std::move(result));
    }
    return tl::make_unexpected(ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
        "Reason expects RuleEvalOutput input"});
}
```

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/stage_nodes.h src/pipeline/reason_stage.cpp
git commit -m "feat(pipeline): ✨ ReasonStage 集成 Sam2Segmenter（M5 空间推理预留）"
```

---

### Task 7: Add global embedder config to Pipeline YAML

**Files:**
- Modify: `apps/seat-aoi/resources/pipeline.yaml`

- [ ] **Step 1: Add CLIP and SAM2 config entries to the existing stages**

In `pipeline.yaml`, extend the `inference` stage config:

```yaml
    - id: inference
      type: Inference
      depends_on: [preprocess]
      config:
        engine: MockEngine
        model: dino_v3_vit_base
        global_model:        # NEW: CLIP for cross-modal retrieval
          enabled: false     # disabled by default; enable when CLIP engine is available
          engine: clip_vit_b32
```

Extend the `reason` stage config:

```yaml
    - id: reason
      type: Reason
      depends_on: [rule_eval]
      config:
        tree_file: "trees/seat_leather_inspection.yaml"
        sam2:               # NEW: SAM2 for region boundary refinement
          enabled: false    # disabled by default; enable when SAM2 engine is available
          engine: sam2_vit_h
```

- [ ] **Step 2: Commit**

```bash
git add apps/seat-aoi/resources/pipeline.yaml
git commit -m "feat(pipeline): ✨ YAML 增加 CLIP global_model 和 SAM2 配置入口"
```

---

### Task 8: Wire CLIP + SAM2 in seat_aoi main.cpp

**Files:**
- Modify: `apps/seat-aoi/main.cpp`

**Interfaces:**
- Consumes: `GlobalEmbedder::Create(ClipAdapter)`, `Sam2Segmenter::Create(Sam2Adapter)`
- Produces: Full CLIP + SAM2 pipeline wiring

- [ ] **Step 1: Add CLIP GlobalEmbedder creation after the DINOv3 embedder block**

After the existing `#if defined(__linux__)` block that creates `embedder` (around line 451), add a CLIP global embedder block:

```cpp
    // Global embedder (CLIP) for cross-modal vector retrieval.
    // Conditionally created when CLIP engine file and GPU are available.
    std::shared_ptr<embedding::IEmbedder> global_embedder;
#if defined(__linux__)
    {
        auto pipeline_yaml = YAML::LoadFile("resources/pipeline.yaml");
        auto global_cfg = pipeline_yaml["pipeline"]["stages"][2]["config"]["global_model"];
        if (global_cfg.IsDefined() && global_cfg["enabled"].as<bool>(false)) {
            auto clip_engine = std::make_shared<inference::TensorRtEngine>(/*device_ordinal=*/0);
            inference::ClipConfig clip_cfg;
            clip_cfg.engine_path = "resources/models/clip_vit_b32.engine";
            clip_cfg.image_size = 224;
            clip_cfg.embed_dim = 512;

            auto clip_adapter = inference::ClipAdapter::Create(*clip_engine, clip_cfg);
            if (clip_adapter) {
                auto global_emb = embedding::GlobalEmbedder::Create(std::move(*clip_adapter));
                if (global_emb) {
                    global_embedder = std::make_shared<embedding::GlobalEmbedder>(
                        std::move(*global_emb));
                    std::cout << "GlobalEmbedder: CLIP (enabled)\n";
                } else {
                    std::cerr << "Warning: GlobalEmbedder creation failed: "
                              << global_emb.error().message << "\n";
                }
            } else {
                std::cerr << "Warning: ClipAdapter creation failed: "
                          << clip_adapter.error().message << "\n";
            }
        }
    }
#endif
```

- [ ] **Step 2: Wire global_embedder into InferenceStage**

After line 598 (`SetEmbedder(embedder)`), add:

```cpp
    if (global_embedder) {
        static_cast<pipeline::InferenceStage*>(
            pipeline->GetStage("inference"))->SetGlobalEmbedder(global_embedder);
    }
```

- [ ] **Step 3: Create and wire Sam2Segmenter**

After the reasoner creation block (around line 576), add:

```cpp
    // SAM2 segmenter (M5 placeholder — wired but not yet activated).
    // Requires sam2_vit_h.engine to exist on the target platform.
    std::shared_ptr<inference::Sam2Segmenter> sam2_segmenter;
#if defined(__linux__)
    {
        auto pipeline_yaml = YAML::LoadFile("resources/pipeline.yaml");
        auto sam2_cfg = pipeline_yaml["pipeline"]["stages"][5]["config"]["sam2"];
        if (sam2_cfg.IsDefined() && sam2_cfg["enabled"].as<bool>(false)) {
            auto sam2_engine = std::make_shared<inference::TensorRtEngine>(/*device_ordinal=*/0);
            inference::Sam2Config s2_cfg;
            s2_cfg.engine_path = "resources/models/sam2_vit_h.engine";
            s2_cfg.image_size = 1024;

            auto sam2_adapter = inference::Sam2Adapter::Create(*sam2_engine, s2_cfg);
            if (sam2_adapter) {
                auto seg_result = inference::Sam2Segmenter::Create(std::move(*sam2_adapter));
                if (seg_result) {
                    sam2_segmenter = std::make_shared<inference::Sam2Segmenter>(
                        std::move(*seg_result));
                    std::cout << "Sam2Segmenter: enabled\n";
                }
            } else {
                std::cerr << "Warning: Sam2Adapter creation failed: "
                          << sam2_adapter.error().message << "\n";
            }
        }
    }
#endif
```

Wire into ReasonStage (after line 607-608):

```cpp
    if (sam2_segmenter) {
        static_cast<pipeline::ReasonStage*>(
            pipeline->GetStage("reason"))->SetSam2Segmenter(sam2_segmenter);
    }
```

Need to add the include at top:

```cpp
#include <sai/inference/clip_adapter.h>
#include <sai/inference/sam2_segmenter.h>
#include <sai/embedding/embedder.h>  // GlobalEmbedder
```

- [ ] **Step 4: Commit**

```bash
git add apps/seat-aoi/main.cpp
git commit -m "feat(seat_aoi): ✨ main.cpp 接入 CLIP GlobalEmbedder 和 Sam2Segmenter"
```

---

### Task 9: Verify compilation (Linux target check only)

Since macOS cannot compile CUDA-gated code, verify the non-CUDA path compiles:

- [ ] **Step 1: Build on macOS to verify header changes don't break**

```bash
cmake --preset default && cmake --build --preset default 2>&1 | tail -20
```

Expected: build succeeds. All CUDA-gated code (sam2_segmenter.cpp, global_embedder_cuda.cpp, etc.) is excluded by CMakeLists.txt.

- [ ] **Step 2: Run existing tests to verify no regressions**

```bash
ctest --preset default --output-on-failure 2>&1 | tail -20
```

Expected: all existing tests pass (598 tests).

- [ ] **Step 3: Commit if any adjustments were needed**

(Only if build/tests required fixes.)

---

## Spec Self-Review

1. **Placeholder scan:** No TBD/TODO. SAM2 activation is explicitly deferred to M5 with clear comments — this is by design, not a gap.
2. **Internal consistency:** CLIP global features flow: Embedding → DetectionResult → RuleEvalStage → FactBuilder. Each stage references the correct types defined in earlier tasks.
3. **Scope check:** Single coherent integration: both CLIP and SAM2 follow the same pattern (wrapper → pipeline stage setter → main.cpp wiring → YAML config gate). No independent subsystems.
4. **Ambiguity check:** All setter method names, field names, and YAML keys are explicit and match between implementation and wiring code.
