# Milestone 3 (AI Inference Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement milestone 3 (batches 3.1 Foundation, 3.2 Embedding, 3.3 Detector) as real C++20 code on top of the frozen M1/M2 libraries, producing `DetectionResult` from `SurfaceImage`/`GpuImage` through the inference pipeline: engine → adapter → embedder → detector.

**Architecture:** Two new libraries — `sai_inference` (STATIC: `IInferenceEngine` + `TensorRtEngine`(gated) + `MockEngine` + adapter structs), `sai_embedding` (STATIC: `Embedding` type + `IEmbedder` + `DimensionReducer` + `FeatureCache`), `sai_detection` (STATIC: `IDetector` + `PatchCore` + `DetectionResult`) — following the same portable-part vs platform-gated-part split as M1/M2. CUDA-gated code (`TensorRtEngine`, adapter `Infer()` implementations, GPU FAISS backend) is written exactly per the frozen contract and excluded from the source list when `CUDAToolkit` is absent. `MockEngine::SetOutputFillCallback` enables the full pipeline to be tested on macOS: test code injects synthetic tensor outputs, adapters and embedders consume them unaware they're not real TensorRT results, and PatchCore's k-NN (via portable faiss-cpu) processes them into real `DetectionResult`.

**Tech Stack:** C++20. vcpkg additions: `faiss` (CPU, portable; `faiss-gpu` optional, gated). Existing deps: `tl-expected`, `yaml-cpp`, `spdlog`, `nlohmann-json`, `gtest`. CUDA Runtime API + TensorRT SDK (gated). FAISS CPU backend (portable, macOS buildable).

## Global Constraints

- C++20 throughout, `sai::` namespace root. Batch sub-namespaces exactly per the frozen spec: `sai::inference` (3.1), `sai::embedding` (3.2), `sai::detection` (3.3).
- **The frozen design baseline is `docs/superpowers/specs/2026-07-12-milestone-3-inference-core-design.md`.** Every class name, method signature, enum, and namespace must match its `## Interfaces` blocks verbatim — except where a frozen signature provably cannot compile (report it, don't silently change it).
- `ErrorCode` (`include/sai/core/error.h`) is a single flat enum; each task appends new members **at the end** (append-only, never reorder or touch other batches' members). The final landed order for milestone 3, after the existing `Io_ImportParseFailed` (last M2 member), is:
  - `Inference_EngineLoadFailed`, `Inference_EngineExecutionFailed`, `Inference_InvalidBinding`, `Inference_ReloadFailed`, `Inference_ModelConfigMismatch`
  - `Embedding_NotGpuImage`, `Embedding_DimensionMismatch`
  - `Detection_FeatureBankLoadFailed`, `Detection_InvalidPatchGrid`
- Coding style: avoid over-defensive code, avoid multi-level nesting (early return), chain `Result<T>` via `and_then`/`or_else`/`map`, recursive tree/chained walks where applicable.
- **Platform gating, not conditional compilation.** Gated `.cpp` files contain exactly one target-platform implementation, no `#ifdef` alternate. Gate with `find_package(CUDAToolkit QUIET)` in the CMakeLists. Headers declaring gated classes may exist even when their `.cpp` isn't compiled, provided nothing portable ODR-uses them. `MockEngine` is the portable alternative for testing the interface flow.
- Every **portable** task ends with `cmake --preset default && cmake --build --preset default && ctest --preset default` — must not reduce the current count (**176 tests** as of M2 completion) — only add. Every **gated** task ends with a compile-review-only report + confirmation that excluding gated sources doesn't break the portable suite.
- Work on a dedicated branch `milestone-3-inference-core` off `main` (HEAD of M2).

## File Structure Overview

```
include/sai/
├── core/error.h                         # Modify: append ErrorCode members across tasks
├── inference/
│   ├── inference_engine.h               # Task 1: IInferenceEngine, TensorBinding
│   ├── mock_engine.h                    # Task 1: MockEngine + OutputFillCallback
│   ├── tensorrt_engine.h                # Task 3 (gated): TensorRtEngine declaration
│   ├── dino_v3_adapter.h                # Task 1: DinoV3Config, PatchFeatures, DinoV3Adapter (header)
│   ├── sam2_adapter.h                   # Task 1: Sam2Config, SegmentationMask, Sam2Adapter (header)
│   └── clip_adapter.h                   # Task 1: ClipConfig, GlobalFeatures, ClipAdapter (header)
├── embedding/
│   ├── embedding.h                      # Task 4: Embedding, EmbeddingMeta, EmbeddingType
│   ├── embedder.h                       # Task 5: IEmbedder, PatchEmbedder, GlobalEmbedder (headers)
│   ├── dimension_reducer.h              # Task 6: DimensionReducer, PcaParams, WhiteningParams
│   └── feature_cache.h                  # Task 6: FeatureCache
└── detection/
    ├── detection_result.h               # Task 7: DetectionResult, AnomalyMap, RegionProposal
    ├── detector.h                       # Task 7: IDetector
    ├── patch_core.h                     # Task 8: PatchCore, FeatureBank, PatchCore::Config
    └── feature_bank.h                   # Task 8: FeatureBank

src/
├── inference/
│   ├── CMakeLists.txt                   # Task 1/3
│   ├── mock_engine.cpp                  # Task 1
│   ├── tensorrt_engine.cpp              # Task 3 (gated)
│   ├── dino_v3_adapter.cpp              # Task 3 (gated)
│   ├── sam2_adapter.cpp                 # Task 3 (gated)
│   └── clip_adapter.cpp                 # Task 3 (gated)
├── embedding/
│   ├── CMakeLists.txt                   # Task 4/5/6
│   ├── embedding.cpp                    # Task 4
│   ├── patch_embedder.cpp               # Task 5 (portable header, gated Infer body split into patch_embedder_cuda.cpp)
│   ├── global_embedder.cpp              # Task 5 (same split pattern)
│   ├── dimension_reducer.cpp            # Task 6
│   └── feature_cache.cpp                # Task 6
└── detection/
    ├── CMakeLists.txt                   # Task 7/8
    ├── detection_result.cpp             # Task 7
    ├── patch_core.cpp                   # Task 8 (post-processing, portable)
    ├── feature_bank.cpp                 # Task 8 (FAISS CPU, portable)
    └── feature_bank_cuda.cpp            # Task 8 (gated: FAISS GPU)

tests/
├── inference/                           # Task 1: mock_engine_test.cpp
├── embedding/                           # Task 4: embedding_test.cpp; Task 5: embedder_test.cpp; Task 6: reducer_cache_test.cpp
├── detection/                           # Task 7: detection_types_test.cpp; Task 8: patch_core_test.cpp
└── integration/                         # Task 9: inference_pipeline_test.cpp
```

## Execution Order

```
Task 1 (Inference engine + MockEngine + adapter structs) [portable]
Task 2 (ErrorCode — Inference_*/Embedding_*/Detection_*)       [portable]
Task 3 (TensorRtEngine + adapter Infer impls)                  [gated]
Task 4 (Embedding type)                                        [portable]
Task 5 (IEmbedder + PatchEmbedder + GlobalEmbedder)            [portable headers + gated Infer split]
Task 6 (DimensionReducer + FeatureCache)                       [portable]
Task 7 (DetectionResult + IDetector + PatchCore Config)        [portable]
Task 8 (FeatureBank + PatchCore Detect)                        [portable FAISS + gated GPU FAISS]
Task 9 (Integration test)                                      [portable]
```

---


### Task 1: Inference engine interface + MockEngine + adapter structs

Batch 3.1 portable subset. Creates the `sai::inference` namespace with `IInferenceEngine`, `TensorBinding`, `MockEngine` (with test data injection), and the adapter configuration structs/types for DINOv3/SAM2/CLIP.

**Files:**
- Create: `include/sai/inference/inference_engine.h`, `mock_engine.h`, `dino_v3_adapter.h`, `sam2_adapter.h`, `clip_adapter.h`
- Create: `src/inference/CMakeLists.txt`, `src/inference/mock_engine.cpp`
- Create: `tests/inference/mock_engine_test.cpp`, `tests/inference/CMakeLists.txt`
- Modify: top-level `CMakeLists.txt` (`add_subdirectory(src/inference)`, `add_subdirectory(tests/inference)`)

**Interfaces:**
- Consumes: `sai::Object` (`object.h`), `sai::Result<T>`/`ErrorInfo` (`error.h`), `sai::image::GpuImage` (fwd-decl only), `sai::inference::IInferenceEngine` (from adapter headers)
- Produces (namespace `sai::inference`):
  - `struct TensorBinding { name, shape, size_bytes, device_ptr; }`; `class IInferenceEngine : public Object` with `Load/Infer/InferAsync/Reload/SetTensorAddress/InputBindings/OutputBindings`
  - `class MockEngine final : public IInferenceEngine` with `using OutputFillCallback = std::function<void(std::string_view,void*,std::size_t)>;` + `SetOutputFillCallback(callback)`
  - `struct DinoV3Config { engine_path, image_size, patch_size, embed_dim; }`; `struct PatchFeatures { device_ptr, grid_h, grid_w, dim; }`; `class DinoV3Adapter` with `static Create(IInferenceEngine&, DinoV3Config)->Result<DinoV3Adapter>`, `Infer(const GpuImage&)->Result<PatchFeatures>` (declared but NOT defined — gated Task 3 defines it), move-only
  - `struct Sam2Config { engine_path, image_size; }`; `struct SegmentationMask { device_ptr, height, width; }`; `class Sam2Adapter` with `static Create` + `Infer` (declared only)
  - `struct ClipConfig { engine_path, image_size, embed_dim; }`; `struct GlobalFeatures { device_ptr, dim; }`; `class ClipAdapter` with `static Create` + `Infer` (declared only)

**Adapter Pattern:** Each adapter is a concrete type (move-only, copy deleted) holding a raw `IInferenceEngine*`. The `Create` factory validates the engine's bindings against the config. `Infer` is DECLARED in the header but DEFINED only in the gated `.cpp` — portable tests use `MockEngine` with `SetOutputFillCallback` to simulate TensorRT, so `Infer()` is never ODR-used in portable tests. This split is the same pattern as M1/M2 gating: declare the method in the header (so callers can write code that compiles), but only define it in the gated `.cpp` (so the linker never tries to resolve it on macOS).

- [ ] **Step 1: Write failing tests for `MockEngine`**

Cover: `Load` records bindings; `Infer` succeeds when outputs have valid device pointers; `Infer` returns error when an output binding has `device_ptr==nullptr`; `SetTensorAddress` updates an output's ptr; `Infer` invokes the `OutputFillCallback` with correct name/ptr/size for each output; `Reload` succeeds. The test allocates a small fake GPU buffer via `std::malloc` (since there's no real GPU) — the callback just memsets the buffer to simulate output.

```cpp
TEST(MockEngine, LoadAndInferInvokesCallback) {
    MockEngine engine;
    std::array<float, 4> buf{};
    auto outputs = std::vector<TensorBinding>{{"features", {1,4}, 16, buf.data()}};
    ASSERT_TRUE(engine.Load("test.engine", {}, outputs).has_value());

    bool called = false;
    engine.SetOutputFillCallback([&](std::string_view name, void* ptr, std::size_t sz) {
        EXPECT_EQ(name, "features");
        EXPECT_EQ(sz, 16U);
        std::memset(ptr, 0xAB, sz);
        called = true;
    });
    ASSERT_TRUE(engine.Infer().has_value());
    EXPECT_TRUE(called);
    EXPECT_EQ(buf[0], *reinterpret_cast<float*>(buf.data()));  // 0xABABABAB
}
```

- [ ] **Step 2: Run to verify it fails** (MockEngine not defined).

- [ ] **Step 3: Implement `inference_engine.h`, `mock_engine.h`, `mock_engine.cpp`, and the three adapter headers.**

`IInferenceEngine`: pure virtuals as per spec §3.4. `MockEngine`: stores inputs_/outputs_/output_fill_ in the cpp file; `Load` copies bindings; `Infer/InferAsync` invoke the callback for each output then return `{}`; `SetTensorAddress` finds by name and updates ptr. `DinoV3Adapter`: stores `IInferenceEngine*` + `DinoV3Config`; `Create` calls `engine.InputBindings()`/`OutputBindings()` to verify names/shapes match config; `Infer` declared only (no body — gated). Same pattern for Sam2/Clip adapter. Each adapter header forward-declares `class GpuImage;` from `sai::image` (no include needed — `Infer` signature uses `const GpuImage&` which only needs a forward decl).

- [ ] **Step 4: Run tests → green, build full suite → 176 prior + new tests all pass.**

`src/inference/CMakeLists.txt`: `sai_inference` STATIC from `mock_engine.cpp` (no gated files in this task), links `sai::core`, alias `sai::inference`, `cxx_std_20`. `tests/inference/CMakeLists.txt`: one test target, links `sai::inference` + `GTest::gtest_main`. Wire top-level CMakeLists.txt.

- [ ] **Step 5: Commit.**
```bash
git add include/sai/inference/ src/inference/ tests/inference/ CMakeLists.txt
git commit -m "feat(inference): ✨ 添加 IInferenceEngine/MockEngine 与模型 adapter 结构体"
```

---


### Task 2: ErrorCode additions

Append all 9 milestone-3 error codes to `error.h`.

**Files:**
- Modify: `include/sai/core/error.h`

**Interfaces:**
- Consumes: existing `ErrorCode` enum (current last member: `Io_ImportParseFailed`)
- Produces: 9 new members appended in exact order

- [ ] **Step 1: Append the 9 codes.** In this exact order after `Io_ImportParseFailed`:
```cpp
Inference_EngineLoadFailed,
Inference_EngineExecutionFailed,
Inference_InvalidBinding,
Inference_ReloadFailed,
Inference_ModelConfigMismatch,
Embedding_NotGpuImage,
Embedding_DimensionMismatch,
Detection_FeatureBankLoadFailed,
Detection_InvalidPatchGrid,
```

- [ ] **Step 2: Rebuild + full suite green.** Confirm 176 prior + Task 1's tests all pass. Append-only rule verification: `git diff include/sai/core/error.h` shows only additions, no reorder.

- [ ] **Step 3: Commit.**
```bash
git add include/sai/core/error.h
git commit -m "feat(core): ✨ 追加 M3 Inference/Embedding/Detection 错误码"
```

---


### Task 3: TensorRtEngine + adapter Infer implementations [gated, no local build]

Batch 3.1 gated portion. Writes the real TensorRT engine implementation and the adapter `Infer()` method bodies. Not compiled on this host.

**Files:**
- Create: `src/inference/tensorrt_engine.cpp`, `dino_v3_adapter.cpp`, `sam2_adapter.cpp`, `clip_adapter.cpp`
- Modify: `src/inference/CMakeLists.txt` (gate all 4 behind `find_package(CUDAToolkit QUIET)`)

- [ ] **Step 1: Implement `tensorrt_engine.cpp`.** `TensorRtEngine::Load`: `nvinfer1::createInferRuntime` → `deserializeCudaEngine` → `createExecutionContext` → validate bindings → store in `atomic<shared_ptr<EngineState>>`. `Infer/InferAsync`: `context->setTensorAddress(name, ptr)` for each binding → `context->enqueueV3(stream)`. `SetTensorAddress`: update `inputs_`/`outputs_` vector entry. `Reload`: deserialize new → validate → `atomic` swap → old context released. Handle all TensorRT error codes → `Inference_EngineLoadFailed`/`EngineExecutionFailed`/`InvalidBinding`/`ReloadFailed`.

- [ ] **Step 2: Implement `dino_v3_adapter.cpp`.** `DinoV3Adapter::Infer(image)`: `engine_.SetTensorAddress("input", image.Data())` → `engine_.Infer()` → read output binding → `PatchFeatures{output.device_ptr, grid_h, grid_w, dim}`. `sam2_adapter.cpp`: similar, with two inputs (image + prompt). `clip_adapter.cpp`: similar, single input.

- [ ] **Step 3: CMake gating.** `src/inference/CMakeLists.txt`: under `if(CUDAToolkit_FOUND)`, append all 4 `.cpp` files + link `CUDA::cudart` + TensorRT libs (`nvinfer`/`nvinfer_plugin` as imported targets). On macOS, these files are absent from compilation — `Infer` remains declared-but-undefined, which is safe because nothing portable ODR-uses them.

- [ ] **Step 4: Confirm exclusion is clean.** Rerun `cmake --preset default && cmake --build --preset default && ctest --preset default` → portable suite still passes at its full count. No new test for gated code.

- [ ] **Step 5: Commit.**
```bash
git add src/inference/ src/inference/CMakeLists.txt
git commit -m "feat(inference): ✨ 添加 TensorRtEngine 与 adapter 推理实现（CUDA 门控）"
```

---


### Task 4: Embedding data type

Batch 3.2 portable subset. Creates `sai::embedding::Embedding` with dual GPU/CPU storage, `EmbeddingMeta`, `EmbeddingType`.

**Files:**
- Create: `include/sai/embedding/embedding.h`, `src/embedding/CMakeLists.txt`, `src/embedding/embedding.cpp`
- Create: `tests/embedding/embedding_test.cpp`, `tests/embedding/CMakeLists.txt`
- Modify: top-level `CMakeLists.txt`

**Note:** `Embedding::ToCpuAsync` is declared but not defined in this task (gated — needs `GpuStreamQueue` + `PinnedPool` which are CUDA-gated). Portable tests cover FromGpu/FromCpu/move/SizeBytes/IsOnGpu/metadata.

- [ ] **Step 1: Write failing tests.** `FromCpu` yields `IsOnGpu()==false`, `Data()!=nullptr`, `Meta()` matches; `FromGpu` yields `IsOnGpu()==true`; move leaves source default (no dangling); `SizeBytes() == meta.count * meta.dim * sizeof(float)`; const Data() returns same ptr.

- [ ] **Step 2: Run to verify it fails → implement → green.**

- [ ] **Step 3: Full suite → 176 prior + Task 1 + Task 4 tests all pass.**

- [ ] **Step 4: Commit.**
```bash
git add include/sai/embedding/ src/embedding/ tests/embedding/ CMakeLists.txt
git commit -m "feat(embedding): ✨ 添加 Embedding 双存储数据类型"
```

---


### Task 5: IEmbedder + PatchEmbedder + GlobalEmbedder

Batch 3.2. `IEmbedder` pure-virtual interface (portable header). `PatchEmbedder` and `GlobalEmbedder` concrete classes (portable headers; `Extract`/`ExtractBatch` bodies gated because they call adapter `Infer` which is Task 3 gated). Portable tests use `MockEngine` with `OutputFillCallback` to inject synthetic features — `Extract` is not ODR-used in portable tests; instead, test the embedder's **metadata handling** (ModelName, factory Create with mock engine validates bindings) and the **Extract GPU guard** (calling Extract on SurfaceImage→Embedding_NotGpuImage).

**Files:**
- Create: `include/sai/embedding/embedder.h`, `src/embedding/patch_embedder.cpp` (portable ctor + Create, gated Extract via separate gated CU file)

- [ ] **Step 1: Write tests.** `PatchEmbedder::Create(mock_adapter)` succeeds when adapter bindings match config; `ModelName()=="DINOv3"`; `Extract(surface_image)` (CPU image) returns `Embedding_NotGpuImage` (this path is portable because the GPU guard is implemented inline in the header or the portable `.cpp` — it checks the image's storage type before dispatching to the gated Infer path).

- [ ] **Step 2: Implement → green → full suite.**

- [ ] **Step 3: Commit.**
```bash
git add include/sai/embedding/embedder.h src/embedding/ tests/embedding/
git commit -m "feat(embedding): ✨ 添加 IEmbedder 接口与 PatchEmbedder/GlobalEmbedder"
```

---


### Task 6: DimensionReducer + FeatureCache

Batch 3.2, fully portable. PCA/Whitening fitting (static), dimension reduction (in-place), spatial pooling, and LRU feature cache.

**Files:**
- Create: `include/sai/embedding/dimension_reducer.h`, `feature_cache.h`, `src/embedding/dimension_reducer.cpp`, `feature_cache.cpp`
- Modify: `tests/embedding/CMakeLists.txt` (add test sources)

- [ ] **Step 1: Write failing tests.** PCA: fit on 4 known 3D vectors → Reduce to 2D → known projection. Whitening: fit on same → variance normalized. Pool: 2x2 patch grid → Average pooling → 1×dim global embedding with element-wise mean. FeatureCache: Get on empty returns nullptr; Put+Get returns non-null matching Embedding; LRU evicts oldest when max_entries exceeded; HitRate() computes correctly.

- [ ] **Step 2: Implement → green.**

- [ ] **Step 3: Full suite green.**

- [ ] **Step 4: Commit.**
```bash
git add include/sai/embedding/dimension_reducer.h include/sai/embedding/feature_cache.h src/embedding/ tests/embedding/
git commit -m "feat(embedding): ✨ 添加 DimensionReducer 与 FeatureCache"
```

---


### Task 7: DetectionResult + IDetector + PatchCore Config

Batch 3.3 portable subset. Data structures and the detector interface.

**Files:**
- Create: `include/sai/detection/detection_result.h`, `detector.h`, `patch_core.h` (Config + class decl, no Detect body yet)
- Create: `src/detection/CMakeLists.txt`, `src/detection/detection_result.cpp`
- Create: `tests/detection/detection_types_test.cpp`, `tests/detection/CMakeLists.txt`
- Modify: top-level `CMakeLists.txt`

- [ ] **Step 1: Write tests.** `AnomalyMap::At(y,x)` returns correct score; `MaxScore()` finds global max; `IsDefective(threshold)` correct. `RegionProposal` fields match. `DetectionResult::IsDefective(threshold)` delegates to `image_level_score`. `PatchCore::Config` default values match spec.

- [ ] **Step 2: Implement → green → full suite.**

- [ ] **Step 3: Commit.**
```bash
git add include/sai/detection/ src/detection/ tests/detection/ CMakeLists.txt
git commit -m "feat(detection): ✨ 添加 DetectionResult/IDetector/PatchCore Config"
```

---


### Task 8: FeatureBank + PatchCore::Detect

Batch 3.3. `FeatureBank` uses FAISS (portable `faiss::IndexFlatL2` CPU + gated `faiss::gpu::StandardGpuResources` GPU). `PatchCore::Detect` is the full pipeline: k-NN search → DtoH → upsample → GaussianBlur → threshold → connectedComponents → RegionProposals. The post-processing (upsample → regions) is pure CPU, fully portable, and testable. The FAISS search uses the CPU backend on macOS (testable, real k-NN). GPU FAISS path is gated.

**Files:**
- Create: `include/sai/detection/feature_bank.h`, `patch_core.h` (modify to add Detect decl)
- Create: `src/detection/feature_bank.cpp` (portable FAISS CPU), `feature_bank_cuda.cpp` (gated FAISS GPU), `patch_core.cpp` (post-processing, portable)
- Modify: `vcpkg.json` (add `"faiss"`)
- Create: `tests/detection/patch_core_test.cpp`

- [ ] **Step 1: Add FAISS to vcpkg.json, reconfigure.** Verify `find_package(faiss CONFIG REQUIRED)` works on macOS.

- [ ] **Step 2: Write tests.** `FeatureBank::LoadFromFile` loads a synthetic coreset (small binary file of known float values) → `NumSamples()` + `Dim()` correct; `Search(single_query, k=1)` returns distance ~0 for query matching a coreset sample, positive distance for distant query. `PatchCore::Detect`: construct with mock FeatureBank, call Detect(embedding) → `DetectionResult` with valid AnomalyMap (all scores in [0,1], MaxScore > 0), at least one RegionProposal when score above threshold, empty regions when all below threshold. Upsample: 2×2 anomaly grid → 4×4 output with correct bilinear interpolation. ConnectedComponents: known binary mask → correct region count + bounding boxes.

- [ ] **Step 3: Implement FeatureBank (portable FAISS + gated GPU split).** `feature_bank.cpp`: `LoadFromFile` reads binary float matrix → `faiss::IndexFlatL2` → `Search` performs `index.search(nq, query, k, distances, labels)`. `feature_bank_cuda.cpp` (gated): same LoadFromFile but wraps in `faiss::gpu::StandardGpuResources` → `faiss::gpu::index_cpu_to_gpu`. CMake gate: `feature_bank_cuda.cpp` only under `CUDAToolkit_FOUND`.

- [ ] **Step 4: Implement PatchCore::Detect.** `patch_core.cpp`: `Detect(embedding)` → validate grid matches config → `feature_bank_.Search(query, k)` → DtoH distances → build `AnomalyMap` → bilinear upsample → `GaussianBlur(sigma)` (separable kernel, CPU) → threshold → 4-connected component labeling → fill `vector<RegionProposal>` sorted by max_score desc → return `DetectionResult`.

- [ ] **Step 5: Full suite → confirm 176 prior + all new tests pass.**

- [ ] **Step 6: Commit.**
```bash
git add vcpkg.json include/sai/detection/ src/detection/ tests/detection/
git commit -m "feat(detection): ✨ 添加 FeatureBank（FAISS）与 PatchCore::Detect"
```

---


### Task 9: Validation integration test

Implements the spec §6 validation point: MockEngine → DinoV3Adapter → PatchEmbedder → PatchCore → DetectionResult, all portable on macOS.

**Files:**
- Create: `tests/integration/inference_pipeline_test.cpp`, add to `tests/integration/CMakeLists.txt`
- Modify: `tests/integration/CMakeLists.txt` (if needed)

- [ ] **Step 1: Write the integration test.** Wire the full flow:
  1. Create `MockEngine`, load with DINOv3 bindings (input: (1,3,518,518), output: (1,1024,37,37))
  2. Set `OutputFillCallback` that fills output with a known test pattern (synthetic patch features)
  3. Create `DinoV3Adapter::Create(mock_engine, config{patch_size=14, embed_dim=1024})`
  4. Create `PatchEmbedder::Create(std::move(adapter))`
  5. Build a test `SurfaceImage` (FromOwnedBuffer, 518×518 RGB8 pattern)
  6. Call `embedder.Extract(image)` → `Embedding` (GPU or CPU, `count==37*37==1369`, `dim==1024`)
  7. Load a synthetic `FeatureBank` (small 2-core-sample bin file)
  8. Create `PatchCore(Config{.image_size=518,.patch_size=14,.embed_dim=1024,.anomaly_threshold=0.5})`
  9. Call `patch_core.Initialize(ctx)` (loads feature bank) → `patch_core.Detect(embedding)`
  10. Assert `DetectionResult.image_level_score > 0`, `anomaly_map.grid_h==37`, `anomaly_map.grid_w==37`

```cpp
TEST(InferencePipeline, MockEngineToDetectionResult) {
    MockEngine engine;
    // DINOv3 typical bindings: 1 input (B,3,H,W), 1 output (B,D,H_p,W_p)
    TensorBinding in{"pixel_values", {1,3,518,518}, /*...*/};
    TensorBinding out{"last_hidden_state", {1,1024,37,37}, /*...*/};
    std::vector<float> output_buffer(1*1024*37*37);
    out.device_ptr = output_buffer.data();
    ASSERT_TRUE(engine.Load("dino.engine", {in}, {out}).has_value());

    // Fill with synthetic embedding: each patch's 1024-dim vector = row-major index * 0.01
    engine.SetOutputFillCallback([&](std::string_view, void* ptr, std::size_t) {
        auto* f = static_cast<float*>(ptr);
        for (size_t i = 0; i < 37*37*1024; ++i) f[i] = static_cast<float>(i % 1024) * 0.001F;
    });

    auto adapter = DinoV3Adapter::Create(engine, {.engine_path="dino.engine", .image_size=518, .patch_size=14, .embed_dim=1024});
    ASSERT_TRUE(adapter.has_value());
    auto embedder = PatchEmbedder::Create(std::move(*adapter));
    ASSERT_TRUE(embedder.has_value());

    auto img = SurfaceImage::FromOwnedBuffer(MakeTestImage(518, 518), /*meta*/);
    auto emb = embedder->Extract(img);
    ASSERT_TRUE(emb.has_value());
    EXPECT_EQ(emb->Meta().count, 1369U);
    EXPECT_EQ(emb->Meta().dim, 1024U);

    // ... FeatureBank + PatchCore + Detect + assertions
}
```

- [ ] **Step 2: Implement → green.**

- [ ] **Step 3: Full suite → all prior + integration test pass.**

- [ ] **Step 4: Commit.**
```bash
git add tests/integration/
git commit -m "test(inference): ✅ 添加 M3 推理管线端到端集成测试（MockEngine→DetectionResult）"
```

---

## Spec Coverage Check

- Spec §3 (3.1 Foundation): IInferenceEngine + TensorBinding + TensorRtEngine(gated) + MockEngine + 3 adapters → Tasks 1, 3. ✅
- Spec §4 (3.2 Embedding): Embedding + IEmbedder + PatchEmbedder/GlobalEmbedder + DimensionReducer + FeatureCache → Tasks 4, 5, 6. ✅
- Spec §5 (3.3 Detector): DetectionResult + IDetector + FeatureBank + PatchCore → Tasks 7, 8. ✅
- Spec §6 validation point: MockEngine → adapter → embedder → PatchCore → DetectionResult → Task 9. ✅
- ErrorCode (§3.12): 9 new codes in append-only order → Task 2. ✅
- FAISS vcpkg dependency → Task 8. ✅
