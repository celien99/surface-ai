# GPU-First Performance Subtraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task.

**Goal:** Remove production-path CPU brute-force feature search and full patch-embedding D2H copies, while moving coreset selection distance computation to batched FAISS GPU queries.

**Architecture:** Keep the existing `FeatureBank`/`PatchCore`/`CoresetEvolution` boundaries. GPU residency is established by the application after the final bank index is ready; rebuilt banks are uploaded before they are swapped into the detector. `PatchCore` passes GPU-resident embeddings directly to FAISS GPU and only returns the small distance matrix to CPU. Greedy coreset selection keeps argmax bookkeeping on CPU but performs all M×D distance evaluation through a batched FeatureBank query.

**Tech Stack:** C++20, FAISS CPU/GPU, CUDA, TensorRT-gated embedding path, tl::expected.

## Global Constraints

- Do not compile, link, run tests, run test binaries, or start the application.
- Before every code edit, reread the current version of each file being edited.
- Each independent functionality ends with its own git commit.
- Do not add compatibility layers, speculative validation, or broad defensive branches.
- Production GPU paths must not silently fall back to CPU brute-force search.
- CPU-only code remains only where the existing non-CUDA build requires it; it is not a production fallback when GPU support is compiled in.
- Static verification only: `git diff --check`, `git diff`, `git diff --stat`, `rg`, `sed`, `git status`, and `git log`.

---

### Task 1: Wire loaded FeatureBanks to GPU

**Files:**
- Modify: `apps/seat-aoi/app_builder.cpp`
- Modify: `src/pipeline/detect_stage.cpp`

**Deliverable:** Every production detector/retrieval/bootstrap FeatureBank is migrated with `ToGpu()` before it is used. A migration error aborts the corresponding assembly/bootstrap operation instead of silently selecting the CPU index.

**Commit:** `perf(feature-bank): wire production banks to gpu`

### Task 2: Preserve GPU residency across rebuild and swap

**Files:**
- Modify: `src/detection/feature_bank.cpp`
- Modify: `src/detection/feature_bank_cuda.cpp`
- Modify: `src/detection/coreset_evolution.cpp`

**Deliverable:** `Rebuild()` and `ConvertToIVF()` cannot retain stale GPU mirrors. CoresetEvolution uploads newly built active banks before `SwapFeatureBank()`; a failed upload leaves the current active bank untouched.

**Commit:** `fix(feature-bank): keep gpu index lifecycle consistent`

### Task 3: Keep PatchCore patch queries on GPU

**Files:**
- Modify: `src/pipeline/inference_stage.cpp`
- Modify: `src/detection/patch_core.cpp`

**Deliverable:** GPU patch embeddings remain GPU-resident through normal PatchCore detection. Only the `query_count × k` FAISS distance output reaches CPU. CPU materialization is retained only for the explicitly enabled evolution context and CPU transforms that already require host data.

**Commit:** `perf(patchcore): remove full patch embedding dtoh`

### Task 4: Replace greedy coreset CPU distance loops with batched FeatureBank queries

**Files:**
- Modify: `src/detection/feature_bank.cpp`
- Modify: `include/sai/detection/feature_bank.h`

**Deliverable:** Greedy furthest-point selection uses one batched nearest-neighbor query per selected point, with GPU FeatureBank residency when CUDA+FAISS-GPU is compiled. CPU fallback keeps the same algorithm only for non-GPU builds; no new runtime compatibility mode is introduced.

**Commit:** `perf(coreset): batch greedy selection through faiss`

### Task 5: Static verification and performance audit

- Confirm every production `FeatureBank` construction path is followed by `ToGpu()` before search/swap.
- Confirm no normal `InferenceStage` path performs a full patch embedding `cudaMemcpyDeviceToHost`.
- Confirm `BuildWithGreedyCoreset()` no longer contains the per-point/per-dimension distance loop in the GPU-enabled path.
- Confirm no stale GPU mirror survives `Rebuild()` or `ConvertToIVF()`.
- Run only static commands listed in the global constraints; do not compile or execute tests.
