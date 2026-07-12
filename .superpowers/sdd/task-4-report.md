# Task 4 Report: Embedding data type

**Status:** Complete

**Commit:** `feat(embedding): ✨ 添加 Embedding 双存储数据类型`

**Tests:** 215 total (201 prior + 14 new), 0 failures.

**New tests (14):**
| Test | What it verifies |
|------|-----------------|
| `Embedding.TypeTraits` | Move-only, nothrow move, final |
| `Embedding.FromGpuSignatureCompiles` | `static_assert` on `FromGpu` signature compileability |
| `Embedding.ToCpuAsyncDeclared` | `static_assert` on `ToCpuAsync` member existence |
| `Embedding.FromCpuCreatesCpuEmbedding` | `IsOnGpu()==false`, `Data()!=nullptr`, `Meta()` all fields, `SizeBytes()` |
| `Embedding.FromCpuEmptyVector` | Empty data → `Data()==nullptr`, `SizeBytes()==0` |
| `Embedding.FromCpuGlobalEmbedding` | `EmbeddingType::Global`, single-vector data |
| `Embedding.SizeBytesMatchesCountTimesDimTimesFloatSize` | `SizeBytes() == count * dim * sizeof(float)` |
| `Embedding.SizeBytesWithZeroCountOrDim` | Zero count or dim → `SizeBytes()==0` |
| `Embedding.ConstDataReturnsSamePointer` | `const Data()` and non-const return same address |
| `Embedding.MoveConstructionLeavesSourceValid` | Move ctor transfers data, source becomes empty |
| `Embedding.MoveAssignmentTransfersOwnership` | Move assignment transfers data and metadata |
| `Embedding.SelfMoveAssignmentSafe` | Move assignment does not crash |
| `Embedding.EmbeddingTypeValues` | Enum values correct (Patch=0, Global=1) |
| `Embedding.EmbeddingMetaDefaultValues` | Default-constructed meta fields zero/empty |

**Files created:**
- `include/sai/embedding/embedding.h` — EmbeddingType, EmbeddingMeta, Embedding (dual storage, move-only)
- `src/embedding/embedding.cpp` — FromGpu, FromCpu, Data, SizeBytes, ~Embedding definitions
- `src/embedding/CMakeLists.txt` — sai_embedding static library
- `tests/embedding/embedding_test.cpp` — 14 test cases
- `tests/embedding/CMakeLists.txt` — test target

**Files modified:**
- `CMakeLists.txt` — added `src/embedding` and `tests/embedding` subdirectories

**Concerns:**
- None. All signatures match the spec (SS 4.4). `ToCpuAsync` is declared but not defined, matching the declare-in-header/gate-in-cpp pattern. `PooledPtr<uint8_t>` requires `memory_pool.h` include (same as `SurfaceImage` pattern in M2).
