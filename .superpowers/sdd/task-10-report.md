# Task 10 Report: M6 E2E Pipeline Integration Test

## Summary

Created `tests/integration/m6_e2e_pipeline_test.cpp` with 5 integration tests for the M6 Pipeline end-to-end flow. All 5 tests pass, 544/544 total tests pass (no regressions).

## Files Changed

1. **Created**: `tests/integration/m6_e2e_pipeline_test.cpp` -- 5 E2E integration tests
2. **Modified**: `tests/integration/CMakeLists.txt` -- Added `sai_integration_m6_e2e_pipeline_test` target linked against `sai::pipeline`
3. **Modified**: `include/sai/pipeline/pipeline.h` -- Added explicit `~Pipeline();` declaration (fixes incomplete-type error when Pipeline is destroyed outside sai_pipeline translation unit)
4. **Modified**: `src/pipeline/pipeline.cpp` -- Added `Pipeline::~Pipeline() = default;` definition
5. **Modified**: `src/pipeline/pipeline_builder.h` -- Changed `ParseStageConfig` parameter from `const YAML::Node&` to `YAML::Node&`
6. **Modified**: `src/pipeline/pipeline_builder.cpp` -- Changed `ParseStageConfig` signature to match; same fix

## Tests

| # | Test | Result |
|---|------|--------|
| 1 | `LoadPipelineFromYaml` | PASS -- Loads 7-stage pipeline from temp YAML, verifies Metrics() returns all stage ids and types |
| 2 | `PipelineLifecycle` | PASS -- Load -> Start -> Submit mock frame (RawImage 10x10 Mono8) -> Drain -> Stop |
| 3 | `InvalidYamlReturnsError` | PASS -- `depends_on` references non-existent stage, returns `Pipeline_InvalidConfig` |
| 4 | `CyclicDependencyRejected` | PASS -- a->b->c->a cycle (all Capture type), returns `Pipeline_InvalidConfig` |
| 5 | `StageTypeFromStringConversion` | PASS -- All 8 types roundtrip correctly, unknown type throws `std::invalid_argument` |

## Defects Found and Fixed

### D1: `~Pipeline()` missing (incomplete type error)
`pipeline.h` forward-declares `PipelineExecutor`, `TaskScheduler`, `WorkerPool` but stores them as `std::unique_ptr`. Without an explicit `~Pipeline()`, the compiler generates an inline destructor that needs complete types -- any translation unit including `pipeline.h` fails to compile. Fixed by declaring `~Pipeline();` in the header and defining `= default;` in `pipeline.cpp` where all runtime headers are included.

### D2: `ParseStageConfig` const `YAML::Node&` parameter causes ZombieNode crash
yaml-cpp's const `operator[]` returns a ZombieNode for non-existent keys. When this ZombieNode is assigned to another `YAML::Node` (via `stage.config = node["config"]`), it throws `YAML::InvalidNode`. The non-const `operator[]` safely auto-creates nodes for missing keys. Fixed by changing the parameter from `const YAML::Node&` to `YAML::Node&`.

### D3: Type compatibility mismatch in test pipeline
The original 6-stage pipeline (`capture -> preprocess -> detect -> rule_eval -> reason -> export`) skipped `Inference`, causing `Preprocess` (output: SurfaceImage) to feed `Detect` (input: DetectionResult) -- type mismatch. Fixed by inserting `inference` stage: `capture -> preprocess -> inference -> detect -> rule_eval -> reason -> export` (7 stages).

### D4: Cyclic test made type-safe
Original cyclic test used mixed types (Capture, Detect, Export). Since `IsAcyclic` is checked before `CheckTypeCompatibility` in `Validate`, this was not a runtime problem, but changed all three stages to `Capture` type for clarity and robustness.

## Verification

```bash
ctest --preset default -R "M6E2E" --output-on-failure
# 5/5 tests passed

ctest --preset default
# 544/544 tests passed, 0 failed
```
