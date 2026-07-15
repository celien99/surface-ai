# Task 4: CoresetUpdater — skeleton report

## Summary

Implemented the CoresetUpdater background thread skeleton and prefilter infrastructure.

## Changes

### 1. Header: `include/sai/detection/coreset_evolution.h`
- Added `EvolutionConfig` struct (self-evolution configuration with normality, novelty, buffer, update, and persistence parameters)
- Added `EvolutionStats` struct (update statistics including frame/patch counts, timing, coverage gain)

### 2. New file: `src/detection/coreset_updater.cpp`
- `LightGreedySelect()` — uniform stride sampling for coarse-grained candidate prefiltering
- `CoresetUpdater` class (anonymous namespace, internal) — background thread with `std::jthread`-compatible `Run(std::stop_token)` loop, `Notify()` trigger, and `LatestStats()` query
- `DoUpdate()` — drains candidates, flattens patch vectors, applies `LightGreedySelect` prefilter, records evolution stats
- Merge + double-buffer swap is a placeholder (deferred to Task 5)

### 3. Build: `src/detection/CMakeLists.txt`
- Added `coreset_updater.cpp` to `SAI_DETECTION_SOURCES`

### 4. Test: `tests/detection/coreset_evolution_test.cpp`
- Added `<memory>` include
- Appended `CoresetUpdaterSkeletonTest.LightGreedyPreservesDim` test verifying CandidateBuffer drain path

## Test Results

All 17 tests pass (16 existing + 1 new):
- NormalityProfileTest: 3/3
- NormalityScorerTest: 2/2
- NoveltyFilterTest: 2/2
- CandidateBufferTest: 4/4
- MultiSignalConsensusTest: 5/5
- CoresetUpdaterSkeletonTest: 1/1

## Build

Build clean with no errors or warnings (beyond pre-existing duplicate library link warnings).
