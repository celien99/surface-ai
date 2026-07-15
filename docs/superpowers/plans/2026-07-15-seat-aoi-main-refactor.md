# Seat AOI main.cpp Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split 1173-line `apps/seat-aoi/main.cpp` into 9 focused files; `main()` shrinks from ~710 to ~30 lines.

**Architecture:** Pure code movement — zero behavior change. Extract each logical section from `main.cpp` into its own `.h`/`.cpp` pair. `AssembledApp` struct holds all object ownership; `AppBuilder` assembles everything from `CliArgs`.

**Tech Stack:** C++20, no new dependencies.

## Global Constraints

- Zero behavior change — no logic, parameter, threshold, or output modifications
- All `std::cout`/`std::cerr` output strings preserved verbatim
- `pipeline.yaml` loaded exactly once (currently 4 times); parse result shared internally
- No changes to any `sai::*` library code
- CMakeLists.txt: only add new `.cpp` files to `add_executable`, no link changes
- Follow existing coding style: early returns, `Result<T>` monadic chains, English identifiers/comments

---

### Task 1: `app_config.h` — Resource path constants

**Files:**
- Create: `apps/seat-aoi/app_config.h`

**Interfaces:**
- Produces: `seat_aoi::config` namespace with `constexpr std::string_view` constants for all hardcoded paths and model parameters

Extract every hardcoded path and magic number from main.cpp into named constants:

- `kPipelineYaml`, `kDinoV3Engine`, `kClipEngine`, `kSam2Engine`
- `kDecisionTree`, `kTuningYaml`
- `kDefaultCoresetOutput`, `kDefaultOutputDir`
- `kEmbedDim = 1024`, `kImageSize = 1024`, `kPatchSize = 14`

- [ ] **Step 1: Write `app_config.h`**

```cpp
#pragma once

#include <cstddef>
#include <string_view>

namespace seat_aoi::config {

// ── Resource paths ──
constexpr std::string_view kPipelineYaml   = "resources/pipeline.yaml";
constexpr std::string_view kDinoV3Engine   = "resources/models/dino_v3_vit_base.engine";
constexpr std::string_view kClipEngine     = "resources/models/clip_vit_b32.engine";
constexpr std::string_view kSam2Engine     = "resources/models/sam2_vit_h.engine";
constexpr std::string_view kDecisionTree   = "resources/trees/seat_leather_inspection.yaml";
constexpr std::string_view kTuningYaml     = "resources/tuning/seat_leather_tuning.yaml";

// ── Defaults ──
constexpr std::string_view kDefaultCoresetOutput = "resources/coreset.bin";
constexpr std::string_view kDefaultOutputDir     = "/tmp/surface-ai/results/";

// ── Model / preprocessing constants ──
constexpr std::size_t kEmbedDim  = 1024;
constexpr std::size_t kImageSize = 1024;
constexpr std::size_t kPatchSize = 14;

}  // namespace seat_aoi::config
```

- [ ] **Step 2: Commit**

```bash
git add apps/seat-aoi/app_config.h
git commit -m "refactor(seat_aoi): ♻️ 抽取 app_config.h 集中管理资源路径与模型参数常量"
```

---

### Task 2: `cli_args.h` + `cli_args.cpp` — CLI argument parsing

**Files:**
- Create: `apps/seat-aoi/cli_args.h`
- Create: `apps/seat-aoi/cli_args.cpp`
- Modify: `apps/seat-aoi/main.cpp` (remove CliArgs + ParseArgs, add `#include "cli_args.h"`)

**Interfaces:**
- Produces: `struct CliArgs` (all fields from L61-71), `auto ParseArgs(int argc, char* argv[]) -> CliArgs` (L73-102 exactly)

Source: `main.cpp` L61-102 (struct + function in anonymous namespace)

- [ ] **Step 1: Write `cli_args.h`**

```cpp
#pragma once

#include <cstddef>
#include <string>

struct CliArgs {
    std::string image_dir;
    std::string output_dir = "/tmp/surface-ai/results/";
    std::string coreset_path;
    std::string dataset_path;
    std::string coreset_output_path;
    std::string coreset_algo = "greedy";
    std::size_t coreset_max_samples = 10000;
    bool headless = false;
    bool cpu_mode = false;
};

auto ParseArgs(int argc, char* argv[]) -> CliArgs;
```

- [ ] **Step 2: Write `cli_args.cpp`** — copy L73-102 verbatim from main.cpp, replacing `std::string_view` usage (already correct), keeping the `dataset_path`+`coreset_output_path` default logic (L98-100)

- [ ] **Step 3: Modify `main.cpp`** — remove L59-102 (anonymous namespace opening, CliArgs struct, ParseArgs function, closing `}`), replace with `#include "cli_args.h"`

- [ ] **Step 4: Commit**

---

### Task 3: `knowledge_seed.h` + `knowledge_seed.cpp` — KG seed data

**Files:**
- Create: `apps/seat-aoi/knowledge_seed.h`
- Create: `apps/seat-aoi/knowledge_seed.cpp`
- Modify: `apps/seat-aoi/main.cpp` (remove SeedKnowledgeGraph, add include)

**Interfaces:**
- Produces: `void SeedKnowledgeGraph(sai::knowledge::KnowledgeGraph& kg)`

Source: `main.cpp` L106-163

- [ ] **Step 1: Write `knowledge_seed.h`**

```cpp
#pragma once

namespace sai::knowledge { class KnowledgeGraph; }

void SeedKnowledgeGraph(sai::knowledge::KnowledgeGraph& kg);
```

- [ ] **Step 2: Write `knowledge_seed.cpp`** — copy L106-163's function body verbatim, add `#include "knowledge_seed.h"` + all required sai headers (`sai/knowledge/knowledge_graph.h`, `sai/knowledge/knowledge_record.h` etc.)

- [ ] **Step 3: Modify `main.cpp`** — remove L106-163, add `#include "knowledge_seed.h"`

- [ ] **Step 4: Commit**

---

### Task 4: `coreset_builder.h` + `coreset_builder.cpp` — Offline coreset build

**Files:**
- Create: `apps/seat-aoi/coreset_builder.h`
- Create: `apps/seat-aoi/coreset_builder.cpp`
- Modify: `apps/seat-aoi/main.cpp` (remove BuildCoreset function + closing `}` of anonymous namespace, add include)
- Modify: `apps/seat-aoi/main.cpp` (remove `#include <sqlite3.h>` — only needed by knowledge_seed now)

**Interfaces:**
- Produces: `int BuildCoreset(const CliArgs& cli)`

Source: `main.cpp` L167-382 + L59 (anonymous namespace `{`) + L462 (`}` closing anonymous namespace)

- [ ] **Step 1: Write `coreset_builder.h`**

```cpp
#pragma once

struct CliArgs;

int BuildCoreset(const CliArgs& cli);
```

- [ ] **Step 2: Write `coreset_builder.cpp`** — copy L167-382 function body. Use `seat_aoi::config::kEmbedDim` etc. instead of local `constexpr`. Keep all GPU upload code (HtoD→infer→DtoH), preprocess chain, FeatureBank build, save, and statistics output verbatim.

- [ ] **Step 3: Modify `main.cpp`** — remove L167-382 (BuildCoreset), remove the anonymous namespace opening `namespace {` (L59) and closing `}` (L462), add `#include "coreset_builder.h"`. Remove `#include <sqlite3.h>` if no longer used.

- [ ] **Step 4: Commit**

---

### Task 5: `tuning_wiring.h` + `tuning_wiring.cpp` — Bayesian tuning helpers

**Files:**
- Create: `apps/seat-aoi/tuning_wiring.h`
- Create: `apps/seat-aoi/tuning_wiring.cpp`
- Modify: `apps/seat-aoi/main.cpp` (remove CollectLeafNodes + ApplyParamToYaml + tuning assembly block)

**Interfaces:**
- Produces: `CollectLeafNodes`, `ApplyParamToYaml` (internal), `TryCreateTuningScheduler` (public)
- `TryCreateTuningScheduler` returns `Result<std::optional<sai::tuning::TuningScheduler>>`

Source: `main.cpp` L386-460 (helpers) + L728-868 (tuning assembly in main)

- [ ] **Step 1: Write `tuning_wiring.h`**

```cpp
#pragma once

#include <optional>
#include <yaml-cpp/yaml.h>
#include <sai/core/error.h>

namespace sai {
namespace knowledge { class KnowledgeGraph; class KnowledgeEvolution; }
namespace reasoner { class IReasoner; }
namespace tuning { class TuningScheduler; }
namespace pipeline { class Pipeline; }
}

auto CollectLeafNodes(const YAML::Node& node) -> std::vector<YAML::Node>;
auto ApplyParamToYaml(YAML::Node& root, std::vector<YAML::Node>& leaf_nodes,
                      const std::string& param_name, double value) -> void;

auto TryCreateTuningScheduler(
    const YAML::Node& pipeline_yaml,
    sai::knowledge::KnowledgeGraph& kg,
    sai::knowledge::KnowledgeEvolution& kg_evolution,
    sai::reasoner::IReasoner& reasoner,
    sai::pipeline::Pipeline& pipeline
) -> sai::Result<std::optional<sai::tuning::TuningScheduler>>;
```

- [ ] **Step 2: Write `tuning_wiring.cpp`** — copy L390-460 (CollectLeafNodes + ApplyParamToYaml) verbatim. Then implement `TryCreateTuningScheduler` by extracting L728-868, replacing inline lambdas with the same logic. The `SetParameterApplier` lambda captures `reasoner` by reference; `SetMetricsPoller` captures `pipeline` by reference — preserve these semantics.

- [ ] **Step 3: Modify `main.cpp`** — remove L386-460 (CollectLeafNodes + ApplyParamToYaml), remove L728-868 (tuning assembly in main), add `#include "tuning_wiring.h"`

- [ ] **Step 4: Commit**

---

### Task 6: `app_builder.h` + `app_builder.cpp` — Application assembly (the big one)

**Files:**
- Create: `apps/seat-aoi/app_builder.h`
- Create: `apps/seat-aoi/app_builder.cpp`
- Modify: `apps/seat-aoi/main.cpp` (remove L464-917 main body except headless/GUI dispatch)

**Interfaces:**
- Produces: `struct AssembledApp` (all fields from spec), `auto AssembleApplication(const CliArgs& cli) -> sai::Result<AssembledApp>`

Source: `main.cpp` L467-917 (everything between `auto main()` and the `run_headless` lambda)

This is the core assembly function. It must:

1. Create Context (L479-481)
2. Parse `pipeline.yaml` once — pass `YAML::Node` to sub-steps
3. Create DINOv3 TensorRtEngine + PatchEmbedder (L486-509)
4. Optionally create CLIP GlobalEmbedder (L514-544) — gated by pipeline.yaml
5. Create PatchCore detector (L548-559)
6. Load FeatureBank + VectorPath if `--coreset` provided (L563-585)
7. Optionally create CoresetEvolution (L593-615) — gated by pipeline.yaml
8. Create KnowledgeStore + SeedKnowledgeGraph (L619-634)
9. Optionally create SAM2 Segmenter (L638-666) — gated by pipeline.yaml
10. Create RuleEngine + Reasoner (L670-681)
11. Create JsonExporter (L686)
12. Load Pipeline (L691-698)
13. Wire all objects to stages (L702-724)
14. Optionally create TuningScheduler via tuning_wiring (L728-868)
15. Set ResultCallback for self-evolution (L873-898)
16. Pipeline->Start() (L902-907)
17. Optionally Evolution->Start() (L912-917)

Any step failure → `return tl::make_unexpected(ErrorInfo{code, message})`.

- [ ] **Step 1: Write `app_builder.h`** — define `AssembledApp` struct + `AssembleApplication` signature

```cpp
#pragma once

#include <memory>
#include <optional>
#include <sai/core/error.h>

// Forward declarations for all sai types used
namespace sai {
class Context;
namespace pipeline { class Pipeline; }
namespace embedding { class PatchEmbedder; class IEmbedder; }
namespace detection { class PatchCore; class FeatureBank; struct CoresetEvolution; }
namespace rule { class RuleEngine; }
namespace reasoner { class IReasoner; }
namespace io { class JsonExporter; }
namespace knowledge { class KnowledgeGraph; class KnowledgeEvolution; class KnowledgeStore; }
namespace inference { class Sam2Segmenter; }
namespace retrieval { class VectorPath; }
namespace tuning { class TuningScheduler; }
}

struct CliArgs;

struct AssembledApp {
    std::unique_ptr<sai::Context> ctx;
    std::unique_ptr<sai::pipeline::Pipeline> pipeline;
    std::shared_ptr<sai::embedding::PatchEmbedder> embedder;
    std::shared_ptr<sai::detection::PatchCore> patch_core;
    std::shared_ptr<sai::rule::RuleEngine> rule_engine;
    std::shared_ptr<sai::io::JsonExporter> exporter;
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg;
    std::shared_ptr<sai::knowledge::KnowledgeEvolution> kg_evolution;
    std::unique_ptr<sai::knowledge::KnowledgeStore> knowledge_store;

    std::shared_ptr<sai::embedding::IEmbedder> global_embedder;
    std::shared_ptr<sai::inference::Sam2Segmenter> sam2_segmenter;
    std::shared_ptr<sai::reasoner::IReasoner> reasoner;
    std::shared_ptr<sai::detection::FeatureBank> feature_bank;
    std::shared_ptr<sai::retrieval::VectorPath> vector_path;

    std::optional<sai::detection::CoresetEvolution> evolution;
    std::optional<sai::tuning::TuningScheduler> tuning_scheduler;
};

auto AssembleApplication(const CliArgs& cli) -> sai::Result<AssembledApp>;
```

- [ ] **Step 2: Write `app_builder.cpp`** — implement `AssembleApplication` by extracting L467-917 from main.cpp. Key details:

  - Replace `constexpr std::size_t kEmbedDim = 1024` with `seat_aoi::config::kEmbedDim`
  - Replace hardcoded paths with `seat_aoi::config::k*` constants
  - Load `pipeline.yaml` once at top, pass `YAML::Node` reference through lambdas
  - Keep the `[](auto*){}` non-owning shared_ptr pattern for kg/kg_evolution
  - Tuning assembly delegates to `TryCreateTuningScheduler` from tuning_wiring
  - Result callback for evolution: extract both branches (with and without evolution) — the evolution branch (L1110-1138) will be set here; the non-evolution branch will be handled by GuiRunner
  - All `std::cerr << ...; return 1;` → `return tl::make_unexpected(ErrorInfo{...})`

- [ ] **Step 3: Modify `main.cpp`** — remove L467-917 (everything from `using namespace sai;` through evolution start), add `#include "app_builder.h"`

- [ ] **Step 4: Update `CMakeLists.txt`** — add `app_builder.cpp` to `add_executable`

- [ ] **Step 5: Commit**

---

### Task 7: `headless_runner.h` + `headless_runner.cpp` — Batch processing

**Files:**
- Create: `apps/seat-aoi/headless_runner.h`
- Create: `apps/seat-aoi/headless_runner.cpp`
- Modify: `apps/seat-aoi/main.cpp` (remove run_headless lambda + two headless entry branches)

**Interfaces:**
- Produces: `int RunHeadless(const CliArgs& cli, AssembledApp& app)`

Source: `main.cpp` L921-1081 (`run_headless` lambda) + L1040-1081 (two headless entry branches: dataset_path and image_dir)

- [ ] **Step 1: Write `headless_runner.h`**

```cpp
#pragma once

struct CliArgs;
struct AssembledApp;

int RunHeadless(const CliArgs& cli, AssembledApp& app);
```

- [ ] **Step 2: Write `headless_runner.cpp`** — extract run_headless lambda (L922-1038) as a named function. Then implement `RunHeadless`:
  1. If `!cli.dataset_path.empty()`: load via `io::BasicImporter::ImportDataset`, call lambda
  2. Else if `!cli.image_dir.empty()`: scan directory for image files, wrap in `DirEntry`, call lambda
  - Keep the `if constexpr (requires { entry.surface_id; })` pattern for metadata stamping — it distinguishes dataset entries from raw directory entries
  - Keep all verdict parsing, per-surface aggregation, summary output verbatim
  - The `constexpr` if-branches that stamp metadata (L940-943, L992-999) must differentiate between DatasetEntry (has surface_id/position_id/light_id) and DirEntry (which doesn't)

- [ ] **Step 3: Modify `main.cpp`** — remove L921-1081, add `#include "headless_runner.h"`

- [ ] **Step 4: Update `CMakeLists.txt`** — add `headless_runner.cpp`

- [ ] **Step 5: Commit**

---

### Task 8: `gui_runner.h` + `gui_runner.cpp` — GUI mode

**Files:**
- Create: `apps/seat-aoi/gui_runner.h`
- Create: `apps/seat-aoi/gui_runner.cpp`
- Modify: `apps/seat-aoi/main.cpp` (remove GUI block L1086-1173)

**Interfaces:**
- Produces: `int RunGui(int argc, char* argv[], AssembledApp& app)`

Source: `main.cpp` L1086-1173

- [ ] **Step 1: Write `gui_runner.h`**

```cpp
#pragma once

struct AssembledApp;

int RunGui(int argc, char* argv[], AssembledApp& app);
```

- [ ] **Step 2: Write `gui_runner.cpp`** — extract L1086-1173. Key:
  - Create `QGuiApplication` + `QQmlApplicationEngine`
  - Create ViewModels (`PipelineViewModel`, `InspectionViewModel`, `ConfigViewModel`, `DashboardViewModel`, `FrameProvider`)
  - Set result callback: if `app.evolution` has value → evolution-aware callback; else → simple callback
  - Wire FakeCamera → frame callback → pipeline Submit
  - `aboutToQuit`: pipeline_vm->StopRefresh(), camera stop, pipeline Drain, evolution Stop, tuning Join, pipeline Stop, ctx Stop
  - `return app.exec()`

- [ ] **Step 3: Modify `main.cpp`** — remove L1086-1173, add `#include "gui_runner.h"`

- [ ] **Step 4: Update `CMakeLists.txt`** — add `gui_runner.cpp`

- [ ] **Step 5: Commit**

---

### Task 9: Finalize `main.cpp` — Slim to ~30 lines

**Files:**
- Modify: `apps/seat-aoi/main.cpp` (rewrite to ~30 lines)

**Interfaces:**
- Consumes: `cli_args.h`, `coreset_builder.h`, `app_builder.h`, `headless_runner.h`, `gui_runner.h`

- [ ] **Step 1: Rewrite `main.cpp`**

```cpp
#include <iostream>

#include "app_builder.h"
#include "app_config.h"
#include "cli_args.h"
#include "coreset_builder.h"
#include "gui_runner.h"
#include "headless_runner.h"

auto main(int argc, char* argv[]) -> int {
    auto cli = ParseArgs(argc, argv);

    // --dataset mode (without --coreset-output): build coreset and exit
    if (!cli.dataset_path.empty() && cli.coreset_output_path.empty()) {
        return BuildCoreset(cli);
    }

    // Assemble the full application
    auto app_result = AssembleApplication(cli);
    if (!app_result) {
        std::cerr << "Application assembly failed: "
                  << app_result.error().message << "\n";
        return 1;
    }
    auto app = std::move(*app_result);

    // Headless batch mode
    if (!cli.image_dir.empty() || !cli.dataset_path.empty()) {
        return RunHeadless(cli, app);
    }

    // GUI mode
    return RunGui(argc, argv, app);
}
```

- [ ] **Step 2: Verify `#include` cleanup** — `main.cpp` should no longer include any sai headers directly (only through our new headers). Remove all stale includes (L1-57) — only `<iostream>` + our 5 new headers remain.

- [ ] **Step 3: Update `CMakeLists.txt`** — final `add_executable` should list all .cpp files

- [ ] **Step 4: Commit**

---

### Task 10: Verification

**Note:** `seat_aoi` only builds on Linux + CUDA. On macOS this is a syntax-only review.

- [ ] **Step 1: Review each new file for correctness** — verify all includes are present, all types resolve, no code was dropped

- [ ] **Step 2: Verify main.cpp line count** — should be ~30 lines (not counting comments/blank lines)

- [ ] **Step 3: Diff check** — `git diff main.cpp` should show only removals; new files should contain the moved code

- [ ] **Step 4: Verify CMakeLists.txt** is correct

- [ ] **Step 5: Commit any review fixes**

---
