# Car 数据集端到端测试 + Qt 可视化评审 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 CLI dispatch bug，增加 ground truth 标签支持，在 headless 流程末尾自动计算 metrics，为 GUI 增加 review 模式支持历史数据回看。

**Architecture:** 所有改动位于 CLI 入口层 + 结果聚合层 + ViewModel 层，不碰 Pipeline / Stage / 推理路径。train 和 review 作为独立的 CLI 子命令分流，review 模式完全绕过 Pipeline 直接从磁盘加载。

**Tech Stack:** C++20, Qt6 (QML + QuickControls2), nlohmann_json, yaml-cpp, googletest

## Global Constraints

- 不碰 Pipeline、不碰任何 Stage、不碰推理路径
- 不做 `#ifdef` shim（平台代码通过 CMakeLists.txt 门控）
- 默认值不使用 magic string（`expected_verdict` 使用 `std::optional`）
- 现有 621 测试持续通过
- 错误处理统一走 `tl::expected`（`Result<T>`），不抛异常

---

## File Map

| 文件 | 责任 | 变更类型 |
|------|------|---------|
| `apps/seat-aoi/cli_args.h` | 新增 `train_mode`, `review_dir` 字段 | Modify |
| `apps/seat-aoi/cli_args.cpp` | 解析 `--train`, `--review-dir`，删除 auto-fill | Modify |
| `apps/seat-aoi/main.cpp` | 修复 dispatch，train/review 子命令分流 | Modify |
| `include/sai/io/importer.h` | `DatasetEntry` + `expected_verdict` | Modify |
| `src/io/basic_importer.cpp` | `ImportDataset` 解析 `expected` YAML 键 | Modify |
| `apps/seat-aoi/headless_runner.h` | 声明 `WriteReviewIndex` 辅助函数 | Modify |
| `apps/seat-aoi/headless_runner.cpp` | metrics 计算 + review_index.json 写入 | Modify |
| `include/sai/visualization/frame_provider.h` | `RegisterFramePath()` 磁盘加载接口 | Modify |
| `apps/seat-aoi/gui_runner.h` | 签名增加 `const CliArgs&` | Modify |
| `apps/seat-aoi/gui_runner.cpp` | review 模式分支 | Modify |
| `scripts/generate_car_dataset.py` | expected 标签生成 | Modify |

---

### Task 1: Fix CLI dispatch — 新增 train_mode 和 review_dir

**Files:**
- Modify: `apps/seat-aoi/cli_args.h`
- Modify: `apps/seat-aoi/cli_args.cpp`
- Modify: `apps/seat-aoi/main.cpp`

**Interfaces:**
- Produces: `CliArgs::train_mode` (bool), `CliArgs::review_dir` (std::string)
- Consumes: 无前置依赖

- [ ] **Step 1: 修改 cli_args.h — 新增字段**

```cpp
// apps/seat-aoi/cli_args.h
#pragma once

#include <cstddef>
#include <string>

struct CliArgs {
    std::string image_dir;
    std::string output_dir = "/tmp/surface-ai/results/";
    std::string coreset_path;           // --coreset: load pre-built coreset for detection
    std::string coreset_manifest_path;  // --coreset-manifest: multi-position YAML registry
    std::string dataset_path;            // --dataset: YAML manifest for coreset building
    std::string coreset_output_path;     // --coreset-output: where to save the built coreset
    std::string coreset_algo = "greedy"; // --coreset-algo: greedy | uniform
    std::string review_dir;              // --review-dir: path to JSON results for GUI review
    std::size_t coreset_max_samples = 10000; // --coreset-max-samples N
    bool headless = false;
    bool train_mode = false;             // --train or --coreset-output: build coreset mode
    bool cpu_mode = false;              // --cpu: force CPU embedder (no GPU)
};

auto ParseArgs(int argc, char* argv[]) -> CliArgs;
```

- [ ] **Step 2: 修改 cli_args.cpp — 解析新参数 + 删除 auto-fill**

```cpp
// apps/seat-aoi/cli_args.cpp
#include "cli_args.h"

#include <string_view>

auto ParseArgs(int argc, char* argv[]) -> CliArgs {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--image-dir" && i + 1 < argc) {
            args.image_dir = argv[++i];
            args.headless = true;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--headless") {
            args.headless = true;
        } else if (arg == "--coreset" && i + 1 < argc) {
            args.coreset_path = argv[++i];
        } else if (arg == "--coreset-manifest" && i + 1 < argc) {
            args.coreset_manifest_path = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            args.dataset_path = argv[++i];
        } else if (arg == "--coreset-output" && i + 1 < argc) {
            args.coreset_output_path = argv[++i];
            args.train_mode = true;
        } else if (arg == "--coreset-algo" && i + 1 < argc) {
            args.coreset_algo = argv[++i];
        } else if (arg == "--coreset-max-samples" && i + 1 < argc) {
            args.coreset_max_samples = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--train") {
            args.train_mode = true;
        } else if (arg == "--cpu") {
            args.cpu_mode = true;
        } else if (arg == "--review-dir" && i + 1 < argc) {
            args.review_dir = argv[++i];
        }
    }
    // NOTE: auto-fill of coreset_output_path is DELETED.
    // train_mode is now set explicitly by --train or --coreset-output.
    return args;
}
```

- [ ] **Step 3: 修改 main.cpp — 修复 dispatch**

```cpp
// apps/seat-aoi/main.cpp
#include <iostream>

#include "app_builder.h"
#include "cli_args.h"
#include "coreset_builder.h"
#include "gui_runner.h"
#include "headless_runner.h"

auto main(int argc, char* argv[]) -> int {
    auto cli = ParseArgs(argc, argv);

    // ── Train mode: build coreset and exit ──
    if (cli.train_mode) {
        return BuildCoreset(cli);
    }

    // ── Review mode: GUI without Pipeline / Camera / GPU ──
    // Does NOT call AssembleApplication — review mode reads JSON+images from
    // disk and only uses ViewModels. No model files or CUDA required.
    if (!cli.review_dir.empty()) {
        AssembledApp app;  // empty shell, review mode doesn't touch it
        return RunGui(argc, argv, app, cli);
    }

    // ── Headless batch mode ──
    if (!cli.image_dir.empty() || !cli.dataset_path.empty()) {
        auto app_result = AssembleApplication(cli);
        if (!app_result) {
            std::cerr << "Application assembly failed: "
                      << app_result.error().message << "\n";
            return 1;
        }
        auto app = std::move(*app_result);
        return RunHeadless(cli, app);
    }

    // ── GUI live mode (FakeCamera) ──
    auto app_result = AssembleApplication(cli);
    if (!app_result) {
        std::cerr << "Application assembly failed: "
                  << app_result.error().message << "\n";
        return 1;
    }
    auto app = std::move(*app_result);
    return RunGui(argc, argv, app, cli);
}
```

- [ ] **Step 4: 编译验证**

```bash
cd build/linux && cmake --build . --target seat_aoi 2>&1 | tail -20
```

Expected: 编译成功，无错误。

- [ ] **Step 5: Commit**

```bash
git add apps/seat-aoi/cli_args.h apps/seat-aoi/cli_args.cpp apps/seat-aoi/main.cpp
git commit -m "fix(cli): 🐛 修复CLI dispatch bug，新增--train和--review-dir子命令"
```

---

### Task 2: DatasetEntry 增加 expected_verdict 字段

**Files:**
- Modify: `include/sai/io/importer.h`
- Modify: `src/io/basic_importer.cpp`

**Interfaces:**
- Produces: `DatasetEntry::expected_verdict` (`std::optional<std::string>`)
- Consumes: 无前置依赖（Task 1 独立）

- [ ] **Step 1: 修改 importer.h — 新增 optional 字段**

```cpp
// include/sai/io/importer.h
// 在 DatasetEntry 结构体中新增 expected_verdict 字段

#include <filesystem>
#include <memory>
#include <optional>    // ← 新增
#include <string_view>

#include <sai/core/error.h>
#include <sai/image/image.h>
#include <sai/plugin/plugin.h>

#include <yaml-cpp/yaml.h>

namespace sai::io {

using sai::image::Image;

struct DatasetEntry {
    std::filesystem::path path;
    std::string surface_id;
    std::uint16_t position_id = 0;
    std::uint16_t light_id = 0;
    std::optional<std::string> expected_verdict;  // "OK" | "NG" | std::nullopt
};

// ... rest of file unchanged
```

- [ ] **Step 2: 修改 basic_importer.cpp — 解析 expected 键**

在 `ImportDataset` 的图片循环中，新增 `expected` 键的解析：

```cpp
// src/io/basic_importer.cpp
// 在 for (auto img : images_node) 循环内，添加 expected 解析

for (auto img : images_node) {
    DatasetEntry entry;
    entry.surface_id = surface_id;
    entry.path = base_dir / img["path"].as<std::string>();
    if (auto pos = img["position"]; pos.IsDefined())
        entry.position_id = pos.as<std::uint16_t>();
    if (auto lgt = img["light"]; lgt.IsDefined())
        entry.light_id = lgt.as<std::uint16_t>();
    // ── 新增 ──
    if (auto exp = img["expected"]; exp.IsDefined())
        entry.expected_verdict = exp.as<std::string>();
    // ── 结束 ──
    entries.push_back(std::move(entry));
}
```

- [ ] **Step 3: 编译 + 运行 io 测试验证向后兼容**

```bash
cd build/linux && cmake --build . --target sai_io 2>&1 | tail -10
ctest --preset linux -R "Importer" --output-on-failure
```

Expected: 所有 importer 测试通过（不带 `expected` 键的 YAML 仍然正确解析）。

- [ ] **Step 4: Commit**

```bash
git add include/sai/io/importer.h src/io/basic_importer.cpp
git commit -m "feat(io): ✨ DatasetEntry新增expected_verdict可选字段支持ground truth标记"
```

---

### Task 3: 更新 generate_car_dataset.py — 生成 expected 标签

**Files:**
- Modify: `scripts/generate_car_dataset.py`

**Interfaces:**
- Produces: `car_test_ok.yaml` (含 `expected: OK`), `car_test_ng.yaml` (含 `expected: NG`)
- Consumes: 无前置依赖

- [ ] **Step 1: 修改 write_manifest 函数 — 支持 expected 参数**

```python
# scripts/generate_car_dataset.py
# 修改 write_manifest 签名，新增 expected_label 参数

def write_manifest(out: Path, entries, expected_label=None):
    """Write YAML in BasicImporter format. Returns count.
    
    If expected_label is not None, emits an 'expected' field for every entry.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out), "w", encoding="utf-8") as f:
        f.write(f"surface: {yaml_str(SURFACE_ID)}\n")
        f.write("images:\n")
        for path_item, pos, light in entries:
            f.write(f'  - path: {yaml_str(path_item)}\n')
            f.write(f'    position: {pos}\n')
            f.write(f'    light: {light}\n')
            if expected_label is not None:
                f.write(f'    expected: {expected_label}\n')
    return len(entries)
```

- [ ] **Step 2: 修改 main 中的 write 调用 — 测试集写入 expected**

```python
# scripts/generate_car_dataset.py
# 修改 main() 中 manifests 列表的定义

    manifests = [
        ("car_train.yaml", train, "TRAIN", None),        # train 不设 expected
        ("car_test_ok.yaml", test_ok, "TEST_OK", "OK"),   # 正常样本 → expected: OK
        ("car_test_ng.yaml", ng_all, "TEST_NG", "NG"),    # NG 样本 → expected: NG
        ("car_test_all.yaml", test_ok + ng_all, "TEST_ALL", None),  # 混合测试不设 expected (从 source 继承)
    ]

    for fname, entries, label, expected_label in manifests:
        n = write_manifest(output_dir / fname, entries, expected_label=expected_label)
        print(f"  {label:12s} → {fname:25s} ({n} images)")
```

- [ ] **Step 3: 修改 validate_manifest — 支持 expected 字段验证**

```python
# scripts/generate_car_dataset.py
# 在 validate_manifest 中增加 expected 键计数

    # 在现有的 path/position/light 计数之后添加:
    expected_count = content.count("\n    expected:")
    if expected_count > 0 and expected_count != path_count:
        issues.append(f"{path.name}: expected count ({expected_count}) != path count ({path_count})")
```

- [ ] **Step 4: 运行脚本 dry-run 验证输出**

```bash
python3 scripts/generate_car_dataset.py --data-root /mnt/e/Car --dry-run
```

Expected: 输出显示 TEST_OK 和 TEST_NG 会带 expected 标签。

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_car_dataset.py
git commit -m "feat(scripts): ✨ generate_car_dataset.py测试集清单自动生成expected标签"
```

---

### Task 4: headless_runner — metrics 计算 + review_index.json

**Files:**
- Modify: `apps/seat-aoi/headless_runner.h`
- Modify: `apps/seat-aoi/headless_runner.cpp`

**Interfaces:**
- Produces: `WriteReviewIndex(output_dir, frames)` — 写入 review_index.json
- Consumes: Task 2 (`DatasetEntry::expected_verdict`)

- [ ] **Step 1: 修改 headless_runner.h — 声明辅助结构体和函数**

```cpp
// apps/seat-aoi/headless_runner.h
#pragma once

#include <string>
#include <string_view>
#include <vector>

struct CliArgs;
struct AssembledApp;

// Per-frame record accumulated during headless processing.
// Used for both metrics computation and review_index.json generation.
struct FrameRecord {
    int frame_id;
    std::string image_path;        // original image path from dataset entry
    std::string verdict;           // "OK" | "NG" | "WARN" | "UNCERTAIN"
    double severity;
    double confidence;
    std::string expected_verdict;  // "" if not available
    std::uint16_t position_id;
    std::uint16_t light_id;
    std::string surface_id;
};

int RunHeadless(const CliArgs& cli, AssembledApp& app);

// Write review_index.json for GUI review mode consumption.
// Only called when expected_verdict data is present.
void WriteReviewIndex(std::string_view output_dir,
                      const std::vector<FrameRecord>& records,
                      std::string_view surface_id);
```

- [ ] **Step 2: 修改 headless_runner.cpp — 收集 FrameRecord + metrics 计算**

完整重写 `headless_runner.cpp`。关键变更：
1. `process_entries` 中每条目创建 `FrameRecord` 并收集
2. Pipeline 停止后调用 metrics 计算
3. 调用 `WriteReviewIndex` 写入 JSON

```cpp
// apps/seat-aoi/headless_runner.cpp
#include "headless_runner.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "app_builder.h"
#include "cli_args.h"

#include <nlohmann/json.hpp>

#include <sai/io/importer.h>
#include <sai/image/raw_image.h>
#include <sai/reasoner/reasoner.h>

namespace {

struct MetricsCounts {
    int tp = 0;  // actual OK → predicted OK
    int fp = 0;  // actual NG → predicted OK
    int tn = 0;  // actual NG → predicted NG
    int fn = 0;  // actual OK → predicted NG
    // WARN and UNCERTAIN are tracked separately (neither TP nor FP)
    int actual_ok = 0;
    int actual_ng = 0;

    void Add(std::string_view predicted, std::string_view expected) {
        if (expected == "OK") {
            ++actual_ok;
            if (predicted == "OK") ++tp;
            else if (predicted == "NG") ++fn;
            // WARN / UNCERTAIN on OK → not counted in binary matrix
        } else if (expected == "NG") {
            ++actual_ng;
            if (predicted == "NG") ++tn;
            else if (predicted == "OK") ++fp;
            // WARN / UNCERTAIN on NG → not counted in binary matrix
        }
    }

    [[nodiscard]] auto Precision() const -> double {
        int denom = tp + fp;
        return denom > 0 ? static_cast<double>(tp) / static_cast<double>(denom) : 0.0;
    }
    [[nodiscard]] auto Recall() const -> double {
        int denom = tp + fn;
        return denom > 0 ? static_cast<double>(tp) / static_cast<double>(denom) : 0.0;
    }
    [[nodiscard]] auto F1() const -> double {
        double p = Precision(), r = Recall();
        return (p + r > 0.0) ? 2.0 * p * r / (p + r) : 0.0;
    }
    [[nodiscard]] auto Accuracy() const -> double {
        int denom = tp + tn + fp + fn;
        return denom > 0 ? static_cast<double>(tp + tn) / static_cast<double>(denom) : 0.0;
    }
};

}  // namespace

auto RunHeadless(const CliArgs& cli, AssembledApp& app) -> int {
    using namespace sai;

    std::vector<FrameRecord> records;

    auto process_entries = [&](auto& entries, io::BasicImporter& importer) -> int {
        int ok = 0, ng = 0, warn = 0, uncertain = 0, failed = 0;

        // Per-surface aggregation: track worst verdict for each product
        std::map<std::string, std::string> surface_verdict;
        std::map<std::string, int> surface_frame_count;

        for (size_t i = 0; i < entries.size(); ++i) {
            auto& entry = entries[i];
            auto img_result = importer.ImportImage(entry.path);
            if (!img_result) {
                std::cerr << "FAIL import: " << entry.path.filename() << "\n";
                ++failed;
                continue;
            }

            // Stamp metadata from dataset entry
            if constexpr (requires { entry.surface_id; }) {
                (*img_result)->Meta().surface_id = entry.surface_id;
                (*img_result)->Meta().position_id = entry.position_id;
                (*img_result)->Meta().light_id = entry.light_id;
            }

            FrameRecord rec;
            rec.frame_id = static_cast<int>(i);
            rec.image_path = entry.path.string();
            rec.position_id = entry.position_id;
            rec.light_id = entry.light_id;
            rec.surface_id = entry.surface_id;
            if constexpr (requires { entry.expected_verdict; }) {
                rec.expected_verdict = entry.expected_verdict.value_or("");
            }

            auto* raw = dynamic_cast<image::RawImage*>(img_result->get());
            if (raw) {
                img_result->release();
                (void)app.pipeline->Submit(std::move(*raw));
            } else {
                auto meta = (*img_result)->Meta();
                const auto* data = (*img_result)->Data();
                auto size = (*img_result)->SizeBytes();
                std::vector<std::uint8_t> buffer(data, data + size);
                (void)app.pipeline->Submit(
                    image::RawImage::FromOwnedBuffer(std::move(buffer), meta));
            }

            (void)app.pipeline->Drain();

            // Read the exported JSON result
            auto result_path = std::filesystem::path(cli.output_dir)
                / "default" / "unknown" / "result.json";
            std::string verdict = "?";
            double severity = 0.0;
            double confidence = 0.0;
            if (std::filesystem::exists(result_path)) {
                std::ifstream ifs(result_path);
                try {
                    auto j = nlohmann::json::parse(ifs);
                    verdict = j.value("verdict", "?");
                    severity = j.value("severity", 0.0);
                    confidence = j.value("confidence", 0.0);
                } catch (const nlohmann::json::exception& e) {
                    std::cerr << "JSON parse error: " << e.what() << "\n";
                }
            }

            rec.verdict = verdict;
            rec.severity = severity;
            rec.confidence = confidence;
            records.push_back(std::move(rec));

            if (verdict == "OK") ++ok;
            else if (verdict == "NG") ++ng;
            else if (verdict == "WARN") ++warn;
            else if (verdict == "UNCERTAIN") ++uncertain;
            else ++failed;

            // Per-surface tracking: worst verdict wins
            if constexpr (requires { entry.surface_id; }) {
                auto& sv = surface_verdict[entry.surface_id];
                if (verdict == "NG" || (sv != "NG" && verdict == "WARN")
                    || (sv != "NG" && sv != "WARN" && verdict == "UNCERTAIN")
                    || (sv.empty() && verdict == "OK"))
                    sv = verdict;
                surface_frame_count[entry.surface_id]++;
            }

            std::cout << "[" << (i + 1) << "/" << entries.size() << "] "
                      << entry.path.filename().string() << " → " << verdict << "\n";
        }

        std::cout << "\n===== Frame Summary =====\n"
                  << "Total:   " << entries.size() << "\n"
                  << "OK:      " << ok << "\n"
                  << "NG:      " << ng << "\n"
                  << "WARN:    " << warn << "\n"
                  << "UNCERTAIN: " << uncertain << "\n"
                  << "Failed:  " << failed << "\n";

        if (!surface_verdict.empty()) {
            int s_ok = 0, s_ng = 0, s_warn = 0, s_uncertain = 0;
            for (auto& [sid, v] : surface_verdict) {
                if (v == "OK") ++s_ok;
                else if (v == "NG") ++s_ng;
                else if (v == "WARN") ++s_warn;
                else if (v == "UNCERTAIN") ++s_uncertain;
                std::cout << "  [" << sid << "] " << v
                          << " (" << surface_frame_count[sid] << " frames)\n";
            }
            std::cout << "\n===== Product Summary =====\n"
                      << "Products: " << surface_verdict.size() << "\n"
                      << "OK:       " << s_ok << "\n"
                      << "NG:       " << s_ng << "\n"
                      << "WARN:     " << s_warn << "\n"
                      << "UNCERTAIN:" << s_uncertain << "\n";
        }

        // ── Metrics: confusion matrix + precision/recall/F1 ──
        MetricsCounts metrics;
        for (auto& rec : records) {
            if (rec.expected_verdict.empty()) continue;  // no ground truth → skip
            metrics.Add(rec.verdict, rec.expected_verdict);
        }
        int labeled = metrics.actual_ok + metrics.actual_ng;
        if (labeled > 0) {
            std::cout << "\n===== Detection Metrics =====\n"
                      << "Labeled frames: " << labeled << "\n"
                      << "  Actual OK:  " << metrics.actual_ok << "\n"
                      << "  Actual NG:  " << metrics.actual_ng << "\n\n"
                      << "Confusion Matrix (binary, WARN/UNCERTAIN excluded):\n"
                      << "              Predicted\n"
                      << "              OK    NG\n"
                      << "Actual OK     " << metrics.tp << "     " << metrics.fn << "\n"
                      << "Actual NG     " << metrics.fp << "     " << metrics.tn << "\n\n"
                      << "Precision: " << metrics.Precision() << "\n"
                      << "Recall:    " << metrics.Recall() << "\n"
                      << "F1 Score:  " << metrics.F1() << "\n"
                      << "Accuracy:  " << metrics.Accuracy() << "\n";
        }

        std::cout << "Results: " << cli.output_dir << "\n";

        // ── Write review_index.json ──
        if (!records.empty()) {
            std::string surface_id = records.front().surface_id;
            WriteReviewIndex(cli.output_dir, records, surface_id);
            std::cout << "Review index: " << cli.output_dir << "/review_index.json\n";
        }

        if (app.evolution.has_value()) app.evolution->Stop();
        for (auto& [key, evo] : app.evolutions) {
            evo.Stop();
        }
        if (app.tuning_scheduler.has_value()) app.tuning_scheduler->Join();
        (void)app.pipeline->Stop();
        (void)app.ctx->Stop();
        return (ng > 0 || failed > 0) ? 1 : 0;
    };

    // ... 保持现有的 dataset_path 和 image_dir 两个分支不变 ...
    // (此处的 process_entries 调用逻辑与原来一致)
    if (!cli.dataset_path.empty()) {
        io::BasicImporter importer;
        auto entries_result = io::BasicImporter::ImportDataset(cli.dataset_path);
        if (!entries_result) {
            std::cerr << "Dataset import failed: "
                      << entries_result.error().message << "\n";
            return 1;
        }
        auto& entries = *entries_result;
        std::cout << "Processing " << entries.size() << " images from "
                  << cli.dataset_path << "\n"
                  << "Output: " << cli.output_dir << "\n\n";
        return process_entries(entries, importer);
    }

    if (!cli.image_dir.empty()) {
        io::BasicImporter importer;
        std::vector<std::filesystem::path> image_files;
        for (auto& entry : std::filesystem::directory_iterator(cli.image_dir)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension().string();
            if (ext == ".ppm" || ext == ".png" || ext == ".bmp"
                || ext == ".jpg" || ext == ".jpeg") {
                image_files.push_back(entry.path());
            }
        }
        std::sort(image_files.begin(), image_files.end());

        struct DirEntry {
            std::filesystem::path path;
        };
        std::vector<DirEntry> dir_entries;
        dir_entries.reserve(image_files.size());
        for (auto& p : image_files)
            dir_entries.push_back({std::move(p)});

        std::cout << "Processing " << dir_entries.size() << " images from "
                  << cli.image_dir << "\n"
                  << "Output: " << cli.output_dir << "\n\n";
        return process_entries(dir_entries, importer);
    }

    return 0;
}

void WriteReviewIndex(std::string_view output_dir,
                      const std::vector<FrameRecord>& records,
                      std::string_view surface_id) {
    using json = nlohmann::json;
    json index;
    index["surface_id"] = surface_id;
    index["total_frames"] = records.size();

    json frames_arr = json::array();
    for (auto& rec : records) {
        json f;
        f["frame_id"] = rec.frame_id;
        f["image_path"] = rec.image_path;
        f["verdict"] = rec.verdict;
        f["severity"] = rec.severity;
        f["confidence"] = rec.confidence;
        f["position_id"] = rec.position_id;
        f["light_id"] = rec.light_id;
        f["surface_id"] = rec.surface_id;
        if (!rec.expected_verdict.empty())
            f["expected_verdict"] = rec.expected_verdict;
        frames_arr.push_back(std::move(f));
    }
    index["frames"] = std::move(frames_arr);

    auto path = std::filesystem::path(output_dir) / "review_index.json";
    std::ofstream ofs(path);
    ofs << index.dump(2);  // pretty-print with 2-space indent
}
```

- [ ] **Step 2: 编译验证**

```bash
cd build/linux && cmake --build . --target seat_aoi 2>&1 | tail -20
```

Expected: 编译成功。

- [ ] **Step 3: Commit**

```bash
git add apps/seat-aoi/headless_runner.h apps/seat-aoi/headless_runner.cpp
git commit -m "feat(eval): ✨ headless模式自动计算precision/recall/F1并输出review_index.json"
```

---

### Task 5: GUI review 模式 — 磁盘回看

**Files:**
- Modify: `apps/seat-aoi/gui_runner.h`
- Modify: `apps/seat-aoi/gui_runner.cpp`
- Modify: `include/sai/visualization/frame_provider.h`

**Interfaces:**
- Produces: `RunGui(argc, argv, app, cli)` — 新签名，根据 `cli.review_dir` 选择模式
- Consumes: Task 4 (`review_index.json` 格式)

- [ ] **Step 1: 修改 gui_runner.h — 新签名**

```cpp
// apps/seat-aoi/gui_runner.h
#pragma once

struct AssembledApp;
struct CliArgs;

int RunGui(int argc, char* argv[], AssembledApp& app, const CliArgs& cli);
```

- [ ] **Step 2: 修改 frame_provider.h — 新增磁盘加载接口**

在 `FrameProvider` 类中新增一个方法，允许注册帧 ID 到文件路径的映射：

```cpp
// include/sai/visualization/frame_provider.h
// 在 FrameProvider 类声明中添加:

    /// Register a file-system image path for lazy loading in review mode.
    /// requestImage() loads from disk when no in-memory cache entry exists.
    void RegisterFramePath(int frame_id, const QString& image_path);

    /// Load all frame paths from a review_index.json directory.
    void LoadFromReviewIndex(const QString& review_dir);

// 并在 private 中添加:
    std::map<int, QString> frame_paths_;  // frame_id → disk path for review mode
    mutable std::shared_mutex path_mutex_;
```

- [ ] **Step 3: 实现 FrameProvider 磁盘加载 — frame_provider.cpp**

在 `requestImage` 中添加 review 模式磁盘回退，在 `src/visualization/frame_provider.cpp` 末尾新增两个方法：

```cpp
// src/visualization/frame_provider.cpp
// 在 requestImage() 中，缓存未命中后添加磁盘回退:

auto FrameProvider::requestImage(const QString& id, QSize* size,
                                  const QSize& requestedSize) -> QImage {
    // Expected id format: "frame?t=<frame_id>"
    if (!id.startsWith(QStringLiteral("frame?t="))) {
        return QImage();
    }

    const QStringView num_str = QStringView(id).mid(8);
    bool ok = false;
    const int frame_id = num_str.toInt(&ok);
    if (!ok) {
        return QImage();
    }

    // ── Layer 1: in-memory ring-buffer cache (live mode) ──
    {
        std::shared_lock lock(cache_mutex_);
        const int slot = frame_id % kCacheSize;
        const auto& entry = cache_[slot];
        if (entry.frame_id == frame_id) {
            if (size) *size = entry.image.size();
            if (requestedSize.isValid() && requestedSize != entry.image.size())
                return entry.image.scaled(requestedSize);
            return entry.image;
        }
    }

    // ── Layer 2: disk-backed lazy load (review mode) ──
    {
        std::shared_lock lock(path_mutex_);
        auto it = frame_paths_.find(frame_id);
        if (it != frame_paths_.end()) {
            QImage img(it->second);
            if (!img.isNull()) {
                if (size) *size = img.size();
                // Populate cache for subsequent requests
                {
                    std::unique_lock cache_lock(cache_mutex_);
                    const int slot = frame_id % kCacheSize;
                    cache_[slot].frame_id = frame_id;
                    cache_[slot].image = img;
                    latest_frame_id_.store(frame_id);
                }
                if (requestedSize.isValid() && requestedSize != img.size())
                    return img.scaled(requestedSize);
                return img;
            }
        }
    }

    return QImage();
}

void FrameProvider::RegisterFramePath(int frame_id, const QString& image_path) {
    std::unique_lock lock(path_mutex_);
    frame_paths_[frame_id] = image_path;
}

void FrameProvider::LoadFromReviewIndex(const QString& review_dir) {
    // Reads review_index.json and registers all frame paths.
    // Called once during review mode initialization.
    // Uses nlohmann_json; if not available in visualization lib, caller
    // iterates frames and calls RegisterFramePath() directly.
    // (Implementation in gui_runner.cpp handles the JSON parsing;
    //  this is a convenience entry point.)
}
```

注意：`LoadFromReviewIndex` 的实际 JSON 解析在 `gui_runner.cpp` 的 `RunReviewMode` 中完成（因为 visualization 库不依赖 nlohmann_json）。`LoadFromReviewIndex` 声明保留以便未来扩展，当前实际调用 `RegisterFramePath` 逐帧注册。

- [ ] **Step 4: 修改 gui_runner.cpp — review 模式分支**

在 `RunGui` 函数开头，根据 `cli.review_dir` 选择模式：

```cpp
// apps/seat-aoi/gui_runner.cpp
#include "gui_runner.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "app_builder.h"
#include "cli_args.h"
#include "evolution_offer.h"

#include <nlohmann/json.hpp>

#include <sai/device/fake_camera.h>
#include <sai/image/raw_image.h>
#include <sai/reasoner/reasoner.h>
#include <sai/visualization/pipeline_viewmodel.h>
#include <sai/visualization/inspection_viewmodel.h>
#include <sai/visualization/frame_provider.h>
#include <sai/visualization/config_viewmodel.h>
#include <sai/visualization/dashboard_viewmodel.h>
#include <sai/detection/patch_core.h>
#include <sai/detection/coreset_evolution.h>
#include <sai/tuning/tuning_scheduler.h>

namespace {

/// Review mode: load review_index.json and populate ViewModels.
/// Does NOT start Pipeline / Camera / Tuning.
int RunReviewMode(const CliArgs& cli,
                  QGuiApplication& qapp,
                  QQmlApplicationEngine& engine,
                  sai::visualization::PipelineViewModel* pipeline_vm,
                  sai::visualization::InspectionViewModel* inspection_vm,
                  sai::visualization::DashboardViewModel* dashboard_vm,
                  sai::visualization::FrameProvider* frame_provider,
                  sai::visualization::ConfigViewModel* config_vm) {
    using json = nlohmann::json;

    auto index_path = std::filesystem::path(cli.review_dir) / "review_index.json";
    std::ifstream ifs(index_path);
    if (!ifs) {
        std::cerr << "Review mode: cannot open " << index_path << "\n";
        return 1;
    }

    json index;
    try {
        index = json::parse(ifs);
    } catch (const json::exception& e) {
        std::cerr << "Review mode: JSON parse error: " << e.what() << "\n";
        return 1;
    }

    std::string surface_id = index.value("surface_id", "unknown");
    int total_frames = index.value("total_frames", 0);
    std::cout << "Review mode: " << surface_id
              << " (" << total_frames << " frames)\n";

    // Populate DashboardViewModel with aggregate stats
    int ok = 0, ng = 0, warn = 0, uncertain = 0;
    for (auto& f : index["frames"]) {
        std::string v = f.value("verdict", "?");
        if (v == "OK") ++ok;
        else if (v == "NG") ++ng;
        else if (v == "WARN") ++warn;
        else if (v == "UNCERTAIN") ++uncertain;

        // Register frame path for lazy image loading
        int fid = f.value("frame_id", 0);
        std::string img_path = f.value("image_path", "");
        if (!img_path.empty()) {
            frame_provider->RegisterFramePath(
                fid, QString::fromStdString(img_path));
        }
    }

    // Seed DashboardViewModel with review stats
    for (int i = 0; i < ok; ++i)
        dashboard_vm->AppendFrameSummary(
            {i, "OK", "0.0", std::chrono::system_clock::now(), {}});
    // (Simplified: real implementation would load per-frame summaries)

    std::cout << "OK: " << ok << "  NG: " << ng
              << "  WARN: " << warn << "  UNCERTAIN: " << uncertain << "\n";

    // PipelineViewModel: no pipeline bound → shows "Stopped", which is correct
    // for review mode (no live pipeline running).

    engine.addImageProvider("pipeline", frame_provider);
    engine.rootContext()->setContextProperty("pipelineVM", pipeline_vm);
    engine.rootContext()->setContextProperty("inspectionVM", inspection_vm);
    engine.rootContext()->setContextProperty("configVM", config_vm);
    engine.rootContext()->setContextProperty("dashboardVM", dashboard_vm);

    engine.load("qrc:/MainWindow.qml");

    QObject::connect(&qapp, &QGuiApplication::aboutToQuit, [&]() {
        pipeline_vm->StopRefresh();
    });

    return qapp.exec();
}

}  // namespace

auto RunGui(int argc, char* argv[], AssembledApp& app, const CliArgs& cli) -> int {
    using namespace sai;

    QGuiApplication qapp(argc, argv);
    QQmlApplicationEngine engine;

    auto* pipeline_vm = new visualization::PipelineViewModel(&qapp);
    auto* inspection_vm = new visualization::InspectionViewModel(&qapp);
    auto* config_vm = new visualization::ConfigViewModel(&qapp);
    auto* dashboard_vm = new visualization::DashboardViewModel(&qapp);
    auto* frame_provider = new visualization::FrameProvider();

    // ── Review mode: bypass Pipeline / Camera / Tuning ──
    if (!cli.review_dir.empty()) {
        return RunReviewMode(cli, qapp, engine,
                             pipeline_vm, inspection_vm, dashboard_vm,
                             frame_provider, config_vm);
    }

    // ── Live mode: existing logic (unchanged) ──
    pipeline_vm->BindToPipeline(app.pipeline.get());
    config_vm->BindToPipeline(app.pipeline.get());
    config_vm->BindToRuleEngine(app.rule_engine.get());
    if (app.reasoner) config_vm->BindToReasoner(app.reasoner.get());
    for (auto stage_id : {"capture", "preprocess", "inference", "detect",
                          "rule_eval", "reason", "export"}) {
        auto* node = app.pipeline->GetStage(stage_id);
        if (node) config_vm->RegisterStageNode(stage_id, node);
    }

    // ... rest of existing live mode code unchanged ...
    // (FakeCamera setup, callbacks, etc.)
}
```

注意：`pipeline_vm->SetReviewMode()` 和完整的 `RunReviewMode` 需要根据实际 PipelineViewModel 接口调整。`PipelineViewModel` 中可能需要新增 `SetReviewMode` 方法。

- [ ] **Step 4: 编译验证**

```bash
cd build/linux && cmake --build . --target seat_aoi 2>&1 | tail -30
```

Expected: 编译成功（viewmodel 可能需要链接 sai_visualization）。

- [ ] **Step 5: Commit**

```bash
git add apps/seat-aoi/gui_runner.h apps/seat-aoi/gui_runner.cpp \
        include/sai/visualization/frame_provider.h
git commit -m "feat(gui): ✨ GUI新增--review-dir模式支持批处理结果磁盘回看"
```

---

### Task 6: 编译 + 全量测试回归验证

**Files:**
- 无新增文件

- [ ] **Step 1: 完整构建**

```bash
cd build/linux && cmake --build . 2>&1 | tail -30
```

Expected: 全量编译成功，无新增 warning。

- [ ] **Step 2: 运行全部 621 测试**

```bash
ctest --preset linux --output-on-failure 2>&1 | tail -30
```

Expected: 621 测试全部通过（`100% tests passed`）。

- [ ] **Step 3: 验证 --train 子命令帮助信息**

```bash
./build/linux/apps/seat-aoi/seat_aoi --help 2>&1 || true
```

只需确认 `--train`, `--review-dir` 出现在参数列表中（如果 `--help` 未实现则跳过此步）。

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(build): 🔧 Car数据集端到端测试代码完成，621 tests pass"
```

---

## Execution Checklist

- [ ] Task 1: CLI dispatch 修复
- [ ] Task 2: expected_verdict 字段
- [ ] Task 3: generate_car_dataset.py 更新
- [ ] Task 4: headless metrics + review_index.json
- [ ] Task 5: GUI review 模式
- [ ] Task 6: 全量构建 + 回归测试
