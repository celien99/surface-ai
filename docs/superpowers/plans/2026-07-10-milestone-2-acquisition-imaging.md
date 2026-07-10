# Milestone 2 (Acquisition & Imaging) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement milestone 2 (batches 2.1 Device, 2.2 Imaging, 2.3 IO) as real C++20 code on top of the frozen milestone-1 libraries, plus repay the three milestone-1 debts (D1/D2/D3), so a (software-triggered) camera frame can walk the preprocessing chain into a `SurfaceImage` and be exported as `result.json`.

**Architecture:** Three new libraries — `sai_device` (header-only INTERFACE: device abstractions + `RingBuffer<T>`), `sai_image` (STATIC: `Image` hierarchy + composable CPU preprocess chain; GPU path gated), `sai_io` (STATIC: pluggable `IExporter`/`IImporter` + built-in `JsonExporter`/`BasicImporter`). Each library follows milestone 1's **portable-part vs platform-gated-part** split: portable code builds and is tested on this macOS arm64 host via `cmake --build --preset default && ctest --preset default`; CUDA-gated code (`GpuImage`, the GPU marshaling step that repays debt D1) is written for the target (Ubuntu + NVIDIA) exactly per the frozen contract and simply excluded from the source list when `CUDAToolkit` is absent — no `#ifdef` dual implementation, no dev-machine fallback. The concrete GenICam/GigE camera driver and serial light-controller driver are **out of scope for this code phase** (need a vendor GenTL SDK + real hardware, unavailable and not meaningfully reviewable here); this milestone delivers the framework-side contract — interfaces, frame buffering, callback threading, and the full pipeline wiring — and verifies it end-to-end with an in-memory fake camera, exactly as the spec's validation point permits ("模拟一次触发（或软件触发）").

**Tech Stack:** C++20 (coroutines for the gated GPU path, templates, `std::function`). vcpkg additions: `nlohmann-json` (JSON report serialization). Existing deps reused: `yaml-cpp` (importer metadata), `tl-expected` (`Result<T>`), `gtest`, `spdlog` (debt fixes). CUDA Runtime API (gated, not installed here). No PNG library is added this milestone — annotated-image output is written as portable binary **PPM** as an interim stand-in; real PNG encoding is deferred to a future gated task.

## Global Constraints

- C++20 throughout, `sai::` namespace root. Batch sub-namespaces exactly per the spec's `## 4. Interfaces` blocks: `sai::device` (2.1), `sai::image` (2.2), `sai::io` (2.3). The plugin base types (`IPlugin`, `PluginManifest`, `Context`, `IModule`, `IReflectable`) live in the `sai` root namespace (verify against the existing headers) and are referenced, never redefined.
- **The frozen design baseline is `docs/superpowers/specs/2026-07-10-milestone-2-acquisition-imaging-design.md`.** Every class name, method signature, enum, and namespace must match its `## 4. Interfaces` blocks verbatim — *except* where a frozen signature provably cannot compile or cannot be implemented against the frozen milestone-1 types. In those cases, do NOT silently change it: implement the minimal viable adjustment AND record the deviation in the task report and in the deviation log (Task 11), per CLAUDE.md ("record the deviation explicitly rather than silently changing it"). The four known such deviations are pre-identified in the tasks below (`FromOwnedBuffer` addition, `MakeFlatField` param type, `MakeHDR` single-frame semantics, PNG→PPM interim) — implement exactly those, do not invent others without recording them.
- Reuse frozen milestone-1 signatures verbatim from the glossary and headers: `Result<T>`/`ErrorInfo`/`ErrorCode` (`include/sai/core/error.h`), `Resource` (`include/sai/core/resource.h`), `IPlugin`/`PluginManifest` (`include/sai/plugin/plugin.h`, `manifest.h`), `Context`/`IModule` (`include/sai/core/`), `IMemoryPool`/`PooledPtr<T>`/`MemoryPoolConfig` (`include/sai/memory/`), `PinnedPool`/`GpuPool` (gated), `GpuStreamQueue`/`CopyDirection`/`Task<T>` (`include/sai/runtime/`, gated). The `SAI_DECLARE_TYPE_ID(name)` macro (see `tests/plugin/fixture_plugin/fixture_plugin.cpp`) supplies `TypeId()` for plugin types.
- `ErrorCode` (`include/sai/core/error.h`) is a single flat enum; each task appends its new members **at the end** (append-only, never reorder or touch other batches' members), matching milestone 1's rule. The final landed order for milestone 2, in append order after the existing `Infra_ConfigKeyTypeMismatch`, is:
  - Task 3 (Device): `Device_ConnectionFailed`, `Device_NotConnected`, `Device_AcquisitionInProgress`
  - Task 4 (Image types): `Image_UnsupportedPixelFormat`, `Image_DimensionMismatch`
  - Task 5/6 (Preprocess): `Image_PreprocessFailed`
  - Task 8 (IO export): `Io_ExportPathCreateFailed`, `Io_SerializationFailed`
  - Task 9 (IO import): `Io_ImportFileNotFound`, `Io_ImportParseFailed`
- Coding style (enforced): avoid over-defensive code, avoid multi-level nesting (early return), chain `Result<T>` via `and_then`/`or_else`/`map` rather than nested `if (r.has_value())`. Preprocess `Compose` and any topological/tree walks are recursive, not nested loops. Design prose in reports must commit ("uses X because…"), never hedge with "supports A/B/C".
- **Platform gating, not conditional compilation.** Gated `.cpp` files (`gpu_image.cpp`, `gpu_preprocess.cpp`) contain exactly one target-platform implementation, no `#ifdef` alternate. Gate with `find_package(CUDAToolkit QUIET)` / `if(CUDAToolkit_FOUND)` in `src/image/CMakeLists.txt`, mirroring `src/memory/CMakeLists.txt` and `src/runtime/CMakeLists.txt` verbatim (including the `-mcx16` arch guard when a gated TU includes a 16-byte-atomic header). Headers may declare gated classes even when their `.cpp` isn't compiled, provided nothing in the portable test suite ODR-uses them.
- Every **portable** task ends with a real `cmake --preset default && cmake --build --preset default && ctest --preset default` run that must not reduce the currently-passing count (**84 tests** as of HEAD `06d380f`) — only add to it. Every **gated** task ends with a compile-review-only report (code written, read-reviewed against the API, not locally built) plus a rerun confirming that excluding the gated sources still configures/builds/passes the portable suite cleanly.
- Work on a dedicated branch `milestone-2-acquisition-imaging` off `main` (HEAD `06d380f`). Commit after every task with the repo's Gitmoji convention (`feat`/`fix`/`refactor`/`test`/`docs` + emoji + Chinese description, one space after emoji).

## File Structure Overview

```
include/sai/
├── core/error.h                    # Modify: append ErrorCode members across tasks (append-only)
├── infra/
│   ├── logger.h                    # Modify: D2 per-category DroppedCount (Task 2)
│   └── daily_and_size_sink.h       # Create: D3 composite rotating sink (Task 1)
├── device/
│   ├── device.h                    # Task 3: IDevice, State, Rect
│   ├── camera.h                    # Task 3: ICamera, TriggerMode, FrameCallback (fwd-decl RawImage)
│   ├── light_controller.h          # Task 3: ILightController, StrobeMode
│   └── ring_buffer.h               # Task 3: RingBuffer<T> (header-only template)
├── image/
│   ├── image.h                     # Task 4: Image (base), ImageMeta, PixelFormat
│   ├── raw_image.h                 # Task 4: RawImage (FromPool/FromBuffer/FromOwnedBuffer)
│   ├── surface_image.h             # Task 4: SurfaceImage (FromPool/FromPinned/FromOwnedBuffer)
│   ├── roi.h                       # Task 4: Rect alias, ROI, ROI::Apply
│   ├── preprocess.h                # Task 5/6: PreprocessFn, Compose, MakeXxx, CalibrationParams
│   └── gpu_image.h                 # Task 7 (gated): GpuImage
├── io/
│   ├── exporter.h                  # Task 8: DefectRecord, InspectionResult, IExporter, JsonExporter
│   └── importer.h                  # Task 9: IImporter, BasicImporter
src/
├── infra/{daily_and_size_sink.cpp?, logger.cpp}   # Task 1/2 (see task notes)
├── device/CMakeLists.txt           # Task 3 (INTERFACE lib, no .cpp)
├── image/
│   ├── CMakeLists.txt              # Task 4/5/6 (+ gated Task 7)
│   ├── image.cpp, raw_image.cpp, surface_image.cpp, roi.cpp   # Task 4
│   ├── preprocess.cpp              # Task 5/6
│   ├── gpu_image.cpp               # Task 7 (gated)
│   └── gpu_preprocess.cpp          # Task 7 (gated, D1 fix)
└── io/
    ├── CMakeLists.txt              # Task 8/9
    ├── json_exporter.cpp           # Task 8
    └── basic_importer.cpp          # Task 9
tests/
├── infra/ (Task 1: daily_and_size_sink_test.cpp; Task 2: logger_dropped_count_test.cpp)
├── device/ (Task 3: ring_buffer_test.cpp, device_interface_test.cpp + fake_camera.{h,cpp}, fake_light_controller.{h,cpp})
├── image/ (Task 4: image_types_test.cpp, roi_test.cpp; Task 5/6: preprocess_test.cpp; reuses tests/memory/host_test_pool)
├── io/ (Task 8: json_exporter_test.cpp; Task 9: basic_importer_test.cpp)
└── integration/ (Task 10: acquisition_to_export_test.cpp)
```

## Execution Order

```
Task 1 (D3 sink) ─┐
Task 2 (D2 drop)  ┘  milestone-1 debt (infra, independent, portable)
Task 3 (Device: interfaces + RingBuffer + fakes)                         [portable]
Task 4 (Image types: Image/RawImage/SurfaceImage/ROI) ──> Task 5 ──> Task 6   [portable]
                                                     └──> Task 7 (GPU imaging + D1) [gated]
Task 8 (IO: IExporter + JsonExporter) ──> Task 9 (IO: IImporter + BasicImporter)  [portable]
Task 10 (validation integration test) — needs Tasks 3,4,5,6,8              [portable]
Task 11 (glossary contract rows + consolidated deviation log)             [docs]
```

Single controller executes in plan order 1→11 (subagent-driven-development dispatches one implementer at a time). D1 is repaid inside Task 7 (gated); D2/D3 are Tasks 1–2.

---

## Task 1: Debt D3 — `DailyAndSizeRotatingFileSink`

Repays milestone-1 debt D3 (rotation was size-only; spec mandates daily **OR** 100 MB, whichever first). Portable, fully testable.

**Files:**
- Create: `include/sai/infra/daily_and_size_sink.h`
- Modify: `src/infra/logger.cpp` (swap the size-only `rotating_file_sink_mt` used in `InitializeGlobalSinks` for `DailyAndSizeRotatingFileSink`), `src/infra/CMakeLists.txt` if a `.cpp` is added
- Create: `tests/infra/daily_and_size_sink_test.cpp`, and register it in `tests/infra/CMakeLists.txt`

**Interfaces:**
- Consumes: `spdlog::sinks::base_sink<Mutex>`, `spdlog::details::file_helper`.
- Produces (namespace `sai::infra`):
  - `template <typename Mutex> class DailyAndSizeRotatingFileSink final : public spdlog::sinks::base_sink<Mutex>` with ctor `(spdlog::filename_t base_filename, std::size_t max_size, std::size_t max_files)`.
  - Aliases `using DailyAndSizeRotatingFileSink_mt = DailyAndSizeRotatingFileSink<std::mutex>;`
  - A protected `virtual auto NowTm() const noexcept -> std::tm;` seam so tests can simulate a date change without waiting a day.

- [ ] **Step 1: Write the failing test.** In `daily_and_size_sink_test.cpp`, define a `TestableSink` subclass overriding `NowTm()` to return a settable `std::tm`. Cases: (a) writing a message that pushes byte count past a tiny `max_size` produces a rotated file (`base.1.log` exists); (b) advancing `NowTm()` to the next day forces a rotation on the next log even when under `max_size` (assert a new dated/rotated file appears and the active file byte count reset); (c) `max_files` cap deletes the oldest.

```cpp
TEST(DailyAndSizeSink, RotatesWhenSizeExceeded) {
    auto dir = MakeTempDir();
    auto sink = std::make_shared<TestableSink>((dir / "app.log").string(),
                                               /*max_size=*/64, /*max_files=*/3);
    spdlog::logger log("t", sink);
    for (int i = 0; i < 50; ++i) log.info("0123456789");  // >64 bytes total
    log.flush();
    EXPECT_TRUE(std::filesystem::exists(dir / "app.1.log"));
}

TEST(DailyAndSizeSink, RotatesOnDayChangeUnderSizeCap) {
    auto dir = MakeTempDir();
    auto sink = std::make_shared<TestableSink>((dir / "app.log").string(), 1'000'000, 3);
    spdlog::logger log("t", sink);
    log.info("day1"); log.flush();
    sink->SetDay(/*next day*/);           // simulate tomorrow
    log.info("day2"); log.flush();
    EXPECT_TRUE(std::filesystem::exists(dir / "app.1.log"));  // day1 archived out
}
```

- [ ] **Step 2: Run to verify it fails.** `ctest --preset default -R DailyAndSizeSink` → FAIL (type undefined).

- [ ] **Step 3: Implement `daily_and_size_sink.h`.** Inherit `base_sink<Mutex>`; hold `file_helper file_helper_`, `max_size_`, `max_files_`, `current_size_`, and `int current_day_` (from `NowTm().tm_yday`/`tm_year`). Override `sink_it_(const details::log_msg&)`:

```cpp
void sink_it_(const spdlog::details::log_msg& msg) override {
    spdlog::memory_buf_t formatted;
    base_sink<Mutex>::formatter_->format(msg, formatted);
    const auto now = NowTm();
    const bool day_changed = now.tm_yday != current_day_ || now.tm_year != current_year_;
    if (day_changed || current_size_ + formatted.size() > max_size_) {
        Rotate_();                       // early-return style: rotate first, then write
        current_day_ = now.tm_yday; current_year_ = now.tm_year; current_size_ = 0;
    }
    file_helper_.write(formatted);
    current_size_ += formatted.size();
}
```
`Rotate_()` renames `base.(n-1).log`→`base.n.log` down to `base.log`→`base.1.log`, deleting `base.max_files_.log`, then reopens `base.log` (reuse `rotating_file_sink`'s calc-filename helper shape). `flush_()` calls `file_helper_.flush()`. `NowTm()` default returns `spdlog::details::os::localtime()`.

- [ ] **Step 4: Wire into `InitializeGlobalSinks`.** In `logger.cpp`, replace the size-only rotating sink construction with `std::make_shared<DailyAndSizeRotatingFileSink_mt>(path, 100 * 1024 * 1024, 10)`. Header-only template → likely no new `.cpp`; if `CMakeLists.txt` needs no change, note that.

- [ ] **Step 5: Build + full suite.** `cmake --preset default && cmake --build --preset default && ctest --preset default` → 84 prior + new sink tests, all pass.

- [ ] **Step 6: Commit.**
```bash
git add include/sai/infra/daily_and_size_sink.h src/infra/logger.cpp tests/infra/
git commit -m "fix(infra): 🐛 补齐日志按天+100MB 双条件轮转（D3）"
```

---

## Task 2: Debt D2 — per-category `DroppedCount`

Repays milestone-1 debt D2 (`DroppedCount()` was process-wide because all categories shared one `spdlog::thread_pool`). Fix: give each category's **drop tier** its own thread pool so `overrun_counter()` is genuinely per-category; add the `DroppedCount(category)` overload. The block tier (Warning+, never drops) may stay shared. **Deviation to record (Task 11):** this changes 1.6 §9's "single shared thread_pool" to a per-category drop-tier pool — sanctioned by the milestone-2 spec §7 D2, which explicitly repays this debt.

**Files:**
- Modify: `include/sai/infra/logger.h` (add static `DroppedCount(std::string_view category)` overload; keep instance `DroppedCount()`; `drop_tier_` now backed by a per-instance `std::shared_ptr<spdlog::details::thread_pool>`), `src/infra/logger.cpp` (`Get` creates a per-category drop pool)
- Create: `tests/infra/logger_dropped_count_test.cpp`, register in `tests/infra/CMakeLists.txt`

**Interfaces:**
- Produces: `Logger::DroppedCount() const noexcept -> std::uint64_t` (unchanged signature, now returns this category's drop-tier `thread_pool->overrun_counter()`); `static Logger::DroppedCount(std::string_view category) -> std::uint64_t` (= `Get(category).DroppedCount()`).

- [ ] **Step 1: Write the failing test.** Initialize sinks to a temp dir; flood category `"A"`'s drop tier (Trace/Debug) hard enough to overflow its small queue; assert `Logger::DroppedCount("A") > 0` and `Logger::DroppedCount("B") == 0`.

```cpp
TEST(LoggerDroppedCount, PerCategoryAttribution) {
    ASSERT_TRUE(Logger::InitializeGlobalSinks(MakeTempDir()).has_value());
    auto& a = Logger::Get("A");
    a.SetLevel(LogLevel::Trace);
    for (int i = 0; i < 200'000; ++i) a.Log(LogLevel::Debug, "flood {}", i);
    EXPECT_GT(Logger::DroppedCount("A"), 0U);
    EXPECT_EQ(Logger::DroppedCount("B"), 0U);
}
```

- [ ] **Step 2: Run to verify it fails.** Fails to compile (no `DroppedCount(category)` overload) or asserts wrongly (shared counter attributes A's drops to B).

- [ ] **Step 3: Implement.** In `logger.cpp`, `Get(category)` constructs a dedicated small-capacity `thread_pool` (e.g. `queue_size=8192, threads=1`) for that category's `drop_tier_` async_logger; keep the shared block-tier pool. Store the drop pool on the `Logger` instance (`std::shared_ptr<spdlog::details::thread_pool> drop_pool_`). `DroppedCount()` returns `drop_pool_ ? drop_pool_->overrun_counter() : 0`. Add the static overload delegating to `Get(category).DroppedCount()`. Remove the now-dead `dropped_count_` atomic (flagged as dead in the milestone-1 final review).

- [ ] **Step 4: Build + full suite.** Run the full `ctest --preset default`; the existing `logger_test.cpp` drop smoke test must still pass alongside the new per-category test.

- [ ] **Step 5: Commit.**
```bash
git add include/sai/infra/logger.h src/infra/logger.cpp tests/infra/
git commit -m "fix(infra): 🐛 DroppedCount 改为按类别归属（D2）"
```

---

## Task 3: Device — interfaces + `Rect` + `RingBuffer<T>` (+ test fakes)

Batch 2.1. `IDevice`/`ICamera`/`ILightController` are pure-abstract (header-only); `Rect` and `RingBuffer<T>` are the only concrete units, and `RingBuffer<T>` is the testable deliverable. Also lands the in-memory test fakes used by Task 10.

**Files:**
- Create: `include/sai/device/device.h`, `camera.h`, `light_controller.h`, `ring_buffer.h`
- Create: `src/device/CMakeLists.txt` (INTERFACE library `sai_device` / alias `sai::device`)
- Create: `tests/device/ring_buffer_test.cpp`, `tests/device/device_interface_test.cpp`, `tests/device/fake_camera.{h,cpp}`, `tests/device/fake_light_controller.{h,cpp}`, `tests/device/CMakeLists.txt`
- Modify: top-level `CMakeLists.txt` (`add_subdirectory(src/device)`, `add_subdirectory(tests/device)`), `include/sai/core/error.h` (append `Device_ConnectionFailed`, `Device_NotConnected`, `Device_AcquisitionInProgress`)

**Interfaces:**
- Consumes: `sai::IPlugin` (`include/sai/plugin/plugin.h`), `sai::Result<T>`, `SAI_DECLARE_TYPE_ID`.
- Produces (namespace `sai::device`), copied verbatim from spec §3.4: `struct Rect { std::size_t x,y,width,height; Area(); IsEmpty(); }`; `class IDevice : public IPlugin` with `enum class State {Disconnected,Connected,Acquiring,Error}` and `Connect/Disconnect/IsConnected/SerialNumber/CurrentState`; `class ICamera : public IDevice` with `enum class TriggerMode {Software,Hardware,FreeRun}`, `using FrameCallback = std::function<void(RawImage)>` (forward-declares `class RawImage;` per spec — no include of image headers), `SetTriggerMode/StartAcquisition/StopAcquisition/RegisterFrameCallback/SetExposureTime/SetGain/SetROI`; `class ILightController : public IDevice` with `enum class StrobeMode {Continuous,OnTrigger,Off}`, `ChannelCount/SetIntensity/GetIntensity/Enable/Disable/SetStrobeMode`; `template <typename T> class RingBuffer final` per spec §3.4 (`explicit RingBuffer(std::size_t)`, `Push(T)`, `TryPop()->std::optional<T>`, `Capacity/Size/DroppedCount`, copy-deleted, mutex-guarded).

- [ ] **Step 1: Write the failing `RingBuffer` test.** Use a move-only `TestFrame{int id; std::unique_ptr<int> tag;}` to prove move-only support. Cases: fresh buffer `Size()==0`, `TryPop()==nullopt`; push→pop FIFO order; filling past `Capacity()` overwrites the oldest (next `TryPop` returns the *newer* value, `DroppedCount()` increments per overwrite); `Size()` never exceeds `Capacity()`.

```cpp
TEST(RingBuffer, OverwritesOldestWhenFull) {
    sai::device::RingBuffer<int> rb(2);
    rb.Push(1); rb.Push(2); rb.Push(3);           // 1 overwritten
    EXPECT_EQ(rb.DroppedCount(), 1U);
    EXPECT_EQ(rb.Size(), 2U);
    EXPECT_EQ(rb.TryPop(), 2);
    EXPECT_EQ(rb.TryPop(), 3);
    EXPECT_EQ(rb.TryPop(), std::nullopt);
}
```

- [ ] **Step 2: Run to verify it fails.** Header does not exist → compile error.

- [ ] **Step 3: Implement the four headers.** `ring_buffer.h`: `std::vector<std::optional<T>> slots_` sized to capacity, `head_`/`tail_`/`count_`/`dropped_count_`, `std::mutex`. `Push` under lock: if `count_ == capacity_`, advance `tail_` and `++dropped_count_` (overwrite path), then move `item` into `slots_[head_]`, advance `head_`, cap `count_`. `TryPop` under lock: if `count_==0` return `std::nullopt`; else move out `slots_[tail_]`, reset it, advance `tail_`, `--count_`. Early-return style, no nesting. The interface headers are pure-virtual declarations copied verbatim from the spec (no bodies except the inline `Rect::Area/IsEmpty`).

- [ ] **Step 4: Add the device interface conformance test + fakes.** `device_interface_test.cpp` instantiates `FakeCamera`/`FakeLightController` (test doubles implementing every pure virtual) and asserts the state machine: `Connect()`→`IsConnected()`, `CurrentState()==Connected`; `StartAcquisition()`→`Acquiring`; a software trigger invokes the registered `FrameCallback`; `StopAcquisition()`→`Connected`; `Disconnect()`→`Disconnected`. `FakeCamera` returns `Device_NotConnected` when acquiring before `Connect`. `FakeCamera`'s frame production is stubbed here (calls the callback with a default-constructed `RawImage` only once Task 4 exists — until then, `device_interface_test` exercises the state machine and light controller; frame-producing wiring lands in Task 10's integration test, which depends on Task 4). Keep the fakes minimal and out of `include/`.

- [ ] **Step 5: CMake + ErrorCode + build.** `src/device/CMakeLists.txt`: `add_library(sai_device INTERFACE)`, `target_include_directories(sai_device INTERFACE ${CMAKE_SOURCE_DIR}/include)`, `target_link_libraries(sai_device INTERFACE sai::plugin)`, alias `sai::device`. Append the three `Device_*` codes to `error.h`. Build + full suite green.

- [ ] **Step 6: Commit.**
```bash
git add include/sai/device/ src/device/ tests/device/ include/sai/core/error.h CMakeLists.txt
git commit -m "feat(device): ✨ 添加设备接口（IDevice/ICamera/ILightController）与 RingBuffer"
```

---

## Task 4: Imaging — `Image` hierarchy + `ROI`

Batch 2.2, part 1: the `Image` base and its three portable concrete factories, plus `ROI`. **Deviation to record (Task 11):** add `FromOwnedBuffer(std::vector<std::uint8_t>, ImageMeta)` to `RawImage`/`SurfaceImage` — an *additive* heap-owning factory. Rationale: the frozen preprocess/ROI/importer signatures have no pool parameter, yet steps that change size/format (Debayer, Resize, Calibration, ROI crop, PPM import) must return an *owning* image; the frozen `Image` types own only via `PooledPtr` (needs a pool) or `FromBuffer` (non-owning). `FromOwnedBuffer` closes that gap portably without a pool and without changing any frozen signature. `GpuImage` gets no such factory (device memory can't back a host vector) and stays gated (Task 7).

**Files:**
- Create: `include/sai/image/image.h`, `raw_image.h`, `surface_image.h`, `roi.h`
- Create: `src/image/CMakeLists.txt`, `src/image/image.cpp`, `raw_image.cpp`, `surface_image.cpp`, `roi.cpp`
- Create: `tests/image/image_types_test.cpp`, `tests/image/roi_test.cpp`, `tests/image/CMakeLists.txt`
- Modify: top-level `CMakeLists.txt`, `include/sai/core/error.h` (append `Image_UnsupportedPixelFormat`, `Image_DimensionMismatch`)

**Interfaces:**
- Consumes: `sai::Resource` (`resource.h`), `sai::Result<T>`, `sai::memory::IMemoryPool`/`PooledPtr<std::uint8_t>` (`memory_pool.h`, `pooled_ptr.h`), `sai::device::Rect` (`device.h`).
- Produces (namespace `sai::image`), from spec §4.4/§4.6 verbatim, plus the two additive factories:
  - `enum class PixelFormat : std::uint16_t {Mono8,Mono10,Mono12,BayerRG8,BayerRG10,BayerRG12,RGB8,BGR8,Undefined=0xFFFF};`
  - `struct ImageMeta {width,height,channels; PixelFormat pixel_format; std::chrono::nanoseconds timestamp; std::uint32_t frame_index;};`
  - `class Image : public Resource` — protected ctor `(std::uint8_t* data, std::size_t size_bytes, ImageMeta)`, `Meta()`, `Data()` (const+mutable), `SizeBytes()`, `IsValid() override {return data_!=nullptr;}`, `Release() override`.
  - `class RawImage final : public Image` — `static FromPool(IMemoryPool&, ImageMeta)->Result<RawImage>`, `static FromBuffer(std::uint8_t*, std::size_t, ImageMeta)->RawImage` (non-owning), **`static FromOwnedBuffer(std::vector<std::uint8_t>, ImageMeta)->RawImage`** (additive), move-only.
  - `class SurfaceImage final : public Image` — `static FromPool(IMemoryPool&, ImageMeta)->Result<SurfaceImage>`, `static FromPinned(PooledPtr<std::uint8_t>, ImageMeta)->SurfaceImage`, **`static FromOwnedBuffer(std::vector<std::uint8_t>, ImageMeta)->SurfaceImage`** (additive), move-only.
  - `roi.h`: `using Rect = sai::device::Rect;`, `struct ROI {std::vector<Rect> regions; IsEmpty(); BoundingBox()->Rect; static Apply(const Image& src, const ROI&)->Result<std::unique_ptr<Image>>;};`

- [ ] **Step 1: Write the failing `image_types_test`.** Cover, using a `tests/memory/host_test_pool` instance (see CMake note): `RawImage::FromPool` returns a valid image whose `Meta()`/`SizeBytes()` match and `Data()!=nullptr`; move leaves the source `!IsValid()`; destroying a pool-backed image raises `AvailableSlabCount()` back; `FromBuffer` (caller buffer) yields a valid non-owning image whose dtor does **not** touch the caller buffer; `FromOwnedBuffer` yields a valid image that owns its bytes (data survives until dtor, `Data()` stable across a move); `Release()` makes `IsValid()==false`.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement the `Image` base + `RawImage`/`SurfaceImage`.** Each concrete class holds `PooledPtr<std::uint8_t> buffer_` **and** `std::vector<std::uint8_t> owned_bytes_` (exactly one is populated). `FromPool`: `pool.Acquire(size)` → `and_then` construct with the handle, `data_=handle.Get()`. `FromBuffer`: `data_=data`, both `buffer_`/`owned_bytes_` empty (dtor no-op). `FromOwnedBuffer`: move the vector into `owned_bytes_`, `data_=owned_bytes_.data()` (stable across `std::vector` move). `Release()`/dtor: reset `buffer_` (returns slab to pool if live) and clear `owned_bytes_`, set `data_=nullptr`. Defaulted move ctor/assign work (both members movable; raw `data_` stays valid because vector move preserves its allocation and `PooledPtr` move transfers). Non-template method bodies in `.cpp`.

- [ ] **Step 4: Implement `ROI` + `roi_test`.** `IsEmpty()` = `regions.empty()`; `BoundingBox()` = min/max over regions (early return empty `Rect` when empty). `ROI::Apply` crops the *first* region (multi-region compositing deferred; document in report) into a fresh `SurfaceImage::FromOwnedBuffer`, copying rows within bounds; out-of-bounds region → `Image_DimensionMismatch`. Test: `BoundingBox` over 2 rects; `Apply` crops a known 4×4 buffer to a 2×2 sub-rect with correct pixels; oversized region → `Image_DimensionMismatch`.

- [ ] **Step 5: CMake + ErrorCode + build.** `src/image/CMakeLists.txt`: `sai_image` STATIC from `image.cpp raw_image.cpp surface_image.cpp roi.cpp`, links `sai::core sai::memory`, `sai::device` (for `Rect`), `cxx_std_20`. `tests/image/CMakeLists.txt`: compile `${CMAKE_SOURCE_DIR}/tests/memory/host_test_pool.cpp` into the image test target and add `${CMAKE_SOURCE_DIR}/tests/memory` to its include dirs (reuse the milestone-1 host pool, DRY); apply the same `if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64") -mcx16` guard the memory tests use (host_test_pool.h static_asserts a 16-byte atomic). Append the two `Image_*` codes. Build + full suite green.

- [ ] **Step 6: Commit.**
```bash
git add include/sai/image/ src/image/ tests/image/ include/sai/core/error.h CMakeLists.txt
git commit -m "feat(image): ✨ 添加图像类型体系（Image/RawImage/SurfaceImage）与 ROI"
```

---

## Task 5: Imaging — preprocess core (`Compose`, `MakeFlatField`, `MakeDebayer`)

Batch 2.2, part 2: the pipeline backbone and the spec's headline correctness point (FlatField **before** Debayer). **Deviation to record (Task 11):** `MakeFlatField`'s frozen param is `Image correction_frame` (by value) — impossible (`Image` is abstract, protected ctor, move-only → slices/won't compile). Changed to `const Image& correction_frame` (read-only, long-lived correction frame captured by pointer), consistent with `preprocess.h`'s own anti-slicing note.

**Files:**
- Create: `include/sai/image/preprocess.h` (full declaration set per spec §4.4)
- Create: `src/image/preprocess.cpp` (this task: `Compose`, `MakeFlatField`, `MakeDebayer`)
- Create: `tests/image/preprocess_test.cpp`, add to `tests/image/CMakeLists.txt`
- Modify: `src/image/CMakeLists.txt` (add `preprocess.cpp`), `include/sai/core/error.h` (append `Image_PreprocessFailed`)

**Interfaces:**
- Produces (namespace `sai::image`): `using PreprocessFn = std::function<auto(std::unique_ptr<Image>) -> Result<std::unique_ptr<Image>>>;`; `auto Compose(std::vector<PreprocessFn>) -> PreprocessFn;`; `auto MakeFlatField(const Image& correction_frame) -> PreprocessFn;`; `auto MakeDebayer() -> PreprocessFn;`. (Remaining `MakeXxx` + `CalibrationParams` are declared in this header now but *implemented* in Task 6.)

- [ ] **Step 1: Write failing tests.** (a) `Compose` runs steps in order and short-circuits: a two-step chain where step 1 returns `Image_PreprocessFailed` never runs step 2 and propagates the error; an empty chain returns the input unchanged. (b) `MakeFlatField`: a Mono8 8×8 input with a uniform correction frame (all == a reference gray) is returned unchanged; a correction frame simulating a dark corner scales those pixels up (assert a known corner pixel brightened). (c) `MakeDebayer`: a `BayerRG8` 4×4 input yields an `RGB8` image with `channels==3`, `SizeBytes()==w*h*3`, and a known pixel demosaiced to the expected RGB triple.

```cpp
TEST(Preprocess, ComposeShortCircuitsOnError) {
    int ran2 = 0;
    auto step1 = [](std::unique_ptr<Image>) -> Result<std::unique_ptr<Image>> {
        return tl::make_unexpected(ErrorInfo{ErrorCode::Image_PreprocessFailed, "boom", {}});
    };
    auto step2 = [&](std::unique_ptr<Image> i) -> Result<std::unique_ptr<Image>> { ++ran2; return i; };
    auto chain = Compose({step1, step2});
    auto out = chain(MakeMono8(2, 2));
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().code, ErrorCode::Image_PreprocessFailed);
    EXPECT_EQ(ran2, 0);
}
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `Compose`.** Recursive fold, no nested loops, chaining with `and_then`:

```cpp
auto Compose(std::vector<PreprocessFn> steps) -> PreprocessFn {
    return [steps = std::move(steps)](std::unique_ptr<Image> img) -> Result<std::unique_ptr<Image>> {
        return RunFrom(steps, 0, std::move(img));   // free recursive helper (file-local)
    };
}
// RunFrom(steps, i, img): if i==size return img; else steps[i](move(img)).and_then(next i+1)
```

- [ ] **Step 4: Implement `MakeFlatField` + `MakeDebayer`.** `MakeFlatField` captures `const Image* corr` (from the ref), asserts matching dims (`Image_DimensionMismatch` otherwise), mutates the input buffer in place (same size) `out[i] = clamp(in[i] * ref_mean / corr[i])`, returns the same `unique_ptr` (in-place). `MakeDebayer` allocates `w*h*3` via `SurfaceImage::FromOwnedBuffer`, demosaics `BayerRG8` by nearest-neighbor per 2×2 quad (RGGB), sets `channels=3, pixel_format=RGB8`; non-Bayer input → `Image_UnsupportedPixelFormat`.

- [ ] **Step 5: Build + full suite green.**

- [ ] **Step 6: Commit.**
```bash
git add include/sai/image/preprocess.h src/image/ tests/image/ include/sai/core/error.h
git commit -m "feat(image): ✨ 添加预处理链 Compose 与 FlatField/Debayer 步骤"
```

---

## Task 6: Imaging — preprocess remainder (`WhiteBalance`, `Resize`, `Calibration`, `HDR`)

Batch 2.2, part 3: the remaining CPU steps. **Deviation to record (Task 11):** `MakeHDR(std::size_t num_exposures)` returns a single-input `PreprocessFn`, which cannot fuse multiple frames (the type takes one `Image`). Implemented as a single-frame dynamic-range/tone expansion (documented interpretation); true multi-exposure fusion needs a multi-input API and is deferred to Future Extension.

**Files:**
- Modify: `src/image/preprocess.cpp` (add `MakeWhiteBalance`, `MakeResize`, `MakeCalibration`, `MakeHDR`; `CalibrationParams` already declared in `preprocess.h` from Task 5)
- Modify: `tests/image/preprocess_test.cpp` (add cases)

**Interfaces:**
- Produces: `struct CalibrationParams {std::array<double,9> camera_matrix; std::array<double,5> dist_coeffs; double pixel_scale_mm=1.0;};`; `MakeWhiteBalance(float r_gain,float g_gain,float b_gain)`, `MakeResize(std::size_t w,std::size_t h)`, `MakeCalibration(CalibrationParams)`, `MakeHDR(std::size_t num_exposures)` — all `-> PreprocessFn`.

- [ ] **Step 1: Write failing tests.** `MakeWhiteBalance`: RGB8 pixel `(100,100,100)` with gains `(1.2,1.0,0.8)` → `(120,100,80)` (in place, clamped). `MakeResize`: 4×4 RGB8 → 2×2 via bilinear, `SizeBytes()==2*2*3`, corner pixel matches expected average. `MakeCalibration`: identity camera matrix + zero distortion returns geometrically-unchanged pixels but stamps `pixel_scale_mm` reachable via meta (or documented no-op-on-identity); `MakeHDR(1)` on a low-contrast frame widens its min/max range (documented single-frame semantics).

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement the four steps.** WhiteBalance: in place, per-channel gain on interleaved RGB8, `Image_UnsupportedPixelFormat` for non-RGB. Resize: allocate `tw*th*channels` via `FromOwnedBuffer`, bilinear sample. Calibration: allocate output via `FromOwnedBuffer`, bilinear inverse-map through `camera_matrix`/`dist_coeffs` (identity/zero → straight copy), carry `pixel_scale_mm` (store in a documented spare `ImageMeta` field or note the carry mechanism). HDR: single-frame min-max stretch. Every step early-returns an `Image_PreprocessFailed`/`Image_UnsupportedPixelFormat` on precondition failure.

- [ ] **Step 4: Build + full suite green.**

- [ ] **Step 5: Commit.**
```bash
git add src/image/preprocess.cpp tests/image/preprocess_test.cpp
git commit -m "feat(image): ✨ 添加 WhiteBalance/Resize/Calibration/HDR 预处理步骤"
```

---

## Task 7: Imaging — GPU path (`GpuImage` + GPU marshaling step, repays D1) [gated, no local build]

Batch 2.2, part 4: CUDA-gated. Repays milestone-1 debt D1 (`GpuStreamQueue::EnqueueAsyncCopy` never populated/drained the pinned transit buffer) by making the marshaling explicit **caller-side** exactly as spec §4.3 prescribes — `GpuStreamQueue`'s interface is not modified. Not compiled on this host; reviewed by careful reading against the CUDA Runtime API + the frozen `GpuStreamQueue`/`PinnedPool` contracts, like milestone-1's gated tasks.

**Files:**
- Create: `include/sai/image/gpu_image.h` (portable to *include*: only `Image` + `PooledPtr`, no CUDA header), `src/image/gpu_image.cpp` (gated: `GpuImage::FromPool` via `GpuPool`), `src/image/gpu_preprocess.cpp` (gated: the HtoD→[GPU op placeholder for M3]→DtoH coroutine step with explicit populate/drain)
- Modify: `src/image/CMakeLists.txt` (`find_package(CUDAToolkit QUIET)`; when found, append `gpu_image.cpp gpu_preprocess.cpp`, link `sai::runtime CUDA::cudart`, add the `-mcx16` x86-64 guard; when not found, neither file is compiled)
- Modify: `include/sai/core/error.h` only if a `Runtime_GpuError` is needed — otherwise reuse existing codes (record the taxonomy note)

**Interfaces:**
- Produces (namespace `sai::image`): `class GpuImage final : public Image { static auto FromPool(IMemoryPool& gpu_pool, ImageMeta) noexcept -> Result<GpuImage>; move-only; ~GpuImage(); }`; and a gated free function `auto MakeGpuUpload(sai::memory::PinnedPool&, sai::runtime::GpuStreamQueue&) -> ...` returning a coroutine-driven marshaling step (exact shape per spec §4.3/§4.8). Consumes gated `GpuPool`, `PinnedPool`, `GpuStreamQueue`, `CopyDirection`, `Task<void>`, `SurfaceImage::FromPinned`.

- [ ] **Step 1: Write `gpu_image.h` + `gpu_image.cpp`.** `GpuImage` mirrors `SurfaceImage`'s pool-backed shape but no `FromOwnedBuffer`/`FromBuffer`; `FromPool` acquires from the `GpuPool` (device memory), `Data()` points at device memory (documented: not host-dereferenceable).

- [ ] **Step 2: Write `gpu_preprocess.cpp` implementing the D1 fix verbatim per spec §4.3.** HtoD: `pinned_pool.Acquire(size)` → `std::memcpy(transit.Get(), cpu_image.Data(), size)` (**populate — the D1 fix**) → `co_await gpu_queue.EnqueueAsyncCopy(transit, size, CopyDirection::HostToDevice, token)`. DtoH: `co_await gpu_queue.EnqueueAsyncCopy(transit, size, CopyDirection::DeviceToHost, token)` → `SurfaceImage::FromPinned(std::move(transit), meta)` (**drain — the D1 fix**). Comment each populate/drain line as the D1 repair. Reuse `CopyDirection`/`Task<void>` verbatim; do not touch `GpuStreamQueue`.

- [ ] **Step 3: CMake gating.** Edit `src/image/CMakeLists.txt` to mirror `src/runtime/CMakeLists.txt`'s gate exactly. Confirm the two gated files are appended only under `if(CUDAToolkit_FOUND)`.

- [ ] **Step 4: Confirm exclusion is clean.** Rerun `cmake --preset default && cmake --build --preset default && ctest --preset default` on this host → the gated files are absent from the build log and the portable suite still passes at its full count. No GPU test is added (gated).

- [ ] **Step 5: Commit.**
```bash
git add include/sai/image/gpu_image.h src/image/gpu_image.cpp src/image/gpu_preprocess.cpp src/image/CMakeLists.txt
git commit -m "feat(image): ✨ 添加 GpuImage 与 GPU 搬运步骤（修复 D1，CUDA 门控）"
```

---

## Task 8: IO — `IExporter` + `JsonExporter` (+ nlohmann-json dep)

Batch 2.3, part 1. Delivers the exporter contract, data structures, and the built-in JSON exporter. **Deviation to record (Task 11):** annotated-image output is written as portable **PPM** (interim) rather than PNG — no PNG library is added this milestone (per the approved dependency decision); local tests exercise the JSON path (nullptr annotated) primarily.

**Files:**
- Modify: `vcpkg.json` (add `"nlohmann-json"`)
- Create: `include/sai/io/exporter.h`, `src/io/CMakeLists.txt`, `src/io/json_exporter.cpp`
- Create: `tests/io/json_exporter_test.cpp`, `tests/io/CMakeLists.txt`
- Modify: top-level `CMakeLists.txt` (`add_subdirectory(src/io)`, `add_subdirectory(tests/io)`), `include/sai/core/error.h` (append `Io_ExportPathCreateFailed`, `Io_SerializationFailed`)

**Interfaces:**
- Consumes: `sai::IPlugin`, `sai::PluginManifest`, `sai::Context`, `SAI_DECLARE_TYPE_ID`, `sai::image::SurfaceImage`, `sai::device::Rect`, `Result<T>`, `nlohmann::json`.
- Produces (namespace `sai::io`), from spec §5.4 verbatim: `struct DefectRecord {std::string label,severity; float confidence; Rect location; std::string evidence_path;};` (add `using sai::device::Rect;`); `struct InspectionResult {std::string sku_id,serial_number; std::chrono::system_clock::time_point timestamp; std::vector<DefectRecord> defects; std::string verdict;};`; `class IExporter : public IPlugin { virtual auto Export(const InspectionResult&, std::filesystem::path output_dir, const SurfaceImage* annotated_image) noexcept -> Result<void> = 0; virtual auto FormatName() const noexcept -> std::string_view = 0; };`; `class JsonExporter final : public IExporter` with the lifecycle no-ops + `manifest_` from the spec, plus `SAI_DECLARE_TYPE_ID(sai.io.json-exporter)`.

- [ ] **Step 1: Write failing `json_exporter_test`.** (a) `Export` of an `InspectionResult` with 2 defects + `verdict="FAIL"` into a temp dir creates `<dir>/<sku_id>/<serial_number>/result.json`; parse it back with `nlohmann::json` and assert `sku_id`, `serial_number`, `verdict`, `defects.size()==2`, and one defect's `label`/`severity`/`confidence`/`location`. (b) `annotated_image==nullptr` writes only `result.json` (no image file). (c) `FormatName()=="json_report"`. (d) directory-create failure on an unwritable path → `Io_ExportPathCreateFailed`.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `JsonExporter::Export`.** Early-return chain: `std::filesystem::create_directories(dir/sku/serial)` (error → `Io_ExportPathCreateFailed`); build `nlohmann::json` from the result (serialize `timestamp` as epoch-count; each defect as an object incl. `location` as `{x,y,width,height}`); write `result.json` (write failure → `Io_SerializationFailed`); if `annotated_image != nullptr`, write `annotated.ppm` from its pixel buffer (P6 header + raw bytes — read-only over the `SurfaceImage`, no allocation) with a `// TODO: real PNG when a PNG lib is added` note. Lifecycle hooks return `{}`; `GetManifest()` returns `manifest_`.

- [ ] **Step 4: CMake + dep + build.** Add `nlohmann-json` to `vcpkg.json`. `src/io/CMakeLists.txt`: `find_package(nlohmann_json CONFIG REQUIRED)`; `sai_io` STATIC (`json_exporter.cpp` this task), links `sai::core sai::image sai::device sai::plugin nlohmann_json::nlohmann_json`. Append the two `Io_*` codes. Reconfigure (vcpkg installs nlohmann-json) + build + full suite green.

- [ ] **Step 5: Commit.**
```bash
git add vcpkg.json include/sai/io/exporter.h src/io/ tests/io/ include/sai/core/error.h CMakeLists.txt
git commit -m "feat(io): ✨ 添加 IExporter 契约与 JsonExporter（JSON 报告，PPM 占位）"
```

---

## Task 9: IO — `IImporter` + `BasicImporter`

Batch 2.3, part 2. `ImportMetadata` (YAML) is fully delivered and tested; `ImportImage` decodes portable **PPM** into an owning `SurfaceImage` via `FromOwnedBuffer` (PNG decode deferred with the same rationale as Task 8).

**Files:**
- Create: `include/sai/io/importer.h`, `src/io/basic_importer.cpp`
- Create: `tests/io/basic_importer_test.cpp`, add to `tests/io/CMakeLists.txt`
- Modify: `src/io/CMakeLists.txt` (add `basic_importer.cpp`, link `yaml-cpp::yaml-cpp`), `include/sai/core/error.h` (append `Io_ImportFileNotFound`, `Io_ImportParseFailed`)

**Interfaces:**
- Consumes: `sai::IPlugin`, `sai::image::Image` (`std::unique_ptr<Image>`), `YAML::Node`, `Result<T>`.
- Produces (namespace `sai::io`), from spec §5.4 verbatim: `class IImporter : public IPlugin { virtual auto ImportImage(std::filesystem::path) noexcept -> Result<std::unique_ptr<Image>> = 0; virtual auto ImportMetadata(std::filesystem::path) noexcept -> Result<YAML::Node> = 0; virtual auto FormatName() const noexcept -> std::string_view = 0; };`; `class BasicImporter final : public IImporter` with lifecycle no-ops + `manifest_` + `SAI_DECLARE_TYPE_ID(sai.io.basic-importer)`, `FormatName()=="basic_import"`.

- [ ] **Step 1: Write failing `basic_importer_test`.** (a) `ImportMetadata` on a small YAML file returns a `YAML::Node` whose known keys read back (e.g. `sku_id`); missing file → `Io_ImportFileNotFound`; malformed YAML → `Io_ImportParseFailed`. (b) `ImportImage` on a hand-written 2×2 P6 PPM returns a `SurfaceImage` with `channels==3`, `pixel_format==RGB8`, `SizeBytes()==2*2*3`, and correct pixel bytes; missing file → `Io_ImportFileNotFound`. (c) `FormatName()=="basic_import"`.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `BasicImporter`.** `ImportMetadata`: existence check (`Io_ImportFileNotFound`) → `YAML::LoadFile` in try/catch (`YAML::Exception` → `Io_ImportParseFailed`) → return node. `ImportImage`: existence check → parse P6 PPM header (magic/width/height/maxval) → read pixels into a `std::vector<std::uint8_t>` → `SurfaceImage::FromOwnedBuffer(std::move(bytes), meta{w,h,3,RGB8})`; header/format error → `Io_ImportParseFailed`. Lifecycle hooks `{}`; `GetManifest()` returns `manifest_`.

- [ ] **Step 4: Build + full suite green.**

- [ ] **Step 5: Commit.**
```bash
git add include/sai/io/importer.h src/io/basic_importer.cpp tests/io/ include/sai/core/error.h
git commit -m "feat(io): ✨ 添加 IImporter 契约与 BasicImporter（YAML 元数据 + PPM 图像）"
```

---

## Task 10: Milestone-2 validation integration test

Implements the spec §6 validation point end-to-end, portable (fake software-triggered camera, CPU chain only, no GPU): **camera frame → RingBuffer → preprocess chain → SurfaceImage → InspectionResult → JsonExporter → result.json on disk.**

**Files:**
- Create: `tests/integration/acquisition_to_export_test.cpp`, `tests/integration/CMakeLists.txt`
- Modify: top-level `CMakeLists.txt` (`add_subdirectory(tests/integration)`); reuse `tests/device/fake_camera.{h,cpp}` (extend it to actually emit a `RawImage` now that Task 4 exists) and `tests/memory/host_test_pool`

**Interfaces:**
- Consumes everything produced above: `FakeCamera` (`ICamera`), `RingBuffer<RawImage>`, `Compose`/`MakeFlatField`/`MakeDebayer`/`MakeWhiteBalance`/`MakeResize`, `SurfaceImage`, `InspectionResult`/`JsonExporter`.

- [ ] **Step 1: Write the failing integration test.** Wire the flow: construct a `FakeCamera` that, on `StartAcquisition()` + a `TriggerSoftware()` test hook, builds a `BayerRG8` `RawImage` (from a host pool or `FromOwnedBuffer`) and invokes the registered `FrameCallback`; the callback pushes into a `RingBuffer<RawImage>`. Pop the frame, run `Compose({MakeFlatField(corr), MakeDebayer(), MakeWhiteBalance(1,1,1), MakeResize(8,8)})`, assert a valid `SurfaceImage` (`RGB8`, 8×8). Build an `InspectionResult{sku_id="Tesla_Model3_Driver", serial_number="SN-0001", verdict="PASS", defects={one DefectRecord}}`, `JsonExporter::Export` into a temp dir, assert `<dir>/Tesla_Model3_Driver/SN-0001/result.json` exists and round-trips.

```cpp
TEST(AcquisitionToExport, SoftwareTriggerToJsonReport) {
    FakeCamera cam; ASSERT_TRUE(cam.Connect().has_value());
    RingBuffer<RawImage> ring(4);
    ASSERT_TRUE(cam.RegisterFrameCallback([&](RawImage f){ ring.Push(std::move(f)); }).has_value());
    ASSERT_TRUE(cam.SetTriggerMode(ICamera::TriggerMode::Software).has_value());
    ASSERT_TRUE(cam.StartAcquisition().has_value());
    cam.TriggerSoftware();                                   // test hook → one frame
    auto raw = ring.TryPop(); ASSERT_TRUE(raw.has_value());
    auto chain = Compose({ /* FlatField, Debayer, WhiteBalance, Resize */ });
    auto surf = chain(std::make_unique<RawImage>(std::move(*raw)));
    ASSERT_TRUE(surf.has_value());
    // ... build InspectionResult, Export, assert result.json exists + parses
}
```

- [ ] **Step 2: Run to verify it fails, then implement the `FakeCamera` frame-emitting path + wiring until green.**

- [ ] **Step 3: Build + full suite green** (this is the milestone-2 acceptance gate; capture the passing test count in the task report).

- [ ] **Step 4: Commit.**
```bash
git add tests/integration/ tests/device/fake_camera.h tests/device/fake_camera.cpp CMakeLists.txt
git commit -m "test(io): ✅ 添加里程碑 2 采集→预处理→导出端到端验证"
```

---

## Task 11: Contracts + consolidated deviation log

Docs-only: register milestone-2 public types in the cross-batch contract table and record every deviation in one place, per the SDD discipline.

**Files:**
- Modify: `docs/surface-ai/glossary-and-contracts.md` (append rows to §1 概念归属表 and §2 核心接口签名表)
- Modify: `.superpowers/sdd/progress.md` (milestone-2 ledger entry) — or create the milestone-2 ledger section

- [ ] **Step 1: Append glossary concept + interface rows** for: `IDevice`/`ICamera`/`ILightController`/`RingBuffer`/`Rect` (2.1); `Image`/`RawImage`/`SurfaceImage`/`GpuImage`/`ImageMeta`/`PixelFormat`/`ROI`/`PreprocessFn` (2.2); `IExporter`/`IImporter`/`JsonExporter`/`BasicImporter`/`DefectRecord`/`InspectionResult` (2.3). Signatures copied verbatim from the landed headers. Do not modify any milestone-1 row.

- [ ] **Step 2: Write the consolidated deviation log** (in the progress ledger and/or a short section in the glossary), listing exactly: (D-a) `FromOwnedBuffer` additive factory on `RawImage`/`SurfaceImage`; (D-b) `MakeFlatField(const Image&)` vs frozen by-value; (D-c) `MakeHDR` single-frame semantics vs multi-exposure fusion; (D-d) annotated image PPM interim vs PNG; (D-e) D2 per-category drop-tier pool vs 1.6 §9 shared pool; (D-f) any `ErrorCode` taxonomy reuse in the gated GPU path. Each with rationale + follow-up.

- [ ] **Step 3: Commit.**
```bash
git add docs/surface-ai/glossary-and-contracts.md .superpowers/sdd/progress.md
git commit -m "docs(spec): 📝 登记里程碑 2 契约行与偏差记录"
```

---

## Spec Coverage Check

- Spec §1.1 debt: D1 → Task 7 (gated), D2 → Task 2, D3 → Task 1. ✅
- Spec §3 (2.1 Device): interfaces + `Rect` + `RingBuffer` → Task 3; concrete GenICam/serial drivers explicitly out-of-scope (Architecture note) — validation via fakes. ✅
- Spec §4 (2.2 Imaging): `Image`/`RawImage`/`SurfaceImage` → Task 4; `ROI` → Task 4; `PreprocessFn`/`Compose`/all `MakeXxx` → Tasks 5–6; `GpuImage` + GPU path → Task 7. ✅
- Spec §5 (2.3 IO): `IExporter`/`JsonExporter`/`DefectRecord`/`InspectionResult` → Task 8; `IImporter`/`BasicImporter` → Task 9. ✅
- Spec §6 validation point → Task 10. ✅
- Spec §7 (D2/D3) → Tasks 1–2. ✅
- Four pre-identified frozen-signature deviations all implemented + logged (Task 11). ✅
