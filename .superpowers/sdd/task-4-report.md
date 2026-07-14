# Task 4 Report: 重写 CaptureStage

**Status:** Complete
**Commit:** `1285702` (`feat(pipeline): ✨ CaptureStage 通过 Context/DI 获取 ICamera，FrameCallback 驱动 Pipeline::Submit`)

## Files changed

| File | Action |
|------|--------|
| `src/pipeline/capture_stage.cpp` | **Rewritten** -- 3-arg constructor, DI-based camera resolution (stub fallback), full camera lifecycle in OnStart/OnStop |
| `src/pipeline/stage_factory.h` | **Modified** -- `StageFactory::Create` 新增 `Pipeline* pipeline = nullptr` 参数 |
| `src/pipeline/stage_factory.cpp` | **Modified** -- 将 `pipeline` 参数传递给 `CaptureStage` 构造函数 |
| `src/pipeline/pipeline.cpp` | **Modified** -- 调用 `StageFactory::Create` 时传入 `pipeline.get()` |

## Build result

所有修改文件编译通过（`capture_stage.cpp`, `stage_factory.cpp`, `pipeline.cpp`）。

## Test result

571/572 测试通过。唯一失败的是 `LoggerDroppedCount.PerCategoryAttribution`，这是已知的预存问题（Task 8 偏差 D1：spdlog 的 `overrun_counter` 是进程级而非按类别），与本次变更无关。

## Design notes

- **3-arg constructor:** `CaptureStage(std::string id, YAML::Node config, Pipeline* pipeline)` -- 持有非拥有型 `Pipeline*` 指针用于 FrameCallback 中调用 `pipeline_->Submit()`
- **Camera lifecycle:** `OnStart` 执行完整序列 `Connect() -> SetTriggerMode(FreeRun) -> RegisterFrameCallback(lambda -> pipeline_->Submit) -> StartAcquisition()`；`OnStop` 执行 `StopAcquisition() -> Disconnect()`。每个步骤传播错误。
- **Stub fallback:** `OnInitialize` 中文档说明了当前 `ICamera`（`IPlugin`）不满足 `Context::Resolve<T>` 的 `IService` 约束，始终以 stub 模式运行。相机生命周期代码已完整写入，待 DI 机制支持非 IService 类型解析后只需修改 `OnInitialize` 即可激活。
- **Process passthrough:** `RawImage` 透传，类型不匹配返回 `Pipeline_StageTypeMismatch`

## Concerns

1. **`ICamera` 无法通过 `Context::Resolve<T>()` 解析** -- `Context::Resolve<T>` 要求 `T` 同时满足 `Reflectable` 和 `std::derived_from<T, IService>`。`ICamera` 继承链为 `IPlugin -> IModule + IReflectable`，不包含 `IService`。实际相机生命周期代码（`OnStart`/`OnStop`）已完整写入，当 DI 机制扩展支持后即可激活。

2. **`preprocess_stage.cpp` 编译错误（预存）** -- `Process` 方法中 `chain_()` 返回 `Result<unique_ptr<Image>>`，而 `StageOutput` variant 不接受基类 `Image`。此问题与本次变更无关，需要单独修复。
