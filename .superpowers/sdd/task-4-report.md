### Task 4 完成报告: headless_runner -- metrics 计算 + review_index.json

**Status:** COMPLETE (code written, build not verified on this Windows machine)

**Commits:**
- `feat(eval): ✨ headless模式自动计算precision/recall/F1并输出review_index.json`

**Files modified:**
- `apps/seat-aoi/headless_runner.h` -- 新增 `FrameRecord` 结构体和 `WriteReviewIndex` 函数声明
- `apps/seat-aoi/headless_runner.cpp` -- 完整重写，新增 `MetricsCounts`、FrameRecord 收集、metrics 计算、`WriteReviewIndex` 实现

**Changes summary:**

1. **headless_runner.h** -- 添加 `<cstdint>`, `<string>`, `<string_view>`, `<vector>` 头文件；声明 `FrameRecord` 结构体（frame_id, image_path, verdict, severity, confidence, expected_verdict, position_id, light_id, surface_id）；声明 `WriteReviewIndex` 函数。

2. **headless_runner.cpp** -- 完整重写：
   - 匿名命名空间中新增 `MetricsCounts` 辅助类（tp/fp/tn/fn + Precision/Recall/F1/Accuracy）
   - `process_entries` lambda 中每条目创建 `FrameRecord` 并收集到 `records` vector
   - `if constexpr (requires { ... })` 守卫正确处理 `DatasetEntry`（有 surface_id/position_id/light_id/expected_verdict）和 `DirEntry`（仅有 path）两种条目类型
   - Pipeline 停止后调用 `MetricsCounts::Add` 遍历 records 计算混淆矩阵 + precision/recall/F1
   - 调用 `WriteReviewIndex` 写入 `review_index.json`
   - `WriteReviewIndex` 实现：遍历 records 构建 JSON，pretty-print 2-space indent

3. **计划偏差：** 计划代码中 FrameRecord 构造时直接访问 `entry.position_id`/`entry.light_id`/`entry.surface_id` 而无 `if constexpr` 守卫。已修正为带 `if constexpr` 守卫的形式，DirEntry 使用默认值（position_id=0, light_id=0, surface_id=""）。

**Build verification:**
- Windows 11 机器无法配置 Linux preset（缺少 vcpkg toolchain 和 Unix Makefiles）
- 代码通过手动审查验证：所有 `if constexpr` 守卫正确，类型匹配，头文件完整

**Test summary:**
- 无新测试用例（headless_runner 是应用层代码，依赖完整 pipeline 的集成测试）
- 代码理论正确性：`if constexpr` 分支确保 DatasetEntry 和 DirEntry 两种路径均可编译；`MetricsCounts` 仅对 WARN/UNCERTAIN 做 passive tracking（不计入二分类矩阵）；`WriteReviewIndex` 兼容空 expected_verdict

**Concerns:**
- 需要在 Linux 机器上实际编译验证（`cmake --build --preset linux --target seat_aoi`）
- `WriteReviewIndex` 从 `records.front().surface_id` 推断 surface_id -- 如果 records 混合多个 surface，review_index.json 的 surface_id 字段仅反映第一个，但 frames 数组中每条仍保留自己的 surface_id
