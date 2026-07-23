# PatchCore Correctness and Performance Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 PatchCore 训练和在线推理的关键正确性问题，并在保持精确贪心 coreset 语义的前提下消除训练热点。

**Architecture:** TensorRT adapter 负责显式预处理和 dtype 转换；FeatureBank 负责确定性的增量 FPS；PatchCore 以不可变共享快照发布特征库，并把检测上下文随帧传递。在线检索只保留精确 FlatL2。

**Tech Stack:** C++20、CUDA、TensorRT、FAISS、GoogleTest、CMake Presets、yaml-cpp。

## Global Constraints

- 最终 coreset 选择必须是全候选集上的贪心最远点采样。
- 默认检测检索必须保持精确 FlatL2，不保留未验证的 IVFFlat 生产接口。
- DINO 输入为 RGB8 HWC，模型张量为 NCHW FP16/FP32，使用 mean `[0.485, 0.456, 0.406]` 和 std `[0.229, 0.224, 0.225]`。
- 下游 PatchFeatures 始终暴露 float32。
- 每个任务先增加能失败的窄测试，再实现，再运行相关测试。
- 模块化以职责边界和公共逻辑复用为准，不以减少文件数量为目标。
- 热循环不得因抽象增加多层调用、重复分配或重复数据搬运。

---

### Task 1: TensorRT 张量契约与 DINO 预处理

**Files:**
- Modify: `include/sai/inference/inference_engine.h`
- Modify: `include/sai/inference/dino_v3_adapter.h`
- Replace: `src/inference/dino_v3_adapter.cpp` with `src/inference/dino_v3_adapter.cu`
- Modify: `src/inference/tensorrt_engine.cpp`
- Modify: `apps/seat-aoi/embedder_factory.h`
- Test: `tests/inference/mock_engine_test.cpp`

- [ ] 增加 `TensorDataType`、元素字节数和 binding 一致性校验测试。
- [ ] 增加纯 CPU RGB8 HWC→NCHW float32 参考变换测试。
- [ ] 增加 CUDA FP16/FP32 预处理和 FP16 输出转 float32 实现。
- [ ] 运行 inference/embedding 相关测试并在 Linux CUDA 上验证真实 engine。

### Task 2: 等价的增量贪心 FPS

**Files:**
- Modify: `include/sai/detection/feature_bank.h`
- Modify: `src/detection/feature_bank.cpp`
- Create: `src/detection/greedy_coreset_cuda.cu`
- Modify: `src/detection/CMakeLists.txt`
- Test: `tests/detection/feature_bank_test.cpp`

- [ ] 写直接参考 FPS 与生产选择索引逐轮一致的测试。
- [ ] 写重复向量早停和距离相同按最小索引选择的测试。
- [ ] CPU 路径只用新选点更新 `min_dist`。
- [ ] CUDA 路径持久化候选和 `min_dist`，每轮只计算新选点距离并归约 argmax。
- [ ] 以固定 N/K/D 基准记录选择阶段耗时和峰值显存。

### Task 3: Release preset 与训练计时

**Files:**
- Modify: `CMakePresets.json`
- Modify: `apps/seat-aoi/coreset_builder.cpp`

- [ ] 新增独立 `linux-release` configure/build/test preset。
- [ ] 为图片读取、GPU 上传、embedding、贪心选择、保存和总耗时增加稳定日志。
- [ ] 验证 Debug preset 未改变，Release cache 为 `CMAKE_BUILD_TYPE=Release`。

### Task 3.1: 按机位流式训练与公共向量入口

**Files:**
- Modify: `apps/seat-aoi/coreset_builder.cpp`
- Modify: `include/sai/detection/feature_bank.h`
- Modify: `src/detection/feature_bank.cpp`
- Modify: `src/detection/coreset_evolution.cpp`
- Modify: `src/pipeline/detect_stage.cpp`

- [ ] 每个机位在完整候选集上执行精确贪心，保存后立即释放该机位向量。
- [ ] 删除跨全部机位常驻的 `pos_embeddings`。
- [ ] 训练、bootstrap、演化统一复用连续 float 向量入口，删除临时 Embedding 包装与二次拼接。
- [ ] 删除未使用的覆盖统计和整份 `min_distances` DtoH。

### Task 4: FeatureBank 发布与逐帧上下文

**Files:**
- Modify: `include/sai/detection/patch_core.h`
- Modify: `src/detection/patch_core.cpp`
- Modify: `include/sai/pipeline/stage_node.h`
- Modify: `src/pipeline/pipeline.cpp`
- Modify: `apps/seat-aoi/app_builder.cpp`
- Modify: `apps/seat-aoi/gui_runner.cpp`
- Test: `tests/detection/patch_core_test.cpp`
- Test: `tests/pipeline/pipeline_test.cpp`

- [ ] 用 `shared_ptr<const FeatureBank>` 快照覆盖 swap 与在途 Detect 测试。
- [ ] 将对应帧的 PatchCore context 附着到 `FrameContext`。
- [ ] 删除应用运行路径对全局 `LastContext()` 的依赖。

### Task 5: Self-evolution 配置与 profile 产物

**Files:**
- Modify: `src/detection/coreset_evolution.cpp`
- Modify: `apps/seat-aoi/coreset_builder.cpp`
- Test: `tests/detection/coreset_evolution_test.cpp`

- [ ] 用真实调用结构测试 `FromYaml(self_evolution_node)`。
- [ ] 统一 parser 契约并拒绝错误层级。
- [ ] 构建 `.bin` 后计算并原子保存 `.bin.profile.yaml`。
- [ ] 测试训练产物可由多机位应用直接加载。

### Task 6: 删除未验证的 IVFFlat 生产路径

**Files:**
- Modify: `include/sai/detection/feature_bank.h`
- Modify: `src/detection/feature_bank.cpp`
- Modify: `src/detection/feature_bank_cuda.cpp`
- Modify: `src/retrieval/vector_path.cpp`

- [ ] 删除 `ConvertToIVF`、`PrepareGpuIvf`、`nprobe` 和 GPU IVF 类型转换。
- [ ] 默认 CPU/GPU 路径均保持精确 FlatL2。

### Task 7: 全量验证

- [ ] 在目标 Linux 环境运行 configure、build、ctest 和静态检查。
- [ ] 在 Linux Release + CUDA + TensorRT + FAISS GPU 环境运行真实 engine smoke test。
- [ ] 用 50 张/机位和 600 张全量数据记录阶段计时、GPU 利用率、峰值显存、coreset 覆盖统计和最终判定回归。
- [ ] 只有在测试输出和真实性能数据支持时才声明完成。
