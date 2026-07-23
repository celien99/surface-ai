# PatchCore 正确性与性能加固设计

## 目标

在不降低 PatchCore 算法覆盖质量的前提下，修复训练与在线推理的数据契约、并发和配置问题，并将当前约 20 分钟的单机位 coreset 构建热点降为真正的增量贪心最远点采样。

## 不可违反的约束

- coreset 最终选择必须保持贪心最远点采样：第 `k` 个点是相对前 `k-1` 个已选点的最近距离最大的全局候选点。
- 不允许用均匀、stride、reservoir 或随机采样替代最终贪心阶段。
- 如未来引入候选预筛选，必须是显式可选，并通过覆盖误差、缺陷召回和最终判定一致性门禁；默认路径仍对完整候选集执行精确贪心。
- TensorRT 输入必须明确颜色顺序、布局、归一化和 dtype；输出必须转换为下游约定的 float32 patch 特征。
- 在线 FeatureBank 必须先完整构建再原子发布；在途检测必须持有自己的不可变快照。
- 演化上下文属于单帧数据，不能通过进程级 `LastContext` 在异步流水线阶段间传递。

## 方案

采用分阶段、正确性优先的方案：

1. 为 Tensor binding 增加 dtype，校验 engine 的名称、方向、形状、dtype 和字节数。DINO 输入统一为 RGB8 HWC 到 NCHW FP16/FP32，执行 `/255` 和 ImageNet/DINO mean/std 归一化；FP16 输出转换为 float32 patch 特征。
2. coreset 使用等价的增量 FPS。只计算“新选点到所有候选点”的距离，并更新持久化的 `min_dist`，复杂度从重复搜索整个已选集合的近似 `O(N*K^2*D)` 恢复为 `O(N*K*D)`。固定首点为索引 0，并以最小索引解决距离相同的情况，保证确定性。
3. 增加 Linux Release preset 和训练分阶段计时，使读取、上传、embedding、候选拼接、贪心选择、保存分别可观测。
4. 将 FeatureBank 改为不可变 `shared_ptr` 快照的原子发布；将 PatchCore 检测上下文附着到对应 `FrameContext`。
5. 统一 `SelfEvolutionConfig::FromYaml` 接收 `self_evolution` 子节点，并在训练产物旁生成匹配的 normality profile。
6. 删除未通过质量门禁的 IVFFlat 生产路径，在线检索统一使用精确 FlatL2。
7. 训练按机位提取、精确贪心、保存并释放；同一机位仍使用完整候选集，不跨机位常驻 embedding。

## 代码组织原则

- 按职责拆分 CPU、CUDA、流水线和业务装配代码，不以减少文件数量为目标。
- 同步/异步路径复用同一实现；公共逻辑只保留一份。
- 热路径保持扁平，不引入多层虚调用、回调包装或临时对象链。
- 删除无质量门禁的 IVFFlat、全局 `LastContext`、重复 binding 查询和无用转换缓冲。

## 错误处理与兼容性

- 对不支持的 TensorRT dtype、错误形状、错误字节数和颜色格式返回明确错误，不进行隐式猜测。
- CUDA/FAISS 异常在 `noexcept` 边界内转换为 `Result` 或安全失败，避免触发 `std::terminate`。
- 现有 Debug preset 保留，Release 使用独立构建目录。
- 无 CUDA 的平台保留 CPU 参考实现和纯 CPU 单元测试；Linux/CUDA 上补充 GPU 一致性与性能验证。

## 验证标准

- 合成图像精确验证通道、NCHW 索引、归一化和 FP16/FP32 字节数。
- 小数据集上的新 FPS 与直接参考实现逐轮选择索引完全一致，并覆盖重复向量早停。
- 记录 68,450×768、最多 10,000 点场景的阶段耗时；不得再出现随已选集合增长而重复全索引搜索。
- 并发测试证明旧 FeatureBank 在检测快照释放前不会销毁，逐帧上下文不会串帧。
- YAML 测试使用应用实际传入的子节点结构；训练后 `.bin.profile.yaml` 可被应用直接加载。
- 在线检索代码中不存在 IVFFlat 转换、`nprobe` 或隐式近似路径。
