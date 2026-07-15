# Surface AI Framework —— 在线 Coreset 自进化 设计文档

> Status: Draft
> Date: 2026-07-15
> Based on: 里程碑 3（PatchCore/FeatureBank/DetectionResult）、里程碑 4（KnowledgeEvolution/KnowledgeStore）、里程碑 5（RuleEngine/Reasoner）、里程碑 6（Pipeline）
> Depends on: `sai::detection::FeatureBank`（Rebuild/ExtractAllVectors/BuildWithGreedyCoreset）、`sai::detection::PatchCore`（SetFeatureBank）、`sai::rule::RuleEngine`（EvaluateAll）、`sai::reasoner::IReasoner`（Reason）

---

## 1. 背景与范围

### 1.1 问题陈述

PatchCore 的 coreset 在初始部署前必须采集正常样本构建，这一冷启动步骤无法避免（单类分类的本质要求）。但产线运行后，以下场景会导致 coreset 逐渐偏离实际正常分布：

- **光源老化**：LED 亮度衰减使图像整体偏暗，原始 coreset 中的亮度特征不再匹配
- **材料批次变化**：同型号皮革在不同批次间存在天然纹理差异，coreset 未覆盖新纹理
- **工艺微调**：产线参数（缝线张力、模具温度）的合理漂移导致产品表面特征变化
- **多 SKU 切换**：座椅型号切换时，coreset 需要适配新 SKU 的正常表面特征

当前系统在上述场景下的应对方式是**人工重新采集样本并重建 coreset**——需要停线、采集、标定、部署，周期为数小时到数天。本设计的目标是让 coreset 在运行中自主适应正常分布的变化，将人工干预频率降到最低。

### 1.2 设计目标

1. **零人工标定**：系统从 coreset 自查询的统计分布中自主推导"正常"的边界，不由人设定任何阈值
2. **运行时零停机更新**：检测管线持续运行，coreset 在后台更新后通过 double-buffer 原子替换
3. **冗余剔除**：只有引入新信息（覆盖正常流形稀疏区）的样本才纳入候选，避免 coreset 体积膨胀但信息不增长
4. **固定容量**：coreset 大小维持不变（K），通过 greedy coreset 重选实现新旧样本的优胜劣汰
5. **空闲全量重建**：Pipeline 停止时（换班/维护窗口），利用积累的候选做精确的全量 greedy coreset 重建并持久化
6. **可审计**：每次 coreset 变更通过 KnowledgeEvolution 记录完整审计链

### 1.3 明确排除项

- **缺陷样本的学习**：只学习"高置信正常"样本，任何疑似缺陷的帧绝不纳入 coreset
- **初始采集的替代**：不改变冷启动流程——首次部署仍需采集正常样本
- **模型权重更新**：不涉及 DINOv3/CLIP/任何推理模型的 fine-tune 或权重修改
- **多产品线迁移**：coreset 不跨产品型号迁移——每个 SKU 维护独立的 coreset 和 profile
- **异常检测算法替换**：不改变 PatchCore 的核心算法，仅扩展其 memory bank 的维护方式

---

## 2. 对已有里程碑的依赖

| 依赖组件 | 位置 | 用途 |
|---------|------|------|
| `FeatureBank::Rebuild` | `detection/feature_bank.h:51` | 用新向量集替换 FAISS 索引 |
| `FeatureBank::ExtractAllVectors` | `detection/feature_bank.h:48` | 提取全部 coreset 向量用于合并 |
| `FeatureBank::BuildWithGreedyCoreset` | `detection/feature_bank.h:70` | 从合并集贪心重选固定大小 coreset |
| `PatchCore::SetFeatureBank` | `detection/patch_core.h:72` | 运行时热替换 FeatureBank（需扩展为 SwapFeatureBank——返回旧 unique_ptr，见 §4.5.5） |
| `IDetector::Detect` | `detection/detector.h:27` | 已有 k-NN distances，NormalityScorer 复用 |
| `DetectionResult` | `detection/detection_result.h:39` | 多信号共识：image_level_score |
| `RuleEngine::EvaluateAll` | `rule/` | 多信号共识：规则命中情况 |
| `IReasoner::Reason` | `reasoner/` | 多信号共识：最终裁决 OK/NG/WARN |
| `KnowledgeStore` | `knowledge/knowledge_store.h` | 记录演化事件 |
| `KnowledgeEvolution::Append` | `knowledge/knowledge_evolution.h` | coreset 变更审计链 |
| `Pipeline::ResultCallback` | `pipeline/pipeline.h` | 每帧完成后触发 AssessAndOffer |

---

## 3. 架构总览

### 3.1 旁路自进化通道

不修改 Pipeline 的线性拓扑（`capture → preprocess → inference → detect → rule_eval → reason → export`），在 Pipeline 旁边添加一条独立的自进化通道：

```
                    ┌──────────────────────────────────────────┐
                    │          Pipeline (不变)                   │
                    │  Capture → ... → Detect → ... → Export     │
                    └──────────────┬───────────────────────────┘
                                   │ ResultCallback (已有钩子)
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │     NormalityScorer (新增)                 │
                    │  coreset 自查询分布 → 帧级正常度评分        │
                    │  复用 Detect 的 k-NN distances，零额外查询   │
                    └──────────────┬───────────────────────────┘
                                   │ NormalityAssessment
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │     MultiSignalConsensus (新增)            │
                    │  k-NN 正常度 + PCA + 规则 + Reasoner       │
                    │  全部通过 → "高置信正常"                    │
                    └──────────────┬───────────────────────────┘
                                   │ 通过
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │     NoveltyFilter (新增)                   │
                    │  帧的 patch 覆盖增益 > 阈值？               │
                    │  冗余 → 跳过；有信息 → 纳入候选             │
                    └──────────────┬───────────────────────────┘
                                   │ 通过的候选帧
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │   CandidateBuffer (新增)                   │
                    │  有界环形缓冲：max 50 帧 / 50000 patch      │
                    │  帧数 ≥ 20 OR patch 数 ≥ 20000 → 触发更新  │
                    └──────────────┬───────────────────────────┘
                                   │ DrainAll()
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │   CoresetUpdater 后台线程 (新增)            │
                    │  候选预压缩(轻量贪心) → 合并 → 精准贪心重选  │
                    │  → standby bank → atomic swap              │
                    └──────────────┬───────────────────────────┘
                                   │ swap 完成
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │   KnowledgeEvolution 记录 (已有)           │
                    │  每次更新追加审计记录                       │
                    └──────────────────────────────────────────┘
```

### 3.2 两个触发路径

| 路径 | 触发条件 | 执行策略 | 是否停机 |
|------|---------|---------|---------|
| **运行时更新** | CandidateBuffer 满（帧数 ≥ 20 或 patch 数 ≥ 20000） | 阶梯合并（候选预压缩 + 精准贪心重选），后台线程 | 否（double buffer swap） |
| **空闲全量重建** | `Pipeline::Stop()` 调用时 | 全量精准贪心重选 + 持久化 | 是（但此时管线已停止） |

---

## 4. 组件详细设计

### 4.1 NormalityScorer — 自主正常度评分

#### 4.1.1 Purpose

用 coreset 自身定义"正常"的统计特征，对每帧输出 0~1 的正常度分数。所有阈值均从 coreset 自查询分布中导出，零人工标定。

#### 4.1.2 Design

**为什么用 coreset 自查询而非预设阈值**：预设阈值（如"k-NN 距离 < 0.5 算正常"）隐含了对特征空间尺度的假设，这个假设在不同模型（DINOv3 vs CLIP）、不同维度（768 vs 1024）、不同产品（皮革 vs 金属）下不成立。coreset 自查询直接测量"正常 patch 之间的典型距离"，不依赖任何外部假设。

**为什么用尾部比例（tail ratio）作为主要判据而非中位数**：中位数只反映中心趋势——一个 90% 正常 + 10% 严重异常的帧，中位数可能仍然接近正常。尾部比例直接度量"不正常的 patch 有多少"，对局部异常更敏感。

**为什么尾部参考 P95 而非 P99**：P99 是对极端值的估计，在小样本下不稳定。P95 更稳健，且产线上真正的异常在特征空间中通常显著偏离 P95（实际缺陷的 k-NN 距离往往是 P95 的 3~10 倍）。

#### 4.1.3 算法

**初始化——coreset 自查询**（与 `ComputeAdaptiveThreshold` 合并为一次扫描）：

```
输入: coreset bank (N 个向量, dim D), k = 5

对 coreset 中每个向量 vi (i = 0..N-1):
  dists = bank.Search(vi, 1, k+1)        // k+1 跳过自身
  di = dists[k]                           // 取第 k 近邻距离

对 {di} 排序:
  P50 = 中位数
  P95 = 第 95 百分位
  P99 = 第 99 百分位
  mean = 均值
  stddev = 标准差

输出: NormalityProfile { k, dim, N, P50, P95, P99, mean, stddev }
```

O(N²·D) 单次，约 1~3 秒，仅初始化时执行一次。`NormalityProfile` 随 coreset 持久化到 `.profile.yaml`，下次启动直接加载。

**每帧——正常度评分**（复用 PatchCore::Detect 的 k-NN distances，零额外查询）：

```
输入: query distances (M = grid_h × grid_w 个距离值), profile

// 1. 集中度
median_dist = median(distances)           // O(M) with nth_element
concentration = median_dist / profile.P50

// 2. 尾部比例
tail_count = count(distances > profile.P95)
tail_ratio = tail_count / M

// 3. 综合评分
if tail_ratio == 0:
    normalcy_score = 1.0                  // 全部 patch 在正常范围内
else:
    // 尾部比例越大，正常度越低。tail_ratio_max 来自 YAML 配置（默认 0.10）
    normalcy_score = 1.0 - min(1.0, tail_ratio / cfg.tail_ratio_max)

// concentration 作为辅助参考（用于调试和监控，不直接参与评分）
```

**为什么用 `tail_ratio / tail_ratio_max` 做线性映射**：`tail_ratio_max` 的含义是"不可接受的尾部比例上限"——当 10% 的 patch 落在 P95 以外时正常度归零。这个值直接从配置读取，用户可根据产品线的纹理一致性调整：纹理高度一致的金属表面可设更严格（如 0.05），纹理天然的皮革可设更宽松（如 0.15）。

#### 4.1.4 Data Structure

```cpp
struct NormalityProfile {
    std::size_t k_nearest;
    std::size_t dim;
    std::size_t num_samples;
    float p50;
    float p95;
    float p99;
    float mean;
    float stddev;
};

struct NormalityAssessment {
    float normalcy_score;         // 0~1，越高越正常
    float concentration_ratio;    // median(query) / profile.P50
    float tail_ratio;             // 超过 P95 的 patch 比例
};
```

### 4.2 MultiSignalConsensus — 多信号共识

#### 4.2.1 Purpose

k-NN 正常度评分只是单维度信号。一个帧可能在 k-NN 空间中看起来正常（低距离），但在 PCA 子空间中异常——或者刚好相反。多信号共识要求所有维度一致确认"正常"才通过。

#### 4.2.2 判定条件

```
consensus_ok = ALL of:
  1. normalcy_score >= 0.9                          // k-NN 维度正常
  2. detection.image_level_score < effective_threshold  // 图像级分数在阈值以下
  3. 规则全部未命中 (rule_eval matched_rules == 0)    // 没有命中任何缺陷规则
  4. reasoner 最终裁决 == "OK"                        // 决策树判为 OK

如果启用了 PCA 混合评分:
  5. pca_image_score < pca_self_query_P95            // PCA 子空间也正常
     (pca_self_query_P95 在初始化时通过对 coreset 自身做 PCA 评分得到)
```

**为什么要求全部通过而非多数投票**：自进化的代价很高——错误地将缺陷纳入 coreset 会导致该缺陷类型在后续检测中永久漏检。宁可保守（少纳入几个正常帧），不能激进（纳入一个缺陷帧）。

### 4.3 NoveltyFilter — 冗余剔除

#### 4.3.1 Purpose

高置信正常的帧不一定都值得加入 coreset。如果一个帧的所有 patch 都紧密围绕已有 coreset 点，加入它不会改善对正常流形的覆盖——单纯增加 FAISS 索引体积和搜索延迟。

#### 4.3.2 Design

**为什么用 P50 作为"已覆盖"的阈值**：P50 是中位数——一半的 coreset 向量与最近邻的距离小于 P50。如果 query patch 的距离也小于 P50，说明这个 patch 已经有一个"比一半 coreset 点之间的典型距离更近"的邻居，无需再添加。P95 太松（大部分 patch 都能通过），P99 太紧（几乎全部被过滤），P50 处于信息增益和筛选精度之间的平衡点。

**为什么保留全帧而非只保留稀疏 patch**：patch 的空间上下文对 greedy coreset 重选有价值——相邻的 patch 才能代表连续的表面纹理特征。只保留稀疏 patch 会破坏这种空间结构，导致后续 greedy coreset 选取的点无法有效代表原始帧的纹理模式。冗余 patch 在 greedy coreset 步骤会被自然淘汰。

#### 4.3.3 算法

```
输入: query distances (M 个), profile

// 统计"已被 coreset 密集覆盖"的 patch 数
covered_count = count(distances < profile.P50)
coverage_ratio = covered_count / M

// 判定
if coverage_ratio < 0.6:    // 至少 40% 的 patch 不在密集覆盖区
    is_novel = true          // 帧带来了新信息
else:
    is_novel = false         // 冗余，跳过

输出: NoveltyResult { is_novel, coverage_ratio, novel_patch_count }
```

**为什么 coverage_threshold 设置为 0.6**：意味着一帧中至少 40% 的 patch 必须落在 coreset 稀疏区，才被认为有信息增益。这个值在典型产线场景下意味着：一个略有纹理变化但仍正常的新批次材料会被纳入（~50% patch 覆盖），但一个与已有样本几乎完全相同的帧会被跳过（~95% patch 覆盖）。该值可通过 YAML 配置调整以适应不同产品线的纹理多样性。

### 4.4 CandidateBuffer — 有界候选缓冲

#### 4.4.1 Purpose

积累通过 NoveltyFilter 的候选帧，在达到阈值时触发 CoresetUpdater。硬上限防止无限积累。

#### 4.4.2 Design

**为什么有界而非无界**：产线上可能连续数小时不触发更新（材料稳定、光源稳定），也可能在换料后一小时内产生数百个高信息增益候选。无界缓冲在后者场景下会耗尽内存，且积累过多候选后单次更新耗时过长。有界缓冲 + 阶梯合并确保了更新耗时的可预测性。

**为什么用帧数和 patch 数双重触发**：不同分辨率/模型下，单个帧的 patch 数量差异很大（DINOv3 518×518 → 37×37=1369 patch；高分辨率 1024×1024 → 73×73=5329 patch）。仅用帧数触发在低分辨率下可能积累不足，仅用 patch 数触发在不同分辨率下触发频率不一致。双重条件取"或"逻辑——任一满足即触发，确保在多种部署配置下行为合理。

#### 4.4.3 Data Structure

```cpp
struct EvolutionCandidate {
    std::shared_ptr<const float> patch_vectors;  // shared_ptr 跨线程安全传递
    std::size_t grid_h;
    std::size_t grid_w;
    std::size_t dim;
    float normalcy_score;
    std::chrono::steady_clock::time_point captured_at;
};

class CandidateBuffer {
public:
    explicit CandidateBuffer(std::size_t max_frames = 50,
                             std::size_t max_patches = 50000,
                             std::size_t trigger_frames = 20,
                             std::size_t trigger_patches = 20000);

    // 检测线程调用（热路径，锁临界区极小）
    // 返回 true = 已加入候选，false = 缓冲区满且不替换
    auto Append(EvolutionCandidate candidate) -> bool;

    // 是否触发更新？（任一条件满足）
    auto IsTriggered() const -> bool;

    // 后台线程调用：一次性取出全部候选并清空
    auto DrainAll() -> std::vector<EvolutionCandidate>;

    // 查询
    auto FrameCount() const -> std::size_t;
    auto PatchCount() const -> std::size_t;

private:
    mutable std::mutex mutex_;
    std::vector<EvolutionCandidate> candidates_;
    std::size_t max_frames_;
    std::size_t max_patches_;
    std::size_t trigger_frames_;
    std::size_t trigger_patches_;
    std::size_t total_patches_ = 0;
};
```

#### 4.4.4 Thread Safety

- `Append`: 检测线程调用。`lock_guard` 保护 `push_back` + counter 更新，临界区 ~微秒
- `DrainAll`: 后台线程调用。`lock_guard` 保护 swap + counter 重置
- `IsTriggered`: 可在任意线程调用，`lock_guard` 保护

### 4.5 CoresetUpdater — 合并与重选

#### 4.5.1 Purpose

后台线程消费 CandidateBuffer 中的候选，与现有 coreset 合并后通过 greedy coreset 重选维持固定大小 K，最终通过 double-buffer swap 实现零停机热替换。

#### 4.5.2 Design

**为什么阶梯合并而非直接全量重选**：候选经过预压缩（轻量贪心）后与 coreset 合并，总向量数控制在 ~15000 以内（coreset 10000 + 预压缩候选 5000）。精准贪心重选的复杂度是 O(K·N·D)，在此规模下约 200~500ms——在后台线程可接受。若不预压缩，连续积累可能导致合并集达到 50000+，精准贪心重选耗时数秒，影响更新的及时性。

**为什么用 double buffer 而非读写锁**：读写锁在"高频读、低频写"场景下表现为写者饥饿——检测线程持续持有读锁（每帧一次 Detect），后台写者可能长时间无法获取写锁。double buffer 完全解耦读写路径——Detect 总是通过稳定指针读取当前 active bank，swap 是几个指针的原子交换，检测路径零阻塞。

**为什么 coreset 大小固定而非动态增长**：FAISS IndexFlatL2 的搜索延迟与索引大小成正比。固定大小保证搜索延迟的可预测性——无论系统运行了多久、纳入了多少候选，每次 Detect 的 k-NN 搜索耗时稳定。若 coreset 可动态增长，延迟会随时间持续上升，最终超出产线节拍要求。

#### 4.5.3 算法

```
Update(candidates):
  // 1. 候选预压缩（轻量贪心）
  all_candidate_patches = Flatten(candidates)           // Nc × D
  target = min(greedy_prefilter, Nc)                    // 默认 5000
  prefiltered = LightGreedySelect(all_candidate_patches, target)
  // LightGreedySelect: 从候选 patch 中均匀采样 target 个种子点，
  // 然后执行一轮迭代——对剩余向量各计算到最近种子点的距离，选距离最大的
  // target/2 个加入种子，形成一个比纯均匀采样更好的初始覆盖。
  // 复杂度 O(Nc·target·D)，在 Nc=20000, target=5000, D=1024 下约 ~200ms

  // 2. 与现有 coreset 合并
  existing = active_bank.ExtractAllVectors()            // K × D
  merged = Concat(existing, prefiltered)                // (K + target) × D

  // 3. 精准贪心重选
  selected = GreedyCoresetSelect(merged, K)             // 复用 BuildWithGreedyCoreset 核心

  // 4. 构建新 bank 到 standby
  standby_bank.Rebuild(selected.data(), K, dim)
  new_profile = NormalityProfile::ComputeFast(standby_bank, old_profile)
  // ComputeFast: 对新 coreset 随机采样 sqrt(K) 个向量做自查询（而非全量），
  // 用采样估计 P50/P95。运行时更新需要控制耗时，完整的全量自查询（O(K²·D)）
  // 仅在初始化（§4.1.3）和 FullRebuild（§4.6）时执行。

  // 5. Double-buffer swap: 注入新 bank，回收旧 bank 作为下一次的 standby
  auto old_bank = detector.SwapFeatureBank(std::move(standby_bank));
  standby_bank = std::move(old_bank);
  // SwapFeatureBank 是 SetFeatureBank 的扩展：返回旧 unique_ptr<FeatureBank>
  // 这样 CoresetEvolution 始终持有一个 standby，避免反复 ExtractAll+拷贝

  // 6. 同步 swap profile
  std::swap(active_profile, standby_profile);

  // 6. 记录演化事件
  RecordEvolutionEvent(stats)
```

#### 4.5.4 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target_size` | 10000 | coreset 固定大小，与初始保持一致 |
| `min_update_interval` | 5s | 两次更新最小间隔，防止频繁重建 |
| `greedy_prefilter` | 5000 | 候选预压缩后的 patch 数 |

#### 4.5.5 Swap 期间的数据安全

CoresetEvolution 内部维护一个 standby FeatureBank（初始为 active 的深拷贝）。更新完成后，通过 `PatchCore::SetFeatureBank()` 将新 bank 注入——`SetFeatureBank` 内部做 `std::unique_ptr` swap，旧 bank 转由 CoresetEvolution 持有（作为下一次更新的 standby），其内存在下一次 swap 或析构时才释放。

Swap 时刻正在执行的 `Detect()` 已经通过裸指针访问了旧 bank 的 FAISS index。`SetFeatureBank` 中的 `unique_ptr` swap 不会释放旧 bank（因为旧 unique_ptr 转给了 standby），因此 Detect 可以安全完成当前搜索。

### 4.6 空闲全量重建 + 持久化

#### 4.6.1 Purpose

运行时阶梯合并为控制延迟而在候选预压缩阶段引入了近似误差。Pipeline 停止时（换班/维护窗口），利用积累的候选做一次不做阶梯压缩的精确全量重建，消除累积的近似误差。

#### 4.6.2 触发与流程

```
触发: Pipeline::Stop() → CoresetEvolution::Stop() 内部

FullRebuild(save_path):
  1. existing = active_bank.ExtractAllVectors()
  2. candidates = buffer.DrainAll()      // 取出所有剩余候选
  3. merged = Concat(existing, Flatten(candidates))
  4. selected = GreedyCoresetSelect(merged, K)  // 不做预压缩，全量精准
  5. active_bank.Rebuild(selected)        // Stop 后无检测线程，直接重建
  6. active_profile = ComputeNormalityProfile(active_bank)
  7. active_bank.SaveToFile(save_path)
  8. active_profile.SaveToYaml(save_path + ".profile.yaml")
  9. 可选：备份旧 bank（save_path + ".backup.{timestamp}.bin"）
```

#### 4.6.3 持久化文件

```
models/
  seat_leather_coreset.bin                 # coreset 原始 float32（与现有格式兼容）
  seat_leather_coreset.bin.profile.yaml    # NormalityProfile
  seat_leather_coreset.bin.backup.20260715T143000Z.bin  # 历史备份（最多 3 个）
```

`profile.yaml` 格式：

```yaml
profile:
  k_nearest: 5
  dim: 1024
  num_samples: 10000
  created_at: "2026-07-15T14:30:00.123456Z"
  statistics:
    p50: 2.347
    p95: 5.891
    p99: 8.234
    mean: 2.712
    stddev: 1.423
  evolution:
    total_updates: 47
    total_candidates_consumed: 312
    last_updated_at: "2026-07-15T16:45:00.654321Z"
```

启动时若 `.profile.yaml` 存在则直接加载（跳过初始化自查询）；若不存在（首次部署）则从 coreset 计算。

### 4.7 KnowledgeEvolution 集成

#### 4.7.1 Purpose

每次 coreset 更新在 KnowledgeGraph 中创建一条 `CoresetEvolutionEvent` 节点，并通过 `KnowledgeEvolution::Append` 追加到 changelog。形成完整的可审计演化历史。

#### 4.7.2 记录字段

| 字段 | 类型 | 含义 | 用途 |
|------|------|------|------|
| `event_type` | string | `"RuntimeUpdate"` 或 `"FullRebuild"` | 区分更新路径 |
| `frames_added` | int64 | 本次合并的候选帧数 | 评估触发频率 |
| `patches_added` | int64 | 合并的总 patch 向量数 | 评估数据流量 |
| `patches_removed` | int64 | 被贪心重选淘汰的 patch 数 | 评估信息更新率 |
| `size_before` | int64 | 更新前 coreset 大小 | 验证固定容量约束 |
| `size_after` | int64 | 更新后 coreset 大小 | 应始终等于 target_size |
| `mean_displacement` | float64 | 新旧 coreset 中心的 L2 距离 | 监测概念漂移 |
| `coverage_gain` | float64 | 稀疏区覆盖率改善幅度 | 评估更新有效性 |
| `update_duration_ms` | int64 | 更新耗时（毫秒） | 性能监控 |
| `before_profile` | string | 旧 NormalityProfile 的 JSON | 可回滚 |

#### 4.7.3 退化告警

`mean_displacement` 是概念漂移的关键指标。CoresetUpdater 维护一个滑动窗口（最近 5 次更新），若连续 3 次更新的位移量均超过历史均值的 2 个标准差，通过 Logger 发出 Warning：

```
"[CoresetEvolution] Potential concept drift detected: mean_displacement=0.347
 exceeds 2σ threshold=0.218 over last 3 updates. Normal manifold may be shifting
 — consider checking lighting/materials. Self-evolution continues."
```

**不阻止更新**——系统继续自适应，仅通知运营人员关注数据质量。

---

## 5. YAML 配置

### 5.1 完整配置项

```yaml
# 在 Detect stage 的 config 中扩展
self_evolution:
  enabled: true

  normality:
    k_self_query: 5
    tail_percentile: 0.95
    tail_ratio_max: 0.10        # 尾部比例 ≥ 0.10 时 normalcy_score 归零（= 不可接受的尾部比例上限）

  novelty:
    coverage_threshold: 0.60    # coverage_ratio 低于此值认为有信息增益
    distance_percentile: 0.50   # P50 作为"已覆盖"的距离阈值

  buffer:
    max_frames: 50
    max_patches: 50000
    trigger_frames: 20
    trigger_patches: 20000

  update:
    target_size: 10000
    min_interval_sec: 5
    greedy_prefilter: 5000

  persistence:
    save_on_stop: true
    backup_old_bank: true
    max_backups: 3
```

### 5.2 推荐默认值（开箱即用，无需调整）

所有值均基于 DINOv3 ViT-B (dim=1024, patch=14, 518×518→37×37=1369 patch) 场景校准。更换模型或分辨率时可能需要调整 `trigger_patches` 和 `greedy_prefilter`。

---

## 6. 公开接口

### 6.1 头文件

新增 `include/sai/detection/coreset_evolution.h`，核心类 `CoresetEvolution`：

```cpp
namespace sai::detection {

class CoresetEvolution final : public Object {
public:
    // 构造时传入初始 FeatureBank（用于计算/加载 profile）和 PatchCore 引用（用于 swap 回写）。
    // active_bank 的引用仅在构造期间使用——拷贝初始向量到 standby，不持有引用。
    // 之后的 double-buffer swap 通过 PatchCore::SetFeatureBank() 回写。
    CoresetEvolution(EvolutionConfig cfg,
                     PatchCore& detector,
                     NormalityProfile profile) noexcept;

    // ── 每帧调用（检测线程，零阻塞，~微秒级） ──
    // distances: PatchCore::Search 返回的 k-NN distances（复用，不额外查询）
    // query_count: grid_h * grid_w
    // k: k_nearest
    // det_result, rule_output, reason_output: 多信号共识
    //
    // 无返回值（void）—— 这是热路径，禁止失败阻塞检测主线。
    // 内部异常全部吞掉 + 打日志 + counter 自增。
    auto AssessAndOffer(const float* distances,
                        std::size_t query_count,
                        std::size_t k,
                        const DetectionResult& det_result,
                        const RuleEvalOutput& rule_output,
                        const ReasoningResult& reason_output) noexcept -> void;

    // ── 启动/停止后台更新线程 ──
    auto Start(std::stop_token token) noexcept -> void;
    auto Stop() noexcept -> void;   // 阻塞直到线程退出 + 触发 FullRebuild

    // ── 查询 ──
    auto IsRunning() const noexcept -> bool;
    auto LatestStats() const noexcept -> EvolutionStats;
    auto Profile() const noexcept -> NormalityProfile;

    // ── Knowledge 集成 ──
    auto BindKnowledgeStore(std::shared_ptr<knowledge::KnowledgeStore> ks) noexcept -> void;

    // ── 显式触发全量重建 ──
    auto FullRebuild(const std::filesystem::path& save_path) noexcept -> Result<void>;

    // Object 约束：禁止移动/拷贝
    CoresetEvolution(CoresetEvolution&&) noexcept = delete;
    CoresetEvolution(const CoresetEvolution&) = delete;
};

}  // namespace sai::detection
```

### 6.2 集成示例（seat_aoi）

```cpp
// 1. 加载 coreset 和 profile
auto bank = FeatureBank::LoadFromFile("models/coreset.bin", 1024);
auto profile_path = "models/coreset.bin.profile.yaml";
auto profile = fs::exists(profile_path)
    ? NormalityProfile::LoadFromYaml(profile_path)
    : NormalityProfile::Compute(*bank);

// 2. 注入到 PatchCore
auto detector = std::make_unique<PatchCore>(cfg);
detector->SetFeatureBank(std::move(bank));

// 3. 创建 CoresetEvolution（传入 detector 引用用于 swap 回写）
auto evo = std::make_unique<CoresetEvolution>(evo_cfg, *detector, std::move(profile));
evo->BindKnowledgeStore(ks);

// 4. Pipeline callback 中调用
pipeline->SetResultCallback([&](const InspectionResult& r) {
    evo->AssessAndOffer(r.knn_distances.data(),
                        r.knn_distances.size() / r.k,
                        r.k,
                        r.detection,
                        r.rule_output,
                        r.reasoning_result);
    updateDashboard(r);
});

// 5. 启动
evo->Start(stop_token);

// 6. 停止（内部触发 FullRebuild + 持久化）
evo->Stop();
```

---

## 7. 线程模型

```
┌─────────────────────────────────────────────────────────────┐
│              检测线程（Pipeline worker）                      │
│                                                             │
│  PatchCore::Detect()  ─→  distances (已有)                  │
│  AssessAndOffer():                                          │
│    NormalityScorer     O(M)    ~微秒                         │
│    MultiSignalConsensus O(1)   ~纳秒                         │
│    NoveltyFilter       O(M)    ~微秒                         │
│    CandidateBuffer     lock+n   ~微秒                        │
│                                                             │
│  总增量: ~数十微秒/帧，对检测延迟的影响 < 0.1%               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│         CoresetUpdater 后台线程 (1 x std::jthread)            │
│                                                             │
│  loop:                                                      │
│    wait(cv, stop_token)                                     │
│    candidates = buffer.DrainAll()        [lock, ~ns]        │
│    PrefilterCandidates(candidates)       [~50ms]            │
│    existing = active.ExtractAllVectors() [O(K·D)]           │
│    merged = Concat                       [O((K+P)·D)]       │
│    selected = GreedyCoreset(merged, K)   [~200-500ms]      │
│    standby.Rebuild(selected)             [O(K·D)]           │
│    new_profile = ComputeFast(standby)    [~50ms 采样]       │
│    old = detector.SwapFeatureBank(       [~1μs]             │
│            std::move(standby))                              │
│    standby = std::move(old)  // 旧 bank 作为下次 standby   │
│    swap(active_profile↔standby_profile)                     │
│    RecordEvolutionEvent()                [IO, ~ms]          │
│                                                             │
│  总耗时: ~500ms-2s，后台运行，零检测中断                     │
└─────────────────────────────────────────────────────────────┘

Swap 时刻:
  - 正在执行的 Detect() 持有旧 active_bank 裸指针 → 继续正常执行
  - 下一个 Detect() 自动走新 active_bank
  - 旧 bank 在本次 Detect 返回后由 standby 回收
  - 检测路径零锁、零阻塞、零原子操作
```

---

## 8. 错误处理

| 场景 | 策略 | 恢复 |
|------|------|------|
| `AssessAndOffer` 内部异常 | 吞掉 + 打 Error 日志 + 内部 counter++ | 不传播，检测主线不受影响 |
| 后台线程更新失败（单次） | 记录错误，保留候选数据，等下次触发重试 | 自动恢复 |
| 后台线程连续 3 次失败 | emit `Detection_CoresetEvolution_UpdateFailed` Alarm，停止自进化 | 需重启 Pipeline 恢复 |
| 候选预压缩超时 | 减少候选用量，只用 coreset 自身的 greedy 维护（跳过本轮候选） | 下轮重试 |
| FAISS rebuild 失败 | 保持旧 active_bank 不变，记录错误 | 下轮重试 |
| 磁盘满（持久化失败） | 降级：不写文件，coreset 在内存中仍然有效 | 清理磁盘后下次 Stop 自动重试 |
| profile 文件损坏/不存在 | 重新从 coreset 计算 | 自动 |
| 概念漂移 Warning | Log Warning + KnowledgeEvolution 记录，不停止更新 | 人工检查数据质量 |

### 新增 ErrorCode

在 `error.h` 末尾追加（follow append-only，不碰已有枚举）：

```
Detection_CoresetEvolution_UpdateFailed    // 运行时 coreset 更新连续失败
Detection_CoresetEvolution_Degraded        // 概念漂移告警（mean_displacement 持续偏高）
Detection_CoresetEvolution_FullRebuildFailed // 空闲全量重建失败
Detection_CoresetEvolution_ProfileLoadFailed // profile 文件损坏且无法重新计算
```

---

## 9. 测试策略

### 9.1 单元测试（GoogleTest，macOS 可运行）

| 测试 | 内容 |
|------|------|
| `NormalityProfile.ComputeFromBank` | 用 MockEngine + SimplePatchEmbedder 构建小型 FeatureBank，验证 P50/P95/P99/mean/stddev 计算正确 |
| `NormalityProfile.RoundTripYaml` | SaveToYaml → LoadFromYaml → 所有字段相等 |
| `NormalityScorer.PerfectNormalGetsHighScore` | 用 coreset 中的向量作为 query，验证 normalcy_score ≈ 1.0 |
| `NormalityScorer.OutlierGetsLowScore` | 用随机噪声向量作为 query，验证 normalcy_score 低 |
| `NormalityScorer.ZeroExtraQueries` | 验证 AssessAndOffer 不调用 FeatureBank::Search（通过 mock） |
| `MultiSignalConsensus.AllPassed` | 所有信号 OK → consensus_ok = true |
| `MultiSignalConsensus.RuleHit` | 规则命中 → consensus_ok = false |
| `MultiSignalConsensus.ReasonerNG` | Reasoner 判 NG → consensus_ok = false |
| `NoveltyFilter.NovelFramePasses` | 构建稀疏 + 密集混合的 distances，验证 coverage_ratio < 0.6 时通过 |
| `NoveltyFilter.RedundantFrameBlocked` | 全部 distances < P50 → coverage_ratio = 1.0 → 被过滤 |
| `CandidateBuffer.TriggerByFrames` | 添加 20 帧 → IsTriggered() = true |
| `CandidateBuffer.TriggerByPatches` | 添加少量大帧（大量 patch）→ IsTriggered() = true |
| `CandidateBuffer.DrainAllClears` | DrainAll 后 FrameCount() = 0 |
| `CoresetUpdater.FixedSizeAfterUpdate` | 更新后 coreset.Size() = target_size |
| `CoresetUpdater.SwapDoesNotAffectSearch` | Swap 前后分别 Search，验证能正常返回结果 |
| `EvolutionStats.MeanDisplacement` | 两次相同 coreset 更新后 displacement ≈ 0 |

### 9.2 集成测试

| 测试 | 内容 |
|------|------|
| `EndToEndSelfEvolution` | 1) 构建初始 coreset → 2) 注入 50 帧"正常但有纹理变化"的 frame → 3) 验证 coreset 被更新 ≥ 2 次 → 4) 验证更新后仍能检测已知缺陷 |
| `DefectNeverIncluded` | 含缺陷的帧在 MultiSignalConsensus 被拒绝，验证 CandidateBuffer 中无缺陷帧 |
| `FullRebuildOnStop` | Start → 注入候选 → Stop → 验证 .bin + .profile.yaml 文件存在且格式正确 |
| `RecoveryFromCorruptProfile` | 删除 .profile.yaml → 重新启动 → 自动从 .bin 计算 profile |

---

## 10. 性能约束

| 指标 | 约束 | 测量方法 |
|------|------|---------|
| `AssessAndOffer` 单帧耗时 | < 100μs | GoogleTest 计时 |
| 检测延迟增量 | < 0.1%（对比未启用自进化的基线） | 集成测试中统计 Detect 延迟分布 |
| 后台更新耗时 | < 2s（target_size=10000, greedy_prefilter=5000） | 单元测试计时 |
| 内存增量（稳态） | < 2× coreset 大小（double buffer） | 进程 RSS 监控 |
| 持久化文件大小 | coreset.bin: K×D×4 bytes; profile.yaml: < 1KB | 文件系统检查 |

---

## 11. 未来扩展

| 方向 | 描述 | 依赖 |
|------|------|------|
| **多 SKU 独立 coreset** | 按 SKU ID 路由到不同的 FeatureBank + profile，CoresetEvolution 管理一组 bank 而非单个 | SKU 识别信号（来自 MES 或 Camera metadata） |
| **FAISS IVF 索引** | 当 coreset 规模超过 10^5 时，IndexFlatL2 搜索延迟不可接受。切换到 IVF 索引做近似搜索 + 在线增量 add | FAISS IVF 在 macOS arm64 上的构建验证 |
| **时序衰减** | 给候选帧附加时间戳权重，greedy coreset 重选时偏好最近的样本——加速适应产线环境变化 | 需要在 greedy coreset 选择算法中引入加权距离 |
| **跨产线 coreset 共享** | 同型号产线 A 的 coreset 更新可推送至产线 B（需人工审批） | 产线间网络 + 审批工作流 |
| **A/B coreset 实验** | 同时维护两个 coreset（stable + experimental），experimental 的检测结果与 stable 对比，达标后 promote | double-buffer 机制已有基础，需额外 Diff 评分 |

---

## 12. 实现批次划分

```
Coreset 在线自进化
├── 批 1: NormalityProfile + NormalityScorer
│         （Compute, Load/Save YAML, Assess，单元测试）
├── 批 2: NoveltyFilter + CandidateBuffer
│         （冗余剔除，有界缓冲，线程安全，单元测试）
├── 批 3: MultiSignalConsensus
│         （多信号联合判定，单元测试）
├── 批 4: CoresetUpdater
│         （后台线程，阶梯合并，double-buffer swap，单元测试）
├── 批 5: 空闲全量重建 + 持久化
│         （FullRebuild, SaveToFile, profile 持久化）
├── 批 6: KnowledgeEvolution 集成
│         （演化事件记录，退化告警）
├── 批 7: YAML 配置解析 + seat_aoi 集成
│         （EvolutionConfig 从 YAML 解析，end-to-end 集成测试）
```

每个批次：设计 review → 代码实现 → 单元测试 → 批次 review。执行顺序必须严格递增（后一批次代码依赖前一批次冻结的接口）。
