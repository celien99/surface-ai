# Seat AOI main.cpp 重构设计

## 1. 背景

`apps/seat-aoi/main.cpp` 当前 1173 行，承担了 6 个独立职责：CLI 解析、KG 种子数据、Coreset 构建、调参 YAML 工具、应用装配、Headless/GUI 双模式调度。三个函数超过 100 行（`BuildCoreset` 215 行，`main` 710 行，`run_headless` lambda 115 行），`pipeline.yaml` 被重复 parse 4 次，硬编码路径散落 12+ 处。

## 2. 目标

将 `main.cpp` 拆分为 9 个聚焦的文件，`main()` 缩减至 ~30 行。行为零变化——只移动代码，不修改逻辑、不优化、不修 bug。

## 3. 文件拆分

```
apps/seat-aoi/
├── main.cpp              # ~30 行：ParseArgs → BuildCoreset | AssembleApplication → RunHeadless | RunGui
├── app_config.h           # 资源路径常量集中定义
├── cli_args.h / cli_args.cpp        # CliArgs + ParseArgs
├── coreset_builder.h / coreset_builder.cpp  # BuildCoreset
├── knowledge_seed.h / knowledge_seed.cpp    # SeedKnowledgeGraph
├── app_builder.h / app_builder.cpp          # AssembleApplication → Result<AssembledApp>
├── headless_runner.h / headless_runner.cpp  # RunHeadless
├── gui_runner.h / gui_runner.cpp            # RunGui
├── tuning_wiring.h / tuning_wiring.cpp      # 调参装配辅助（被 AppBuilder 调用）
├── CMakeLists.txt         # add_executable 加入新 .cpp 文件
├── qml.qrc
└── resources/
```

## 4. AssembledApp 结构体

```cpp
struct AssembledApp {
    // 核心（必须存在）
    std::unique_ptr<sai::Context> ctx;
    std::unique_ptr<sai::pipeline::Pipeline> pipeline;
    std::shared_ptr<sai::embedding::PatchEmbedder> embedder;
    std::shared_ptr<sai::detection::PatchCore> patch_core;
    std::shared_ptr<sai::rule::RuleEngine> rule_engine;
    std::shared_ptr<sai::io::JsonExporter> exporter;
    std::shared_ptr<sai::knowledge::KnowledgeGraph> kg;
    std::shared_ptr<sai::knowledge::KnowledgeEvolution> kg_evolution;
    std::unique_ptr<sai::knowledge::KnowledgeStore> knowledge_store;

    // 可选组件
    std::shared_ptr<sai::embedding::IEmbedder> global_embedder;
    std::shared_ptr<sai::inference::Sam2Segmenter> sam2_segmenter;
    std::shared_ptr<sai::reasoner::IReasoner> reasoner;
    std::shared_ptr<sai::detection::FeatureBank> feature_bank;
    std::shared_ptr<sai::retrieval::VectorPath> vector_path;
    std::optional<sai::detection::CoresetEvolution> evolution;
    std::optional<sai::tuning::TuningScheduler> tuning_scheduler;
};
```

- `knowledge_store` 放在最前面声明（但在结构体末尾位置），确保析构时最后释放——`kg` 和 `kg_evolution` 是它的 non-owning 引用
- 可选组件用 `std::optional` 表达"启用/未启用"
- `AssembleApplication` 返回 `Result<AssembledApp>`，装配失败时传播错误而非 `exit(1)`

## 5. 各文件职责

### 5.1 `app_config.h`

集中定义所有资源路径常量：

```cpp
namespace seat_aoi::config {
    constexpr std::string_view kPipelineYaml = "resources/pipeline.yaml";
    constexpr std::string_view kDinoV3Engine = "resources/models/dino_v3_vit_base.engine";
    constexpr std::string_view kClipEngine = "resources/models/clip_vit_b32.engine";
    constexpr std::string_view kSam2Engine = "resources/models/sam2_vit_h.engine";
    constexpr std::string_view kDecisionTree = "resources/trees/seat_leather_inspection.yaml";
    constexpr std::string_view kTuningYaml = "resources/tuning/seat_leather_tuning.yaml";
    constexpr std::string_view kDefaultCoresetOutput = "resources/coreset.bin";
    constexpr std::string_view kDefaultOutputDir = "/tmp/surface-ai/results/";
    // 模型/预处理参数常量
    constexpr std::size_t kEmbedDim = 1024;
    constexpr std::size_t kImageSize = 1024;
    constexpr std::size_t kPatchSize = 14;
}
```

### 5.2 `cli_args.h/cpp`

从 `main.cpp` L61-102 直接抽取，逻辑不变。

### 5.3 `knowledge_seed.h/cpp`

从 `main.cpp` L107-163 抽取。签名：

```cpp
void SeedKnowledgeGraph(sai::knowledge::KnowledgeGraph& kg);
```

### 5.4 `coreset_builder.h/cpp`

从 `main.cpp` L167-382 抽取。签名：

```cpp
int BuildCoreset(const CliArgs& cli);
```

内部负责 GPU 初始化、DINOv3 embedder 创建、数据集加载、前处理、特征提取、FeatureBank 构建、保存。独立于 `AssembleApplication`。

### 5.5 `app_builder.h/cpp`（核心）

```cpp
auto AssembleApplication(const CliArgs& cli) -> sai::Result<AssembledApp>;
```

内部流程：
1. 创建 Context → Initialize → Start
2. Parse `pipeline.yaml` **一次**，后续所有子阶段共享 parse 结果
3. 创建推理引擎 + DINOv3 PatchEmbedder
4. 可选：CLIP GlobalEmbedder（取决于 pipeline.yaml）
5. 创建 PatchCore 检测器
6. 加载 FeatureBank + VectorPath（如果 `--coreset` 提供）
7. 可选：CoresetEvolution（取决于 pipeline.yaml 的 self_evolution 配置）
8. 创建 KnowledgeStore + SeedKnowledgeGraph
9. 可选：SAM2 Segmenter（取决于 pipeline.yaml）
10. 创建 RuleEngine + Reasoner
11. 创建 JsonExporter
12. Load Pipeline from YAML
13. Wire 所有对象到 Pipeline stages
14. 可选：TuningScheduler（调用 tuning_wiring 辅助）
15. 设置 ResultCallback（含 evolution 集成）
16. Pipeline->Start()
17. 可选：Evolution->Start()

任何步骤失败 → `return tl::make_unexpected(ErrorInfo{...})`。

### 5.6 `headless_runner.h/cpp`

```cpp
auto RunHeadless(const CliArgs& cli, AssembledApp& app) -> int;
```

从 `main.cpp` L922-1081 的 `run_headless` lambda + 两个 headless 入口分支抽取。负责：
- 根据 `cli.dataset_path` 或 `cli.image_dir` 加载图像列表
- 遍历帧 → Submit → Drain → 读取 JSON 结果 → 统计
- 多 surface 汇总（per-surface worst verdict）
- 结束时按序清理：evolution->Stop(), tuning->Join(), pipeline->Stop(), ctx->Stop()

### 5.7 `gui_runner.h/cpp`

```cpp
auto RunGui(int argc, char* argv[], AssembledApp& app) -> int;
```

从 `main.cpp` L1086-1173 抽取。负责：
- 创建 QGuiApplication + QQmlApplicationEngine
- 创建并绑定 ViewModels、FrameProvider
- 创建 FakeCamera + 注册帧回调
- 根据 evolution 是否存在设置 ResultCallback（两分支）
- `aboutToQuit` 信号中按序清理
- 调用 `app.exec()` 进入事件循环

### 5.8 `tuning_wiring.h/cpp`

从 `main.cpp` L386-460（`CollectLeafNodes`, `ApplyParamToYaml`）+ L728-868（调参装配）抽取。作为 `AppBuilder` 的内部辅助，提供：

```cpp
auto TryCreateTuningScheduler(
    const YAML::Node& pipeline_yaml,
    sai::knowledge::KnowledgeGraph& kg,
    sai::knowledge::KnowledgeEvolution& kg_evolution,
    sai::reasoner::IReasoner& reasoner,
    sai::pipeline::Pipeline& pipeline
) -> sai::Result<std::optional<sai::tuning::TuningScheduler>>;
```

## 6. 错误处理统一

所有 `cerr << ...; return 1;` 替换为 `return tl::make_unexpected(ErrorInfo{...})`。错误打印统一在 `main()` 调用点：

```cpp
auto app_result = AssembleApplication(cli);
if (!app_result) {
    std::cerr << "Application assembly failed: "
              << app_result.error().message << "\n";
    return 1;
}
```

## 7. 不变项

- 不修改任何 `sai::*` 库代码
- 不修改 `pipeline.yaml` / `qml.qrc` / resources
- 不修改 CMakeLists.txt 的 target_link_libraries
- 不修改任何业务逻辑、算法参数、阈值
- 所有 `std::cout` / `std::cerr` 输出内容不变
- GPU 上传路径（HtoD → infer → DtoH）不变

## 8. 验证标准

- `cmake --build --preset default --target seat_aoi` 编译通过
- 所有已有测试通过（`ctest --preset default`）
- `--dataset` 模式 coreset 构建行为不变
- `--image-dir` / `--dataset` headless 批量推理输出不变
