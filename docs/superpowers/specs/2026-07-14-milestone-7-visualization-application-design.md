# Surface AI Framework —— 里程碑 7 呈现与首个应用 设计文档

> Status: Draft
> Date: 2026-07-14
> Based on: `docs/superpowers/specs/2026-07-07-surface-ai-framework-phased-plan-design.md` §4 里程碑 7
> Depends on: 里程碑 1（1.1-1.6 全部冻结接口）、里程碑 2（2.1-2.3 全部冻结接口）、里程碑 3（3.1-3.3 全部冻结接口）、里程碑 4（4.1-4.2 全部冻结接口）、里程碑 5（5.1-5.2 全部冻结接口）、里程碑 6（6.1-6.2 全部冻结接口）

---

## 1. 背景与范围

里程碑 7 是 Surface AI Framework 的最后一个里程碑，覆盖"呈现与首个应用"：在 M1-M6 构建的完整计算管道之上，提供 Qt 6 + QML 工业级实时可视化界面和 Seat AOI 参考应用，验证"Product 只是 metadata"的核心设计原则。

### 1.1 架构总览

两层分离：

```
7.1 Visualization（呈现层）
    └── ViewModel 桥接层：C++ 核心数据 → QML Q_PROPERTY/QAbstractListModel
    └── QQuickImageProvider：SurfaceImage 像素 → QML Image 源
    └── QML 界面：实时监控 / 历史回顾 / 产线配置 / 统计仪表盘

7.2 Application（参考应用）
    └── Seat AOI：纯粹的模块组装——零业务逻辑
    └── 验证"Product 只是 metadata"——换产品只需换 YAML 配置
```

### 1.2 与 M1-M6 的关系

M7 不修改 M1-M6 的任何接口或实现。所有 Qt 依赖限定在 `src/visualization/` 和 `apps/seat-aoi/` 内，M1-M6 核心库保持纯 C++20（零 Qt 依赖）。

M7 通过以下方式读取 M1-M6 数据：

| 数据 | 来源 | 读取方式 |
|------|------|---------|
| Pipeline 阶段指标 | `Pipeline::Metrics()` | ViewModel 定时轮询，读取 `std::atomic<size_t>` 字段（wait-free） |
| 当前帧图像 | `IStageNode` 内部持有的 `SurfaceImage*` | Capture 阶段暴露 `frame_id → SurfaceImage*` 环形缓存，`FrameProvider` 按 ID 查找并拷贝到 `QImage` |
| 检测结果 | `ReasoningResult` | 通过 Pipeline 的结果回调或 Export 阶段的 JSON 输出回读 |
| 规则/决策树文件 | 文件系统 + `ConfigStore` | ViewModel 直接读取 YAML 文本内容供 QML 编辑器展示 |

### 1.3 批次划分与执行顺序

```
里程碑 7：呈现与首个应用
├── 7.1 Visualization（ViewModel + QQuickImageProvider + QML 界面）
└── 7.2 Application（Seat AOI 参考应用组装）
```

执行顺序：**7.1 设计 → 7.1 代码 → 7.2 设计 → 7.2 代码 → E2E 集成 + 契约更新**

### 1.4 项目锚点

延续 M1-M6 的汽车座椅 AOI 场景：

- **实时监控主屏**：相机预览 + 缺陷叠加框（bounding box overlay）+ 当前帧 verdict（OK/NG/WARN/UNCERTAIN）+ Pipeline 各阶段实时指标
- **历史回顾屏**：缺陷图库（按时间/SKU/缺陷类型筛选）+ 单帧详情（原始图/预处理图/缺陷标注图）
- **产线配置屏**：Pipeline YAML 编辑器 + 规则文件编辑器 + 参数调优面板 + 热生效/重启控制
- **统计仪表盘**：缺陷趋势图、PPM 曲线、阶段延迟分布

### 1.5 明确排除项

- 自定义 QML Chart 组件（使用 Qt Charts 模块，已可用）
- 3D 点云可视化（v1 不做，Future Extension）
- Web 远程监控（v1 不做，Future Extension）
- 历史数据 SQLite 持久化（M7 只做内存内历史摘要缓存最近 1000 帧，持久化延后至 Future Extension）
- Pipeline 拓扑热重载（M6 §13 列为 Future Extension，M7 配置屏提供"确认重启"按钮触发优雅重启）
- 多窗口/多屏布局（v1 单窗口四 Tab）

---

## 2. 对里程碑 1/2/3/4/5/6 的依赖

| M7 批次 | 依赖接口 | 用途 |
|---------|---------|------|
| 7.1 | `Pipeline::Metrics()` / `Pipeline::Submit()` / `Pipeline::Start()` / `Pipeline::Stop()`（6.1） | 获取实时阶段指标、控制 Pipeline 生命周期 |
| 7.1 | `StageMetrics`（6.2） | per-stage 原子计数器（frames_processed/failed/dropped/queue_depth） |
| 7.1 | `ReasoningResult`（5.2） | 判决结果（verdict/severity/recommendation/confidence）+ Trace + Evidence |
| 7.1 | `DetectionResult`（3.3） | 缺陷位置（bounding box）+ 得分 + 特征相似度排行 |
| 7.1 | `SurfaceImage`（2.2） | 预处理完毕帧的像素缓冲，供 QML 渲染 |
| 7.1 | `RawImage`（2.2） | 相机原始帧，供历史回顾屏展示原始图 |
| 7.1 | `RuleEngine`（5.1） | 规则文件内容读取 + `ReloadRules()` 热重载 |
| 7.1 | `IReasoner`（5.2） | 决策树文件内容读取 + `ReloadTree()` 热重载 |
| 7.1 | `IStageNode`（6.1） | 阶段参数 `ReloadConfig(YAML::Node)` 热生效 |
| 7.1 | `ConfigStore`（1.6） | 读取/写入 YAML 配置文件 |
| 7.1 | `Logger`（1.6） | 日志级别调整 + 错误日志展示 |
| 7.2 | 以上所有 + `Context`（1.2） | Seat AOI 的模块装配：Register → Initialize → Start |

---

## 3. 架构设计

### 3.1 核心设计原则

**ViewModel 桥接层隔离 Qt 依赖**。M1-M6 核心模块保持纯 C++20，不引入任何 Qt 头文件。`src/visualization/` 内的 ViewModel 层持有核心对象的裸指针，通过 `Q_PROPERTY` 和 `QAbstractListModel` 将数据暴露给 QML。QML 层只做声明式绑定，不调用框架 C++ API。

```
┌─────────────────────────────────────────────────────┐
│ QML 层（apps/seat-aoi/qml/）                         │
│ MonitorScreen / HistoryScreen / ConfigScreen /       │
│ DashboardScreen                                      │
│ 只做声明式绑定：Image { source: "image://..." }       │
│                Text { text: viewmodel.verdict }       │
├─────────────────────────────────────────────────────┤
│ ViewModel 层（src/visualization/viewmodels/）         │
│ PipelineViewModel / InspectionViewModel /            │
│ ConfigViewModel / DashboardViewModel                 │
│ Q_PROPERTY 暴露数据，QAbstractListModel 暴露列表       │
├─────────────────────────────────────────────────────┤
│ 核心层（src/*/）M1-M6                                  │
│ Pipeline / IStageNode / RuleEngine / IReasoner /     │
│ DetectionResult / SurfaceImage                        │
│ 纯 C++20，无 Qt 依赖                                  │
└─────────────────────────────────────────────────────┘
```

### 3.2 架构取舍

| 决策 | 采用 | 拒绝 | 理由 |
|------|------|------|------|
| UI 技术 | Qt 6 + QML（声明式） | Qt Widgets / imgui / Web 前端 | QML 的声明式绑定天然适合数据驱动的监控界面；Widgets 需要大量手动布局代码，视觉效果差且难以维护；Web 前端引入额外网络栈，不满足工业产线本地运行需求 |
| C++/QML 桥接 | ViewModel (`Q_PROPERTY` + `QAbstractListModel`) | 直接暴露 C++ 类型给 QML / JSON 桥接 | ViewModel 隔离 Qt 依赖到单一库，核心模块零改动；直接暴露需要 M1-M6 引入 Qt 头文件（破坏模块纯洁性）；JSON 桥接性能差（60fps 场景不可接受）且丢失类型安全 |
| 图像渲染 | `QQuickImageProvider` + 像素拷贝 | `QQuickFramebufferObject`（GPU 纹理共享）/ QPixmap 直接传递 | 1024×1024×3 单帧拷贝 ~3MB，即使 30fps 也只有 90MB/s——现代硬件的内存带宽完全可忽略；`QQuickFramebufferObject` 零拷贝但耦合 Qt 渲染管线细节，开发/调试成本高 |
| 数据刷新 | 轮询（QML `Timer` + `std::atomic` 读取） | 信号槽推送 / 共享内存 | 轮询避免跨线程信号槽的队列竞争和事件循环阻塞风险；`std::atomic` wait-free 读取保证 QML 主线程永不阻塞；30Hz 帧刷新 + 2Hz 指标刷新不会造成 CPU 热点 |
| 配置热生效 | 参数级即时生效 + 拓扑级确认重启 | 全部热生效 / 全部要求重启 | 调参（阈值/规则）是高频操作必须即时生效；改拓扑是低频操作（小时/天级），优雅重启成本可接受 |
| QML Chart | Qt Charts 模块（QtCharts QML plugin） | 自定义 Canvas / Chart.js WebView | Qt Charts 与 QML 原生集成，样式/动画/交互一致性好，减少自定义开发工作量 |

### 3.3 C++/QML 类型映射

| C++ 核心类型 | ViewModel 暴露方式 | QML 消费方式 |
|-------------|-------------------|-------------|
| `StageMetrics`（原子计数器） | `PipelineViewModel::StageMetrics` 为 `QObject` + `Q_PROPERTY(int framesProcessed ...)` | `Text { text: stage.framesProcessed }` 属性绑定 |
| `ReasoningResult` | `InspectionViewModel` 的 `Q_PROPERTY`（verdict/severity/recommendation/confidence） | 文本/颜色绑定（OK=绿, NG=红） |
| `DetectionResult.defects[]` | `QAbstractListModel` → `DefectModel` | `ListView` / `Repeater` 渲染缺陷叠加框 |
| `SurfaceImage`（像素缓冲） | `FrameProvider::requestImage()` → `QImage` | `Image { source: "image://pipeline/frame?t=" + Date.now() }` |
| YAML 文件内容 | `ConfigViewModel` 的 `Q_PROPERTY(QString pipelineYaml ...)` | `TextArea { text: configVM.pipelineYaml }` |
| 历史摘要 | `DashboardViewModel` 的 `QAbstractListModel` | `ChartView` 绑定 model 数据 |
| `vector<EvidenceItem>` | `QAbstractListModel` → `EvidenceModel` | `ListView` 展示证据链 |

---

## 4. QML UI/UX 设计

### 4.1 设计语言

**暗色工业风主题**。工厂车间光照条件复杂（强光/弱光/频闪），暗色 UI 减少视觉疲劳、降低屏幕反光干扰。整体风格参考现代工业 HMI（人机界面）：信息密度高但不杂乱，关键状态一目了然。

```
色彩系统
────────────────────────────────────────────
背景层:
  主背景        #1a1a2e (深蓝黑)
  卡片/面板     #16213e (深蓝)
  输入框        #0f3460 (蓝灰)
  分割线        #1f3460 (暗蓝)

文本层:
  主文本        #e4e6eb (浅灰白)  — 正文、标签
  次文本        #a0a8b8 (灰蓝)    — 辅助说明、单位
  占位文本      #5a6478 (暗灰蓝)  — placeholder
  高亮文本      #ffffff           — 关键数值

语义色:
  OK/通过       #00c853 (绿)      — verdict OK、阶段正常
  NG/缺陷       #ff1744 (红)      — verdict NG、阶段失败
  WARN/警告     #ff9100 (橙)      — verdict WARN
  UNCERTAIN     #ffc107 (琥珀)    — verdict UNCERTAIN
  信息/高亮     #448aff (蓝)      — 选中态、链接
  Pipeline运行  #00e676 (亮绿)    — 运行中指示灯
  Pipeline停止  #757575 (灰)      — 已停止指示灯

Chart 调色板:
  series[0]     #448aff (蓝)
  series[1]     #ff9100 (橙)
  series[2]     #00c853 (绿)
  series[3]     #ff1744 (红)
  series[4]     #e040fb (紫)
  series[5]     #00e5ff (青)
```

**字体系统**：全局使用系统默认无衬线字体（macOS: SF Pro, Linux: Noto Sans, Windows: Segoe UI）。数值/代码使用等宽字体（SF Mono / Noto Sans Mono / Consolas）。

| 层级 | 字号 | 字重 | 用途 |
|------|------|------|------|
| H1 | 28px | Bold | 页面标题 |
| H2 | 20px | DemiBold | 区块标题 |
| H3 | 16px | Medium | 面板标题 |
| Body | 14px | Regular | 正文、标签 |
| Caption | 12px | Regular | 辅助信息、时间戳 |
| Data | 32px | Bold | 核心数据（verdict、FPS） |
| Metric | 24px | DemiBold | 关键指标（PPM、帧数） |
| Mono | 13px | Regular | YAML 代码编辑 |

**间距系统**（8px 基准）：
- xs: 4px / sm: 8px / md: 16px / lg: 24px / xl: 32px / 2xl: 48px

**圆角系统**：
- 卡片: 12px / 按钮: 8px / 输入框: 6px / 缺陷框: 4px / 指示灯: 50%（圆形）

### 4.2 交互规范

**所有可交互元素必须显示鼠标手型光标**，通过以下 QML 模式统一应用：

```qml
// 自定义 cursor 样式 — 所有可点击控件的基础
HoverCursor { cursorShape: Qt.PointingHandCursor }

// 具体用法：
Button { cursorShape: Qt.PointingHandCursor }
TabBar { cursorShape: Qt.PointingHandCursor }  // 每个 TabButton
ComboBox { cursorShape: Qt.PointingHandCursor }
SpinBox { cursorShape: Qt.PointingHandCursor }
Slider { cursorShape: Qt.PointingHandCursor }
Switch { cursorShape: Qt.PointingHandCursor }
TextArea { cursorShape: Qt.IBeamCursor }  // 文本编辑区用 I 型光标
```

**交互反馈三层**：所有可交互元素在以下三种状态必须有视觉区分——

| 状态 | 效果 | 用途 |
|------|------|------|
| Default | 基础色 | 正常态 |
| Hovered（悬浮） | 亮度 +15%，轻微 scale(1.02)，cursor 变手型 | 用户鼠标悬停 |
| Pressed（按下） | 亮度 -10%，scale(0.98) | 用户点击确认 |
| Disabled | 透明度 0.4，cursor 变箭头（DefaultCursor） | 不可操作 |

实现方式：封装 `InteractiveButton`、`InteractiveTab` 等基础 QML 组件，内置 hover/pressed/disabled 状态机，避免每个控件重复写 `MouseArea`。

### 4.3 主窗口布局

```
┌──────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────┐│
│  │  Surface AI — Seat AOI Inspection         ● Running  ││  ← 标题栏
│  │  2026-07-14 14:32:15  │  SKU: LA-001  │  FPS: 4.8    ││     高度 48px
│  └──────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────┐│
│  │ [实时监控] [历史回顾] [产线配置] [统计仪表盘]          ││  ← TabBar
│  └──────────────────────────────────────────────────────┘│     高度 40px
│                                                          │
│  ┌──────────────────────────────────────────────────────┐│
│  │                    Tab 内容区                         ││
│  │                  (切换时带 crossfade 动画)             ││
│  │                                                      ││
│  └──────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────┐│
│  │  Status: Pipeline running │ Last frame: #0421 NG     ││  ← 状态栏
│  └──────────────────────────────────────────────────────┘│     高度 32px
└──────────────────────────────────────────────────────────┘
```

**标题栏** (`TitleBar`)：
- 左侧：应用名称 + 当前产线名称（从 Pipeline YAML `name` 读取）
- 右侧：运行指示灯（8px 圆形，Running = 绿色脉冲动画，Stopped = 灰色常亮）+ 文本 + 当前时间 + 当前 SKU + 实时 FPS
- 背景色比主背景略深 `#141e30`，形成层次区分
- 状态指示灯脉冲动画：
  ```qml
  Rectangle {
      id: statusDot
      radius: width / 2; width: 8; height: 8
      color: running ? "#00e676" : "#757575"
      SequentialAnimation on opacity {
          running: parent.running; loops: Animation.Infinite
          NumberAnimation { from: 1.0; to: 0.3; duration: 800 }
          NumberAnimation { from: 0.3; to: 1.0; duration: 800 }
      }
  }
  ```

**TabBar**：
- 4 个 TabButton：实时监控 / 历史回顾 / 产线配置 / 统计仪表盘
- 选中态：底部 2px 蓝色下划线 + 文字高亮白色
- 未选中态：次文本色 #a0a8b8
- 切换动画：`TabView` content 做 200ms crossfade（`opacity: 0→1`）
- 每个 TabButton `cursorShape: Qt.PointingHandCursor`

**状态栏** (`StatusBar`)：
- 左侧：Pipeline 运行状态文本
- 右侧：最近帧 ID + Verdict（OK=绿字/NG=红字）
- 背景透明，字体 12px Caption

### 4.4 实时监控屏 (MonitorScreen)

```
┌──────────────────────────────────────────────────────────┐
│ ┌─────────────────────────┐  ┌──────────────────────────┐│
│ │                         │  │  ● VERDICT               ││  ← 左侧 70%
│ │                         │  │                          ││     右侧 30%
│ │                         │  │    OK                    ││
│ │     相机预览画面          │  │                          ││  ← verdict 120px 大字
│ │     (1024×1024)         │  │  severity: 0.12          ││
│ │                         │  │  confidence: 0.98        ││
│ │     ┌──────┐            │  │  "表面无缺陷"              ││
│ │     │缺陷框│            │  │                          ││
│ │     └──────┘            │  ├──────────────────────────┤│
│ │                         │  │  Stage Metrics           ││
│ │                         │  │  Capture    ████  12ms   ││
│ │                         │  │  Preprocess ██     8ms   ││
│ │                         │  │  Inference  █████ 45ms   ││
│ │                         │  │  Detect     ███   15ms   ││
│ │                         │  │  RuleEval   ████  18ms   ││
│ │                         │  │  Reason     █      2ms   ││
│ │                         │  │  Export     ██    10ms   ││
│ └─────────────────────────┘  └──────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

**左侧 — 相机预览区**：
- 背景 `#0a0a14`（几乎纯黑），让画面成为视觉焦点
- 图像居中显示，保持宽高比
- 图像四周 2px 边框 `#1f3460`，与周围面板分隔
- 缺陷叠加框（`DefectOverlay`）：半透明红色矩形框 + 右上角缺陷标签浮层

  ```qml
  // DefectOverlay.qml
  Rectangle {
      property string defectLabel
      property string severity
      property rect boundingBox

      x: boundingBox.x; y: boundingBox.y
      width: boundingBox.width; height: boundingBox.height
      color: "transparent"
      border { color: "#ff1744"; width: 2 }

      // 右上角标签
      Rectangle {
          anchors { top: parent.top; right: parent.right }
          color: "#ff1744"; radius: 4
          Label { text: defectLabel; color: "white"; font.pixelSize: 12 }
      }
  }
  ```

**右侧 — 判决面板**：
- 顶部 VerdictBadge 组件：
  - `OK`：绿色大字 120px Bold + 绿色背景圆角卡片（`#00c853` 20%透明度）
  - `NG`：红色大字 120px Bold + 红色背景圆角卡片（`#ff1744` 20%透明度）
  - `WARN`：橙色 / `UNCERTAIN`：琥珀色
  - verdict 切换时做 300ms 颜色渐变动画（`ColorAnimation`）
- 判决文本下方：severity（数值 + 进度条）、confidence（数值 + 百分比）
- 推荐操作文本（从 ReasoningResult.recommendation 读取），最多两行，超出省略号

**右侧 — 阶段延迟条** (`StageMetricsBar`)：
- 每个阶段一行：阶段名称（左对齐） + 水平条形图（宽度 = latency / max_latency）+ 延迟数字（右对齐，monospace）

  ```qml
  // 单行阶段指标
  RowLayout {
      Label { text: "Inference"; Layout.preferredWidth: 90; color: "#a0a8b8" }
      Rectangle {
          Layout.fillWidth: true; height: 6; radius: 3
          color: "#1f3460"
          Rectangle {
              width: parent.width * (latency / maxLatency)
              height: parent.height; radius: 3
              color: latency > threshold ? "#ff1744" : "#448aff"
              Behavior on width { NumberAnimation { duration: 200 } }
          }
      }
      Label { text: latency + "ms"; Layout.preferredWidth: 50; font.family: "monospace" }
  }
  ```

- 条形图颜色：延迟超过阈值时变红（如 Inference > 100ms），否则蓝色
- 宽度变化动画 200ms `NumberAnimation`

### 4.5 历史回顾屏 (HistoryScreen)

```
┌──────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────┐│
│  │ 筛选: [时间范围 ▼] [SKU ▼] [缺陷类型 ▼]    🔍 搜索   ││  ← 工具栏
│  └──────────────────────────────────────────────────────┘│
│  ┌────────────┐  ┌──────────────────────────────────────┐│
│  │ 帧列表      │  │                                      ││  ← 左侧 280px
│  │            │  │     单帧详情                           ││     右侧填充
│  │ ┌────────┐ │  │                                      ││
│  │ │ #0423  │ │  │   ┌─────────────┐  ┌─────────────┐   ││
│  │ │ NG ●   │ │  │   │  原始图      │  │  预处理图    │   ││
│  │ │ 14:32  │ │  │   │  (RawImage)  │  │(SurfaceImage)│   ││
│  │ └────────┘ │  │   └─────────────┘  └─────────────┘   ││
│  │ ┌────────┐ │  │                                      ││
│  │ │ #0422  │ │  │  Defects:                            ││
│  │ │ OK  ●  │ │  │  ┌──────────────────────────────────┐││
│  │ │ 14:32  │ │  │  │ ● scratch  severity:0.7  conf:0.9│││
│  │ └────────┘ │  │  │ ● dent     severity:0.3  conf:0.8│││
│  │ ┌────────┐ │  │  └──────────────────────────────────┘││
│  │ │ #0421  │ │  │                                      ││
│  │ │ NG ●   │ │  │  Evidence Chain:                     ││
│  │ │ 14:31  │ │  │  ├─ surface.refectance → 0.43        ││
│  │ └────────┘ │  │  ├─ material.defect_history → 23%    ││
│  │            │  │  └─ batch.reject_rate → 5.2%         ││
│  └────────────┘  └──────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

**工具栏**：
- 水平排列：ComboBox × 3（时间范围、SKU、缺陷类型）+ Spacer + 搜索输入框
- 每个 ComboBox `cursorShape: Qt.PointingHandCursor`
- 筛选条件变更时，帧列表做 200ms 淡入刷新（`OpacityAnimator`）

**左侧 — 帧列表** (`ListView`)：
- 每行一个帧摘要卡片（高度 64px）：帧 ID + verdict 色点（8px 圆形） + 时间戳 + 检出缺陷数
- 选中行：左侧 3px 蓝色边框 + 背景 `#0f3460`（比默认卡片色亮）
- `cursorShape: Qt.PointingHandCursor` on each delegate
- 虚拟列表（只渲染可见行），支持键盘上下键导航
- 滚动条样式：4px 宽，半透明，hover 时变亮

**右侧 — 单帧详情**：
- 双图并排：原始图（RawImage）+ 预处理图（SurfaceImage），之间 2px 蓝色竖线分隔
- 每张图周围 8px padding + `#0f3460` 背景
- 缺陷列表：每条缺陷用 `RowLayout`，带 severity 彩色圆点 + 缺陷名 + severity 进度条
- 证据链 (`EvidenceItem[]`)：树形结构，每条左边有连接线（用 `Canvas` 绘制），key: value 用 monospace 字体

### 4.6 产线配置屏 (ConfigScreen)

```
┌──────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────┐│
│  │ 配置文件: [pipeline.yaml ▼]  [rules.yaml ▼]           ││  ← 文件选择
│  │                               [tree.yaml ▼]           ││
│  └──────────────────────────────────────────────────────┘│
│  ┌────────────┐  ┌──────────────────────────────────────┐│
│  │ 结构树      │  │  YAML 编辑器                          ││  ← 左侧 240px
│  │            │  │                                      ││     右侧填充
│  │ ▼ pipeline │  │  pipeline:                           ││
│  │   name     │  │    name: "seat_aoi_inspection"       ││
│  │   version  │  │    version: "1.0"                    ││
│  │ ▶ backpres │  │    backpressure:                     ││
│  │ ▼ stages   │  │      default: block                  ││
│  │   capture  │  │      stage_overrides:                ││
│  │   preproc  │  │        capture: drop_oldest            ││
│  │   inferenc │  │    stages:                           ││
│  │   detect   │  │      - id: capture                   ││
│  │   rule_eva │  │        type: Capture                 ││
│  │   reason   │  │        ...                           ││
│  │   export   │  │                                      ││
│  └────────────┘  └──────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────┐│
│  │  💡 此改动仅涉及参数，可即时生效      [校验YAML] [应用] ││  ← 操作栏
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

**文件选择 TabBar**（二级 Tab）：
- 三个 ToggleButton：pipeline.yaml / rules.yaml / tree.yaml
- 选中态高亮，带底部蓝色下划线
- 切换时编辑器内容同步切换，`TextArea.text` 绑定到对应 ConfigViewModel property

**左侧结构树** (`TreeView`)：
- YAML 结构的可折叠树形视图，方便快速定位配置位置
- 展开/折叠箭头用 `▶` / `▼` 字符，点击时带 200ms 旋转动画
- 叶子节点点击 → 右侧编辑器自动滚动到对应行并高亮
- `cursorShape: Qt.PointingHandCursor` on each tree node

**右侧 YAML 编辑器** (`TextArea` + 语法高亮)：
- 暗色背景 `#0d1117`（GitHub Dark 风格），等宽字体 13px
- 语法高亮：key 蓝色、string 绿色、number 橙色、comment 灰色斜体、bool 紫色
- 行号显示在左侧 gutter（`#161b22` 背景）
- 当前行高亮（整行 `#1a2332` 背景）
- 括号匹配高亮
- `cursorShape: Qt.IBeamCursor`（文本编辑光标）

**底部操作栏**：
- 提示文字：分析变更类型，告知用户是"即时生效"还是"需要重启"
  - 参数变更（阈值/ROI/k_nearest）→ "💡 此改动仅涉及参数，可即时生效" 绿色图标
  - 拓扑变更 → "⚠ 此改动涉及 Pipeline 结构，需要重启生效" 橙色图标
- 两个按钮：`[校验 YAML]`（次要按钮，白色边框） + `[应用]`（主要按钮，蓝色填充）
- 按钮最小宽度 120px，高度 36px，圆角 8px
- `cursorShape: Qt.PointingHandCursor` on both buttons
- "应用"按钮 pressed 后做 600ms loading spinner 动画 + "应用成功 ✓" / "应用失败 ✗" 反馈

```qml
// 应用按钮反馈状态机
Button {
    text: status === "idle" ? "应用" : status === "loading" ? "应用中..." : status === "success" ? "✓ 已生效" : "✗ 失败"
    enabled: status !== "loading"
    cursorShape: Qt.PointingHandCursor
    background: Rectangle {
        color: status === "success" ? "#00c853" : status === "error" ? "#ff1744" : "#448aff"
        radius: 8
        Behavior on color { ColorAnimation { duration: 300 } }
    }
}
```

### 4.7 统计仪表盘 (DashboardScreen)

```
┌──────────────────────────────────────────────────────────┐
│  ┌─────────────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐│
│  │  Total Frames    │ │  OK Rate  │ │  PPM     │ │ NG    ││  ← 顶部 KPI 卡片行
│  │    12,847        │ │  98.2%    │ │  1,847   │ │  236  ││
│  └─────────────────┘ └──────────┘ └──────────┘ └───────┘│
│  ┌────────────────────────────────┐ ┌───────────────────┐│
│  │  PPM Trend (Last 24h)          │ │  Defect Type      ││  ← 左侧 60%
│  │                                │ │  Distribution     ││     右侧 40%
│  │  📈 折线图                      │ │  ┌──────────────┐ ││
│  │                                │ │  │ scratch  42% │ ││
│  │                                │ │  │ dent     28% │ ││
│  │                                │ │  │ stain    15% │ ││
│  │                                │ │  │ crack     8% │ ││
│  │                                │ │  │ other     7% │ ││
│  │                                │ │  └──────────────┘ ││
│  └────────────────────────────────┘ └───────────────────┘│
│  ┌──────────────────────────────────────────────────────┐│
│  │  Stage Latency Distribution (p50 / p95 / p99)        ││
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         ││
│  │  │Capture │ │Preproc │ │Inferenc│ │Detect  │ ...     ││  ← 阶段延迟
│  │  │ 5/8/12 │ │ 3/6/10 │ │35/55/80│ │ 8/12/18│         ││     分布卡
│  │  │   ms   │ │   ms   │ │   ms   │ │   ms   │         ││
│  │  └────────┘ └────────┘ └────────┘ └────────┘         ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

**顶部 KPI 卡片行**（4 个指标卡片，等宽分布）：
- 每个卡片：圆角 12px，背景 `#16213e`，内部 padding 16px
- 数字用 Data 字号（32px Bold），标签用 Caption 字号（12px）
- 卡片 hover 时轻微上浮（`translate y: -2px` + 阴影），`cursorShape: Qt.PointingHandCursor`
- 数字刷新时做 500ms 数字滚动动画（`NumberAnimation` from old_value to new_value）

```qml
// KPI 卡片组件
Rectangle {
    color: "#16213e"; radius: 12
    Behavior on y { NumberAnimation { duration: 200 } }
    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        hoverEnabled: true
        onEntered: parent.y = -2
        onExited: parent.y = 0
    }
    Column {
        anchors { centerIn: parent; margins: 16 }
        Label { text: kpiValue; font.pixelSize: 32; font.bold: true; color: "#ffffff" }
        Label { text: kpiLabel; font.pixelSize: 12; color: "#a0a8b8" }
    }
}
```

**PPM 趋势图** (`TrendChart` — Qt Charts `ChartView`)：
- 暗色背景 `#16213e`，白色网格线 10% 透明度
- X 轴：时间（自动格式化为 HH:mm），Y 轴：PPM 值
- 折线颜色 `#448aff`，线宽 2px，数据点 4px 圆形
- 支持 hover 显示 tooltip（数据点的精确时间+PPM值），`cursorShape: Qt.PointingHandCursor` on chart area
- chart 主题：`ChartView.ChartThemeDark` / 关闭 legend 边框

**缺陷类型分布**（水平条形图列表）：
- 每个缺陷类型：标签 + 百分比 + 水平进度条（宽度 = 百分比，颜色取自 chart 调色板轮转）
- 条形图动画：`Behavior on width { NumberAnimation { duration: 800; easing.type: Easing.OutCubic } }`

**阶段延迟分布卡片行**：
- 每个阶段一张小卡片（160px × 100px）：阶段名 + p50/p95/p99 三行数据
- p50 绿色、p95 橙色、p99 红色（体现延迟尾部的严重程度）
- 用 monospace 字体展示数字

### 4.8 动画与过渡

| 动画 | 时机 | 时长 | 缓动曲线 | 说明 |
|------|------|------|---------|------|
| Tab 切换 crossfade | Tab 切换 | 200ms | `EaseInOut` | 旧内容 opacity 1→0，新内容 opacity 0→1 |
| VerdictBadge 颜色切换 | 新帧 verdict 与上帧不同 | 300ms | `EaseInOut` | `ColorAnimation` 背景色渐变 |
| 阶段延迟条宽度 | 新 metrics 到达 | 200ms | `EaseOutCubic` | 条形图宽度平滑过渡 |
| KPI 数字滚动 | 统计数据更新 | 500ms | `EaseOutCubic` | `NumberAnimation` from→to |
| 缺陷框出现 | 新帧 defects 数据到达 | 150ms | `EaseOutBack` | scale 0→1 + opacity 同时动画 |
| 按钮反馈 | 点击 | 100ms | `EaseInOut` | scale 1→0.98→1 |
| 状态灯脉冲 | Pipeline Running | 800ms × ∞ | `Linear` | opacity 1.0↔0.3 |
| 列表项 hover | 鼠标进入列表项 | 150ms | `EaseOut` | 背景色过渡 |
| YAML 校验结果 | 校验完成 | 300ms | `EaseOutCubic` | 校验状态图标 scale 0→1 弹出 |
| Tooltip 出现 | chart 数据点 hover | 150ms | `EaseOut` | 透明度 0→1 + translate y: -10→0 |

### 4.9 响应式布局

- **最小窗口尺寸**：1280×800（低于此尺寸显示提示"请增大窗口"）
- **推荐窗口尺寸**：1920×1080（全屏运行于产线监控显示器）
- 左侧/右侧比例：实时监控屏 70/30、历史回顾屏 280px/剩余、配置屏 240px/剩余
- 窗口 resize 时各面板比例通过 `Layout.fillWidth` + `preferredWidth` 自适应
- 字体大小不随窗口缩放——保持可读性

---

## 5. 模块结构

### 5.1 文件布局

```
include/sai/visualization/
    pipeline_viewmodel.h         # PipelineViewModel：Pipeline 状态 + 阶段指标
    inspection_viewmodel.h       # InspectionViewModel：ReasoningResult + DetectionResult
    frame_provider.h             # FrameProvider：QQuickImageProvider 实现
    config_viewmodel.h           # ConfigViewModel：YAML 编辑 + 热重载控制
    dashboard_viewmodel.h        # DashboardViewModel：历史统计聚合

src/visualization/
    CMakeLists.txt
    pipeline_viewmodel.cpp
    inspection_viewmodel.cpp
    frame_provider.cpp
    config_viewmodel.cpp
    dashboard_viewmodel.cpp

src/visualization/qml/
    MainWindow.qml               # 四屏 TabBar 容器
    MonitorScreen.qml            # 实时监控主屏
    HistoryScreen.qml            # 历史回顾屏
    ConfigScreen.qml             # 产线配置屏
    DashboardScreen.qml          # 统计仪表盘
    components/
        DefectOverlay.qml        # 缺陷叠加框组件
        StageMetricsBar.qml      # 阶段延迟柱状图组件
        TrendChart.qml           # PPM / 缺陷趋势折线图组件
        VerdictBadge.qml         # OK/NG/WARN/UNCERTAIN 判决徽章

apps/seat-aoi/
    CMakeLists.txt               # 可执行目标，link sai_visualization + 所有 M1-M6 库
    main.cpp                     # QGuiApplication + QQmlApplicationEngine + 模块装配
    resources/
        pipeline.yaml            # Seat AOI 默认 Pipeline
        rules/
            seat_leather_defects.yaml
        trees/
            seat_leather_inspection.yaml
    qml.qrc                     # QML 资源文件

tests/visualization/
    pipeline_viewmodel_test.cpp
    inspection_viewmodel_test.cpp
    frame_provider_test.cpp
    config_viewmodel_test.cpp
    dashboard_viewmodel_test.cpp

tests/app/
    seat_aoi_startup_test.cpp    # 验证应用能启动 + Pipeline 构建成功
```

### 5.2 命名空间

- `sai::visualization`：7.1 Visualization 的所有公共类型（ViewModel + FrameProvider）
- Application 不引入新命名空间，`main.cpp` 中直接使用 `sai::*` 各模块命名空间

### 5.3 CMake 集成

```cmake
# src/visualization/CMakeLists.txt
find_package(Qt6 REQUIRED COMPONENTS Quick QuickControls2 Charts)

add_library(sai_visualization STATIC)
add_library(sai::visualization ALIAS sai_visualization)

target_sources(sai_visualization PRIVATE
    pipeline_viewmodel.cpp
    inspection_viewmodel.cpp
    frame_provider.cpp
    config_viewmodel.cpp
    dashboard_viewmodel.cpp
)

target_link_libraries(sai_visualization PUBLIC
    Qt6::Quick
    Qt6::QuickControls2
    Qt6::Charts
    sai::pipeline       # Pipeline / IStageNode / StageMetrics
    sai::rule           # RuleEngine::ReloadRules
    sai::reasoner       # IReasoner / ReasoningResult
    sai::detection      # DetectionResult
    sai::image          # SurfaceImage / RawImage
    sai::infra          # ConfigStore / Logger
)

# apps/seat-aoi/CMakeLists.txt
add_executable(seat_aoi main.cpp)

target_link_libraries(seat_aoi PRIVATE
    sai::visualization
    sai::pipeline
    sai::rule
    sai::reasoner
    sai::detection
    sai::inference
    sai::embedding
    sai::knowledge
    sai::retrieval
    sai::device
    sai::image
    sai::io
    sai::infra
    sai::core
    sai::memory
    sai::plugin
    sai::runtime
)
```

---

## 6. 接口设计

### 6.1 PipelineViewModel

```cpp
namespace sai::visualization {

class StageMetricsObject : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString stageId READ stageId CONSTANT)
    Q_PROPERTY(QString stageType READ stageType CONSTANT)
    Q_PROPERTY(int framesProcessed READ framesProcessed NOTIFY metricsChanged)
    Q_PROPERTY(int framesFailed READ framesFailed NOTIFY metricsChanged)
    Q_PROPERTY(int framesDropped READ framesDropped NOTIFY metricsChanged)
    Q_PROPERTY(int queueDepth READ queueDepth NOTIFY metricsChanged)
    Q_PROPERTY(double avgLatencyMs READ avgLatencyMs NOTIFY metricsChanged)
    Q_PROPERTY(double p99LatencyMs READ p99LatencyMs NOTIFY metricsChanged)
public:
    // 由 PipelineViewModel 内部调用，非 QML 直接访问
    void UpdateFromStageMetrics(const sai::pipeline::StageMetrics&);
signals:
    void metricsChanged();
private:
    // 使用 std::atomic 字段 + QTimer 驱动的刷新
};

class PipelineViewModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString pipelineName READ pipelineName CONSTANT)
    Q_PROPERTY(QString pipelineStatus READ pipelineStatus NOTIFY statusChanged)
    Q_PROPERTY(double overallFps READ overallFps NOTIFY metricsChanged)
    Q_PROPERTY(int totalFramesProcessed READ totalFramesProcessed NOTIFY metricsChanged)
    Q_PROPERTY(int totalFramesFailed READ totalFramesFailed NOTIFY metricsChanged)
    Q_PROPERTY(int inFlightFrames READ inFlightFrames NOTIFY metricsChanged)
    Q_PROPERTY(QList<QObject*> stageMetrics READ stageMetrics NOTIFY stageMetricsChanged)
public:
    explicit PipelineViewModel(QObject* parent = nullptr);

    // 绑定到 Pipeline 实例（不持有所有权）
    void BindToPipeline(const sai::pipeline::Pipeline* pipeline);

    // 启动定时刷新（QTimer，30Hz）
    void StartRefresh(int intervalMs = 33);
    void StopRefresh();

signals:
    void statusChanged();
    void metricsChanged();
    void stageMetricsChanged();

private:
    const sai::pipeline::Pipeline* pipeline_;  // 不持有所有权
    QTimer refresh_timer_;
    QList<StageMetricsObject*> stage_metrics_objects_;
};
```

### 6.2 InspectionViewModel

```cpp
namespace sai::visualization {

class DefectItem : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString label READ label CONSTANT)
    Q_PROPERTY(QString severity READ severity CONSTANT)
    Q_PROPERTY(double confidence READ confidence CONSTANT)
    Q_PROPERTY(QRectF boundingBox READ boundingBox CONSTANT)
public:
    // 从 sai::detection::DetectionResult 构造
    static DefectItem* FromDetectionResult(const sai::detection::DetectionResult::Defect&);
};

class DefectModel : public QAbstractListModel {
    Q_OBJECT
public:
    enum Roles {
        LabelRole = Qt::UserRole + 1,
        SeverityRole,
        ConfidenceRole,
        BoundingBoxRole
    };
    // 标准 QAbstractListModel 实现：rowCount / data / roleNames
    void UpdateDefects(const std::vector<sai::detection::DetectionResult::Defect>& defects);
};

class EvidenceModel : public QAbstractListModel {
    Q_OBJECT
public:
    enum Roles {
        KeyRole = Qt::UserRole + 1,
        ValueRole,
        SourceRole,
        TraceRole
    };
    void UpdateEvidence(const std::vector<sai::rule::EvidenceItem>& evidence);
};

class InspectionViewModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString verdict READ verdict NOTIFY resultChanged)
    Q_PROPERTY(QString severity READ severity NOTIFY resultChanged)
    Q_PROPERTY(QString recommendation READ recommendation NOTIFY resultChanged)
    Q_PROPERTY(double confidence READ confidence NOTIFY resultChanged)
    Q_PROPERTY(QObject* defectModel READ defectModel CONSTANT)
    Q_PROPERTY(QObject* evidenceModel READ evidenceModel CONSTANT)
public:
    explicit InspectionViewModel(QObject* parent = nullptr);

    // 更新最新帧的检测结果（由 Pipeline 结果回调触发）
    void UpdateResult(const sai::rule::ReasoningResult& result);

signals:
    void resultChanged();

private:
    DefectModel* defect_model_;
    EvidenceModel* evidence_model_;
    // 缓存最新结果字段
    std::string verdict_;
    std::string severity_;
    std::string recommendation_;
    double confidence_;
};
```

### 6.3 FrameProvider

```cpp
namespace sai::visualization {

class FrameProvider : public QQuickImageProvider {
public:
    explicit FrameProvider();

    // QQuickImageProvider 接口
    auto requestImage(const QString& id, QSize* size,
                      const QSize& requestedSize) -> QImage override;

    // C++ 端注册帧（由 WorkerPool 线程调用，线程安全）
    // frame_id 单调递增，环形缓存存储最近 N 帧
    void RegisterFrame(int frame_id, const sai::image::SurfaceImage* image);
    void RegisterRawFrame(int frame_id, const sai::image::RawImage* image);

private:
    // 环形缓存：frame_id → QImage 快照
    // 容量 = 16 帧（≈ 0.5s @ 30fps）
    static constexpr int kCacheSize = 16;
    struct CachedFrame {
        int frame_id;
        QImage image;
    };
    std::array<CachedFrame, kCacheSize> cache_;
    std::atomic<int> latest_frame_id_{0};
    std::shared_mutex cache_mutex_;  // 读多写少——QML 30Hz 读 vs 产线 5Hz 写
};
```

### 6.4 ConfigViewModel

```cpp
namespace sai::visualization {

class ConfigViewModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString pipelineYaml READ pipelineYaml WRITE setPipelineYaml NOTIFY pipelineYamlChanged)
    Q_PROPERTY(QString rulesYaml READ rulesYaml WRITE setRulesYaml NOTIFY rulesYamlChanged)
    Q_PROPERTY(QString treeYaml READ treeYaml WRITE setTreeYaml NOTIFY treeYamlChanged)
    Q_PROPERTY(QString reloadStatus READ reloadStatus NOTIFY reloadStatusChanged)
    Q_PROPERTY(bool needsRestart READ needsRestart NOTIFY needsRestartChanged)
    Q_PROPERTY(QStringList validationErrors READ validationErrors NOTIFY validationErrorsChanged)
public:
    explicit ConfigViewModel(QObject* parent = nullptr);

    // 绑定依赖对象（不持有所有权，生命周期由 Context 管理）
    void BindToPipeline(sai::pipeline::Pipeline* pipeline);
    void BindToRuleEngine(sai::rule::RuleEngine* rule_engine);
    void BindToReasoner(sai::reasoner::IReasoner* reasoner);

    // 从文件系统加载 YAML 文本到 Q_PROPERTY
    Q_INVOKABLE void LoadConfig();
    Q_INVOKABLE void LoadRules();
    Q_INVOKABLE void LoadTree();

    // 应用变更
    Q_INVOKABLE void ApplyParameterChanges();   // 参数级热生效（阈值/k_nearest/ROI等）
    Q_INVOKABLE void ApplyRuleChanges();        // 规则热重载
    Q_INVOKABLE void ApplyTreeChanges();        // 决策树热重载
    Q_INVOKABLE void RestartPipeline();         // 拓扑变更 → 优雅重启

    // YAML 语法校验
    Q_INVOKABLE void ValidateYaml(const QString& text);

signals:
    void pipelineYamlChanged();
    void rulesYamlChanged();
    void treeYamlChanged();
    void reloadStatusChanged();
    void needsRestartChanged();
    void validationErrorsChanged();

private:
    sai::pipeline::Pipeline* pipeline_;
    sai::rule::RuleEngine* rule_engine_;
    sai::reasoner::IReasoner* reasoner_;
    bool needs_restart_{false};
};
```

### 6.5 DashboardViewModel

```cpp
namespace sai::visualization {

struct FrameSummary {
    int frame_id;
    std::string verdict;
    std::string severity;
    std::chrono::system_clock::time_point timestamp;
    std::vector<std::string> defect_types;  // 本帧检出的缺陷类型
};

class DashboardViewModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(int totalFrames READ totalFrames NOTIFY summaryChanged)
    Q_PROPERTY(int okFrames READ okFrames NOTIFY summaryChanged)
    Q_PROPERTY(int ngFrames READ ngFrames NOTIFY summaryChanged)
    Q_PROPERTY(double ppm READ ppm NOTIFY summaryChanged)
    Q_PROPERTY(QObject* defectTypeModel READ defectTypeModel CONSTANT)
    Q_PROPERTY(QVariantList ppmTrend READ ppmTrend NOTIFY summaryChanged)
    Q_PROPERTY(QVariantList latencyTrend READ latencyTrend NOTIFY summaryChanged)
public:
    explicit DashboardViewModel(QObject* parent = nullptr);

    // 每帧完成时追加摘要（由 Pipeline 结果回调触发）
    void AppendFrameSummary(FrameSummary summary);

    // 最大缓存帧数
    static constexpr int kMaxFrameSummaries = 1000;

signals:
    void summaryChanged();

private:
    // 环形缓冲区存储最近 kMaxFrameSummaries 帧摘要
    std::deque<FrameSummary> frame_summaries_;
    // 聚合统计（增量更新，不遍历全量）
    std::atomic<int> total_frames_{0};
    std::atomic<int> ok_frames_{0};
    std::atomic<int> ng_frames_{0};
    // 缺陷类型频次（QAbstractListModel）
};
```

### 6.6 Pipeline 结果回调接口

为支持 ViewModel 获取每帧检测结果，在 `sai::pipeline` 中新增一个轻量回调：

```cpp
namespace sai::pipeline {

// M7 新增：Pipeline 结果回调
using ResultCallback = std::function<void(int frame_id, const sai::rule::ReasoningResult&)>;

// Pipeline 接口追加一个方法（向后兼容，默认 nullptr = 不回调）
class Pipeline {
public:
    // ... 现有接口不变 ...
    auto SetResultCallback(ResultCallback callback) -> void;  // M7 新增
};

} // namespace sai::pipeline
```

**设计决策**：`ResultCallback` 在 `Export` 阶段的 WorkerPool 线程中调用，回调内部只做原子写入或 `QMutex` 保护下的数据更新，不做耗时操作。回调持有 `const ReasoningResult&` 引用，在回调返回后该引用失效——ViewModel 必须在回调内拷贝所需字段。

### 6.7 IStageNode 参数热生效接口（M6 补充）

为支持配置屏参数级热生效，在 `IStageNode` 接口中新增可选方法：

```cpp
namespace sai::pipeline {

class IStageNode : public Object {
public:
    // ... 现有接口不变 ...

    // M7 新增：参数热重载（可选实现，默认返回不支持）
    // 调用时机：运行期，下一帧 Process() 即生效
    // 参数：config 为 YAML 中该 stage 的 config 子节点
    virtual auto ReloadConfig(const YAML::Node& config) -> Result<void> {
        return Result<void>{};  // 默认：不支持热重载
    }
};

} // namespace sai::pipeline
```

---

## 7. 数据流

### 7.1 实时监控屏数据流

```
WorkerPool 线程（Capture）
  │
  ├── Pipeline::Submit(raw_image)
  │   └── frame_id++
  │
  ├── FrameProvider::RegisterRawFrame(frame_id, raw_image)
  │   └── 拷贝像素到环形缓存 CachedFrame
  │       └── latest_frame_id_.store(frame_id, relaxed)
  │
  └── [Capture 阶段 Process 完成，Push 到下游]

WorkerPool 线程（Export）
  │
  ├── 本帧完成
  │
  └── ResultCallback(frame_id, reasoning_result)
      ├── InspectionViewModel::UpdateResult(reasoning_result)
      │   └── std::atomic 更新 verdict/severity/confidence
      │   └── DefectModel 追加本帧 defects
      └── DashboardViewModel::AppendFrameSummary(summary)
          └── 追加到环形 dequeue
          └── 增量更新 total/ok/ng/ppm 原子计数器

QML 主线程（30Hz Timer）
  │
  ├── FrameProvider::requestImage()
  │   └── 读取 latest_frame_id_，从环形缓存取 QImage
  │       └── 拷贝返回（~3MB，<3ms）
  │
  ├── PipelineViewModel 读取
  │   └── Pipeline::Metrics() 中的 StageMetrics 原子字段
  │       └── 更新 StageMetricsObject 的 Q_PROPERTY → NOTIFY
  │
  └── InspectionViewModel 读取
      └── 读取原子字段 → NOTIFY → QML 绑定刷新
```

### 7.2 配置屏数据流

```
ConfigScreen.qml
  │ 用户编辑 YAML → 点击"应用"
  ▼
ConfigViewModel::ApplyParameterChanges()
  ├── 遍历 stages[] 中每个 stage
  │   ├── 解析 config 段提取变更字段
  │   └── stage->ReloadConfig(sub_config)  // IStageNode 新增接口
  │       └── 有实现 → 下一帧 Process() 用新参数
  │       └── 默认实现 → 记录警告（该 stage 不支持热重载）
  └── reloadStatus = "参数已应用，下一帧生效"

ConfigViewModel::ApplyRuleChanges()
  ├── 写入规则 YAML 到临时文件
  └── RuleEngine::ReloadRules(temp_path)  // M5 已有接口
      └── reloadStatus = "规则已热重载"

ConfigViewModel::ApplyTreeChanges()
  ├── 写入决策树 YAML 到临时文件
  └── IReasoner::ReloadTree(temp_path)    // M7 新增 IReasoner 方法
      └── reloadStatus = "决策树已热重载"

ConfigViewModel::RestartPipeline()
  ├── Pipeline::Drain()  // 等待在途帧完成
  ├── Pipeline::Stop()
  ├── Pipeline::LoadFromYAML(new_path, context)  // 新拓扑
  └── Pipeline::Start()
      └── reloadStatus = "Pipeline 已重启"
```

### 7.3 帧数据所有权

```
Camera 回调线程
  │
  └── unique_ptr<RawImage> → Pipeline::Submit(std::move(raw_image))
      │
      └── Capture Queue → Preprocess Queue → ... → Export Queue
          │
          └── unique_ptr 所有权沿队列传递
              每帧的 SurfaceImage 在 Export 后析构

FrameProvider::RegisterFrame(frame_id, const SurfaceImage*)
  │ 只读访问 SurfaceImage 像素缓冲区
  │ 拷贝到 QImage（独立于 SurfaceImage 的生命周期）
  └── 拷贝完成后不持有任何 SurfaceImage 引用
```

**关键约束**：`FrameProvider::RegisterFrame()` 在 `ResultCallback` 中调用（Export WorkerPool 线程），与 `Pipeline::Process()` 同线程——此时 `SurfaceImage` 尚未析构，拷贝安全。拷贝完成后，`QImage` 独立于 `SurfaceImage` 生命周期，后续 QML 主线程读写无竞争。

---

## 8. 工作流

### 8.1 应用启动

```
Seat AOI main()
  ├── Step 1: QGuiApplication + QQmlApplicationEngine 初始化
  ├── Step 2: 创建 Context（M1 DI 容器）
  ├── Step 3: 注册所有内建 Module（M1-M6）
  │   ├── sai::core / sai::memory / sai::plugin / sai::runtime / sai::infra
  │   ├── sai::device（MockCamera + MockLightController）
  │   ├── sai::image / sai::io
  │   ├── sai::inference（MockEngine）
  │   ├── sai::embedding / sai::detection（MockDetector）
  │   ├── sai::knowledge / sai::retrieval
  │   ├── sai::rule（RuleEngine）
  │   ├── sai::reasoner（DefaultReasoner）
  │   └── sai::pipeline（StageFactory 注册 7 个 mock stage 实现）
  ├── Step 4: Context::Initialize() + Context::Start()
  ├── Step 5: Pipeline::LoadFromYAML("pipeline.yaml", context)
  │   └── 失败 → 日志错误 + 弹窗提示 + 退出
  ├── Step 6: Pipeline::SetResultCallback(ViewModel 方法)
  ├── Step 7: Pipeline::Start()
  ├── Step 8: 创建 ViewModel 实例 + 绑定 Pipeline
  │   ├── PipelineViewModel::BindToPipeline(pipeline)
  │   ├── InspectionViewModel（绑定到 ResultCallback 内部）
  │   ├── ConfigViewModel::BindToPipeline(pipeline) +
  │   │   BindToRuleEngine(rule_engine) + BindToReasoner(reasoner)
  │   └── DashboardViewModel（绑定到 ResultCallback 内部）
  ├── Step 9: 注册 QML 类型 + 设置 Context Property
  │   ├── qmlRegisterType<StageMetricsObject>("sai.visualization", 1, 0, "StageMetrics")
  │   ├── qmlRegisterType<DefectModel>("sai.visualization", 1, 0, "DefectModel")
  │   ├── engine.addImageProvider("pipeline", frame_provider)
  │   └── engine.rootContext()->setContextProperty("pipelineVM", pipeline_vm)
  │       engine.rootContext()->setContextProperty("inspectionVM", inspection_vm)
  │       engine.rootContext()->setContextProperty("configVM", config_vm)
  │       engine.rootContext()->setContextProperty("dashboardVM", dashboard_vm)
  ├── Step 10: engine.load("qrc:/MainWindow.qml")
  └── Step 11: app.exec()  // 进入 Qt 事件循环
```

### 8.2 运行期主循环

```
Qt 事件循环（QML 主线程）
  │
  ├── 30Hz 刷新 Timer
  │   ├── PipelineViewModel::Refresh()
  │   │   ├── 读取 Pipeline::Metrics() 中各 StageMetrics 的 atomic 字段
  │   │   ├── 计算 overallFps（增量 frames_processed / (now - last_refresh)）
  │   │   └── emit metricsChanged() → QML 绑定刷新
  │   │
  │   └── InspectionViewModel → emit resultChanged() → QML 绑定刷新
  │
  ├── FrameProvider::requestImage()（QML Image 组件触发）
  │   ├── 从 latest_frame_id_ 确定最新帧索引
  │   ├── shared_lock 读环形缓存
  │   └── 返回 QImage 拷贝
  │
  └── 用户交互（Tab 切换、历史筛选、配置编辑...）
      └── 调用 ViewModel 的 Q_INVOKABLE 方法
          └── ViewModel 内部只做轻量操作（读缓存/写文件/调热重载）
```

### 8.3 应用关闭

```
main() 退出前
  ├── Pipeline::Drain()       // 等待在途帧完成
  ├── Pipeline::Stop()        // 阶段逆序停止
  ├── PipelineViewModel::StopRefresh()  // 停止 QTimer
  ├── Context::Stop()         // 停止所有 Module
  └── QGuiApplication::quit()
```

---

## 9. 线程模型

### 9.1 线程分布

| 线程 | 职责 | 关键约束 |
|------|------|---------|
| QML 主线程（Qt GUI） | QML 渲染、ViewModel 属性读取、用户交互响应 | 永不阻塞——只读 `std::atomic` 或 `shared_lock`；不做文件 I/O、不做 Pipeline 操作 |
| WorkerPool：Capture | 相机回调 → `Pipeline::Submit()` + `FrameProvider::RegisterFrame()` | RegisterFrame 内部 `unique_lock` 写环形缓存（写频 = 产线节拍 ≤ 5Hz） |
| WorkerPool：Inference | GPU 推理（MockEngine 或 TensorRtEngine） | 独占 CUDA stream |
| WorkerPool：Detect | PatchCore 检测 | 共享 Inference 池 |
| WorkerPool：Reason | RuleEval + Reason（决策树遍历） | CPU 密集型，不涉及 GPU |
| WorkerPool：Export | JSON 导出 + `ResultCallback(frame_id, reasoning_result)` | 回调内 ViewModel 原子写入（~100ns）+ DashboardViewModel deque 追加（~1μs） |

### 9.2 线程安全约定

| 对象 | 线程模型 | 同步机制 |
|------|---------|---------|
| `PipelineViewModel` | QML 主线程读，Export 线程无写入（只通过 Pipeline::Metrics 间接读） | 内部 `StageMetricsObject` 字段由 `UpdateFromStageMetrics` 更新（QML 主线程，Timer 回调），无竞争 |
| `InspectionViewModel` | Export 线程（ResultCallback）写 `std::atomic` 字段，QML 主线程读 | `std::atomic` 字段（verdict/severity/confidence），wait-free on all platforms |
| `FrameProvider` 环形缓存 | Export 线程（Capture）写，QML 主线程读 | `std::shared_mutex`：写 `unique_lock`，读 `shared_lock`——写频 5Hz、读频 30Hz，锁竞争可忽略 |
| `ConfigViewModel` | QML 主线程读写（用户操作触发） | 单线程，无需同步 |
| `DashboardViewModel` | Export 线程（ResultCallback）写 deque + atomic，QML 主线程读 | deque 写 `unique_lock(mutex_)`（写频 5Hz），atomic 字段 wait-free |
| `DefectModel` / `EvidenceModel` | Export 线程写（ResultCallback），QML 主线程读 | `QAbstractListModel::beginResetModel/endResetModel` 内部处理竞争 |

### 9.3 死锁预防

- ViewModel 的 `Q_PROPERTY` getter 只读 `std::atomic` 或 `shared_lock`——永不尝试获取第二个锁
- `FrameProvider::requestImage()` 只 `shared_lock` 读环形缓存，不调任何可能获取锁的回调
- `ConfigViewModel` 的热重载方法调用 `Pipeline::Drain()/Stop()/Start()`（可能需要等待 WorkerPool 线程），但这些方法在 QML 主线程通过信号槽异步调用，不阻塞 UI
- `ResultCallback` 中不调 `Pipeline::Submit()` 或其他可能获取 Pipeline 内部锁的操作

---

## 10. 性能

### 10.1 性能目标

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| QML 渲染帧率 | ≥ 30 fps | Qt Quick Profiler / `QQuickWindow::frameSwapped` 信号 |
| `requestImage()` 延迟 | < 3ms | 1024×1024 RGBA QImage 拷贝 + 格式转换 |
| 配置热生效延迟 | < 1 帧 | 下一帧 `Process()` 即生效，无需等 Drain |
| 应用冷启动（含 Pipeline 构建） | < 3s | `QElapsedTimer` main() 入口到 `app.exec()` |
| ViewModel 属性读取 | < 1μs | `std::atomic` load + 简单算术 |
| Pipeline 结果回调 | < 10μs | ViewModel 原子写入 + deque 追加 |

### 10.2 性能策略

- **帧缓存容量 = 16 帧**——环形缓存覆盖 ~0.5s 窗口（@30fps QML 刷新），QML 端永不因等待帧拷贝而丢帧
- **Metrics 轮询在 QML 主线程**——读取 `std::atomic`（wait-free），避免跨线程信号槽的锁竞争
- **QML 绑定最小化**——只绑定关键属性（verdict/severity/fps），非关键指标（各阶段 p99 延迟）通过 `QTimer` 低频刷新（2Hz）
- **DefectOverlay Repeater 上限**——单帧最多展示 20 个缺陷框（最大缺陷数），避免 QML 端 `Repeater` 失控渲染
- **历史屏延迟加载**——仅在用户切换到历史 Tab 时才加载缺陷图库数据，避免启动时全量加载

---

## 11. 内存

| 对象 | 生命周期 | 分配策略 |
|------|---------|---------|
| `PipelineViewModel` + `StageMetricsObject[]` | Application 启动 → 退出 | 栈上创建，`QObject` 父节点管理，7 个 `StageMetricsObject` 不增不减 |
| `InspectionViewModel` | Application 启动 → 退出 | 单实例，内部 `DefectModel`/`EvidenceModel` 每帧 `resetModel` 覆盖 |
| `FrameProvider` 环形缓存 | Application 启动 → 退出 | 16 个 QImage，每个 ~4MB（1024×1024×RGBA8888 = 4 bytes/pixel），总计 ~64MB。QML 端 `Image.source` 持有最近一帧的隐式缓存（由 Qt Quick 场景图管理，通常 2-3 帧） |
| `DashboardViewModel` 帧摘要环形缓冲区 | Application 启动 → 退出 | `std::deque<FrameSummary>` 上限 1000 帧。每帧摘要 ~200 字节（frame_id + verdict + timestamp + 平均 2 个 defect_type），总计 ~200KB |
| `ConfigViewModel` YAML 字符串 | 加载时 → 下次加载 | `QString`（UTF-16），一个 5KB pipeline.yaml → ~10KB QString。三个文件总计 ~30KB |

---

## 12. ErrorCode

M7 新增 `Visualization_*` 错误码，append-only 追加在 M6 错误码 `Scheduler_QueueCreateFailed` 之后：

| 错误码 | 含义 | 触发条件 |
|--------|------|---------|
| `Visualization_FrameBufferFull` | 帧环形缓存满 | `RegisterFrame` 时缓存未及时消费（QML 刷新停滞 > 0.5s） |
| `Visualization_ConfigReloadFailed` | 参数热重载失败 | `IStageNode::ReloadConfig` 返回失败（参数值不合法等） |
| `Visualization_PipelineRestartFailed` | Pipeline 重启失败 | 拓扑变更后 `Pipeline::LoadFromYAML` 或 `Start()` 失败 |

3 个错误码，按表内顺序追加。轻量——M7 主要是读取和展示，不以产生新错误路径为主要目的。

---

## 13. 未来扩展

| 扩展点 | 方向 | 触发条件 |
|--------|------|---------|
| 3D 点云可视化 | 激光三角测量/结构光的 3D 点云在 QML 中用 Qt Quick 3D 渲染 | 3D 传感器集成 |
| Web 远程监控 | `PipelineViewModel` 数据通过 WebSocket 推送到浏览器端，用 Web 图表库渲染 | 远程产线监控/多产线集中管理 |
| 历史数据持久化 | `DashboardViewModel` 的帧摘要落 SQLite，支持跨启动历史查询 | 缺陷趋势分析需要跨天/周数据 |
| 多窗口布局 | 监控屏 + 配置屏分离到不同显示器（产线操作员 vs 工程师视角） | 实际产线部署 |
| 自定义 QML Chart 主题 | 深色/高对比度工厂车间主题 | 不同光照条件的车间环境 |
| Pipeline 拓扑热重载 | `ConfigViewModel::RestartPipeline` 升级为无 Drain 的原子拓扑替换 | M6 §13 触发条件满足 |
| GPU 纹理共享渲染 | `FrameProvider` 升级为 `QQuickFramebufferObject` 零拷贝路径 | 8K 线扫相机大分辨率帧场景 |

---

## 14. 最佳实践

1. **ViewModel 只持有裸指针，不持有所有权**——Pipeline/RuleEngine/Reasoner 的生命周期由 Context 管理，ViewModel 不应干预
2. **QML 端只做声明式绑定，不调 C++ API**——`Image.source: "image://pipeline/frame?t=" + Date.now()`，不通过 `Button.onClicked` 调 `pipeline.submit()`
3. **ResultCallback 在 Export 线程执行，只做原子写入**——不在回调中做文件 I/O、数据库查询、YAML 解析
4. **帧缓存容量 = 刷新率 × 可容忍延迟**——16 帧 × 33ms = 528ms 窗口，覆盖临时 QML 卡顿
5. **配置屏的文本编辑器加 YAML 语法高亮和校验**——`ConfigViewModel::ValidateYaml()` 在用户点击"应用"前做校验，错误精确到行号
6. **历史屏数据仅加载筛选范围内的帧**——默认展示最近 100 帧，用户切换筛选条件后增量加载
7. **启动失败优雅降级**——Pipeline 构建失败时不 crash，弹出错误对话框并记录日志，允许用户修改配置后重试

---

## 15. 反模式

1. **在 QML 中调用 `pipelineVM.start()` / `pipelineVM.submit()`**——Pipeline 生命周期控制应该通过 `ConfigViewModel` 的封装方法，不直接暴露给 QML
2. **在 `requestImage()` 中做像素格式转换**——必须在 `RegisterFrame()` 时预转换为 `QImage::Format_RGBA8888`，`requestImage()` 只做纯拷贝
3. **在 QML `Timer` 回调中 `setContextProperty`**——Context Property 只在 `main()` 初始化时设置一次，运行期不增删
4. **ViewModel 持有 Pipeline 的 `shared_ptr`**——ViewModel 用裸指针 + 不持有所有权。Pipeline 的析构时机由 `Context::Stop()` 决定，ViewModel 不应延长其生命周期
5. **在 `ResultCallback` 中 `emit Q_OBJECT signal`**——`ResultCallback` 在 WorkerPool 线程执行，emit signal 会通过 `Qt::AutoConnection` 排队到 QML 主线程，但 Qt 事件循环的排队延迟不可控。正确做法是回调中只写 atomic 字段，由 QML 主线程的 QTimer 读取
6. **将 QML 文件硬编码路径**——使用 Qt Resource System（`.qrc`），QML 编译进可执行文件，不依赖运行时文件路径
7. **假设所有 stage 都支持 `ReloadConfig`**——`IStageNode::ReloadConfig` 的默认实现返回成功但不做任何事（表示"不支持热重载"），调用方应检查日志而非假设生效

---

## 16. 验证点

**主验证点**：Seat AOI 应用启动，四屏切换正常，模拟一帧走完 Pipeline 全链路，实时监控屏正确展示检测结果。

具体场景：

1. `seat_aoi` 启动 → `Pipeline::LoadFromYAML("pipeline.yaml")` → 校验通过
2. `Pipeline::Start()` → WorkerPool × 4 创建，StageQueue × 6 创建
3. 切换到 **实时监控屏** → 模拟一帧 `Pipeline::Submit(mock_raw_image)` → 30Hz Timer 刷新阶段指标
4. 切换 **历史回顾屏** → 缺陷图库展示模拟数据
5. 切换到 **产线配置屏** → 编辑检测阈值 → 点击"应用" → 参数热生效
6. 切换到 **统计仪表盘** → PPM 曲线、缺陷趋势展示模拟数据
7. 切换回实时监控屏 → `Pipeline::Drain()` → `Pipeline::Stop()` → 应用退出

**子验证点**：
- FrameProvider 环形缓存：Submit 帧后 `requestImage()` 返回有效的 QImage（非 null、尺寸正确）
- InspectionViewModel 数据流：ResultCallback 写入 → QML 属性绑定刷新（verdict/severity 文本正确）
- 配置热生效：修改阈值 → 下一帧 Process() 使用新值（通过 mock stage 的检测阈值断言验证）
- 规则热重载：修改 YAML 规则文件 → `ReloadRules()` → 下一帧 RuleEval 用新规则
- 优雅重启：`RestartPipeline()` 不崩溃、不丢帧（Drain 完成所有在途帧才 Stop）
- 启动失败：故意加载无效 YAML → 错误对话框 + 日志，crash-free

---

## 17. 契约增量

### 17.1 概念归属表新增

| 概念名称 | 归属批次 | 定义所在文档 | 说明 |
|---------|---------|-------------|------|
| `PipelineViewModel` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | QML 桥接层：Pipeline 状态 + StageMetrics → Q_PROPERTY |
| `InspectionViewModel` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | QML 桥接层：ReasoningResult + DetectionResult → Q_PROPERTY + QAbstractListModel |
| `FrameProvider` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | QQuickImageProvider：SurfaceImage/RawImage 像素 → QML Image 源 |
| `ConfigViewModel` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | QML 桥接层：YAML 编辑 + RuleEngine/Reasoner/IStageNode 热重载控制 |
| `DashboardViewModel` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | QML 桥接层：历史帧摘要聚合（PPM/趋势/延迟分布） |
| `ResultCallback` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | Pipeline 结果回调：每帧完成时通知 ViewModel |
| `Seat AOI` | 7.2 | design/milestone-07-visualization-application/7.2-application.md | 参考应用：零业务逻辑的模块组装，验证"Product 只是 metadata" |

### 17.2 接口签名表新增

| 接口名称 | 归属批次 | 定义所在文档 | 签名摘要 |
|---------|---------|-------------|---------|
| `PipelineViewModel` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | `class PipelineViewModel : public QObject { Q_PROPERTY(QString pipelineStatus ...) Q_PROPERTY(double overallFps ...) Q_PROPERTY(QList<QObject*> stageMetrics ...); void BindToPipeline(const Pipeline*); void StartRefresh(int ms); void StopRefresh(); }` |
| `InspectionViewModel` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | `class InspectionViewModel : public QObject { Q_PROPERTY(QString verdict ...) Q_PROPERTY(QString severity ...) Q_PROPERTY(QObject* defectModel ...); void UpdateResult(const ReasoningResult&); }` |
| `FrameProvider` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | `class FrameProvider : public QQuickImageProvider { QImage requestImage(const QString&, QSize*, const QSize&) override; void RegisterFrame(int, const SurfaceImage*); void RegisterRawFrame(int, const RawImage*); }` |
| `ConfigViewModel` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | `class ConfigViewModel : public QObject { Q_PROPERTY(QString pipelineYaml ...) Q_PROPERTY(QString reloadStatus ...) Q_PROPERTY(bool needsRestart ...); Q_INVOKABLE void LoadConfig/LoadRules/LoadTree(); Q_INVOKABLE void ApplyParameterChanges/ApplyRuleChanges/ApplyTreeChanges(); Q_INVOKABLE void RestartPipeline(); }` |
| `DashboardViewModel` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | `class DashboardViewModel : public QObject { Q_PROPERTY(int totalFrames ...) Q_PROPERTY(double ppm ...) Q_PROPERTY(QVariantList ppmTrend ...); void AppendFrameSummary(FrameSummary); }` |
| `ResultCallback` | 7.1 | design/milestone-07-visualization-application/7.1-visualization.md | `using ResultCallback = std::function<void(int frame_id, const ReasoningResult&)>;` — Pipeline::SetResultCallback 参数类型 |
| `IStageNode::ReloadConfig` | 6.1 (M7补充) | design/milestone-07-visualization-application/7.1-visualization.md | `virtual auto ReloadConfig(const YAML::Node& config) -> Result<void> { return Result<void>{}; }` — 阶段参数热重载（默认不支持） |
