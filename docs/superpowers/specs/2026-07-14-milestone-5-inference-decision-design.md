# Surface AI Framework —— 里程碑 5 推理决策 设计文档

> Status: Draft
> Date: 2026-07-14
> Based on: `docs/superpowers/specs/2026-07-07-surface-ai-framework-phased-plan-design.md` §4 里程碑 5
> Depends on: 里程碑 1（1.1-1.6 全部冻结接口）、里程碑 2（2.1-2.3 全部冻结接口）、里程碑 3（3.1-3.3 全部冻结接口）、里程碑 4（4.1-4.2 全部冻结接口）

---

## 1. 背景与范围

里程碑 5 覆盖"推理决策"：在 M3 的 DetectionResult 和 M4 的 KnowledgeGraph/RetrievalResult 之上，构建工业规则引擎（自研表达式 AST + YAML 规则文件 + 显式冲突消解）和分层推理机（YAML 配置决策树 + 加权评分 + 算子级 Trace + 全链路 Evidence）。

### 1.1 架构总览

两层分离，每层定义于独立批次：

```
5.1 Rule（规则引擎）
    IExpression（AST 接口 + 具体节点类型）
    Value（double/string/bool/list 类型系统）
    Lexer/Parser（递归下降解析器，中缀表达式 → AST）
    FactBase（扁平键值表 + FactSource 溯源元数据）
    FactBuilder（图路径 SQL 物化 + 向量检索注入 + DetectionResult 直接映射）
    RuleEngine（批量求值 + Memoization + 短路求值 + 多 rule_set 并行）
    ConflictResolver（显式 overrides/overridden_by + AST 重叠检测 + 循环覆盖拒绝）

5.2 Reasoner（分层推理机）
    DecisionTree（YAML → BranchNode/LeafNode 树结构）
    DefaultReasoner（树遍历 + 分支匹配 + 叶子加权评分）
    ScoreCalculator（sigmoid(Σw·x - t) + 多公式 max 聚合）
    TraceRecorder（算子级溯源：Expression/Rule/TreeBranch/Scoring）
    EvidenceCollector（全链路 FactBase + FactSource + Trace 关联打包）
    ReasoningResult（verdict/severity/recommendation/confidence/trace/evidence）
```

### 1.2 批次划分与执行顺序

```
里程碑 5：推理决策
├── 5.1 Rule（Value → AST → Lexer/Parser → FactBase → FactBuilder → RuleEngine → ConflictResolver）
└── 5.2 Reasoner（DecisionTree → ScoreCalculator → TraceRecorder → EvidenceCollector → DefaultReasoner）
```

执行顺序：**5.1 设计 → 5.1 代码 → 5.2 设计 → 5.2 代码 → 最终整体回顾**

### 1.3 项目锚点

延续 M1/M2/M3/M4 的汽车座椅 AOI 场景：

- **Rule**：工艺工程师为每种缺陷类型编写判定规则（如"划痕面积 > 10mm² 且位于座椅中心区域 → NG"），规则存储在 YAML 中，支持热重载
- **图路径规则**：`material->supplier->batch.reject_rate > 0.05 AND defect.confidence > 0.9 → 供应商质量预警`——跨 KnowledgeGraph 边关系查询
- **Reasoner**：分层决策树先按缺陷类型分支（scratch/bubble/crack/…），再按位置/面积/供应商质量进一步分支，叶子节点用加权评分公式综合多维度得分
- **Trace & Evidence**：每个 NG 判定可回溯到具体哪条规则、哪个字段取了什么值、知识图谱中哪条 SQL 产生了哪个中间结果

### 1.4 明确排除项

- 规则训练/自动发现（规则由工艺工程师手工编写）
- 自然语言规则编写（M7 GUI 处理）
- 规则版本管理/SemVer（v1 不做，Future Extension）
- 时序窗口函数（`defect.count IN last_7_days`——v1 不做，但统计查询可通过 KnowledgeGraph SQL 实现等价语义）
- 多 Surface 跨工位关联规则（v1 不做）
- 分布式 Evidence 存储（单进程内完成）

---

## 2. 对里程碑 1/2/3/4 的依赖

| M5 批次 | 依赖接口 | 用途 |
|---------|---------|------|
| 5.1 | `Result<T>`、`Object`（1.1） | 错误处理 + AST 接口基类 |
| 5.1 | `ConfigStore`（1.6） | Rule YAML 路径 + Schema 校验 + 热重载 |
| 5.1 | `Logger`（1.6） | 规则求值耗时、冲突诊断、PathExpr 解析日志 |
| 5.1 | `WorkerPool`（1.4） | 多 rule_set 并行求值 |
| 5.1 | `KnowledgeGraph`（4.1） | 图路径解析：`material→supplier→batch.reject_rate` → SQL JOIN 物化 |
| 5.1 | `KnowledgeStore`（4.1） | 知识记录批量查询 |
| 5.1 | `RetrievalResult`、`VectorPath`（4.2） | 向量检索 Top-K 注入 FactBase |
| 5.1 | `DetectionResult`（3.3） | 检测产出映射到 FactBase |
| 5.1 | `Embedding`（3.2） | 向量检索的 query embedding |
| 5.2 | `IService`（1.2） | Reasoner 作为 Service 注册到 DI 容器 |
| 5.2 | `Context`（1.2） | Reasoner 的依赖注入入口 |
| 5.2 | 5.1 全部接口 | `IExpression`、`FactBase`、`FactSource`、`ResolvedRule` |

---

## 3. 架构设计

### 3.1 核心设计原则

**FactBase 物化是架构的核心**。规则表达式和图路径在求值前完成一次性的批量物化，求值阶段仅做纯 CPU 表查找，不触发外部 I/O。

```
DetectionResult ──┐
KnowledgeGraph ───┤
VectorPath ───────┘
        │
        ▼
FactBuilder::Build(surface_id, detection, graph_paths)
        │
        ├── 直接映射：defect.score, defect.area_mm2, ...
        ├── 图路径物化：material→supplier→batch → SQL JOIN → 扁平键
        ├── 向量检索：Top-K 相似案例 → retrieval.top1..topK
        └── 批量完成，所有结果注入 FactBase（键 + 值 + FactSource 溯源）
        │
        ▼
FactBase（扁平 unordered_map<string, Value>）
        │
        ├──► RuleEngine::EvaluateAll(FactBase)  → ResolvedRule[]
        │      ├── 多 rule_set 并行求值（各自 FactBase copy-on-write 快照）
        │      ├── AST 解释执行 + Memoization + 短路求值
        │      └── ConflictResolver: 显式 overrides DAG + 循环检测
        │
        └──► Reasoner::Reason(FactBase, ResolvedRule[])
               ├── 决策树遍历（BranchNode 字段匹配 + LeafNode 加权评分）
               ├── sigmoid(Σw·x - t) + 多公式 max 聚合
               ├── TraceRecorder: 算子级溯源
               └── EvidenceCollector: 全链路打包
```

### 3.2 架构取舍

| 决策 | 采用 | 拒绝 | 理由 |
|------|------|------|------|
| 求值策略 | AST 解释执行 + Memoization | 字节码编译 | 工业 AOI 规则数（几十到几百）+ 秒级产线节拍，不需要微秒级优化；解释执行的 Trace 直接映射源码，调试友好 |
| 数据注入 | FactBase 预先物化 | 求值时惰性查询 | 避免 N+1 问题（每条规则的图路径各自触发 SQL），物化后纯 CPU 表查找确定性高 |
| 冲突消解 | 显式 overrides/overridden_by | 隐式优先级数字 | 优先级数字在规则数量增长后难以维护；显式声明自带文档效果，且支持 AST 重叠检测自动告警 |
| 规则语言 | 自研中缀表达式 + YAML | Lua / Python 嵌入 | 锁定技术栈要求；自研语言可控性强，避免任意代码执行风险 |
| Confidence | 复用 leaf score | 独立置信度公式 | M3 的 `defect.confidence` 已作为 FactBase 一等字段参与加权评分；拆分不增加表达能力（Future Extension 可加可选 `confidence_formula`） |

---

## 4. 模块结构

### 4.1 文件布局

```
include/sai/rule/
    value.h                 # Value, Scalar 类型
    expression.h             # IExpression, ExprKind, BinaryOp, UnaryOp, BuiltinFn
    ast_nodes.h              # LiteralExpr, FieldRefExpr, BinaryExpr, UnaryExpr, FunctionExpr, PathExpr
    fact_base.h              # FactBase, FactSource, FactSourceKind
    fact_builder.h           # FactBuilder
    rule_engine.h            # Rule, RuleAction, ResolvedRule, RuleEngine, OverlapWarning

include/sai/reasoner/
    decision_tree.h          # DecisionTree, IDecisionNode, BranchNode, LeafNode, ScoreFormula
    reasoner.h               # IReasoner, DefaultReasoner, ReasoningResult
    evidence_collector.h     # EvidenceItem, EvidenceCollector（EvidenceItem 是公开 API 类型）

src/rule/
    CMakeLists.txt
    lexer.h                  # 内部：词法分析器
    parser.h                 # 内部：递归下降解析器
    conflict_resolver.h      # 内部：冲突消解
    value.cpp
    lexer.cpp
    parser.cpp
    expression.cpp
    fact_base.cpp
    fact_builder.cpp
    rule_engine.cpp
    conflict_resolver.cpp

src/reasoner/
    CMakeLists.txt
    tree_walker.h            # 内部：决策树遍历
    score_calculator.h       # 内部：评分计算
    trace_recorder.h         # 内部：溯源记录
    decision_tree.cpp
    tree_walker.cpp
    score_calculator.cpp
    trace_recorder.cpp
    evidence_collector.cpp
    reasoner.cpp

tests/rule/
    lexer_test.cpp
    parser_test.cpp
    expression_test.cpp
    fact_base_test.cpp
    fact_builder_test.cpp
    rule_engine_test.cpp
    conflict_resolver_test.cpp
    integration_test.cpp

tests/reasoner/
    decision_tree_test.cpp
    score_calculator_test.cpp
    reasoner_test.cpp
    integration_test.cpp
```

### 4.2 命名空间

- `sai::rule`：5.1 Rule Engine 的所有公共类型
- `sai::reasoner`：5.2 Reasoner 的所有公共类型

---

## 5. 接口设计

### 5.1 Value 类型系统

```cpp
namespace sai::rule {

using Scalar = std::variant<double, std::string, bool>;

class Value {
public:
    enum class Kind { Null, Scalar, List };

    static auto Null() -> Value;
    static auto Of(double) -> Value;
    static auto Of(std::string) -> Value;
    static auto Of(bool) -> Value;
    static auto OfList(std::vector<Value>) -> Value;

    auto GetKind() const noexcept -> Kind;
    auto IsNull() const -> bool;
    auto AsDouble() const -> std::optional<double>;
    auto AsString() const -> std::optional<std::string_view>;
    auto AsBool() const -> std::optional<bool>;
    auto AsList() const -> const std::vector<Value>*;  // nullptr = not a list (optional<T&> illegal in C++)

    // 算术（Null 传播）
    auto operator+(const Value&) const -> Value;
    auto operator-(const Value&) const -> Value;
    auto operator*(const Value&) const -> Value;
    auto operator/(const Value&) const -> Value;

    // 比较
    auto operator==(const Value&) const -> bool;
    auto operator<(const Value&) const -> bool;
    auto operator>(const Value&) const -> bool;
};

} // namespace sai::rule
```

### 5.2 表达式 AST

```cpp
namespace sai::rule {

enum class ExprKind { Literal, FieldRef, Binary, Unary, Function, Path };
enum class BinaryOp { And, Or, Add, Sub, Mul, Div, Eq, Neq, Lt, Le, Gt, Ge, In };
enum class UnaryOp { Not, Neg };
enum class BuiltinFn { Avg, Count, Max, Min, Sum, Len };

class IExpression : public Object {
public:
    virtual auto GetKind() const noexcept -> ExprKind = 0;
    virtual auto Evaluate(FactBase& ctx) const -> Result<Value> = 0;
    virtual auto CollectFieldRefs() const -> std::vector<std::string> = 0;
    virtual auto SourceText() const -> std::string_view = 0;
};

class LiteralExpr final : public IExpression {
    // Value value_;
};

class FieldRefExpr final : public IExpression {
    // std::string key_;  // FactBase 中的键
};

class BinaryExpr final : public IExpression {
    // BinaryOp op_; unique_ptr<IExpression> lhs_, rhs_;
};

class UnaryExpr final : public IExpression {
    // UnaryOp op_; unique_ptr<IExpression> operand_;
};

class FunctionExpr final : public IExpression {
    // BuiltinFn fn_; std::vector<unique_ptr<IExpression>> args_;
    // 列表解包：args 中的 ListUnpackExpr 表示对列表元素逐个应用函数
};

class PathExpr final : public IExpression {
    // std::string path_;  // 原始路径字符串 "material->supplier->batch.reject_rate"
    // 加载期解析：调用 KnowledgeGraph::ResolvePath() → 存储物化后的 FactBase 键列表
    // 求值时：直接 FactBase::Get(flattened_key)
};

} // namespace sai::rule
```

### 5.3 FactBase & FactBuilder

```cpp
namespace sai::rule {

enum class FactSourceKind {
    Direct,        // 直接映射（defect.score ← DetectionResult）
    GraphPath,     // 图路径查询（material.supplier.name ← KnowledgeGraph JOIN）
    VectorSearch,  // 向量检索（retrieval.top3_similar ← FAISS TopK）
    Computed,      // 表达式中间结果（Memoization）
    Default        // 默认值
};

struct FactSource {
    FactSourceKind kind;
    std::string description;
    std::chrono::microseconds elapsed;
    std::optional<std::string> sql;    // GraphPath 时的实际 SQL
    std::optional<int> top_k;          // VectorSearch 时的 K 值
};

class FactBase {
public:
    auto Set(std::string_view key, Value value, FactSource source) -> void;
    auto SetDefault(std::string_view key, Value value) -> void;

    auto Get(std::string_view key) const -> std::optional<Value>;
    auto GetOr(std::string_view key, Value default_val) const -> Value;
    auto Has(std::string_view key) const -> bool;

    // 图路径 → 扁平键映射
    auto AddPathMapping(std::string_view graph_path, std::string_view flat_key) -> void;
    auto ResolvePath(std::string_view graph_path) const -> std::optional<std::string>;

    auto SourceOf(std::string_view key) const -> const FactSource&;
    auto AllEntries() const -> std::vector<std::pair<std::string, Value>>;
    auto AllSources() const -> std::vector<std::pair<std::string, FactSource>>;

    // 创建不可变快照（用于多 rule_set 并行求值）
    auto Snapshot() const -> FactBase;
};

class FactBuilder {
public:
    explicit FactBuilder(
        std::shared_ptr<knowledge::KnowledgeGraph> kg,
        std::shared_ptr<retrieval::VectorPath> vp
    );

    auto Build(
        std::string_view surface_id,
        const detection::DetectionResult& detection,
        const std::vector<std::string>& graph_paths_to_resolve
    ) -> Result<FactBase>;

private:
    auto MapDetection(FactBase&, const detection::DetectionResult&) -> void;
    auto ResolveGraphPaths(FactBase&, const std::vector<std::string>&) -> Result<void>;
    auto RunVectorRetrieval(FactBase&, std::string_view embedding_key) -> Result<void>;
};

} // namespace sai::rule
```

### 5.4 Rule & RuleEngine

```cpp
namespace sai::rule {

struct RuleAction {
    std::string label;
    double base_severity;        // 0.0-1.0
    std::string recommendation;
};

struct Rule {
    std::string name;
    uint32_t priority;           // tiebreaker within rule_set
    std::string rule_set;
    std::vector<std::string> overrides;
    std::vector<std::string> overridden_by;
    std::unique_ptr<IExpression> condition;
    RuleAction action;
};

// TraceStep 定义在 sai::rule 命名空间（5.1 是下层，5.2 复用）
// 四个 level: Expression/Rule 由 RuleEngine 填充，TreeBranch/Scoring 由 Reasoner 填充
struct TraceStep {
    std::string id;
    enum class Level { Expression, Rule, TreeBranch, Scoring };
    Level level;
    std::string description;
    std::string source_location;
    std::chrono::microseconds timestamp;
    std::optional<std::string> parent_id;
};

struct ResolvedRule {
    std::string name;
    bool matched;
    RuleAction action;
    std::vector<TraceStep> eval_trace;  // Expression + Rule level 溯源
};

struct OverlapWarning {
    std::string rule_a;
    std::string rule_b;
    std::vector<std::string> common_fields;
};

class RuleEngine {
public:
    // 加载与热重载
    auto LoadFromYAML(std::filesystem::path) -> Result<void>;
    auto EnableHotReload(std::filesystem::path, std::stop_token) -> Result<void>;

    // 求值
    auto EvaluateAll(FactBase&) -> Result<std::vector<ResolvedRule>>;

    // 冲突消解
    auto ResolveConflicts(const std::vector<ResolvedRule>&) -> std::vector<ResolvedRule>;

    // 静态分析
    auto DetectOverlaps() const -> std::vector<OverlapWarning>;

private:
    std::map<std::string, std::vector<Rule>> rule_sets_;
    std::filesystem::path current_path_;
};

} // namespace sai::rule
```

### 5.5 DecisionTree

```cpp
namespace sai::reasoner {

struct ScoreFormula {
    std::vector<double> weights;
    std::vector<std::string> features;   // FactBase 键
    double threshold;
    // score = sigmoid(Σ(w[i] * FactBase[features[i]]) - threshold)
};

class IDecisionNode : public Object {
public:
    enum class Kind { Branch, Leaf };
    virtual auto GetKind() const noexcept -> Kind = 0;
};

class BranchNode final : public IDecisionNode {
public:
    auto FieldName() const -> std::string_view;
    auto Branches() const -> const std::map<std::string, std::unique_ptr<IDecisionNode>>&;
    auto DefaultBranch() const -> const IDecisionNode*;
};

class LeafNode final : public IDecisionNode {
public:
    auto Formulas() const -> const std::vector<ScoreFormula>&;
    auto Label() const -> std::string_view;
    auto Recommendation() const -> std::string_view;
};

class DecisionTree {
public:
    static auto LoadFromYAML(std::filesystem::path) -> Result<std::unique_ptr<DecisionTree>>;
    auto Root() const -> const IDecisionNode&;
};

} // namespace sai::reasoner
```

### 5.6 Reasoner

```cpp
namespace sai::reasoner {

// TraceStep 复用 sai::rule::TraceStep（5.2 不重新定义）

struct EvidenceItem {
    std::string key;
    rule::Value value;
    rule::FactSource source;
};

struct ReasoningResult {
    std::string verdict;          // "OK" | "NG" | "WARN" | "UNCERTAIN"
    double severity;              // 0.0-1.0
    std::string recommendation;
    double confidence;            // 0.0-1.0，复用 leaf score
    std::vector<rule::TraceStep> trace;
    std::vector<EvidenceItem> evidence;
    std::vector<std::string> triggered_rules;
    std::vector<std::string> overridden_rules;
};

class IReasoner : public IService {
public:
    virtual auto Reason(
        const rule::FactBase& facts,
        const std::vector<rule::ResolvedRule>& rules
    ) -> Result<ReasoningResult> = 0;
};

class DefaultReasoner final : public IReasoner {
public:
    explicit DefaultReasoner(std::unique_ptr<DecisionTree> tree);
    auto Reason(const rule::FactBase&, const std::vector<rule::ResolvedRule>&)
        -> Result<ReasoningResult> override;
};

} // namespace sai::reasoner
```

---

## 6. 表达式语法规范

### 6.1 词法单元

| Token | 示例 | 说明 |
|-------|------|------|
| Number | `123`, `3.14` | 整数或浮点 |
| String | `"hello"`, `'single'` | 双引号或单引号 |
| Bool | `true`, `false` | 布尔字面量 |
| Identifier | `defect.score`, `material.name` | 字段引用（`.` 分隔层级） |
| Arrow | `->` | 图路径分隔符 |
| Operator | `+`, `-`, `*`, `/`, `==`, `!=`, `>`, `<`, `>=`, `<=` | 算术 + 比较 |
| Keyword | `AND`, `OR`, `NOT`, `IN` | 逻辑运算 |
| Paren | `(`, `)` | 分组 |
| Bracket | `[`, `]` | 列表字面量 / 解包 |
| Comma | `,` | 函数参数分隔 |

### 6.2 运算符优先级（从高到低）

| 优先级 | 运算符 | 结合性 |
|--------|--------|--------|
| 7 | `*`, `/` | 左 |
| 6 | `+`, `-` | 左 |
| 5 | `==`, `!=`, `>`, `<`, `>=`, `<=`, `IN` | 左 |
| 4 | `NOT` | 右 |
| 3 | `AND` | 左 |
| 2 | `OR` | 左 |

### 6.3 图路径语法

```
material->supplier->batch.reject_rate
```

- `->` 表示 KnowledgeGraph 中的有向边遍历
- 最后一个段（`batch.reject_rate`）是终节点的属性访问
- 加载期通过 `KnowledgeGraph::ResolvePath()` 解析为 SQL JOIN 并校验边关系存在性
- 物化后的扁平键：`"material.supplier.batch.reject_rate"`（所有 `->` 替换为 `.`）

### 6.4 内置函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `AVG` | `AVG(v1, v2, ...)` | 算术平均 |
| `COUNT` | `COUNT(list)` | 列表元素个数 |
| `MAX` | `MAX(v1, v2, ...)` | 最大值 |
| `MIN` | `MIN(v1, v2, ...)` | 最小值 |
| `SUM` | `SUM(v1, v2, ...)` | 求和 |
| `LEN` | `LEN(list)` | 列表长度 |

### 6.5 列表解包

```
retrieval.top5[*].score     → 对 top5 每个元素取 .score 字段，展开为列表
MAX(retrieval.top5[*].score) → 等价于 MAX(top5[0].score, ..., top5[4].score)
```

---

## 7. YAML Schema

### 7.1 规则文件格式

```yaml
rule_sets:
  seat_leather_defects:
    rules:
      - name: scratch_major
        priority: 100
        overrides: [scratch_minor]
        condition: |
          defect.type == "scratch" AND
          defect.area_mm2 > 10 AND
          defect.position IN ["seat_center", "side_bolster"]
        action:
          label: "scratch_major"
          base_severity: 0.8
          recommendation: "严重划痕，建议立即返工"

      - name: scratch_minor
        priority: 50
        overridden_by: [scratch_major]
        condition: |
          defect.type == "scratch" AND
          defect.area_mm2 > 2
        action:
          label: "scratch_minor"
          base_severity: 0.3
          recommendation: "轻微划痕，标记跟踪"

      - name: supplier_alert
        priority: 200
        condition: |
          defect.severity > 0.7 AND
          material->supplier->batch.quality_score < 0.3
        action:
          label: "supplier_quality_alert"
          base_severity: 0.9
          recommendation: "供应商质量预警，建议暂停该批次"
```

### 7.2 决策树文件格式

```yaml
decision_tree:
  name: "seat_leather_inspection"
  version: "1.0"
  root:
    type: branch
    field: defect.type
    branches:
      scratch:
        type: branch
        field: defect.area_mm2
        branches:
          ">10":
            type: leaf
            label: "NG"
            recommendation: "严重划痕区域 >10mm²"
            formulas:
              - weights: [0.5, 0.3, 0.2]
                features:
                  - defect.score
                  - defect.confidence
                  - material->supplier->batch.reject_rate
                threshold: 0.5
          default:
            type: leaf
            label: "WARN"
            recommendation: "轻微划痕，跟踪观察"
            formulas:
              - weights: [0.6, 0.4]
                features: [defect.score, defect.confidence]
                threshold: 0.3
      bubble:
        type: leaf
        label: "NG"
        recommendation: "气泡缺陷直接判定NG"
        formulas:
          - weights: [1.0]
            features: [defect.score]
            threshold: 0.6
      default:
        type: leaf
        label: "UNCERTAIN"
        recommendation: "未知缺陷类型，人工复核"
        formulas:
          - weights: [1.0]
            features: [defect.score]
            threshold: 0.5
```

### 7.3 决策树语义

- **BranchNode**：取 `field` 在 FactBase 中的值，匹配 `branches` 的键。数值比较用 `>N`/`<N`/`>=N` 前缀语法内嵌在 YAML 键中（YAML 键只能是字符串）。无匹配走 `default`
- **LeafNode**：`formulas` 是并行评分公式族，每个公式独立计算 `sigmoid(Σw·x - t)`，最终 score = max(各公式得分)。`label` 覆盖 verdict 输出
- **default 分支必须存在**——不完整的决策树拒绝加载（`Reasoner_InvalidTree`）

### 7.4 热重载语义

- Rule YAML 与 Decision Tree YAML 均支持热重载（复用 1.6 `ConfigStore::EnableHotReload` 的 inotify 模式）
- 重载流程：解析新文件 → 校验（路径合法性 + AST 语法 + 循环覆盖检测）→ 原子替换（成功则 swap，失败保留旧配置 + 日志 Error）
- 热重载后自动运行 `DetectOverlaps()`：发现新的未声明条件重叠 → Warning 日志，不阻止重载

---

## 8. 工作流

### 8.1 规则加载流程

```
RuleParser::ParseYAML(path)
  ├── YAML::Load(rule_file)
  ├── 遍历 rule_sets → 对每个 rule:
  │   ├── Lexer::Tokenize(condition_string) → Token[]
  │   ├── Parser::Parse(Token[]) → unique_ptr<IExpression>
  │   │   递归下降：ParseOr → ParseAnd → ParseNot → ParseCompare →
  │   │   ParseAddSub → ParseMulDiv → ParseUnary → ParseAtom
  │   ├── CollectFieldRefs() → 提取所有字段引用
  │   ├── PathExpr::Resolve() → 调用 KnowledgeGraph::ValidatePath() 校验边关系
  │   └── 构造 Rule 对象 → rules_[rule_set].push_back(rule)
  └── DetectOverlaps() → 同 rule_set 内字段引用交集检查
```

### 8.2 FactBase 构建流程

```
FactBuilder::Build(surface_id, detection, graph_paths)
  ├── Step 1: 直接映射
  │   defect.type/score/confidence/area_mm2/position/passed ← DetectionResult
  ├── Step 2: 图路径物化（批量）
  │   └── 对每个 graph_path: KnowledgeGraph::ResolvePath() → SQL JOIN → 扁平键
  └── Step 3: 向量检索（按需）
      └── VectorPath::TopK(embedding, K) → retrieval.top1..topK
```

### 8.3 规则求值与冲突消解

```
RuleEngine::EvaluateAll(FactBase& facts)
  └── 多 rule_set 并行（各自 FactBase snapshot）
      └── 每条 rule（按 priority 降序）:
          ├── condition->Evaluate(facts)
          │   ├── Memoization: 子表达式结果缓存至 FactBase
          │   ├── 短路求值: A AND B → A=false 跳过 B
          │   └── TraceStep 记录每个操作数求值和运算符结果
          └── matched ? ResolvedRule { matched, action, eval_trace }

ConflictResolver::Resolve(ResolvedRule[])
  └── 每个 rule_set 独立:
      ├── 构造 overrides DAG（拓扑消解）
      ├── 循环覆盖检测 → Rule_CyclicOverride
      └── 输出: triggered_rules[] + overridden_rules[]
```

### 8.4 决策树遍历与评分

```
DefaultReasoner::Reason(FactBase& facts, ResolvedRule[] rules)
  ├── Step 1: 合并规则结果到 FactBase（rule.X.matched = true/false）
  ├── Step 2: WalkNode(tree.Root(), facts)
  │   ├── BranchNode: facts.Get(field) → 匹配 branches 键 → 递归
  │   └── LeafNode:
  │       ├── 每个 ScoreFormula: sigmoid(Σw·x - t)
  │       ├── score = max(formula_scores)
  │       └── 返回 { label, score, recommendation, trace_nodes }
  ├── Step 3: 组装 ReasoningResult
  └── Step 4: EvidenceCollector::Pack（FactBase::AllEntries + AllSources + Trace step ID 关联）
```

### 8.5 端到端数据流

```
M3 DetectionResult ──┐
M4 KnowledgeGraph ───┤
M4 VectorPath ───────┘
        │
        ▼
FactBuilder::Build()                     [5.1 Rule]
        │
        ▼
FactBase
        │
        ├──► RuleEngine::EvaluateAll()   [5.1 Rule]
        │       │
        │       ▼
        │    ResolvedRule[]
        │
        └──► Reasoner::Reason()          [5.2 Reasoner]
                │
                ▼
            ReasoningResult { verdict, severity, trace[], evidence[] }
```

---

## 9. 线程模型

### 9.1 Rule 并行求值

```
RuleEngine::EvaluateAll(FactBase& facts)
  ├── rule_sets.size() == 1 → 串行求值
  └── rule_sets.size() > 1 → 并行
      ├── FactBase snapshot = facts.Snapshot()  // 共享不可变快照
      ├── 提交 N 个任务到 WorkerPool(Reason)
      │   └── 每个任务: EvaluateRuleSet(snapshot, rule_set[i])
      └── merge: 合并所有 ResolvedRule[] 后统一冲突消解
```

### 9.2 线程安全约定

| 对象 | 线程模型 | 说明 |
|------|---------|------|
| `FactBase` | 单写者多读者 | Build 阶段单线程写入；求值阶段多读者（并行时 snapshot 共享） |
| `RuleEngine` | 单线程调用 | `EvaluateAll()` 内部并行，对外单线程 |
| `DecisionTree` | 只读 | 加载后不可变，遍历期间无锁 |
| `DefaultReasoner` | 单线程 | `Reason()` 调用期间独占 |

### 9.3 错误恢复

单条规则求值失败（如类型错误）→ 该规则标记为 `matched=false`，错误信息写入 TraceStep（level=Expression, description=错误详情），继续求值剩余规则。**工业产线不因单条规则配置错误而整体停机。**

---

## 10. ErrorCode

M5 引入 `Rule_*` / `Reasoner_*` 错误码前缀，append-only 追加在 M4 错误码之后：

| 错误码 | 含义 | 触发条件 |
|--------|------|---------|
| `Rule_ParseError` | 规则表达式语法错误 | Lexer/Parser 无法识别的 token |
| `Rule_InvalidPath` | 图路径不存在 | PathExpr 引用的知识图谱边关系不存在 |
| `Rule_TypeMismatch` | 表达式类型错误 | 无意义的类型间比较（如 `"hello" > 5`） |
| `Rule_CyclicOverride` | 覆盖声明成环 | `A overrides B, B overrides C, C overrides A` |
| `Rule_ReloadFailed` | 规则热重载失败 | 新 YAML 解析/校验失败，旧规则保留 |
| `Reasoner_TreeLoadFailed` | 决策树加载失败 | YAML 解析失败/结构不符合 schema |
| `Reasoner_InvalidTree` | 决策树结构错误 | 叶子节点无 formulas/分支节点缺少 field/default |
| `Reasoner_ScoreComputationFailed` | 评分计算失败 | 特征键在 FactBase 中缺失 |

以上 8 个错误码按表内顺序追加。M1 的 `error.h` 附加规则同样适用——永不重排、永不碰其他批次的成员。

---

## 11. 冲突消解算法

### 11.1 显式覆盖

```
rule_a.overrides 包含 rule_b → rule_a 淘汰 rule_b
rule_b.overridden_by 包含 rule_a → rule_a 淘汰 rule_b

双向声明不一致：以 overridden_by 为准，发出 Warning
```

### 11.2 循环检测

```
构造有向图: overrides 边 from 覆盖者 to 被覆盖者
DFS 检测环 → Rule_CyclicOverride → 拒绝加载
```

### 11.3 AST 重叠检测（DetectOverlaps）

```
同 rule_set 内两两比较:
  field_refs_a = CollectFieldRefs(rule_a.condition)
  field_refs_b = CollectFieldRefs(rule_b.condition)
  common = intersect(field_refs_a, field_refs_b)
  if common 非空 AND 无显式覆盖声明 → OverlapWarning
```

---

## 12. 评分计算

### 12.1 单公式评分

```
raw_score = Σ(w[i] * FactBase[features[i]]) - threshold
score = sigmoid(raw_score) = 1 / (1 + exp(-raw_score))
```

`sigmoid` 将任意实数值映射到 `(0, 1)`，阈值 `threshold` 决定了 sigmoid 的中点偏移。

### 12.2 多公式聚合

```
leaf_score = max(formula_scores)
```

同一个 leaf 的多个 formulas 取最大值——每个公式代表一个独立的评分维度，max 语义是"最严重维度决定结果"。

### 12.3 Verdict 判定

```
if score > 0.7  → "NG"
if score > 0.3  → "WARN"（若 leaf.label 为 "WARN"，否则视 leaf.label 覆盖）
else            → "OK"
```

`leaf.label` 可以覆盖 verdict：label 为 `"NG"` 的叶子即使 score < 0.7 也输出 NG（如"气泡直接判定 NG"）。

---

## 13. 性能目标

| 指标 | 目标 | 条件 |
|------|------|------|
| 单条规则表达式求值 | < 1ms | 表达式复杂度 ≤ 10 AST 节点 |
| FactBase 构建 | < 10ms | 10 条图路径 + Top-5 向量检索, SQLite 1000 节点 |
| 决策树遍历 + 评分 | < 1ms | 深度 ≤ 5 层, ≤ 20 叶子 |
| 端到端 (Build→Evaluate→Reason) | < 50ms | 100 条规则 + 10 图路径 + 5 层决策树 |
| 热重载延迟 | < 100ms | 100 条规则 YAML 重解析 |

### 13.1 性能策略

- **Memoization**：AST 节点求值结果缓存到 FactBase（以 SourceText hash 为 key），跨 rule 共享子表达式只算一次
- **图路径批量预取**：加载期 `CollectFieldRefs()` 收集所有规则的全部图路径 → 合并去重 → 一次性提交 KnowledgeGraph
- **短路求值**：`AND`/`OR` 操作数短路
- **决策树缓存**：BranchNode 匹配结果在同一 FactBase 下不变，首次匹配后缓存子节点引用

---

## 14. 内存策略

| 对象 | 生命周期 | 分配策略 |
|------|---------|---------|
| `Rule` + `IExpression` AST | 加载→下次热重载 | `unique_ptr` 树形结构，重载时整体替换旧树 |
| `FactBase` | 单次请求 | 栈分配 `unordered_map<string, Value>`，典型规模 ~200 键 < 16KB |
| `DecisionTree` | 加载→下次热重载 | 同 Rule，`unique_ptr` 树形结构 |
| `TraceStep[]` + `EvidenceItem[]` | 单次 Reason 调用 | `vector` 移动给 `ReasoningResult` |
| `Evidence` | 随 ReasoningResult | 直接复用 FactBase 的 AllEntries，零额外拷贝 |

---

## 15. 未来扩展

| 扩展点 | 方向 | 触发条件 |
|--------|------|---------|
| 时序窗口函数 | `COUNT(defect IN last_7_days)` | 需要生产数据趋势分析 |
| 多 Surface 关联规则 | 跨工位引用 FactBase 字段 | 多工位 AOI 线 |
| 规则版本管理 | `Rule` 带 `SemVer` + 变更 diff | 多产线不同规则版本共存 |
| 独立 Confidence 公式 | leaf 可选 `confidence_formula: {...}` | 需要独立建模置信度的 AOI 场景 |
| 决策树自动学习 | 从历史 ReasoningResult 学习树结构 | 积累足够标注数据 |
| 规则市场 | 预置行业规则模板 | 跨行业部署（PCB/Glass/Steel） |
| 分布式 Evidence 存储 | Evidence → InfluxDB/TimescaleDB | M6 Scheduler 跨进程场景 |

---

## 16. 最佳实践

1. **规则粒度**：一条规则 = 一个独立判定条件 + 一个明确 action。不把多种缺陷类型的判定塞进单条规则
2. **双向覆盖声明**：`A overrides B` 时 B 应同时声明 `overridden_by: [A]`，`DetectOverlaps()` 告警单向声明
3. **决策树 default 必填**：每个分支节点必须有 default，拒绝不完整树
4. **图路径先验证后引用**：加载时 `KnowledgeGraph::ValidatePath()` 校验边关系存在性
5. **FactBase 优先物化**：所有外部 I/O 在 `FactBuilder::Build()` 一次性完成，求值阶段不做 I/O
6. **Trace 即文档**：规则 condition 字符串 + 算子级 Trace 可完整还原判定推理过程

## 17. 反模式

1. **规则中嵌入业务 ID**：`material.supplier_id == 5` → 应使用图路径 `material->supplier->batch.reject_rate > 0.05`
2. **过度嵌套的决策树**：> 6 层 → 拆分为多个 rule_set + Reasoner 组合
3. **循环覆盖声明**：`A→B→C→A` → 加载时被拒绝
4. **Severity 硬编码**：`base_severity: 0.9` 只应作为 fallback，应通过决策树加权评分动态计算
5. **表达式中做副作用**：`Evaluate()` 是纯函数，外部数据通过 FactBase 注入
6. **Default 分支缺失**：`Reasoner_InvalidTree` 拒绝加载

---

## 18. 验证点

**主验证点**：一条 YAML 规则触发，Reasoner 输出带算子级 Trace 的结论。

具体场景：汽车座椅表面检测到划痕（面积 12.3mm²，位于 seat_center），判断 NG。

验证链：
1. M3 DetectionResult → FactBase（defect.type="scratch", defect.area_mm2=12.3, defect.position="seat_center"）
2. FactBuilder 物化图路径 `material->supplier->batch.reject_rate` = 0.032
3. RuleEngine::EvaluateAll → scratch_major 匹配（area > 10 AND position IN [...]）→ scratch_minor 被 overridden
4. DefaultReasoner::Reason → 决策树: scratch 分支 → area > 10 分支 → Leaf NG
5. ReasoningResult:
   - verdict = "NG"
   - severity > 0.7（sigmoid(0.5*0.92 + 0.3*0.85 + 0.2*0.032 - 0.5) ≈ sigmoid(0.2214) ≈ 0.555...待实际数据校准）
   - trace[] 包含：Expression 级别的字段求值 + RuleMatch 级别的规则命中 + TreeBranch 级别的树分支 + Scoring 级别的公式计算
   - evidence[] 包含：defect 全部字段 + 图路径物化的 SQL + 向量检索的 TopK

---

## 19. 契约增量

M5 新增以下概念与接口到 `glossary-and-contracts.md`：

### 19.1 概念归属表新增

| 概念名称 | 归属批次 | 定义所在文档 | 说明 |
|---------|---------|-------------|------|
| `FactBase` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | 扁平键值表，求值前批量物化外部数据（Detection + Knowledge + Retrieval），每条记录带 `FactSource` 溯源元数据 |
| `FactSource` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | 事实来源元数据（Direct/GraphPath/VectorSearch/Computed/Default），含 SQL 语句/FAISS 参数/耗时 |
| `Rule` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | YAML 定义的工业判定规则（name + condition AST + action + overrides/overridden_by + rule_set） |
| `RuleEngine` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | 规则求值引擎，批量加载 YAML、并行求值、冲突消解、热重载 |
| `DecisionTree` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | YAML 配置的分层决策树（BranchNode + LeafNode + ScoreFormula），叶子用加权 sigmoid 评分 |
| `ReasoningResult` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | 推理产出：verdict（OK/NG/WARN/UNCERTAIN）+ severity + recommendation + confidence + trace[] + evidence[] |
| `TraceStep` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | 算子级溯源步骤（Expression/Rule/TreeBranch/Scoring 四级），含描述、源码位置、父节点引用。定义于 `sai::rule` 命名空间，5.2 Reasoner 复用 |
| `EvidenceItem` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | 全链路证据项：FactBase 键值对 + FactSource 溯源 + TraceStep 关联 |

### 19.2 接口签名表新增

| 接口名称 | 归属批次 | 定义所在文档 | 签名摘要 |
|---------|---------|-------------|---------|
| `IExpression` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | `class IExpression : public Object { virtual auto Evaluate(FactBase&) const -> Result<Value> = 0; virtual auto CollectFieldRefs() const -> vector<string> = 0; virtual auto SourceText() const -> string_view = 0; }` |
| `FactBase` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | `class FactBase { auto Set(string_view, Value, FactSource) -> void; auto Get(string_view) const -> optional<Value>; auto Has(string_view) const -> bool; auto SourceOf(string_view) const -> const FactSource&; auto AllEntries() const -> vector<pair<string,Value>>; auto AllSources() const -> vector<pair<string,FactSource>>; }` |
| `FactBuilder` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | `class FactBuilder { explicit FactBuilder(shared_ptr<KnowledgeGraph>, shared_ptr<VectorPath>); auto Build(string_view surface_id, const DetectionResult&, const vector<string>& graph_paths) -> Result<FactBase>; }` |
| `RuleEngine` | 5.1 | design/milestone-05-inference-decision/5.1-rule-engine.md | `class RuleEngine { auto LoadFromYAML(path) -> Result<void>; auto EvaluateAll(FactBase&) -> Result<vector<ResolvedRule>>; auto ResolveConflicts(const vector<ResolvedRule>&) -> vector<ResolvedRule>; auto EnableHotReload(path, stop_token) -> Result<void>; auto DetectOverlaps() const -> vector<OverlapWarning>; }` |
| `DecisionTree` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | `class DecisionTree { static auto LoadFromYAML(path) -> Result<unique_ptr<DecisionTree>>; auto Root() const -> const IDecisionNode&; }` |
| `IReasoner` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | `class IReasoner : public IService { virtual auto Reason(const FactBase&, const vector<ResolvedRule>&) -> Result<ReasoningResult> = 0; }` |
| `DefaultReasoner` | 5.2 | design/milestone-05-inference-decision/5.2-reasoner.md | `class DefaultReasoner final : public IReasoner { explicit DefaultReasoner(unique_ptr<DecisionTree>); auto Reason(const FactBase&, const vector<ResolvedRule>&) -> Result<ReasoningResult> override; }` |
