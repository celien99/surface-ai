# Surface AI Framework —— 里程碑 4 知识与检索 设计文档

> Status: Draft
> Date: 2026-07-13
> Based on: `docs/superpowers/specs/2026-07-07-surface-ai-framework-phased-plan-design.md` §4 里程碑 4
> Depends on: 里程碑 1（1.1-1.6 全部冻结接口）、里程碑 2（2.1-2.3 全部冻结接口）、里程碑 3（3.1-3.3 全部冻结接口）

---

## 1. 背景与范围

里程碑 4 覆盖"知识与检索"：在 M3 的 Embedding/Detection 产出之上，构建结构化知识管理系统（SQLite 属性图 + Evolution/Snapshot）和混合检索引擎（FAISS 向量路 + SQL metadata 路 + 可插拔分数融合）。

### 1.1 架构总览

两层分离，每层定义于独立批次：

```
4.1 Knowledge（结构化知识管理）
    KnowledgeRecord（类型化字段容器）
    KnowledgeGraph（SQLite 属性图：节点 + 有向边 + 属性）
    KnowledgeEvolution（变更日志 + 版本链）
    KnowledgeSnapshot（时间点快照：SAVEPOINT + 时间戳表）
    KnowledgeStore（统一读写/查询门面）

4.2 Retrieval（混合检索）
    VectorPath（FAISS TopK/Range/Hybrid，复用 M3 FeatureBank）
    MetadataPath（SQLite WHERE + JOIN 结构化过滤 + 打分）
    IScoreFusion（可插拔融合策略接口）
    WeightedFusion / RRFFusion（内置实现）
    RetrievalResult（排序结果 + 分数溯源）
```

### 1.2 批次划分与执行顺序

```
里程碑 4：知识与检索
├── 4.1 Knowledge（KnowledgeRecord → KnowledgeGraph → KnowledgeStore）
└── 4.2 Retrieval（VectorPath + MetadataPath → IScoreFusion → HybridRetriever）
```

执行顺序：**4.1 设计 → 4.1 review → 4.1 代码 → 4.2 设计 → 4.2 review → 4.2 代码 → 最终整体回顾**

### 1.3 项目锚点

延续 M1/M2/M3 的汽车座椅 AOI 场景：

- **Material**：座椅面料类型（真皮/PVC/织物），每类关联典型缺陷模式和检测参数
- **Supplier**：面料供应商，关联批次质量历史、退货率
- **Batch**：生产批次，关联检测结果统计、不合格率趋势
- **Knowledge Graph**：构建 Material→supplied_by→Supplier→produces→Batch→has_defect→DefectType 的因果追溯链
- **CLIP 全局嵌入**（M3 产出）作为知识记录的向量表示，支持"查找类似缺陷模式的历史案例"
- **Hybrid Retrieval**：向量相似度（CLIP 嵌入）+ metadata 过滤（"供应商 X 的批次 Y"）→ 融合排序

### 1.4 明确排除项

- 训练/微调知识图谱嵌入模型（使用 M3 的预训练 CLIP 嵌入，不做 KG embedding 训练）
- 外部知识库集成（ERP/MES/Wiki——M5/M6 处理）
- 自然语言查询接口（M7 GUI 处理）
- 知识推理/规则引擎（M5 Rule 处理）
- 分布式知识存储（单进程 SQLite，不做多节点同步）

---

## 2. 对里程碑 1/2/3 的依赖

| M4 批次 | 依赖接口 | 用途 |
|---------|---------|------|
| 4.1 | `Result<T>`、`Object`（1.1） | 错误处理 + 接口基类 |
| 4.1 | `ConfigStore`（1.6） | SQLite 路径 + schema 版本配置 |
| 4.1 | `Logger`（1.6） | 变更日志、查询耗时记录 |
| 4.1 | `IService`、`Context`（1.2） | KnowledgeStore 作为 Service 注册到 DI 容器 |
| 4.2 | `FeatureBank`（3.3） | 复用 FAISS IndexFlatL2 封装，扩展到 Range/Hybrid 搜索 |
| 4.2 | `Embedding`（3.2） | 向量存储与比较（double 精度） |
| 4.2 | `DetectionResult`（3.3） | 检测结果关联到 KnowledgeRecord 作为 evidence |
| 4.2 | `InspectionResult`、`DefectRecord`（2.3） | 检测产出写入知识库 |
| 4.2 | `WorkerPool`（1.4） | 双路检索并行执行 |

---

## 3. 批次 4.1 Knowledge

### 3.1 Purpose

定义结构化知识管理子系统：将 Material/Supplier/Batch/DefectType 等工业 AOI 领域的实体建模为 SQLite 属性图（节点 + 有向边 + 键值属性），支持变更追溯（Evolution）和时间点快照（Snapshot），通过 `KnowledgeStore` 提供统一读写门面。

### 3.2 Responsibilities

- `KnowledgeRecord`：类型化字段容器——一个 record 是一个实体的属性集合（key → typed value），不绑定到特定 schema
- `KnowledgeGraph`：属性图存储引擎——节点表 + 边表 + 属性索引，支持按类型/属性/关系遍历
- `KnowledgeEvolution`：变更日志——每次 Create/Update/Delete 记录 (entity_id, version, timestamp, operation, snapshot_blob)
- `KnowledgeSnapshot`：时间点快照——通过 SQLite SAVEPOINT + 时间戳标记实现轻量级快照，支持按时间点还原查询
- `KnowledgeStore`：统一门面——封装上述三者，对外暴露 Insert/Update/Delete/Query/GetSnapshot/ListEvolution 等操作

### 3.3 Design

**为什么用 SQLite 属性图而非专用图数据库**：技术栈已锁定 SQLite（§2），且 AOI 场景的知识图谱规模是百~万节点级，不是百万~亿级。属性图在 SQLite 上用两张表（nodes + edges）+ JSON 属性列即可高效实现，不引入 Neo4j/ArangoDB 的运维复杂度。Gremlin/Cypher 遍历通过 C++ 递归函数实现，不引入图查询语言。

**为什么 KnowledgeRecord 不用固定 schema**：Material/Supplier/Batch 的字段集合随行业变化（汽车座椅 vs PCB vs 玻璃），固定 schema 迫使每次扩展改表结构。`KnowledgeRecord` 用 `map<string, FieldValue>`（variant<int64_t, double, string, vector<uint8_t>>）存储属性，SQLite 侧存为 JSON 列 + 虚拟列索引常用字段（material_code, batch_id 等）。灵活性与查询性能的折中：虚拟列上建索引保证常用字段过滤性能，JSON 列兜底覆盖扩展字段。

**为什么 Snapshot 用 SAVEPOINT + 时间戳表而非 WAL 文件拷贝**：SQLite 的 SAVEPOINT 是嵌套事务机制——创建快照时不阻塞读写，回滚到快照是 O(1) 操作（回滚日志中该 savepoint 之后的所有页面变更）。配合 `snapshots` 元数据表记录 (snapshot_id, timestamp, label, savepoint_name)，实现"命名快照 + 时间点查询"。WAL 文件拷贝方案的优势是独立归档文件，但需要暂停写入或承受 checkpoint 竞态——SAVEPOINT 方案零停机。

**为什么 Evolution 与 Snapshot 互补而非合并**：Snapshot 是时刻状态（"2026-07-13 10:00 的知识库长什么样"），Evolution 是变更流（"Batch B-2026-0001 的 status 从 pending→inspected→approved，每一步谁改的"）。一个用于回溯审计，一个用于追溯因果。存储上，Evolution 的 `snapshot_blob` 只存变更实体的 before-image（单行 JSON），不是全库快照。

**为什么 KnowledgeStore 是 Service 而非静态函数集**：需要 `Context::Resolve<KnowledgeStore>()` 注入到 M5 Reasoner 和 M6 Pipeline 中，作为 DI 容器管理的单例服务。生命周期由 `Context` 统一管理（OnInitialize 打开 SQLite 连接并 migrate schema，OnStop 关闭连接并做最终 checkpoint）。

### 3.4 Interfaces

```cpp
// knowledge_record.h
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace sai::knowledge {

// 类型化字段值：覆盖 AOI 知识记录的常见数据类型
using FieldValue = std::variant<
    std::int64_t,       // 整数（batch 数量、缺陷计数）
    double,             // 浮点（阈值、统计值）
    std::string,        // 字符串（名称、编码、描述）
    std::vector<std::uint8_t>  // 二进制（嵌入向量、缩略图）
>;

// 知识记录：一组具名字段的类型化容器
struct KnowledgeRecord {
    std::map<std::string, FieldValue> fields;
};

}  // namespace sai::knowledge
```

```cpp
// knowledge_graph.h
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/knowledge/knowledge_record.h>

namespace sai::knowledge {

// 节点类型标识
using NodeId = std::int64_t;
using EdgeId = std::int64_t;

// 属性图节点
struct KnowledgeNode {
    NodeId id = 0;
    std::string type;         // "Material" | "Supplier" | "Batch" | "DefectType" | custom
    KnowledgeRecord properties;
};

// 属性图有向边
struct KnowledgeEdge {
    EdgeId id = 0;
    NodeId source_id;
    NodeId target_id;
    std::string relationship; // "supplied_by" | "produces" | "has_defect" | "belongs_to" | custom
    KnowledgeRecord properties;
};

// 图遍历结果：从某节点出发，沿指定关系类型走到达的节点
struct GraphPath {
    NodeId source;
    std::string relationship;
    std::vector<KnowledgeNode> targets;
};

class KnowledgeGraph final {
public:
    // 构造时不打开连接——由 KnowledgeStore 注入 SQLite 句柄
    explicit KnowledgeGraph(struct sqlite3* db) noexcept;

    // 节点 CRUD
    [[nodiscard]] auto InsertNode(std::string type, KnowledgeRecord properties) noexcept
        -> Result<NodeId>;
    [[nodiscard]] auto UpdateNode(NodeId id, KnowledgeRecord properties) noexcept -> Result<void>;
    [[nodiscard]] auto DeleteNode(NodeId id) noexcept -> Result<void>;
    [[nodiscard]] auto GetNode(NodeId id) const noexcept -> Result<KnowledgeNode>;
    [[nodiscard]] auto FindNodesByType(std::string_view type) const noexcept
        -> Result<std::vector<KnowledgeNode>>;

    // 边 CRUD
    [[nodiscard]] auto InsertEdge(NodeId source, NodeId target,
                                   std::string relationship,
                                   KnowledgeRecord properties) noexcept -> Result<EdgeId>;
    [[nodiscard]] auto DeleteEdge(EdgeId id) noexcept -> Result<void>;
    [[nodiscard]] auto GetEdge(EdgeId id) const noexcept -> Result<KnowledgeEdge>;

    // 图遍历
    [[nodiscard]] auto Traverse(NodeId from, std::string_view relationship) const noexcept
        -> Result<std::vector<GraphPath>>;
    // 反向遍历：查找指向该节点的所有边
    [[nodiscard]] auto ReverseTraverse(NodeId to, std::string_view relationship) const noexcept
        -> Result<std::vector<GraphPath>>;

    // 统计
    [[nodiscard]] auto NodeCount() const noexcept -> std::size_t;
    [[nodiscard]] auto EdgeCount() const noexcept -> std::size_t;

    KnowledgeGraph(const KnowledgeGraph&) = delete;
    auto operator=(const KnowledgeGraph&) -> KnowledgeGraph& = delete;
    KnowledgeGraph(KnowledgeGraph&&) noexcept = default;
    auto operator=(KnowledgeGraph&&) noexcept -> KnowledgeGraph& = default;

private:
    struct sqlite3* db_;  // 不拥有所有权
};

}  // namespace sai::knowledge
```

```cpp
// knowledge_evolution.h
#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/knowledge/knowledge_record.h>

namespace sai::knowledge {

// 变更操作类型
enum class EvolutionOp : std::uint8_t {
    Insert,
    Update,
    Delete,
};

// 单条变更记录
struct EvolutionEntry {
    std::int64_t entry_id = 0;
    std::string entity_type;    // "Node" | "Edge"
    std::int64_t entity_id;
    EvolutionOp operation;
    std::int64_t version;       // 该实体的单调递增版本号
    std::chrono::system_clock::time_point timestamp;
    std::string changed_by;     // 变更来源（"importer" | "detector" | "manual"）
    KnowledgeRecord before_image;  // 变更前状态（Insert 时为空）
};

class KnowledgeEvolution final {
public:
    explicit KnowledgeEvolution(struct sqlite3* db) noexcept;

    // 追加变更日志条目（由 KnowledgeStore 在每次写操作后调用）
    [[nodiscard]] auto Append(std::string entity_type, std::int64_t entity_id,
                               EvolutionOp op, KnowledgeRecord before_image,
                               std::string changed_by) noexcept -> Result<void>;

    // 查询某实体的完整变更历史（按版本号升序）
    [[nodiscard]] auto GetHistory(std::string_view entity_type,
                                    std::int64_t entity_id) const noexcept
        -> Result<std::vector<EvolutionEntry>>;

    // 查询某时间范围内的所有变更
    [[nodiscard]] auto GetChangesSince(
        std::chrono::system_clock::time_point since) const noexcept
        -> Result<std::vector<EvolutionEntry>>;

    KnowledgeEvolution(const KnowledgeEvolution&) = delete;
    auto operator=(const KnowledgeEvolution&) -> KnowledgeEvolution& = delete;
    KnowledgeEvolution(KnowledgeEvolution&&) noexcept = default;
    auto operator=(KnowledgeEvolution&&) noexcept -> KnowledgeEvolution& = default;

private:
    struct sqlite3* db_;
};

}  // namespace sai::knowledge
```

```cpp
// knowledge_snapshot.h
#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <sai/core/error.h>

namespace sai::knowledge {

// 快照元数据
struct SnapshotInfo {
    std::int64_t snapshot_id = 0;
    std::string label;          // 用户可读标签（"MES 同步前快照"）
    std::chrono::system_clock::time_point created_at;
    std::int64_t node_count = 0;
    std::int64_t edge_count = 0;
};

class KnowledgeSnapshot final {
public:
    explicit KnowledgeSnapshot(struct sqlite3* db) noexcept;

    // 创建命名快照（内部使用 SAVEPOINT）
    [[nodiscard]] auto Create(std::string label) noexcept -> Result<std::int64_t>;

    // 列出所有快照
    [[nodiscard]] auto List() const noexcept -> Result<std::vector<SnapshotInfo>>;

    // 回滚到指定快照（内部使用 ROLLBACK TO SAVEPOINT）
    // 注意：会丢弃该快照之后的所有变更，包括 Evolution 日志
    [[nodiscard]] auto Restore(std::int64_t snapshot_id) noexcept -> Result<void>;

    // 删除快照（RELEASE SAVEPOINT）
    [[nodiscard]] auto Delete(std::int64_t snapshot_id) noexcept -> Result<void>;

    KnowledgeSnapshot(const KnowledgeSnapshot&) = delete;
    auto operator=(const KnowledgeSnapshot&) -> KnowledgeSnapshot& = delete;
    KnowledgeSnapshot(KnowledgeSnapshot&&) noexcept = default;
    auto operator=(KnowledgeSnapshot&&) noexcept -> KnowledgeSnapshot& = default;

private:
    struct sqlite3* db_;
};

}  // namespace sai::knowledge
```

```cpp
// knowledge_store.h
#pragma once
#include <filesystem>
#include <memory>
#include <string>

#include <sai/core/error.h>
#include <sai/core/object.h>
#include <sai/knowledge/knowledge_graph.h>
#include <sai/knowledge/knowledge_evolution.h>
#include <sai/knowledge/knowledge_snapshot.h>
#include <sai/knowledge/knowledge_record.h>

struct sqlite3;  // 前向声明

namespace sai::knowledge {

// KnowledgeStore：知识子系统统一门面。
// 作为 IService 注册到 Context，生命周期由 DI 容器管理。
//
// 内部持有 sqlite3* 连接——Graph/Evolution/Snapshot 三个子组件共享同一连接，
// 各自仅持有裸指针（不拥有所有权），保证事务一致性。
class KnowledgeStore final : public Object {
public:
    struct Config {
        std::filesystem::path db_path;
        bool enable_evolution = true;   // 是否开启变更日志（生产环境建议开启）
    };

    [[nodiscard]] static auto Create(const Config& cfg) noexcept
        -> Result<std::unique_ptr<KnowledgeStore>>;

    // 图操作（委托给 KnowledgeGraph）
    [[nodiscard]] auto InsertNode(std::string type, KnowledgeRecord properties,
                                    std::string changed_by = "system") noexcept -> Result<NodeId>;
    [[nodiscard]] auto UpdateNode(NodeId id, KnowledgeRecord properties,
                                    std::string changed_by = "system") noexcept -> Result<void>;
    [[nodiscard]] auto DeleteNode(NodeId id,
                                    std::string changed_by = "system") noexcept -> Result<void>;
    [[nodiscard]] auto GetNode(NodeId id) const noexcept -> Result<KnowledgeNode>;
    [[nodiscard]] auto FindNodesByType(std::string_view type) const noexcept
        -> Result<std::vector<KnowledgeNode>>;
    [[nodiscard]] auto InsertEdge(NodeId source, NodeId target,
                                    std::string relationship, KnowledgeRecord properties,
                                    std::string changed_by = "system") noexcept -> Result<EdgeId>;
    [[nodiscard]] auto GetEdge(EdgeId id) const noexcept -> Result<KnowledgeEdge>;
    [[nodiscard]] auto Traverse(NodeId from, std::string_view relationship) const noexcept
        -> Result<std::vector<GraphPath>>;

    // 快照操作（委托给 KnowledgeSnapshot）
    [[nodiscard]] auto CreateSnapshot(std::string label) noexcept -> Result<std::int64_t>;
    [[nodiscard]] auto ListSnapshots() const noexcept -> Result<std::vector<SnapshotInfo>>;
    [[nodiscard]] auto RestoreSnapshot(std::int64_t snapshot_id) noexcept -> Result<void>;

    // 演化查询（委托给 KnowledgeEvolution）
    [[nodiscard]] auto GetEntityHistory(std::string_view entity_type,
                                          std::int64_t entity_id) const noexcept
        -> Result<std::vector<EvolutionEntry>>;
    [[nodiscard]] auto GetChangesSince(
        std::chrono::system_clock::time_point since) const noexcept
        -> Result<std::vector<EvolutionEntry>>;

    // 原始 SQLite 句柄（供 Retrieval 层做 metadata 过滤）
    [[nodiscard]] auto DbHandle() const noexcept -> sqlite3* { return db_.get(); }

    // 获取子组件引用（供测试和内部使用）
    [[nodiscard]] auto Graph() noexcept -> KnowledgeGraph& { return graph_; }
    [[nodiscard]] auto Graph() const noexcept -> const KnowledgeGraph& { return graph_; }
    [[nodiscard]] auto Evolution() noexcept -> KnowledgeEvolution& { return evolution_; }
    [[nodiscard]] auto Snapshot() noexcept -> KnowledgeSnapshot& { return snapshot_; }

    ~KnowledgeStore() override;

    KnowledgeStore(const KnowledgeStore&) = delete;
    auto operator=(const KnowledgeStore&) -> KnowledgeStore& = delete;
    KnowledgeStore(KnowledgeStore&&) = delete;
    auto operator=(KnowledgeStore&&) -> KnowledgeStore& = delete;

private:
    KnowledgeStore() noexcept = default;

    struct Sqlite3Deleter {
        auto operator()(sqlite3* db) const noexcept -> void;
    };
    std::unique_ptr<sqlite3, Sqlite3Deleter> db_;
    KnowledgeGraph graph_{nullptr};
    KnowledgeEvolution evolution_{nullptr};
    KnowledgeSnapshot snapshot_{nullptr};
    Config config_;
};

}  // namespace sai::knowledge
```

### 3.5 Workflow

**Schema 迁移**：`KnowledgeStore::Create()` 打开 SQLite 连接后立即执行 schema 迁移——检查 `schema_version` 表，若不存在则创建初始表结构（nodes / edges / evolution_log / snapshots），若版本落后则执行 migrate SQL。迁移脚本硬编码在代码中，不依赖外部 .sql 文件。

**节点插入流程**：
1. `KnowledgeStore::InsertNode(type, properties, changed_by)` 被调用
2. 属性序列化为 JSON → `INSERT INTO nodes (type, properties_json) VALUES (?, ?)`
3. 若 `enable_evolution` 为 true → `evolution_.Append("Node", new_id, EvolutionOp::Insert, {}, changed_by)`
4. 返回 `NodeId`

**快照创建与恢复流程**：
1. `CreateSnapshot(label)` → 执行 `SAVEPOINT sp_<id>`，在 `snapshots` 表插入元数据行（关联 savepoint 名称）
2. `RestoreSnapshot(id)` → 查找对应 savepoint 名称 → 执行 `ROLLBACK TO SAVEPOINT sp_<id>` → 删除该 savepoint 之后的所有 evolution 条目（因为对应的写操作已被回滚）
3. `DeleteSnapshot(id)` → 执行 `RELEASE SAVEPOINT sp_<id>` → 删除元数据行

**图遍历流程**：
1. `Traverse(from_id, "produces")` → `SELECT target_id FROM edges WHERE source_id = ? AND relationship = ?`
2. 对每个 target_id → `SELECT * FROM nodes WHERE id = ?`
3. 递归沿路径追踪（深度由 `max_depth` 参数控制，默认 3 层）

### 3.6 Data Structure

**SQLite Schema**：

```sql
-- 版本管理
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 属性图节点
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_nodes_type ON nodes(type);

-- 属性图边
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    relationship TEXT NOT NULL,
    properties_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_relationship ON edges(relationship);

-- 演化日志
CREATE TABLE evolution_log (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,       -- "Node" | "Edge"
    entity_id INTEGER NOT NULL,
    operation TEXT NOT NULL,          -- "Insert" | "Update" | "Delete"
    version INTEGER NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    changed_by TEXT NOT NULL DEFAULT 'system',
    before_image_json TEXT            -- 变更前状态（Insert 为 NULL）
);
CREATE INDEX idx_evolution_entity ON evolution_log(entity_type, entity_id);
CREATE INDEX idx_evolution_timestamp ON evolution_log(timestamp);

-- 快照元数据
CREATE TABLE snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    savepoint_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    node_count INTEGER NOT NULL DEFAULT 0,
    edge_count INTEGER NOT NULL DEFAULT 0
);
```

### 3.7 Class Diagram

```
┌──────────────────────────────────────────────────┐
│              KnowledgeStore                       │
│  (统一门面，持有 sqlite3* 和三个子组件)              │
├──────────────────────────────────────────────────┤
│  + Create(Config) → unique_ptr<KnowledgeStore>    │
│  + InsertNode / UpdateNode / DeleteNode           │
│  + InsertEdge / GetEdge                           │
│  + Traverse / FindNodesByType                     │
│  + CreateSnapshot / RestoreSnapshot               │
│  + GetEntityHistory / GetChangesSince             │
│  + DbHandle() → sqlite3*                          │
├──────────────────────────────────────────────────┤
│  - db_: unique_ptr<sqlite3, Deleter>              │
│  - graph_: KnowledgeGraph                         │
│  - evolution_: KnowledgeEvolution                 │
│  - snapshot_: KnowledgeSnapshot                   │
└──────┬──────────────┬──────────────────┬─────────┘
       │              │                  │
       ▼              ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│KnowledgeGraph│ │KnowledgeEvol │ │KnowledgeSnapshot │
│ (持有 db_*)   │ │ (持有 db_*)   │ │ (持有 db_*)       │
├──────────────┤ ├──────────────┤ ├──────────────────┤
│+ InsertNode  │ │+ Append      │ │+ Create          │
│+ InsertEdge  │ │+ GetHistory  │ │+ Restore         │
│+ Traverse    │ │+ GetChanges  │ │+ List            │
│+ FindNodes   │ │  Since       │ │+ Delete          │
└──────────────┘ └──────────────┘ └──────────────────┘
       │              │                  │
       └──────────────┼──────────────────┘
                      │
              ┌───────▼────────┐
              │    sqlite3*    │
              │ (共享连接)      │
              └────────────────┘

┌──────────────────────┐
│   KnowledgeRecord    │
├──────────────────────┤
│ + fields: map<string,│
│     FieldValue>      │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│     FieldValue       │
│ variant<int64,double,│
│   string,vector<u8>> │
└──────────────────────┘
```

### 3.8 Sequence Diagram

```
调用方        KnowledgeStore    KnowledgeGraph   KnowledgeEvolution  SQLite
  │                │                  │                 │               │
  │ InsertNode()   │                  │                 │               │
  │───────────────>│                  │                 │               │
  │                │ InsertNode()     │                 │               │
  │                │─────────────────>│                 │               │
  │                │                  │ INSERT          │               │
  │                │                  │────────────────────────────────>│
  │                │                  │ node_id         │               │
  │                │                  │<────────────────────────────────│
  │                │                  │                 │               │
  │                │ Append("Node",   │                 │               │
  │                │   id, Insert,    │                 │               │
  │                │   {}, "importer")│                 │               │
  │                │─────────────────────────────────>│               │
  │                │                  │                 │ INSERT        │
  │                │                  │                 │──────────────>│
  │                │                  │                 │ ok            │
  │                │                  │                 │<──────────────│
  │                │                  │                 │               │
  │    node_id     │                  │                 │               │
  │<───────────────│                  │                 │               │
```

### 3.9 Thread Model

- `KnowledgeStore` 的所有写操作（Insert/Update/Delete）必须在**单线程**中调用——SQLite 默认的 serialized 模式可以处理多线程读，但写操作需要外部串行化以避免 `SQLITE_BUSY`。
- 设计决策：不在 `KnowledgeStore` 内部加互斥锁——由调用方（M6 Pipeline 的 Knowledge 阶段 `WorkerPool`）保证单写者。
- 读操作（GetNode/FindNodes/Traverse/GetHistory）可以多线程并发——SQLite 在 WAL 模式下支持一写多读。
- `KnowledgeSnapshot::Create()` 和 `Restore()` 是写操作，受同样单写者约束。

### 3.10 Performance

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| 节点插入 | < 1ms（含 JSON 序列化 + Evolution 日志） | 插入 1000 节点取 p99 |
| 图遍历（3 层深度，出度 ≤ 10） | < 5ms | 10 次随机起点的 3 层遍历取 p99 |
| 快照创建 | < 10ms | SAVEPOINT + 元数据行写入 |
| 快照恢复 | < 50ms | ROLLBACK TO SAVEPOINT（与变更量相关） |
| 数据库文件大小（空库） | < 100KB | 初始 schema 迁移后 |

### 3.11 Memory

- `KnowledgeStore` 持有 `sqlite3*` 的唯一所有权（通过 `unique_ptr<sqlite3, Deleter>`）
- `KnowledgeGraph` / `KnowledgeEvolution` / `KnowledgeSnapshot` 各持有裸 `sqlite3*`——不拥有所有权，生命周期绑定于 `KnowledgeStore`
- 查询结果（`vector<KnowledgeNode>` / `vector<EvolutionEntry>`）在栈上返回，由调用方管理生命周期
- SQLite 内部页面缓存通过 `PRAGMA cache_size` 控制（默认 2000 页 ≈ 8MB），在 `Create()` 中设置

### 3.12 Future Extension

- 全文检索：对 `properties_json` 中的文本字段建 FTS5 索引，支持模糊搜索和关键词查询
- 图嵌入：对 Knowledge Graph 做 Node2Vec/TransE 训练，产出图结构嵌入作为向量路的补充特征
- 增量同步：将 Evolution 日志流式推送到 MES/ERP，实现知识库的准实时外部同步
- 自定义遍历深度和剪枝条件：Traverse 当前固定递归逻辑，未来支持注入 `TraversePredicate` 回调

### 3.13 Best Practice

- 节点类型用常量定义（`constexpr auto kNodeTypeMaterial = "Material"`），避免字符串散落各处
- 每次写操作传入有意义的 `changed_by`（"importer"/"detector"/"manual"），否则 Evolution 追溯失去意义
- 生产环境 `enable_evolution = true`——关闭变更日志则无法回溯和审计
- 定期创建快照（如每批检测任务结束后），作为 rollback 检查点
- 在 `Context::OnStop` 中先调用 `KnowledgeStore` 的析构（关闭 SQLite 连接并做 WAL checkpoint），再停止其他服务

### 3.14 Anti Pattern

- 不要在热路径（每帧检测）上逐条写入 KnowledgeStore——批量 Insert 后用单次快照
- 不要在多线程中并发写入——SQLite 串行写约束，违者会触发 `SQLITE_BUSY` 重试风暴
- 不要用 `RestoreSnapshot` 替代普通数据删除——回滚会丢弃 Evolution 日志，破坏审计链
- 不要在 `properties_json` 中存储大 BLOB（> 1MB 的图像/嵌入）——大二进制数据存独立文件，knowledge record 只存文件路径引用

---

## 4. 批次 4.2 Retrieval

### 4.1 Purpose

定义混合检索引擎：复用 M3 FeatureBank 做向量检索，叠加 SQLite metadata 过滤，通过可插拔的 `IScoreFusion` 策略融合两路分数，返回带溯源信息的排序结果。

### 4.2 Responsibilities

- `VectorPath`：封装 FAISS 向量搜索——TopK（k-NN）、Range（距离阈值内全返回）、Hybrid（在指定 ID 子集内搜索）
- `MetadataPath`：封装 SQLite结构化查询——WHERE 条件 + JOIN + 自定义打分函数
- `IScoreFusion`：可插拔融合策略接口
- `WeightedFusion`：加权线性组合（α × normalized_vec_score + (1-α) × normalized_meta_score）
- `RRFFusion`：倒数排名融合（Σ 1/(k + rank_i)）
- `HybridRetriever`：编排双路检索 + 融合，返回 `RetrievalResult`
- `RetrievalResult`：排序结果集，每项含 knowledge_node_id、融合分数、向量分、metadata 分、溯源信息

### 4.3 Design

**为什么 VectorPath 复用 FeatureBank 而非重新封装 FAISS**：M3 的 `FeatureBank` 已经是 FAISS IndexFlatL2 的成熟封装，支持 Load/Search。M4 的 VectorPath 扩展其功能——在 FeatureBank 基础上增加 Range Search（`index_->range_search()`）和 IDSelector 预过滤（Hybrid 模式），而不重复 Load/Search 的实现。FeatureBank 的 `faiss::Index*` 通过友元或 protected 成员暴露给 VectorPath。

**为什么 MetadataPath 自己构造 SQL 而非用 ORM**：Metadata 过滤的本质是动态 WHERE 子句拼接——调用方传入 `vector<FilterCondition>`（field_name + op + value），MetadataPath 构造参数化 SQL（`WHERE material_code = ? AND supplier_id = ?`）。这个逻辑很简单（< 100 行），引入 ORM 库（sqlite_orm/hiberlite）反而增加依赖和编译时间。参数化查询防止 SQL 注入。

**为什么 IScoreFusion 是接口而非 std::function**：融合策略需要配置参数（α 权重、k 常数）和状态（归一化参数的在线估计——如向量分数的均值和方差），`std::function` 无法持有状态。接口类允许每个策略维护内部状态并在 `Configure(YAML::Node)` 中接收参数。

**为什么双路并行而非串行**：FAISS 搜索和 SQLite metadata 查询是独立的——没有数据依赖。通过 `WorkerPool` 并行提交两路任务，总延迟是 max(向量延迟, SQL 延迟) 而非 sum。双路并行要求 `KnowledgeStore` 的 WAL 模式支持并发读（见 3.9）。

**为什么 RetrievalResult 包含分数溯源**：每条结果需要知道"为什么排在这个位置"——向量相似度贡献了多少，metadata 匹配贡献了多少，最终融合分数怎么算的。这对 M5 Reasoner 的决策可解释性是硬需求（"这个 defect 被判定为 NG 因为向量路找到了 3 个高度相似的历史缺陷"）。

### 4.4 Interfaces

```cpp
// vector_path.h
#pragma once
#include <cstddef>
#include <vector>

#include <sai/core/error.h>
#include <sai/detection/feature_bank.h>

namespace sai::retrieval {

struct VectorResult {
    std::size_t index;       // 在 FeatureBank 中的样本索引
    float distance;          // L2 距离
};

// VectorPath：FAISS 向量搜索路径。
// 复用 M3 FeatureBank，扩展 Range/Hybrid 搜索模式。
class VectorPath final {
public:
    enum class Mode {
        TopK,    // 返回距离最小的 K 个结果
        Range,   // 返回距离 < threshold 的所有结果
        Hybrid,  // 在指定 ID 子集内做 TopK 搜索
    };

    struct Config {
        Mode mode = Mode::TopK;
        std::size_t k = 10;              // TopK 模式：返回数量
        float range_threshold = 1.0F;    // Range 模式：距离阈值
        std::vector<std::size_t> id_subset;  // Hybrid 模式：候选 ID 列表
    };

    explicit VectorPath(const sai::detection::FeatureBank& bank) noexcept;

    // 执行向量搜索
    // query: 查询向量（dim 必须与 FeatureBank 一致）
    [[nodiscard]] auto Search(const float* query, const Config& cfg) const noexcept
        -> Result<std::vector<VectorResult>>;

    [[nodiscard]] auto Dim() const noexcept -> std::size_t { return bank_.Dim(); }

private:
    const sai::detection::FeatureBank& bank_;
};

}  // namespace sai::retrieval
```

```cpp
// metadata_path.h
#pragma once
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include <sai/core/error.h>

struct sqlite3;

namespace sai::retrieval {

// 过滤操作符
enum class FilterOp : std::uint8_t {
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
    Like,        // SQL LIKE 模式匹配
    In,          // 值在给定列表中
};

// 单个过滤条件
struct FilterCondition {
    std::string field;           // 字段名（对应 properties_json 中的 key）
    FilterOp op;
    std::variant<std::int64_t, double, std::string, std::vector<std::int64_t>> value;
};

// Metadata 搜索结果
struct MetadataResult {
    std::int64_t node_id;
    float score;                 // metadata 匹配分数 [0, 1]
};

// MetadataPath：SQLite 结构化查询路径。
class MetadataPath final {
public:
    struct Config {
        std::vector<FilterCondition> filters;   // 过滤条件列表（AND 逻辑）
        std::vector<std::string> node_types;     // 限制节点类型（空 = 所有类型）
        std::size_t max_results = 100;
    };

    // 构造时注入 SQLite 句柄（来自 KnowledgeStore::DbHandle()）
    explicit MetadataPath(sqlite3* db) noexcept;

    // 执行结构化查询
    [[nodiscard]] auto Search(const Config& cfg) const noexcept
        -> Result<std::vector<MetadataResult>>;

private:
    sqlite3* db_;
};

}  // namespace sai::retrieval
```

```cpp
// score_fusion.h
#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include <sai/core/error.h>

namespace YAML { class Node; }

namespace sai::retrieval {

struct VectorResult;
struct MetadataResult;

// 单条检索结果的分数分解
struct ScoreBreakdown {
    float vector_score = 0.0F;       // 归一化后的向量相似度分
    float metadata_score = 0.0F;     // metadata 匹配分
    float fused_score = 0.0F;        // 融合后的最终分
    std::string fusion_strategy;     // 使用的融合策略名称
};

// IScoreFusion：可插拔的分数融合策略接口
class IScoreFusion {
public:
    virtual ~IScoreFusion() = default;

    // 从 YAML 节点读取策略参数
    [[nodiscard]] virtual auto Configure(const YAML::Node& params) noexcept -> Result<void> = 0;

    // 融合两路结果，返回排序后的 NodeId → ScoreBreakdown 映射
    // 调用方拥有返回值的所有权
    [[nodiscard]] virtual auto Fuse(
        const std::vector<VectorResult>& vec_results,
        const std::vector<std::int64_t>& vec_node_ids,   // vec_results[i] 对应的 node_id
        const std::vector<MetadataResult>& meta_results) const noexcept
        -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> = 0;

    [[nodiscard]] virtual auto Name() const noexcept -> std::string_view = 0;
};

// WeightedFusion：加权线性组合
// 参数：alpha (0~1，向量分权重)，默认 0.5
class WeightedFusion final : public IScoreFusion {
public:
    [[nodiscard]] auto Configure(const YAML::Node& params) noexcept -> Result<void> override;
    [[nodiscard]] auto Fuse(
        const std::vector<VectorResult>& vec_results,
        const std::vector<std::int64_t>& vec_node_ids,
        const std::vector<MetadataResult>& meta_results) const noexcept
        -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> override;
    [[nodiscard]] auto Name() const noexcept -> std::string_view override { return "WeightedFusion"; }
private:
    float alpha_ = 0.5F;
};

// RRFFusion：倒数排名融合
// 参数：k（平滑常数，默认 60）
class RRFFusion final : public IScoreFusion {
public:
    [[nodiscard]] auto Configure(const YAML::Node& params) noexcept -> Result<void> override;
    [[nodiscard]] auto Fuse(
        const std::vector<VectorResult>& vec_results,
        const std::vector<std::int64_t>& vec_node_ids,
        const std::vector<MetadataResult>& meta_results) const noexcept
        -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> override;
    [[nodiscard]] auto Name() const noexcept -> std::string_view override { return "RRFFusion"; }
private:
    float k_ = 60.0F;
};

}  // namespace sai::retrieval
```

```cpp
// hybrid_retriever.h
#pragma once
#include <memory>
#include <vector>

#include <sai/core/error.h>
#include <sai/retrieval/score_fusion.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>

namespace sai::retrieval {

// 检索结果：单条命中的完整信息
struct RetrievalItem {
    std::int64_t node_id;
    ScoreBreakdown scores;
    std::string node_type;       // 从 KnowledgeGraph 回填
};

// HybridRetriever：编排双路检索 + 分数融合
class HybridRetriever final {
public:
    struct Config {
        VectorPath::Config vector;
        MetadataPath::Config metadata;
    };

    // 构造时注入检索组件和融合策略（所有权归调用方）
    HybridRetriever(VectorPath& vec_path, MetadataPath& meta_path,
                     std::unique_ptr<IScoreFusion> fusion) noexcept;

    // 执行混合检索
    // query_vec: 查询向量（CLIP 嵌入等）
    // vec_to_node_ids: FeatureBank 样本索引 → KnowledgeGraph node_id 的映射
    [[nodiscard]] auto Retrieve(const float* query_vec,
                                  const Config& cfg,
                                  const std::vector<std::int64_t>& vec_to_node_ids) const noexcept
        -> Result<std::vector<RetrievalItem>>;

    // 运行时替换融合策略（配合 ConfigStore 热重载）
    auto SetFusion(std::unique_ptr<IScoreFusion> fusion) noexcept -> void;

    ~HybridRetriever();

    HybridRetriever(const HybridRetriever&) = delete;
    auto operator=(const HybridRetriever&) -> HybridRetriever& = delete;
    HybridRetriever(HybridRetriever&&) noexcept = default;
    auto operator=(HybridRetriever&&) noexcept -> HybridRetriever& = default;

private:
    VectorPath& vec_path_;
    MetadataPath& meta_path_;
    std::unique_ptr<IScoreFusion> fusion_;
};

}  // namespace sai::retrieval
```

### 4.5 Workflow

**混合检索完整流程**：

1. 调用方持有从 `KnowledgeStore` 获取的 `sqlite3*` 和已加载的 `FeatureBank`
2. 构造 `VectorPath(feature_bank)` + `MetadataPath(db)` + 选择的 `IScoreFusion` 实现
3. 构造 `HybridRetriever(vec_path, meta_path, fusion)`
4. 调用 `Retrieve(query_vec, config, vec_to_node_ids)`：
   - **并行阶段**：通过 `WorkerPool` 同时提交 VectorPath::Search 和 MetadataPath::Search
   - **映射阶段**：将向量搜索结果（FeatureBank 样本索引）通过 `vec_to_node_ids` 映射为 KnowledgeGraph node_id
   - **融合阶段**：调用 `fusion_->Fuse(vec_results, meta_results)` → 排序
   - **回填阶段**：对 top-N 结果回填 node_type（从 KnowledgeGraph 一次批量 SELECT）
5. 返回 `vector<RetrievalItem>`（按 fused_score 降序排列）

### 4.6 Data Structure

**向量索引 → 知识图谱的桥接表**：

```sql
-- 向量索引映射：FeatureBank 样本索引 ↔ KnowledgeGraph node_id
CREATE TABLE IF NOT EXISTS vec_to_knowledge (
    vec_index INTEGER PRIMARY KEY,   -- FeatureBank 中的样本位置
    node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE
);
CREATE INDEX idx_v2k_node ON vec_to_knowledge(node_id);
```

此表由 KnowledgeStore 在插入带嵌入向量的 KnowledgeRecord 时同步维护。调用方通过 `SELECT node_id FROM vec_to_knowledge ORDER BY vec_index` 获取 `vec_to_node_ids` 列表。

### 4.7 Class Diagram

```
┌──────────────────────────────────────────────────────┐
│                  HybridRetriever                      │
│  (双路编排 + 融合)                                     │
├──────────────────────────────────────────────────────┤
│  + Retrieve(query, cfg, vec_to_node_ids)              │
│    → vector<RetrievalItem>                            │
│  + SetFusion(IScoreFusion)                            │
├──────────────────────────────────────────────────────┤
│  - vec_path_: VectorPath&                             │
│  - meta_path_: MetadataPath&                          │
│  - fusion_: unique_ptr<IScoreFusion>                  │
└──────┬────────────────────┬──────────────────┬───────┘
       │                    │                  │
       ▼                    ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│  VectorPath   │  │ MetadataPath │  │   IScoreFusion    │
│ (复用         │  │ (SQLite 查询) │  │   (策略接口)       │
│  FeatureBank) │  │              │  ├──────────────────┤
├──────────────┤  ├──────────────┤  │+ Configure(YAML)  │
│+ Search(q,cfg)│  │+ Search(cfg) │  │+ Fuse(vec,meta)   │
│  → VectorRes  │  │  → MetaRes   │  │+ Name()           │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                   │
       ▼                 ▼           ┌───────┴────────┐
┌──────────────┐  ┌──────────────┐   │                │
│ FeatureBank  │  │   sqlite3*   │   ▼                ▼
│  (M3)        │  │ (Knowledge   │ WeightedFusion  RRFFusion
│              │  │  Store)      │ (α·vec+(1-α)·meta) (Σ1/(k+r))
└──────────────┘  └──────────────┘
```

### 4.8 Sequence Diagram

```
调用方      HybridRetriever   VectorPath   MetadataPath   IScoreFusion
  │               │               │             │              │
  │ Retrieve()    │               │             │              │
  │──────────────>│               │             │              │
  │               │               │             │              │
  │               │ Search(query) │             │              │
  │               │──────────────>│             │              │
  │               │               │             │              │
  │               │ Search(cfg)   │             │              │
  │               │────────────────────────────>│              │
  │               │               │             │              │
  │               │   vec_results │   meta_results             │
  │               │<──────────────│             │              │
  │               │<────────────────────────────│              │
  │               │               │             │              │
  │               │ 映射 vec_index → node_id    │              │
  │               │               │             │              │
  │               │ Fuse(vec_node_ids, meta_results)           │
  │               │──────────────────────────────────────────>│
  │               │               │             │              │
  │               │  sorted [(node_id, ScoreBreakdown)]        │
  │               │<──────────────────────────────────────────│
  │               │               │             │              │
  │               │ 回填 node_type (批量 SELECT)               │
  │               │               │             │              │
  │  vector<RetrievalItem>        │             │              │
  │<──────────────│               │             │              │
```

### 4.9 Thread Model

- `HybridRetriever::Retrieve()` 内部将 VectorPath 和 MetadataPath 的搜索提交到两个独立的 `WorkerPool` 线程并行执行
- `VectorPath::Search()` 读 FAISS Index——FAISS CPU Index 不支持并发写入但没有写操作（特征库是只读的），多线程并发 Search 是安全的
- `MetadataPath::Search()` 在 SQLite WAL 模式下执行 SELECT——可以与写操作并发（一写多读）
- `IScoreFusion::Fuse()` 在调用方线程同步执行（纯计算，无 I/O，< 1ms）
- 不使用协程 `Task<T>`——Retrieval 是同步 CPU 操作（FAISS 的 `index_->search()` 和 SQLite SELECT 都是阻塞的），M6 Pipeline 层面通过 `WorkerPool` + `Task<T>` 包裹整个 `Retrieve()` 调用实现异步

### 4.10 Performance

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| 向量搜索（TopK=10, dim=768, 10000 样本） | < 2ms | FAISS IndexFlatL2 暴力搜索 |
| Metadata 过滤（3 条件 AND, 10000 节点） | < 1ms | SQLite 索引查询 |
| 融合排序（100 候选 → top 10） | < 0.1ms | 纯内存计算 |
| 端到端 Retrieve() | < 5ms | 并行双路 + 融合（取 max(vec, meta) + fusion） |

### 4.11 Memory

- `VectorPath` 不持有 FAISS Index 所有权——`FeatureBank` 由调用方管理生命周期
- `MetadataPath` 不持有 `sqlite3*` 所有权——来自 `KnowledgeStore::DbHandle()`
- `HybridRetriever` 持有 `unique_ptr<IScoreFusion>` 所有权
- 检索中间结果（`vector<VectorResult>` + `vector<MetadataResult>`）在 `Retrieve()` 栈帧内分配，返回前释放
- `RetrievalResult` 按值返回（`vector<RetrievalItem>`），由调用方管理

### 4.12 Future Extension

- 近似最近邻（ANN）：将 FeatureBank 从 IndexFlatL2 升级为 IndexIVFFlat/IndexHNSW，支持百万级向量库的低延迟检索
- 多模态向量融合：同时使用 CLIP（全局语义）和 DINOv3（局部纹理）的嵌入做加权向量搜索
- 检索缓存：对高频查询（同 SKU + 同检测点位）的检索结果做 LRU 缓存，减少 FAISS 搜索次数
- 分布式检索：FAISS Index 分片到多进程/多节点（M7+ 考虑）

### 4.13 Best Practice

- `vec_to_node_ids` 映射表在 FeatureBank Load 后立即全量加载到内存（`vector<int64_t>`），避免检索热路径上查 SQLite
- `VectorPath::Config::id_subset` 用于 Hybrid 模式——先通过 MetadataPath 过滤出候选 node_id 集合，再映射为 vec_index 子集，传给 VectorPath 做受限 k-NN
- MetadataPath 的 `FilterCondition::field` 应与 KnowledgeRecord 的字段名一致——使用常量而非字符串字面量
- 融合策略通过 `ConfigStore` 的 YAML 配置选择：`fusion: { strategy: "WeightedFusion", params: { alpha: 0.7 } }`

### 4.14 Anti Pattern

- 不要在每帧检测中对全量向量库做 Range Search（大阈值可能返回数千结果）——除非特征库很小（< 1000 样本）
- 不要忘记 `vec_to_node_ids` 映射——FeatureBank 返回的是样本索引，不是 knowledge node_id，漏掉映射会导致检索到错误的实体
- 不要在 `IScoreFusion::Fuse()` 中做 I/O 或内存分配——融合是纯计算函数，应在 < 1ms 内完成
- 不要直接用字符串拼接构造 SQL——用参数化查询（`sqlite3_bind_*`），防止 SQL 注入

---

## 5. ErrorCode

M4 引入 `Knowledge_*` / `Retrieval_*` 错误码前缀（append-only，追加在 M3 的 `Detection_InvalidPatchGrid` 之后）：

| 错误码 | 含义 | 触发条件 |
|--------|------|---------|
| `Knowledge_DbOpenFailed` | SQLite 数据库打开失败 | 路径不可写/权限不足/文件损坏 |
| `Knowledge_SchemaMigrationFailed` | Schema 迁移失败 | 迁移 SQL 语法错误/版本跳跃 |
| `Knowledge_NodeNotFound` | 节点 ID 不存在 | GetNode/UpdateNode/DeleteNode 的 id 无效 |
| `Knowledge_EdgeNotFound` | 边 ID 不存在 | GetEdge/DeleteEdge 的 id 无效 |
| `Knowledge_InvalidRelationship` | 无效的关系类型 | InsertEdge 的关系字符串为空 |
| `Knowledge_SnapshotNotFound` | 快照 ID 不存在 | RestoreSnapshot/DeleteSnapshot 的 id 无效 |
| `Knowledge_SnapshotRestoreFailed` | 快照恢复失败 | SAVEPOINT 已被释放/事务冲突 |
| `Retrieval_DimensionMismatch` | 查询向量维度与索引不匹配 | VectorPath::Search 的 query dim ≠ FeatureBank dim |
| `Retrieval_EmptyIndex` | FeatureBank 为空 | 对空索引做搜索 |
| `Retrieval_FusionConfigInvalid` | 融合策略配置无效 | YAML 参数缺失/类型错误/超出范围 |

以上 10 个错误码按表内顺序追加。M1 的 `error.h` 附加规则同样适用——永不重排、永不碰其他批次的成员。

---

## 6. 构建与依赖

### 6.1 vcpkg 新增依赖

```json
{
  "name": "surface-ai",
  "version": "0.1.0",
  "dependencies": [
    "tl-expected",
    "gtest",
    "yaml-cpp",
    "spdlog",
    "nlohmann-json",
    "faiss",
    "sqlite3"
  ]
}
```

`sqlite3` 通过 vcpkg 的 `sqlite3` port 引入（封装 `sqlite3.h` C API，无 C++ 包装器依赖）。

### 6.2 CMake 新增目标

- `sai_knowledge`：静态库，link `sai::core sai::infra sqlite3`
- `sai_retrieval`：静态库，link `sai::core sai::knowledge sai::detection sai::infra faiss`
- 两个库均为全平台可移植（SQLite 和 FAISS CPU 在 macOS arm64 上均可编译和测试）

---

## 7. 验证点（里程碑 4）

SPEC 要求的验证点：**写入一条知识记录，检索命中并返回融合分数。**

具体场景：
1. 创建 KnowledgeStore（内存数据库 `:memory:`）
2. 插入 Material 节点：`{material_code: "LEATHER-001", name: "Nappa 真皮", supplier_code: "SUP-2026"}`，关联 CLIP 嵌入向量
3. 插入 Supplier 节点：`{supplier_code: "SUP-2026", name: "某供应商"}`，建立边 `Material -[supplied_by]-> Supplier`
4. 将 CLIP 嵌入向量写入 FeatureBank，建立 `vec_to_knowledge` 映射
5. 用新图像的 CLIP 嵌入调用 `HybridRetriever::Retrieve()`——向量路 + metadata 路（过滤 material_code = "LEATHER-001"）
6. 断言：返回结果非空，第一条 `node_id` 为插入的 Material 节点，`fused_score > 0`

---

## 8. 到里程碑 5 的接口预留

M5 Reasoner 将从 M4 获取：
- `KnowledgeStore` 的 `DbHandle()` ——直接查询知识图谱做因果追溯
- `KnowledgeStore::Traverse()` ——沿 `has_defect → caused_by → resolved_by` 链路追溯缺陷根因
- `HybridRetriever` ——"找到历史上类似缺陷"的检索能力，作为 Reasoner 的 evidence 来源
- `EvolutionEntry` ——缺陷判定变化的时间线

M5 不在 M4 的交付范围内，但这些接口已为 M5 预留。
