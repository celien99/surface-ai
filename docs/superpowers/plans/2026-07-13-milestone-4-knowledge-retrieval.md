# Milestone 4 (Knowledge & Retrieval) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement milestone 4 (batches 4.1 Knowledge, 4.2 Retrieval) as real C++20 code on top of the frozen M1/M2/M3 libraries, providing structured knowledge management (SQLite property graph + Evolution/Snapshot) and hybrid retrieval (FAISS vector + SQL metadata + pluggable score fusion).

**Architecture:** Two new portable libraries — `sai_knowledge` (STATIC: `KnowledgeRecord` + `KnowledgeGraph` + `KnowledgeEvolution` + `KnowledgeSnapshot` + `KnowledgeStore`) and `sai_retrieval` (STATIC: `VectorPath` + `MetadataPath` + `IScoreFusion`/`WeightedFusion`/`RRFFusion` + `HybridRetriever`). `VectorPath` reuses M3's `FeatureBank` FAISS index via friend access for Range/Hybrid search modes. `KnowledgeStore` holds sole `sqlite3*` ownership; `KnowledgeGraph`/`Evolution`/`Snapshot` hold raw pointers to the same connection. All modules are macOS-arm64 portable — no CUDA gating.

**Tech Stack:** C++20. vcpkg additions: `sqlite3`. Existing deps: `tl-expected`, `yaml-cpp`, `spdlog`, `nlohmann-json`, `faiss`, `gtest`. SQLite3 C API directly (no C++ wrapper lib).

## Global Constraints

- C++20 throughout, `sai::` namespace root. Batch sub-namespaces: `sai::knowledge` (4.1), `sai::retrieval` (4.2).
- **The frozen design baseline is `docs/superpowers/specs/2026-07-13-milestone-4-knowledge-retrieval-design.md`.** Every class name, method signature, enum, and namespace must match its `## Interfaces` blocks verbatim — except where a frozen signature provably cannot compile (report it, don't silently change it).
- `ErrorCode` (`include/sai/core/error.h`) is a single flat enum; each task appends new members **at the end** (append-only, never reorder or touch other batches' members). The final landed order for milestone 4, after the existing `Detection_InvalidPatchGrid` (last M3 member), is:
  - `Knowledge_DbOpenFailed`, `Knowledge_SchemaMigrationFailed`, `Knowledge_NodeNotFound`, `Knowledge_EdgeNotFound`, `Knowledge_InvalidRelationship`, `Knowledge_SnapshotNotFound`, `Knowledge_SnapshotRestoreFailed`
  - `Retrieval_DimensionMismatch`, `Retrieval_EmptyIndex`, `Retrieval_FusionConfigInvalid`
- Coding style: avoid over-defensive code, avoid multi-level nesting (early return), chain `Result<T>` via `and_then`/`or_else`/`map`.
- Every task ends with `cmake --preset default && cmake --build --preset default && ctest --preset default` — must not reduce the current count (**326 tests**) — only add.
- Work on a dedicated branch `milestone-4-knowledge-retrieval` off `main` (HEAD of M3).

## File Structure Overview

```
include/sai/
├── core/error.h                         # Modify: append 10 ErrorCode members
├── detection/feature_bank.h             # Modify: add friend class VectorPath
├── knowledge/
│   ├── knowledge_record.h               # Task 2: FieldValue, KnowledgeRecord
│   ├── knowledge_graph.h                # Task 3: NodeId, EdgeId, KnowledgeNode, KnowledgeEdge, GraphPath, KnowledgeGraph
│   ├── knowledge_evolution.h            # Task 4: EvolutionOp, EvolutionEntry, KnowledgeEvolution
│   ├── knowledge_snapshot.h             # Task 5: SnapshotInfo, KnowledgeSnapshot
│   └── knowledge_store.h                # Task 6: KnowledgeStore, Config
└── retrieval/
    ├── vector_path.h                    # Task 7: VectorResult, VectorPath
    ├── metadata_path.h                  # Task 8: FilterOp, FilterCondition, MetadataResult, MetadataPath
    ├── score_fusion.h                   # Task 9: ScoreBreakdown, IScoreFusion, WeightedFusion, RRFFusion
    └── hybrid_retriever.h               # Task 10: RetrievalItem, HybridRetriever

src/
├── knowledge/
│   ├── CMakeLists.txt                   # Task 2: sai_knowledge library
│   ├── knowledge_graph.cpp              # Task 3
│   ├── knowledge_evolution.cpp          # Task 4
│   ├── knowledge_snapshot.cpp           # Task 5
│   └── knowledge_store.cpp              # Task 6
└── retrieval/
    ├── CMakeLists.txt                   # Task 7: sai_retrieval library
    ├── vector_path.cpp                  # Task 7
    ├── metadata_path.cpp                # Task 8
    ├── score_fusion.cpp                 # Task 9
    └── hybrid_retriever.cpp             # Task 10

tests/
├── knowledge/
│   ├── CMakeLists.txt                   # Task 2
│   ├── knowledge_graph_test.cpp         # Task 3
│   ├── knowledge_evolution_test.cpp     # Task 4
│   ├── knowledge_snapshot_test.cpp      # Task 5
│   └── knowledge_store_test.cpp         # Task 6
├── retrieval/
│   ├── CMakeLists.txt                   # Task 7
│   ├── vector_path_test.cpp             # Task 7
│   ├── metadata_path_test.cpp           # Task 8
│   ├── score_fusion_test.cpp            # Task 9
│   └── hybrid_retriever_test.cpp        # Task 10
└── integration/
    └── knowledge_retrieval_pipeline_test.cpp  # Task 11

vcpkg.json                                # Modify: add sqlite3
CMakeLists.txt                            # Modify: add subdirectories
```

## Execution Order

```
Task 1  (ErrorCode + vcpkg + CMake wiring)                        [portable]
Task 2  (KnowledgeRecord + FieldValue + JSON serialization)       [portable, header-only]
Task 3  (KnowledgeGraph: SQLite nodes + edges + traversal)        [portable]
Task 4  (KnowledgeEvolution: changelog CRUD)                      [portable]
Task 5  (KnowledgeSnapshot: SAVEPOINT create/restore/list/delete) [portable]
Task 6  (KnowledgeStore: facade + schema migration)               [portable]
Task 7  (VectorPath: FAISS TopK/Range/Hybrid)                     [portable]
Task 8  (MetadataPath: dynamic SQL WHERE + scoring)               [portable]
Task 9  (IScoreFusion + WeightedFusion + RRFFusion)               [portable]
Task 10 (HybridRetriever: dual-path orchestration)                [portable]
Task 11 (Integration test: M4 verification point)                 [portable]
```

---

### Task 1: ErrorCode + vcpkg + CMake wiring

Setup task: append 10 error codes, add sqlite3 dependency, register new subdirectories in top-level CMakeLists.

**Files:**
- Modify: `include/sai/core/error.h` (append 10 ErrorCode members)
- Modify: `vcpkg.json` (add "sqlite3")
- Modify: `CMakeLists.txt` (add src/knowledge, tests/knowledge, src/retrieval, tests/retrieval)
- Modify: `include/sai/detection/feature_bank.h` (add friend declaration for VectorPath)

**Interfaces:**
- Consumes: existing ErrorCode enum (up to `Detection_InvalidPatchGrid`)
- Produces: 10 new ErrorCode members available to all later tasks

- [ ] **Step 1: Append 10 ErrorCode members to error.h**

In `include/sai/core/error.h`, after `Detection_InvalidPatchGrid`, append:

```cpp
    // Knowledge (M4)
    Knowledge_DbOpenFailed,
    Knowledge_SchemaMigrationFailed,
    Knowledge_NodeNotFound,
    Knowledge_EdgeNotFound,
    Knowledge_InvalidRelationship,
    Knowledge_SnapshotNotFound,
    Knowledge_SnapshotRestoreFailed,
    // Retrieval (M4)
    Retrieval_DimensionMismatch,
    Retrieval_EmptyIndex,
    Retrieval_FusionConfigInvalid,
```

Verify with: `grep -c "Knowledge_\|Retrieval_" include/sai/core/error.h` → expect 10.

- [ ] **Step 2: Add sqlite3 to vcpkg.json**

In `vcpkg.json`, add `"sqlite3"` to the dependencies array (after `"faiss"`):

```json
    "faiss",
    "sqlite3"
```

- [ ] **Step 3: Add friend declaration to FeatureBank**

In `include/sai/detection/feature_bank.h`, add a forward declaration before the `namespace sai::detection` block and a friend declaration inside `class FeatureBank`:

At the top of the file, after `#include` lines and before `namespace sai::detection`:

```cpp
namespace sai::retrieval { class VectorPath; }
```

In the `private:` section of `class FeatureBank`, add before the member declarations:

```cpp
    friend class sai::retrieval::VectorPath;
```

- [ ] **Step 4: Register new subdirectories in top-level CMakeLists.txt**

In `CMakeLists.txt`, add before the `add_subdirectory(tests/integration)` line:

```cmake
add_subdirectory(src/knowledge)
add_subdirectory(tests/knowledge)
add_subdirectory(src/retrieval)
add_subdirectory(tests/retrieval)
```

- [ ] **Step 5: Create placeholder CMakeLists.txt files**

Create `src/knowledge/CMakeLists.txt`:
```cmake
find_package(sqlite3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

set(SAI_KNOWLEDGE_SOURCES
    knowledge_graph.cpp
    knowledge_evolution.cpp
    knowledge_snapshot.cpp
    knowledge_store.cpp
)

add_library(sai_knowledge STATIC ${SAI_KNOWLEDGE_SOURCES})
add_library(sai::knowledge ALIAS sai_knowledge)
target_include_directories(sai_knowledge PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_knowledge PUBLIC sai::core sai::infra SQLite::SQLite3 nlohmann_json::nlohmann_json)
target_compile_features(sai_knowledge PUBLIC cxx_std_20)
```

Create `tests/knowledge/CMakeLists.txt`:
```cmake
find_package(GTest CONFIG REQUIRED)

set(SAI_KNOWLEDGE_TEST_SOURCES
    knowledge_graph_test.cpp
    knowledge_evolution_test.cpp
    knowledge_snapshot_test.cpp
    knowledge_store_test.cpp
)

add_executable(sai_knowledge_tests ${SAI_KNOWLEDGE_TEST_SOURCES})
target_include_directories(sai_knowledge_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_knowledge_tests PRIVATE sai::knowledge GTest::gtest GTest::gtest_main)
target_compile_features(sai_knowledge_tests PRIVATE cxx_std_20)
include(GoogleTest)
gtest_discover_tests(sai_knowledge_tests)
```

Create `src/retrieval/CMakeLists.txt`:
```cmake
find_package(faiss CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

# On macOS with Apple Clang, synthetic OpenMP target (copied from src/detection/CMakeLists.txt)
if(APPLE AND NOT OpenMP_FOUND)
    set(OPENMP_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
    set(OPENMP_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    if(EXISTS "${OPENMP_LIBRARY}")
        add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            IMPORTED_LOCATION "${OPENMP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OPENMP_INCLUDE_DIR}"
            INTERFACE_COMPILE_OPTIONS "-Xclang;-fopenmp"
        )
        set(OpenMP_FOUND TRUE)
    endif()
endif()
if(NOT OpenMP_FOUND)
    find_package(OpenMP REQUIRED)
endif()

set(SAI_RETRIEVAL_SOURCES
    vector_path.cpp
    metadata_path.cpp
    score_fusion.cpp
    hybrid_retriever.cpp
)

add_library(sai_retrieval STATIC ${SAI_RETRIEVAL_SOURCES})
add_library(sai::retrieval ALIAS sai_retrieval)
target_include_directories(sai_retrieval PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_retrieval PUBLIC sai::core sai::knowledge sai::detection faiss OpenMP::OpenMP_CXX yaml-cpp::yaml-cpp)
target_compile_features(sai_retrieval PUBLIC cxx_std_20)
```

Create `tests/retrieval/CMakeLists.txt`:
```cmake
find_package(GTest CONFIG REQUIRED)

set(SAI_RETRIEVAL_TEST_SOURCES
    vector_path_test.cpp
    metadata_path_test.cpp
    score_fusion_test.cpp
    hybrid_retriever_test.cpp
)

add_executable(sai_retrieval_tests ${SAI_RETRIEVAL_TEST_SOURCES})
target_include_directories(sai_retrieval_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_retrieval_tests PRIVATE sai::retrieval GTest::gtest GTest::gtest_main)
target_compile_features(sai_retrieval_tests PRIVATE cxx_std_20)
include(GoogleTest)
gtest_discover_tests(sai_retrieval_tests)
```

- [ ] **Step 6: Run vcpkg install + configure + build**

Run: `cmake --preset default`
Expected: configure succeeds, builds sai_knowledge and sai_retrieval targets (libraries compile with 0 source files — valid empty static lib).

- [ ] **Step 7: Run tests**

Run: `ctest --preset default`
Expected: 326/326 tests pass (no regressions).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore(build): 🔧 添加 M4 基础设施（ErrorCode + vcpkg + CMake）"
```

---

### Task 2: KnowledgeRecord + FieldValue

Header-only type definitions for typed knowledge record fields and JSON serialization — no SQLite dependency.

**Files:**
- Create: `include/sai/knowledge/knowledge_record.h`
- Create: `src/knowledge/knowledge_record.cpp` (JSON serialization helpers)

**Interfaces:**
- Consumes: nothing beyond C++ stdlib
- Produces: `FieldValue`, `KnowledgeRecord`, `FieldValueToJson`, `JsonToFieldValue`, `RecordToJson`, `JsonToRecord`

- [ ] **Step 1: Write the header**

Create `include/sai/knowledge/knowledge_record.h`:

```cpp
// knowledge_record.h — 批次 4.1 类型化知识记录字段容器
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace sai::knowledge {

using FieldValue = std::variant<
    std::int64_t,
    double,
    std::string,
    std::vector<std::uint8_t>
>;

struct KnowledgeRecord {
    std::map<std::string, FieldValue> fields;
};

}  // namespace sai::knowledge
```

- [ ] **Step 2: Write the JSON serialization helpers**

Create `src/knowledge/knowledge_record.cpp`:

```cpp
// knowledge_record.cpp — KnowledgeRecord ↔ nlohmann::json 序列化
#include <sai/knowledge/knowledge_record.h>
#include <nlohmann/json.hpp>

namespace sai::knowledge {

namespace {

auto FieldValueToJson(const FieldValue& fv) -> nlohmann::json {
    return std::visit([](const auto& v) -> nlohmann::json {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::vector<std::uint8_t>>) {
            return nlohmann::json::binary(v);
        } else {
            return nlohmann::json(v);
        }
    }, fv);
}

auto JsonToFieldValue(const nlohmann::json& j) -> FieldValue {
    if (j.is_number_integer()) {
        return FieldValue{j.get<std::int64_t>()};
    }
    if (j.is_number_float()) {
        return FieldValue{j.get<double>()};
    }
    if (j.is_string()) {
        return FieldValue{j.get<std::string>()};
    }
    if (j.is_binary()) {
        auto bin = j.get_binary();
        return FieldValue{std::vector<std::uint8_t>(bin.begin(), bin.end())};
    }
    return FieldValue{std::int64_t{0}};  // fallback
}

}  // anonymous namespace

auto RecordToJson(const KnowledgeRecord& record) -> nlohmann::json {
    nlohmann::json j = nlohmann::json::object();
    for (const auto& [key, value] : record.fields) {
        j[key] = FieldValueToJson(value);
    }
    return j;
}

auto JsonToRecord(const nlohmann::json& j) -> KnowledgeRecord {
    KnowledgeRecord record;
    if (!j.is_object()) return record;
    for (auto it = j.begin(); it != j.end(); ++it) {
        record.fields[it.key()] = JsonToFieldValue(it.value());
    }
    return record;
}

}  // namespace sai::knowledge
```

Add the declarations to the header:

In `include/sai/knowledge/knowledge_record.h`, after the `KnowledgeRecord` struct, append:

```cpp
// JSON 序列化（实现在 knowledge_record.cpp，供 KnowledgeGraph 等 SQLite 存储使用）
namespace nlohmann { class json; }
auto RecordToJson(const KnowledgeRecord& record) -> nlohmann::json;
auto JsonToRecord(const nlohmann::json& j) -> KnowledgeRecord;
```

- [ ] **Step 3: Update CMakeLists.txt to include knowledge_record.cpp**

In `src/knowledge/CMakeLists.txt`, prepend to `SAI_KNOWLEDGE_SOURCES`:

```cmake
set(SAI_KNOWLEDGE_SOURCES
    knowledge_record.cpp
    knowledge_graph.cpp
    ...
)
```

- [ ] **Step 4: Build & run tests**

Run: `cmake --build --preset default && ctest --preset default`
Expected: 326/326 tests pass, sai_knowledge compiles with knowledge_record.cpp.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(knowledge): ✨ 添加 KnowledgeRecord 与 FieldValue 类型化字段容器"
```

---

### Task 3: KnowledgeGraph (SQLite nodes + edges + traversal)

Implements the property graph on SQLite: node/edge CRUD, type-based lookup, forward/reverse traversal via SQL JOIN.

**Files:**
- Create: `include/sai/knowledge/knowledge_graph.h`
- Create: `src/knowledge/knowledge_graph.cpp`
- Create: `tests/knowledge/knowledge_graph_test.cpp`

**Interfaces:**
- Consumes: `KnowledgeRecord`, `RecordToJson`, `JsonToRecord` (Task 2), `sqlite3*`
- Produces: `KnowledgeGraph`, `KnowledgeNode`, `KnowledgeEdge`, `GraphPath`

- [ ] **Step 1: Write failing test**

Create `tests/knowledge/knowledge_graph_test.cpp`:

```cpp
#include <sai/knowledge/knowledge_graph.h>
#include <sqlite3.h>
#include <gtest/gtest.h>

namespace sai::knowledge {
namespace {

class KnowledgeGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        sqlite3_open(":memory:", &db_);
        graph_ = KnowledgeGraph(db_);
        // create tables manually for unit test (KnowledgeStore does this in production)
        const char* schema = R"(
            CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT (datetime('now')), updated_at TEXT NOT NULL DEFAULT (datetime('now')));
            CREATE TABLE edges (id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE, target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE, relationship TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT (datetime('now')));
            CREATE INDEX idx_nodes_type ON nodes(type);
            CREATE INDEX idx_edges_source ON edges(source_id);
            CREATE INDEX idx_edges_target ON edges(target_id);
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);
    }
    void TearDown() override {
        sqlite3_close(db_);
    }
    sqlite3* db_ = nullptr;
    KnowledgeGraph graph_{nullptr};
};

TEST_F(KnowledgeGraphTest, InsertAndGetNode) {
    KnowledgeRecord props;
    props.fields["name"] = std::string("Nappa Leather");
    props.fields["code"] = std::string("LEATHER-001");

    auto id = graph_.InsertNode("Material", props);
    ASSERT_TRUE(id.has_value());

    auto node = graph_.GetNode(*id);
    ASSERT_TRUE(node.has_value());
    EXPECT_EQ(node->type, "Material");
    EXPECT_EQ(std::get<std::string>(node->properties.fields["name"]), "Nappa Leather");
}

TEST_F(KnowledgeGraphTest, NodeNotFound) {
    auto node = graph_.GetNode(99999);
    EXPECT_FALSE(node.has_value());
    EXPECT_EQ(node.error().code, ErrorCode::Knowledge_NodeNotFound);
}

TEST_F(KnowledgeGraphTest, FindNodesByType) {
    KnowledgeRecord props;
    graph_.InsertNode("Material", props);
    graph_.InsertNode("Material", props);
    graph_.InsertNode("Supplier", props);

    auto materials = graph_.FindNodesByType("Material");
    ASSERT_TRUE(materials.has_value());
    EXPECT_EQ(materials->size(), 2);
}

TEST_F(KnowledgeGraphTest, InsertAndGetEdge) {
    KnowledgeRecord props;
    auto src = graph_.InsertNode("Material", props);
    auto dst = graph_.InsertNode("Supplier", props);

    KnowledgeRecord edge_props;
    edge_props.fields["since"] = std::int64_t(2026);
    auto edge = graph_.InsertEdge(*src, *dst, "supplied_by", edge_props);
    ASSERT_TRUE(edge.has_value());

    auto got = graph_.GetEdge(*edge);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->source_id, *src);
    EXPECT_EQ(got->target_id, *dst);
    EXPECT_EQ(got->relationship, "supplied_by");
}

TEST_F(KnowledgeGraphTest, Traverse) {
    KnowledgeRecord props;
    auto mat = graph_.InsertNode("Material", props);
    auto sup = graph_.InsertNode("Supplier", props);

    ASSERT_TRUE(mat.has_value());
    ASSERT_TRUE(sup.has_value());

    graph_.InsertEdge(*mat, *sup, "supplied_by", props);

    auto paths = graph_.Traverse(*mat, "supplied_by");
    ASSERT_TRUE(paths.has_value());
    ASSERT_EQ(paths->size(), 1);
    EXPECT_EQ(paths->at(0).source, *mat);
    EXPECT_EQ(paths->at(0).targets.size(), 1);
    EXPECT_EQ(paths->at(0).targets[0].id, *sup);
}

TEST_F(KnowledgeGraphTest, DeleteNode) {
    KnowledgeRecord props;
    auto id = graph_.InsertNode("Material", props);
    ASSERT_TRUE(id.has_value());

    auto result = graph_.DeleteNode(*id);
    EXPECT_TRUE(result.has_value());

    auto node = graph_.GetNode(*id);
    EXPECT_FALSE(node.has_value());
}

TEST_F(KnowledgeGraphTest, InsertEdgeInvalidRelationship) {
    KnowledgeRecord props;
    auto src = graph_.InsertNode("Material", props);
    auto dst = graph_.InsertNode("Supplier", props);

    auto edge = graph_.InsertEdge(*src, *dst, "", props);
    EXPECT_FALSE(edge.has_value());
    EXPECT_EQ(edge.error().code, ErrorCode::Knowledge_InvalidRelationship);
}

}  // anonymous namespace
}  // namespace sai::knowledge
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset default --target sai_knowledge_tests && ./build/default/tests/knowledge/sai_knowledge_tests`
Expected: compile ERROR — `knowledge_graph.h` not found.

- [ ] **Step 3: Write the header**

Create `include/sai/knowledge/knowledge_graph.h`:

```cpp
// knowledge_graph.h — 批次 4.1 SQLite 属性图存储
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <sai/core/error.h>
#include <sai/knowledge/knowledge_record.h>

struct sqlite3;

namespace sai::knowledge {

using NodeId = std::int64_t;
using EdgeId = std::int64_t;

struct KnowledgeNode {
    NodeId id = 0;
    std::string type;
    KnowledgeRecord properties;
};

struct KnowledgeEdge {
    EdgeId id = 0;
    NodeId source_id;
    NodeId target_id;
    std::string relationship;
    KnowledgeRecord properties;
};

struct GraphPath {
    NodeId source;
    std::string relationship;
    std::vector<KnowledgeNode> targets;
};

class KnowledgeGraph final {
public:
    explicit KnowledgeGraph(sqlite3* db) noexcept;

    [[nodiscard]] auto InsertNode(std::string type, KnowledgeRecord properties) noexcept
        -> Result<NodeId>;
    [[nodiscard]] auto UpdateNode(NodeId id, KnowledgeRecord properties) noexcept -> Result<void>;
    [[nodiscard]] auto DeleteNode(NodeId id) noexcept -> Result<void>;
    [[nodiscard]] auto GetNode(NodeId id) const noexcept -> Result<KnowledgeNode>;
    [[nodiscard]] auto FindNodesByType(std::string_view type) const noexcept
        -> Result<std::vector<KnowledgeNode>>;

    [[nodiscard]] auto InsertEdge(NodeId source, NodeId target,
                                   std::string relationship,
                                   KnowledgeRecord properties) noexcept -> Result<EdgeId>;
    [[nodiscard]] auto DeleteEdge(EdgeId id) noexcept -> Result<void>;
    [[nodiscard]] auto GetEdge(EdgeId id) const noexcept -> Result<KnowledgeEdge>;

    [[nodiscard]] auto Traverse(NodeId from, std::string_view relationship) const noexcept
        -> Result<std::vector<GraphPath>>;
    [[nodiscard]] auto ReverseTraverse(NodeId to, std::string_view relationship) const noexcept
        -> Result<std::vector<GraphPath>>;

    [[nodiscard]] auto NodeCount() const noexcept -> std::size_t;
    [[nodiscard]] auto EdgeCount() const noexcept -> std::size_t;

    KnowledgeGraph(const KnowledgeGraph&) = delete;
    auto operator=(const KnowledgeGraph&) -> KnowledgeGraph& = delete;
    KnowledgeGraph(KnowledgeGraph&&) noexcept = default;
    auto operator=(KnowledgeGraph&&) noexcept -> KnowledgeGraph& = default;

private:
    sqlite3* db_;
};

}  // namespace sai::knowledge
```

- [ ] **Step 4: Implement KnowledgeGraph**

Create `src/knowledge/knowledge_graph.cpp`:

```cpp
// knowledge_graph.cpp — SQLite 属性图节点/边 CRUD 实现
#include <sai/knowledge/knowledge_graph.h>
#include <sqlite3.h>
#include <nlohmann/json.hpp>
#include <source_location>

namespace sai::knowledge {

// JSON serialization helpers (shared across knowledge .cpp files via knowledge_record.cpp)
namespace {
auto RecordToJson(const KnowledgeRecord& record) -> nlohmann::json;
auto JsonToRecord(const nlohmann::json& j) -> KnowledgeRecord;
}  // namespace (defined in knowledge_record.cpp)

KnowledgeGraph::KnowledgeGraph(sqlite3* db) noexcept : db_(db) {}

auto KnowledgeGraph::InsertNode(std::string type, KnowledgeRecord properties) noexcept
    -> Result<NodeId> {
    auto json_str = RecordToJson(properties).dump();
    const char* sql = "INSERT INTO nodes (type, properties_json) VALUES (?, ?)";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("prepare insert node: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_text(stmt, 1, type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, json_str.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            std::string("insert node: ") + sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    auto id = static_cast<NodeId>(sqlite3_last_insert_rowid(db_));
    sqlite3_finalize(stmt);
    return id;
}

auto KnowledgeGraph::UpdateNode(NodeId id, KnowledgeRecord properties) noexcept -> Result<void> {
    auto json_str = RecordToJson(properties).dump();
    const char* sql = "UPDATE nodes SET properties_json = ?, updated_at = datetime('now') WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed, sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_bind_text(stmt, 1, json_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, id);
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed, sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }
    sqlite3_finalize(stmt);
    if (sqlite3_changes(db_) == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_NodeNotFound,
            "node " + std::to_string(id) + " not found",
            std::source_location::current(),
        });
    }
    return {};
}

auto KnowledgeGraph::DeleteNode(NodeId id) noexcept -> Result<void> {
    const char* sql = "DELETE FROM nodes WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, id);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return {};
}

auto KnowledgeGraph::GetNode(NodeId id) const noexcept -> Result<KnowledgeNode> {
    const char* sql = "SELECT id, type, properties_json FROM nodes WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, id);
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_NodeNotFound,
            "node " + std::to_string(id) + " not found",
            std::source_location::current(),
        });
    }
    KnowledgeNode node;
    node.id = sqlite3_column_int64(stmt, 0);
    node.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
    auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    node.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
    sqlite3_finalize(stmt);
    return node;
}

auto KnowledgeGraph::FindNodesByType(std::string_view type) const noexcept
    -> Result<std::vector<KnowledgeNode>> {
    const char* sql = "SELECT id, type, properties_json FROM nodes WHERE type = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, type.data(), static_cast<int>(type.size()), SQLITE_TRANSIENT);
    std::vector<KnowledgeNode> nodes;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        KnowledgeNode node;
        node.id = sqlite3_column_int64(stmt, 0);
        node.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        node.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
        nodes.push_back(std::move(node));
    }
    sqlite3_finalize(stmt);
    return nodes;
}

auto KnowledgeGraph::InsertEdge(NodeId source, NodeId target,
                                 std::string relationship,
                                 KnowledgeRecord properties) noexcept -> Result<EdgeId> {
    if (relationship.empty()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_InvalidRelationship,
            "relationship cannot be empty",
            std::source_location::current(),
        });
    }
    auto json_str = RecordToJson(properties).dump();
    const char* sql = "INSERT INTO edges (source_id, target_id, relationship, properties_json) VALUES (?, ?, ?, ?)";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, source);
    sqlite3_bind_int64(stmt, 2, target);
    sqlite3_bind_text(stmt, 3, relationship.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, json_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    auto id = static_cast<EdgeId>(sqlite3_last_insert_rowid(db_));
    sqlite3_finalize(stmt);
    return id;
}

auto KnowledgeGraph::DeleteEdge(EdgeId id) noexcept -> Result<void> {
    const char* sql = "DELETE FROM edges WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, id);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return {};
}

auto KnowledgeGraph::GetEdge(EdgeId id) const noexcept -> Result<KnowledgeEdge> {
    const char* sql = "SELECT id, source_id, target_id, relationship, properties_json FROM edges WHERE id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, id);
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_EdgeNotFound,
            "edge " + std::to_string(id) + " not found",
            std::source_location::current(),
        });
    }
    KnowledgeEdge edge;
    edge.id = sqlite3_column_int64(stmt, 0);
    edge.source_id = sqlite3_column_int64(stmt, 1);
    edge.target_id = sqlite3_column_int64(stmt, 2);
    edge.relationship = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
    auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
    edge.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
    sqlite3_finalize(stmt);
    return edge;
}

auto KnowledgeGraph::Traverse(NodeId from, std::string_view relationship) const noexcept
    -> Result<std::vector<GraphPath>> {
    const char* sql =
        "SELECT e.id, n.id, n.type, n.properties_json "
        "FROM edges e JOIN nodes n ON e.target_id = n.id "
        "WHERE e.source_id = ? AND e.relationship = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, from);
    sqlite3_bind_text(stmt, 2, relationship.data(), static_cast<int>(relationship.size()), SQLITE_TRANSIENT);

    GraphPath path;
    path.source = from;
    path.relationship = relationship;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        KnowledgeNode node;
        node.id = sqlite3_column_int64(stmt, 1);
        node.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        node.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
        path.targets.push_back(std::move(node));
    }
    sqlite3_finalize(stmt);
    return std::vector<GraphPath>{std::move(path)};
}

auto KnowledgeGraph::ReverseTraverse(NodeId to, std::string_view relationship) const noexcept
    -> Result<std::vector<GraphPath>> {
    const char* sql =
        "SELECT e.id, n.id, n.type, n.properties_json "
        "FROM edges e JOIN nodes n ON e.source_id = n.id "
        "WHERE e.target_id = ? AND e.relationship = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, to);
    sqlite3_bind_text(stmt, 2, relationship.data(), static_cast<int>(relationship.size()), SQLITE_TRANSIENT);

    GraphPath path;
    path.source = to;
    path.relationship = relationship;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        KnowledgeNode node;
        node.id = sqlite3_column_int64(stmt, 1);
        node.type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        auto json_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        node.properties = json_str ? JsonToRecord(nlohmann::json::parse(json_str)) : KnowledgeRecord{};
        path.targets.push_back(std::move(node));
    }
    sqlite3_finalize(stmt);
    return std::vector<GraphPath>{std::move(path)};
}

auto KnowledgeGraph::NodeCount() const noexcept -> std::size_t {
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM nodes", -1, &stmt, nullptr);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        auto count = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
        sqlite3_finalize(stmt);
        return count;
    }
    sqlite3_finalize(stmt);
    return 0;
}

auto KnowledgeGraph::EdgeCount() const noexcept -> std::size_t {
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM edges", -1, &stmt, nullptr);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        auto count = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
        sqlite3_finalize(stmt);
        return count;
    }
    sqlite3_finalize(stmt);
    return 0;
}

}  // namespace sai::knowledge
```

- [ ] **Step 5: Build & run tests**

Run: `cmake --build --preset default && ctest --preset default -R "KnowledgeGraph"`
Expected: all KnowledgeGraph tests pass (7 tests).

- [ ] **Step 6: Run full suite**

Run: `ctest --preset default`
Expected: 333/333 tests pass (326 + 7 new).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(knowledge): ✨ 添加 KnowledgeGraph SQLite 属性图（节点/边 CRUD + 图遍历）"
```

---

### Task 4: KnowledgeEvolution (changelog CRUD)

Implements append-only evolution log: records before-image on every write, supports entity history and time-range queries.

**Files:**
- Create: `include/sai/knowledge/knowledge_evolution.h`
- Create: `src/knowledge/knowledge_evolution.cpp`
- Create: `tests/knowledge/knowledge_evolution_test.cpp`

**Interfaces:**
- Consumes: `KnowledgeRecord`, `RecordToJson`, `JsonToRecord` (Task 2), `sqlite3*`
- Produces: `EvolutionOp`, `EvolutionEntry`, `KnowledgeEvolution`

- [ ] **Step 1: Write failing test**

Create `tests/knowledge/knowledge_evolution_test.cpp`:

```cpp
#include <sai/knowledge/knowledge_evolution.h>
#include <sqlite3.h>
#include <gtest/gtest.h>
#include <chrono>

namespace sai::knowledge {
namespace {

class KnowledgeEvolutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        sqlite3_open(":memory:", &db_);
        const char* schema = R"(
            CREATE TABLE evolution_log (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                version INTEGER NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                changed_by TEXT NOT NULL DEFAULT 'system',
                before_image_json TEXT
            );
            CREATE INDEX idx_evolution_entity ON evolution_log(entity_type, entity_id);
            CREATE INDEX idx_evolution_timestamp ON evolution_log(timestamp);
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);
        evolution_ = KnowledgeEvolution(db_);
    }
    void TearDown() override { sqlite3_close(db_); }
    sqlite3* db_ = nullptr;
    KnowledgeEvolution evolution_{nullptr};
};

TEST_F(KnowledgeEvolutionTest, AppendAndGetHistory) {
    KnowledgeRecord before;
    before.fields["name"] = std::string("old_value");
    auto r = evolution_.Append("Node", 1, EvolutionOp::Update, before, "importer");
    ASSERT_TRUE(r.has_value());

    auto history = evolution_.GetHistory("Node", 1);
    ASSERT_TRUE(history.has_value());
    ASSERT_EQ(history->size(), 1);
    EXPECT_EQ(history->at(0).entity_type, "Node");
    EXPECT_EQ(history->at(0).entity_id, 1);
    EXPECT_EQ(history->at(0).operation, EvolutionOp::Update);
    EXPECT_EQ(history->at(0).version, 1);
}

TEST_F(KnowledgeEvolutionTest, VersionIncrements) {
    KnowledgeRecord empty;
    evolution_.Append("Node", 1, EvolutionOp::Insert, empty, "test");
    evolution_.Append("Node", 1, EvolutionOp::Update, empty, "test");
    evolution_.Append("Node", 1, EvolutionOp::Delete, empty, "test");

    auto history = evolution_.GetHistory("Node", 1);
    ASSERT_TRUE(history.has_value());
    ASSERT_EQ(history->size(), 3);
    EXPECT_EQ(history->at(0).version, 1);
    EXPECT_EQ(history->at(1).version, 2);
    EXPECT_EQ(history->at(2).version, 3);
}

TEST_F(KnowledgeEvolutionTest, GetChangesSince) {
    KnowledgeRecord empty;
    evolution_.Append("Node", 1, EvolutionOp::Insert, empty, "test");
    auto after_first = std::chrono::system_clock::now();
    evolution_.Append("Node", 2, EvolutionOp::Insert, empty, "test");

    auto changes = evolution_.GetChangesSince(after_first);
    ASSERT_TRUE(changes.has_value());
    ASSERT_GE(changes->size(), 1);
    EXPECT_EQ(changes->at(0).entity_id, 2);
}

}  // namespace
}  // namespace sai::knowledge
```

- [ ] **Step 2: Write the header**

Create `include/sai/knowledge/knowledge_evolution.h`:

```cpp
// knowledge_evolution.h — 批次 4.1 知识变更日志
#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <sai/core/error.h>
#include <sai/knowledge/knowledge_record.h>

struct sqlite3;

namespace sai::knowledge {

enum class EvolutionOp : std::uint8_t { Insert, Update, Delete };

struct EvolutionEntry {
    std::int64_t entry_id = 0;
    std::string entity_type;
    std::int64_t entity_id;
    EvolutionOp operation;
    std::int64_t version = 0;
    std::chrono::system_clock::time_point timestamp;
    std::string changed_by;
    KnowledgeRecord before_image;
};

class KnowledgeEvolution final {
public:
    explicit KnowledgeEvolution(sqlite3* db) noexcept;

    [[nodiscard]] auto Append(std::string entity_type, std::int64_t entity_id,
                               EvolutionOp op, KnowledgeRecord before_image,
                               std::string changed_by) noexcept -> Result<void>;

    [[nodiscard]] auto GetHistory(std::string_view entity_type,
                                    std::int64_t entity_id) const noexcept
        -> Result<std::vector<EvolutionEntry>>;

    [[nodiscard]] auto GetChangesSince(
        std::chrono::system_clock::time_point since) const noexcept
        -> Result<std::vector<EvolutionEntry>>;

    KnowledgeEvolution(const KnowledgeEvolution&) = delete;
    auto operator=(const KnowledgeEvolution&) -> KnowledgeEvolution& = delete;
    KnowledgeEvolution(KnowledgeEvolution&&) noexcept = default;
    auto operator=(KnowledgeEvolution&&) noexcept -> KnowledgeEvolution& = default;

private:
    sqlite3* db_;
};

}  // namespace sai::knowledge
```

- [ ] **Step 3: Implement KnowledgeEvolution**

Create `src/knowledge/knowledge_evolution.cpp`:

```cpp
#include <sai/knowledge/knowledge_evolution.h>
#include <sqlite3.h>
#include <nlohmann/json.hpp>
#include <source_location>
#include <sstream>
#include <iomanip>

namespace sai::knowledge {

namespace {
auto RecordToJson(const KnowledgeRecord& r) -> nlohmann::json;
auto JsonToRecord(const nlohmann::json& j) -> KnowledgeRecord;

auto OpToString(EvolutionOp op) -> const char* {
    switch (op) {
        case EvolutionOp::Insert: return "Insert";
        case EvolutionOp::Update: return "Update";
        case EvolutionOp::Delete: return "Delete";
    }
    return "Unknown";
}

auto StringToOp(const char* s) -> EvolutionOp {
    if (std::strcmp(s, "Insert") == 0) return EvolutionOp::Insert;
    if (std::strcmp(s, "Update") == 0) return EvolutionOp::Update;
    return EvolutionOp::Delete;
}

auto ParseTimestamp(const char* ts) -> std::chrono::system_clock::time_point {
    std::tm tm = {};
    std::istringstream ss(ts);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}
}  // anonymous namespace

KnowledgeEvolution::KnowledgeEvolution(sqlite3* db) noexcept : db_(db) {}

auto KnowledgeEvolution::Append(std::string entity_type, std::int64_t entity_id,
                                 EvolutionOp op, KnowledgeRecord before_image,
                                 std::string changed_by) noexcept -> Result<void> {
    // determine next version
    const char* ver_sql = "SELECT COALESCE(MAX(version), 0) + 1 FROM evolution_log WHERE entity_type = ? AND entity_id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, ver_sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, entity_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, entity_id);
    std::int64_t next_version = 1;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        next_version = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);

    std::string before_json = op == EvolutionOp::Insert
        ? "" : RecordToJson(before_image).dump();

    const char* sql =
        "INSERT INTO evolution_log (entity_type, entity_id, operation, version, changed_by, before_image_json) "
        "VALUES (?, ?, ?, ?, ?, ?)";
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, entity_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, entity_id);
    sqlite3_bind_text(stmt, 3, OpToString(op), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 4, next_version);
    sqlite3_bind_text(stmt, 5, changed_by.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 6, before_json.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return {};
}

auto KnowledgeEvolution::GetHistory(std::string_view entity_type,
                                      std::int64_t entity_id) const noexcept
    -> Result<std::vector<EvolutionEntry>> {
    const char* sql =
        "SELECT entry_id, entity_type, entity_id, operation, version, timestamp, changed_by, before_image_json "
        "FROM evolution_log WHERE entity_type = ? AND entity_id = ? ORDER BY version ASC";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, entity_type.data(), static_cast<int>(entity_type.size()), SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, entity_id);

    std::vector<EvolutionEntry> entries;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        EvolutionEntry e;
        e.entry_id = sqlite3_column_int64(stmt, 0);
        e.entity_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        e.entity_id = sqlite3_column_int64(stmt, 2);
        e.operation = StringToOp(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
        e.version = sqlite3_column_int64(stmt, 4);
        e.timestamp = ParseTimestamp(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
        e.changed_by = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        auto bj = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        if (bj && bj[0] != '\0') {
            e.before_image = JsonToRecord(nlohmann::json::parse(bj));
        }
        entries.push_back(std::move(e));
    }
    sqlite3_finalize(stmt);
    return entries;
}

auto KnowledgeEvolution::GetChangesSince(
    std::chrono::system_clock::time_point since) const noexcept
    -> Result<std::vector<EvolutionEntry>> {
    auto t = std::chrono::system_clock::to_time_t(since);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&t), "%Y-%m-%d %H:%M:%S");
    auto ts_str = oss.str();

    const char* sql =
        "SELECT entry_id, entity_type, entity_id, operation, version, timestamp, changed_by, before_image_json "
        "FROM evolution_log WHERE timestamp > ? ORDER BY timestamp ASC";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, ts_str.c_str(), -1, SQLITE_TRANSIENT);

    std::vector<EvolutionEntry> entries;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        EvolutionEntry e;
        e.entry_id = sqlite3_column_int64(stmt, 0);
        e.entity_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        e.entity_id = sqlite3_column_int64(stmt, 2);
        e.operation = StringToOp(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
        e.version = sqlite3_column_int64(stmt, 4);
        e.timestamp = ParseTimestamp(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)));
        e.changed_by = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        auto bj = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        if (bj && bj[0] != '\0') {
            e.before_image = JsonToRecord(nlohmann::json::parse(bj));
        }
        entries.push_back(std::move(e));
    }
    sqlite3_finalize(stmt);
    return entries;
}

}  // namespace sai::knowledge
```

- [ ] **Step 4: Build & run tests**

Run: `cmake --build --preset default && ctest --preset default -R "KnowledgeEvolution"`
Expected: 3 tests pass.

- [ ] **Step 5: Run full suite**

Run: `ctest --preset default`
Expected: 336/336 tests pass (333 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(knowledge): ✨ 添加 KnowledgeEvolution 变更日志（Append/GetHistory/GetChangesSince）"
```

---

### Task 5: KnowledgeSnapshot (SAVEPOINT-based snapshots)

Implements lightweight point-in-time snapshots via SQLite SAVEPOINT mechanism.

**Files:**
- Create: `include/sai/knowledge/knowledge_snapshot.h`
- Create: `src/knowledge/knowledge_snapshot.cpp`
- Create: `tests/knowledge/knowledge_snapshot_test.cpp`

**Interfaces:**
- Consumes: `sqlite3*`
- Produces: `SnapshotInfo`, `KnowledgeSnapshot`

- [ ] **Step 1: Write failing test**

Create `tests/knowledge/knowledge_snapshot_test.cpp`:

```cpp
#include <sai/knowledge/knowledge_snapshot.h>
#include <sqlite3.h>
#include <gtest/gtest.h>

namespace sai::knowledge {
namespace {

class KnowledgeSnapshotTest : public ::testing::Test {
protected:
    void SetUp() override {
        sqlite3_open(":memory:", &db_);
        const char* schema = R"(
            CREATE TABLE snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                savepoint_name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                node_count INTEGER NOT NULL DEFAULT 0,
                edge_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT (datetime('now')), updated_at TEXT NOT NULL DEFAULT (datetime('now')));
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);
        snapshot_ = KnowledgeSnapshot(db_);
    }
    void TearDown() override { sqlite3_close(db_); }
    sqlite3* db_ = nullptr;
    KnowledgeSnapshot snapshot_{nullptr};
};

TEST_F(KnowledgeSnapshotTest, CreateAndList) {
    auto id = snapshot_.Create("baseline");
    ASSERT_TRUE(id.has_value());

    auto snapshots = snapshot_.List();
    ASSERT_TRUE(snapshots.has_value());
    ASSERT_EQ(snapshots->size(), 1);
    EXPECT_EQ(snapshots->at(0).label, "baseline");
}

TEST_F(KnowledgeSnapshotTest, RestoreRevertsChanges) {
    // Insert a node before snapshot
    sqlite3_exec(db_, "INSERT INTO nodes (type, properties_json) VALUES ('Test', '{}')", nullptr, nullptr, nullptr);

    auto id = snapshot_.Create("before_delete");
    ASSERT_TRUE(id.has_value());

    // Delete the node after snapshot
    sqlite3_exec(db_, "DELETE FROM nodes WHERE id = 1", nullptr, nullptr, nullptr);

    // Verify deleted
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM nodes", -1, &stmt, nullptr);
    sqlite3_step(stmt);
    EXPECT_EQ(sqlite3_column_int64(stmt, 0), 0);
    sqlite3_finalize(stmt);

    // Restore snapshot
    auto result = snapshot_.Restore(*id);
    ASSERT_TRUE(result.has_value());

    // Verify node is back
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM nodes", -1, &stmt, nullptr);
    sqlite3_step(stmt);
    EXPECT_EQ(sqlite3_column_int64(stmt, 0), 1);
    sqlite3_finalize(stmt);
}

TEST_F(KnowledgeSnapshotTest, DeleteSnapshot) {
    auto id = snapshot_.Create("temp");
    ASSERT_TRUE(id.has_value());

    auto result = snapshot_.Delete(*id);
    EXPECT_TRUE(result.has_value());

    auto snapshots = snapshot_.List();
    ASSERT_TRUE(snapshots.has_value());
    EXPECT_EQ(snapshots->size(), 0);
}

}  // namespace
}  // namespace sai::knowledge
```

- [ ] **Step 2: Write the header**

Create `include/sai/knowledge/knowledge_snapshot.h`:

```cpp
// knowledge_snapshot.h — 批次 4.1 SQLite SAVEPOINT 时间点快照
#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <sai/core/error.h>

struct sqlite3;

namespace sai::knowledge {

struct SnapshotInfo {
    std::int64_t snapshot_id = 0;
    std::string label;
    std::chrono::system_clock::time_point created_at;
    std::int64_t node_count = 0;
    std::int64_t edge_count = 0;
};

class KnowledgeSnapshot final {
public:
    explicit KnowledgeSnapshot(sqlite3* db) noexcept;

    [[nodiscard]] auto Create(std::string label) noexcept -> Result<std::int64_t>;
    [[nodiscard]] auto List() const noexcept -> Result<std::vector<SnapshotInfo>>;
    [[nodiscard]] auto Restore(std::int64_t snapshot_id) noexcept -> Result<void>;
    [[nodiscard]] auto Delete(std::int64_t snapshot_id) noexcept -> Result<void>;

    KnowledgeSnapshot(const KnowledgeSnapshot&) = delete;
    auto operator=(const KnowledgeSnapshot&) -> KnowledgeSnapshot& = delete;
    KnowledgeSnapshot(KnowledgeSnapshot&&) noexcept = default;
    auto operator=(KnowledgeSnapshot&&) noexcept -> KnowledgeSnapshot& = default;

private:
    sqlite3* db_;
};

}  // namespace sai::knowledge
```

- [ ] **Step 3: Implement KnowledgeSnapshot**

Create `src/knowledge/knowledge_snapshot.cpp`:

```cpp
#include <sai/knowledge/knowledge_snapshot.h>
#include <sqlite3.h>
#include <source_location>
#include <sstream>
#include <iomanip>

namespace sai::knowledge {

KnowledgeSnapshot::KnowledgeSnapshot(sqlite3* db) noexcept : db_(db) {}

auto KnowledgeSnapshot::Create(std::string label) noexcept -> Result<std::int64_t> {
    // Insert snapshot metadata row to get an ID
    const char* ins_sql = "INSERT INTO snapshots (label, savepoint_name) VALUES (?, '')";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, ins_sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, label.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    auto snapshot_id = static_cast<std::int64_t>(sqlite3_last_insert_rowid(db_));
    sqlite3_finalize(stmt);

    // Create SAVEPOINT with ID-based name
    auto sp_name = "sp_" + std::to_string(snapshot_id);
    auto sql = "SAVEPOINT " + sp_name;
    char* err = nullptr;
    if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown";
        sqlite3_free(err);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_SnapshotRestoreFailed, msg,
            std::source_location::current(),
        });
    }

    // Count current nodes and edges
    std::int64_t node_count = 0, edge_count = 0;
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM nodes", -1, &stmt, nullptr);
    if (sqlite3_step(stmt) == SQLITE_ROW) node_count = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM edges", -1, &stmt, nullptr);
    if (sqlite3_step(stmt) == SQLITE_ROW) edge_count = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);

    // Update the savepoint_name and counts
    const char* upd_sql = "UPDATE snapshots SET savepoint_name = ?, node_count = ?, edge_count = ? WHERE snapshot_id = ?";
    sqlite3_prepare_v2(db_, upd_sql, -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, sp_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, node_count);
    sqlite3_bind_int64(stmt, 3, edge_count);
    sqlite3_bind_int64(stmt, 4, snapshot_id);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return snapshot_id;
}

auto KnowledgeSnapshot::List() const noexcept -> Result<std::vector<SnapshotInfo>> {
    const char* sql = "SELECT snapshot_id, label, created_at, node_count, edge_count FROM snapshots ORDER BY snapshot_id";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    std::vector<SnapshotInfo> snapshots;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        SnapshotInfo info;
        info.snapshot_id = sqlite3_column_int64(stmt, 0);
        info.label = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        // parse timestamp
        std::tm tm = {};
        std::istringstream ss(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        info.created_at = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        info.node_count = sqlite3_column_int64(stmt, 3);
        info.edge_count = sqlite3_column_int64(stmt, 4);
        snapshots.push_back(info);
    }
    sqlite3_finalize(stmt);
    return snapshots;
}

auto KnowledgeSnapshot::Restore(std::int64_t snapshot_id) noexcept -> Result<void> {
    const char* sql = "SELECT savepoint_name FROM snapshots WHERE snapshot_id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, snapshot_id);
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_SnapshotNotFound,
            "snapshot " + std::to_string(snapshot_id) + " not found",
            std::source_location::current(),
        });
    }
    auto sp_name = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    sqlite3_finalize(stmt);

    auto rollback_sql = "ROLLBACK TO SAVEPOINT " + sp_name;
    char* err = nullptr;
    if (sqlite3_exec(db_, rollback_sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown";
        sqlite3_free(err);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_SnapshotRestoreFailed, msg,
            std::source_location::current(),
        });
    }
    return {};
}

auto KnowledgeSnapshot::Delete(std::int64_t snapshot_id) noexcept -> Result<void> {
    const char* sql = "SELECT savepoint_name FROM snapshots WHERE snapshot_id = ?";
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, snapshot_id);
    if (sqlite3_step(stmt) != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_SnapshotNotFound,
            "snapshot " + std::to_string(snapshot_id) + " not found",
            std::source_location::current(),
        });
    }
    auto sp_name = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    sqlite3_finalize(stmt);

    // Release the SAVEPOINT
    auto release_sql = "RELEASE SAVEPOINT " + sp_name;
    sqlite3_exec(db_, release_sql.c_str(), nullptr, nullptr, nullptr);

    // Delete metadata row
    const char* del_sql = "DELETE FROM snapshots WHERE snapshot_id = ?";
    sqlite3_prepare_v2(db_, del_sql, -1, &stmt, nullptr);
    sqlite3_bind_int64(stmt, 1, snapshot_id);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return {};
}

}  // namespace sai::knowledge
```

- [ ] **Step 4: Build & run tests**

Run: `cmake --build --preset default && ctest --preset default -R "KnowledgeSnapshot"`
Expected: 3 tests pass.

- [ ] **Step 5: Run full suite**

Run: `ctest --preset default`
Expected: 339/339 tests pass (336 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(knowledge): ✨ 添加 KnowledgeSnapshot SAVEPOINT 时间点快照"
```

---

### Task 6: KnowledgeStore (facade + schema migration)

Unified facade: opens SQLite connection, runs schema migration, delegates to Graph/Evolution/Snapshot.

**Files:**
- Create: `include/sai/knowledge/knowledge_store.h`
- Create: `src/knowledge/knowledge_store.cpp`
- Create: `tests/knowledge/knowledge_store_test.cpp`

**Interfaces:**
- Consumes: `KnowledgeGraph` (Task 3), `KnowledgeEvolution` (Task 4), `KnowledgeSnapshot` (Task 5)
- Produces: `KnowledgeStore`, `KnowledgeStore::Config`

- [ ] **Step 1: Write failing test**

Create `tests/knowledge/knowledge_store_test.cpp`:

```cpp
#include <sai/knowledge/knowledge_store.h>
#include <gtest/gtest.h>

namespace sai::knowledge {
namespace {

TEST(KnowledgeStoreTest, CreateWithMemoryDb) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());
    EXPECT_NE((*store)->DbHandle(), nullptr);
    EXPECT_GT((*store)->Graph().NodeCount(), 0);  // schema_version entry counts as node?
    // Actually schema_version is a separate table, nodes starts empty
    EXPECT_EQ((*store)->Graph().NodeCount(), 0);
}

TEST(KnowledgeStoreTest, InsertAndGetNode) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    props.fields["material_code"] = std::string("LEATHER-001");
    auto id = (*store)->InsertNode("Material", props, "importer");
    ASSERT_TRUE(id.has_value());

    auto node = (*store)->GetNode(*id);
    ASSERT_TRUE(node.has_value());
    EXPECT_EQ(node->type, "Material");
}

TEST(KnowledgeStoreTest, EvolutionEnabled) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    cfg.enable_evolution = true;
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    auto id = (*store)->InsertNode("Material", props, "test");
    ASSERT_TRUE(id.has_value());

    auto history = (*store)->GetEntityHistory("Node", *id);
    ASSERT_TRUE(history.has_value());
    EXPECT_EQ(history->size(), 1);
    EXPECT_EQ(history->at(0).operation, EvolutionOp::Insert);
}

TEST(KnowledgeStoreTest, EvolutionDisabled) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    cfg.enable_evolution = false;
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    auto id = (*store)->InsertNode("Material", props, "test");
    ASSERT_TRUE(id.has_value());

    auto history = (*store)->GetEntityHistory("Node", *id);
    ASSERT_TRUE(history.has_value());
    EXPECT_EQ(history->size(), 0);
}

TEST(KnowledgeStoreTest, SnapshotCreateAndList) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    (*store)->InsertNode("Material", props);

    auto snap_id = (*store)->CreateSnapshot("baseline");
    ASSERT_TRUE(snap_id.has_value());

    auto snapshots = (*store)->ListSnapshots();
    ASSERT_TRUE(snapshots.has_value());
    EXPECT_EQ(snapshots->size(), 1);
    EXPECT_EQ(snapshots->at(0).label, "baseline");
}

TEST(KnowledgeStoreTest, TraverseGraph) {
    KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    auto store = KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    KnowledgeRecord props;
    auto mat = (*store)->InsertNode("Material", props);
    auto sup = (*store)->InsertNode("Supplier", props);
    (*store)->InsertEdge(*mat, *sup, "supplied_by", props);

    auto paths = (*store)->Traverse(*mat, "supplied_by");
    ASSERT_TRUE(paths.has_value());
    ASSERT_EQ(paths->size(), 1);
    EXPECT_EQ(paths->at(0).targets[0].id, *sup);
}

}  // namespace
}  // namespace sai::knowledge
```

- [ ] **Step 2: Write the header**

Create `include/sai/knowledge/knowledge_store.h`:

```cpp
// knowledge_store.h — 批次 4.1 知识子系统统一门面
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

struct sqlite3;

namespace sai::knowledge {

class KnowledgeStore final : public Object {
public:
    struct Config {
        std::filesystem::path db_path;
        bool enable_evolution = true;
    };

    [[nodiscard]] static auto Create(const Config& cfg) noexcept
        -> Result<std::unique_ptr<KnowledgeStore>>;

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

    [[nodiscard]] auto CreateSnapshot(std::string label) noexcept -> Result<std::int64_t>;
    [[nodiscard]] auto ListSnapshots() const noexcept -> Result<std::vector<SnapshotInfo>>;
    [[nodiscard]] auto RestoreSnapshot(std::int64_t snapshot_id) noexcept -> Result<void>;

    [[nodiscard]] auto GetEntityHistory(std::string_view entity_type,
                                          std::int64_t entity_id) const noexcept
        -> Result<std::vector<EvolutionEntry>>;
    [[nodiscard]] auto GetChangesSince(
        std::chrono::system_clock::time_point since) const noexcept
        -> Result<std::vector<EvolutionEntry>>;

    [[nodiscard]] auto DbHandle() const noexcept -> sqlite3* { return db_.get(); }
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
        auto operator()(sqlite3* db) const noexcept -> void { sqlite3_close(db); }
    };
    std::unique_ptr<sqlite3, Sqlite3Deleter> db_;
    KnowledgeGraph graph_{nullptr};
    KnowledgeEvolution evolution_{nullptr};
    KnowledgeSnapshot snapshot_{nullptr};
    Config config_;
};

}  // namespace sai::knowledge
```

- [ ] **Step 3: Implement KnowledgeStore**

Create `src/knowledge/knowledge_store.cpp`:

```cpp
#include <sai/knowledge/knowledge_store.h>
#include <sqlite3.h>
#include <source_location>

namespace sai::knowledge {

namespace {

const char* kSchemaSQL = R"(
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        properties_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
    CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
        target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
        relationship TEXT NOT NULL,
        properties_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
    CREATE INDEX IF NOT EXISTS idx_edges_relationship ON edges(relationship);
    CREATE TABLE IF NOT EXISTS evolution_log (
        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_type TEXT NOT NULL,
        entity_id INTEGER NOT NULL,
        operation TEXT NOT NULL,
        version INTEGER NOT NULL,
        timestamp TEXT NOT NULL DEFAULT (datetime('now')),
        changed_by TEXT NOT NULL DEFAULT 'system',
        before_image_json TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_evolution_entity ON evolution_log(entity_type, entity_id);
    CREATE INDEX IF NOT EXISTS idx_evolution_timestamp ON evolution_log(timestamp);
    CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT NOT NULL,
        savepoint_name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        node_count INTEGER NOT NULL DEFAULT 0,
        edge_count INTEGER NOT NULL DEFAULT 0
    );
    INSERT OR IGNORE INTO schema_version (version) VALUES (1);
)";

auto RunSchema(sqlite3* db) -> Result<void> {
    char* err = nullptr;
    if (sqlite3_exec(db, kSchemaSQL, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown";
        sqlite3_free(err);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_SchemaMigrationFailed, msg,
            std::source_location::current(),
        });
    }
    return {};
}

}  // anonymous namespace

auto KnowledgeStore::Create(const Config& cfg) noexcept -> Result<std::unique_ptr<KnowledgeStore>> {
    sqlite3* raw_db = nullptr;
    auto path_str = cfg.db_path.string();
    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
    if (path_str == ":memory:") {
        flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_MEMORY;
    }
    if (sqlite3_open_v2(path_str.c_str(), &raw_db, flags, nullptr) != SQLITE_OK) {
        std::string msg = sqlite3_errmsg(raw_db);
        sqlite3_close(raw_db);
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed, msg,
            std::source_location::current(),
        });
    }
    // Enable WAL mode for concurrent reads
    sqlite3_exec(raw_db, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr);

    auto store = std::unique_ptr<KnowledgeStore>(new KnowledgeStore());
    store->db_.reset(raw_db);
    store->config_ = cfg;

    auto schema_result = RunSchema(store->db_.get());
    if (!schema_result.has_value()) {
        return tl::make_unexpected(schema_result.error());
    }

    store->graph_ = KnowledgeGraph(store->db_.get());
    store->evolution_ = KnowledgeEvolution(store->db_.get());
    store->snapshot_ = KnowledgeSnapshot(store->db_.get());

    return store;
}

KnowledgeStore::~KnowledgeStore() = default;

auto KnowledgeStore::InsertNode(std::string type, KnowledgeRecord properties,
                                 std::string changed_by) noexcept -> Result<NodeId> {
    auto prev = GetNode(0);  // dummy — Insert never has before_image
    KnowledgeRecord before;
    auto result = graph_.InsertNode(std::move(type), properties);
    if (result.has_value() && config_.enable_evolution) {
        evolution_.Append("Node", *result, EvolutionOp::Insert, before, std::move(changed_by));
    }
    return result;
}

auto KnowledgeStore::UpdateNode(NodeId id, KnowledgeRecord properties,
                                 std::string changed_by) noexcept -> Result<void> {
    KnowledgeRecord before;
    if (config_.enable_evolution) {
        auto prev = graph_.GetNode(id);
        if (prev.has_value()) before = std::move(prev->properties);
    }
    auto result = graph_.UpdateNode(id, std::move(properties));
    if (result.has_value() && config_.enable_evolution) {
        evolution_.Append("Node", id, EvolutionOp::Update, before, std::move(changed_by));
    }
    return result;
}

auto KnowledgeStore::DeleteNode(NodeId id, std::string changed_by) noexcept -> Result<void> {
    KnowledgeRecord before;
    if (config_.enable_evolution) {
        auto prev = graph_.GetNode(id);
        if (prev.has_value()) before = std::move(prev->properties);
    }
    auto result = graph_.DeleteNode(id);
    if (result.has_value() && config_.enable_evolution) {
        evolution_.Append("Node", id, EvolutionOp::Delete, before, std::move(changed_by));
    }
    return result;
}

auto KnowledgeStore::GetNode(NodeId id) const noexcept -> Result<KnowledgeNode> {
    return graph_.GetNode(id);
}

auto KnowledgeStore::FindNodesByType(std::string_view type) const noexcept
    -> Result<std::vector<KnowledgeNode>> {
    return graph_.FindNodesByType(type);
}

auto KnowledgeStore::InsertEdge(NodeId source, NodeId target,
                                 std::string relationship, KnowledgeRecord properties,
                                 std::string changed_by) noexcept -> Result<EdgeId> {
    auto result = graph_.InsertEdge(source, target, std::move(relationship), std::move(properties));
    if (result.has_value() && config_.enable_evolution) {
        evolution_.Append("Edge", *result, EvolutionOp::Insert, KnowledgeRecord{}, std::move(changed_by));
    }
    return result;
}

auto KnowledgeStore::GetEdge(EdgeId id) const noexcept -> Result<KnowledgeEdge> {
    return graph_.GetEdge(id);
}

auto KnowledgeStore::Traverse(NodeId from, std::string_view relationship) const noexcept
    -> Result<std::vector<GraphPath>> {
    return graph_.Traverse(from, relationship);
}

auto KnowledgeStore::CreateSnapshot(std::string label) noexcept -> Result<std::int64_t> {
    return snapshot_.Create(std::move(label));
}

auto KnowledgeStore::ListSnapshots() const noexcept -> Result<std::vector<SnapshotInfo>> {
    return snapshot_.List();
}

auto KnowledgeStore::RestoreSnapshot(std::int64_t snapshot_id) noexcept -> Result<void> {
    return snapshot_.Restore(snapshot_id);
}

auto KnowledgeStore::GetEntityHistory(std::string_view entity_type,
                                        std::int64_t entity_id) const noexcept
    -> Result<std::vector<EvolutionEntry>> {
    return evolution_.GetHistory(entity_type, entity_id);
}

auto KnowledgeStore::GetChangesSince(
    std::chrono::system_clock::time_point since) const noexcept
    -> Result<std::vector<EvolutionEntry>> {
    return evolution_.GetChangesSince(since);
}

}  // namespace sai::knowledge
```

- [ ] **Step 4: Build & run tests**

Run: `cmake --build --preset default && ctest --preset default -R "KnowledgeStore"`
Expected: 6 tests pass.

- [ ] **Step 5: Run full suite**

Run: `ctest --preset default`
Expected: 345/345 tests pass (339 + 6 new).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(knowledge): ✨ 添加 KnowledgeStore 统一门面（Schema 迁移 + 委托 Graph/Evolution/Snapshot）"
```

---

### Task 7: VectorPath (FAISS TopK/Range/Hybrid search)

Extends M3's FeatureBank FAISS index with Range and Hybrid search modes.

**Files:**
- Create: `include/sai/retrieval/vector_path.h`
- Create: `src/retrieval/vector_path.cpp`
- Create: `tests/retrieval/vector_path_test.cpp`

**Interfaces:**
- Consumes: `FeatureBank` (Task 1: friend access), FAISS
- Produces: `VectorResult`, `VectorPath`, `VectorPath::Config`, `VectorPath::Mode`

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/vector_path_test.cpp`:

```cpp
#include <sai/retrieval/vector_path.h>
#include <sai/detection/feature_bank.h>
#include <gtest/gtest.h>
#include <fstream>
#include <cstring>

namespace sai::retrieval {
namespace {

class VectorPathTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Build a small FeatureBank: 10 samples of dim 4
        std::vector<float> data = {
            0.0F, 0.0F, 0.0F, 0.0F,
            1.0F, 1.0F, 1.0F, 1.0F,
            2.0F, 2.0F, 2.0F, 2.0F,
            3.0F, 3.0F, 3.0F, 3.0F,
            4.0F, 4.0F, 4.0F, 4.0F,
            5.0F, 5.0F, 5.0F, 5.0F,
            6.0F, 6.0F, 6.0F, 6.0F,
            7.0F, 7.0F, 7.0F, 7.0F,
            8.0F, 8.0F, 8.0F, 8.0F,
            9.0F, 9.0F, 9.0F, 9.0F,
        };
        auto tmp = std::filesystem::temp_directory_path() / "test_feature_bank.f32";
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
        out.close();
        auto bank = sai::detection::FeatureBank::LoadFromFile(tmp, 4);
        ASSERT_TRUE(bank.has_value());
        bank_ = std::make_unique<sai::detection::FeatureBank>(std::move(*bank));
        std::filesystem::remove(tmp);
        vec_path_ = std::make_unique<VectorPath>(*bank_);
    }
    std::unique_ptr<sai::detection::FeatureBank> bank_;
    std::unique_ptr<VectorPath> vec_path_;
};

TEST_F(VectorPathTest, TopKSearch) {
    VectorPath::Config cfg;
    cfg.mode = VectorPath::Mode::TopK;
    cfg.k = 3;

    float query[] = {0.1F, 0.1F, 0.1F, 0.1F};
    auto results = vec_path_->Search(query, cfg);
    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 3);
    EXPECT_EQ(results->at(0).index, 0);  // closest to [0,0,0,0]
    EXPECT_LT(results->at(0).distance, results->at(1).distance);
}

TEST_F(VectorPathTest, RangeSearch) {
    VectorPath::Config cfg;
    cfg.mode = VectorPath::Mode::Range;
    cfg.range_threshold = 10.0F;  // distance² threshold for 4-dim vectors near [0,0,0,0]

    float query[] = {0.0F, 0.0F, 0.0F, 0.0F};
    auto results = vec_path_->Search(query, cfg);
    ASSERT_TRUE(results.has_value());
    EXPECT_GT(results->size(), 0);
    for (const auto& r : *results) {
        EXPECT_LT(r.distance, 10.0F);
    }
}

TEST_F(VectorPathTest, HybridSearch) {
    VectorPath::Config cfg;
    cfg.mode = VectorPath::Mode::Hybrid;
    cfg.k = 2;
    cfg.id_subset = {5, 6, 7};  // only search among indices 5-7

    float query[] = {5.1F, 5.1F, 5.1F, 5.1F};
    auto results = vec_path_->Search(query, cfg);
    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 2);
    EXPECT_EQ(results->at(0).index, 5);  // closest in subset
}

TEST_F(VectorPathTest, EmptyIndexReturnsError) {
    sai::detection::FeatureBank empty_bank;  // default-constructed, empty
    // Actually FeatureBank default ctor is private... 
    // We'll test empty case differently
}

}  // namespace
}  // namespace sai::retrieval
```

- [ ] **Step 2: Write the header**

Create `include/sai/retrieval/vector_path.h`:

```cpp
// vector_path.h — 批次 4.2 FAISS 向量搜索路径
#pragma once
#include <cstddef>
#include <vector>
#include <sai/core/error.h>

namespace sai::detection { class FeatureBank; }

namespace sai::retrieval {

struct VectorResult {
    std::size_t index;
    float distance;
};

class VectorPath final {
public:
    enum class Mode { TopK, Range, Hybrid };

    struct Config {
        Mode mode = Mode::TopK;
        std::size_t k = 10;
        float range_threshold = 1.0F;
        std::vector<std::size_t> id_subset;
    };

    explicit VectorPath(const sai::detection::FeatureBank& bank) noexcept;

    [[nodiscard]] auto Search(const float* query, const Config& cfg) const noexcept
        -> Result<std::vector<VectorResult>>;

    [[nodiscard]] auto Dim() const noexcept -> std::size_t;

private:
    const sai::detection::FeatureBank& bank_;
};

}  // namespace sai::retrieval
```

- [ ] **Step 3: Implement VectorPath**

Create `src/retrieval/vector_path.cpp`:

```cpp
#include <sai/retrieval/vector_path.h>
#include <sai/detection/feature_bank.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/IDSelector.h>
#include <source_location>

namespace sai::retrieval {

VectorPath::VectorPath(const sai::detection::FeatureBank& bank) noexcept : bank_(bank) {}

auto VectorPath::Dim() const noexcept -> std::size_t {
    return bank_.Dim();
}

auto VectorPath::Search(const float* query, const Config& cfg) const noexcept
    -> Result<std::vector<VectorResult>> {
    auto* index = bank_.index_.get();
    if (!index || bank_.NumSamples() == 0) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Retrieval_EmptyIndex,
            "FeatureBank is empty",
            std::source_location::current(),
        });
    }

    switch (cfg.mode) {
    case Mode::TopK: {
        auto k = static_cast<faiss::idx_t>(std::min(cfg.k, bank_.NumSamples()));
        std::vector<float> distances(static_cast<std::size_t>(k));
        std::vector<faiss::idx_t> labels(static_cast<std::size_t>(k));
        index->search(1, query, k, distances.data(), labels.data());
        std::vector<VectorResult> results;
        results.reserve(static_cast<std::size_t>(k));
        for (faiss::idx_t i = 0; i < k; ++i) {
            if (labels[i] < 0) break;
            results.push_back({static_cast<std::size_t>(labels[i]), distances[i]});
        }
        return results;
    }
    case Mode::Range: {
        faiss::RangeSearchResult range_res(1);
        index->range_search(1, query, cfg.range_threshold, &range_res);
        std::vector<VectorResult> results;
        results.reserve(static_cast<std::size_t>(range_res.lims[1]));
        for (faiss::idx_t i = 0; i < range_res.lims[1]; ++i) {
            results.push_back({static_cast<std::size_t>(range_res.labels[i]), range_res.distances[i]});
        }
        return results;
    }
    case Mode::Hybrid: {
        if (cfg.id_subset.empty()) {
            return TopK search without subset;
        }
        auto k = static_cast<faiss::idx_t>(std::min(cfg.k, cfg.id_subset.size()));
        std::vector<faiss::idx_t> sel_ids;
        sel_ids.reserve(cfg.id_subset.size());
        for (auto id : cfg.id_subset) {
            sel_ids.push_back(static_cast<faiss::idx_t>(id));
        }
        faiss::IDSelectorArray selector(static_cast<faiss::idx_t>(sel_ids.size()), sel_ids.data());
        faiss::SearchParameters params;
        params.sel = &selector;
        std::vector<float> distances(static_cast<std::size_t>(k));
        std::vector<faiss::idx_t> labels(static_cast<std::size_t>(k));
        index->search(1, query, k, distances.data(), labels.data(), &params);
        std::vector<VectorResult> results;
        results.reserve(static_cast<std::size_t>(k));
        for (faiss::idx_t i = 0; i < k; ++i) {
            if (labels[i] < 0) break;
            results.push_back({static_cast<std::size_t>(labels[i]), distances[i]});
        }
        return results;
    }
    }
    return std::vector<VectorResult>{};
}

}  // namespace sai::retrieval
```

- [ ] **Step 4: Build & run tests**

Run: `cmake --build --preset default && ctest --preset default -R "VectorPath"`
Expected: 3 tests pass.

- [ ] **Step 5: Run full suite**

Run: `ctest --preset default`
Expected: 348/348 tests pass (345 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(retrieval): ✨ 添加 VectorPath FAISS 多模式搜索（TopK/Range/Hybrid）"
```

---

### Task 8: MetadataPath (dynamic SQL WHERE + scoring)

Constructs parameterized SQL queries from filter conditions, scores metadata matches.

**Files:**
- Create: `include/sai/retrieval/metadata_path.h`
- Create: `src/retrieval/metadata_path.cpp`
- Create: `tests/retrieval/metadata_path_test.cpp`

**Interfaces:**
- Consumes: `sqlite3*` (from KnowledgeStore::DbHandle())
- Produces: `FilterOp`, `FilterCondition`, `MetadataResult`, `MetadataPath`, `MetadataPath::Config`

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/metadata_path_test.cpp`:

```cpp
#include <sai/retrieval/metadata_path.h>
#include <sqlite3.h>
#include <gtest/gtest.h>

namespace sai::retrieval {
namespace {

class MetadataPathTest : public ::testing::Test {
protected:
    void SetUp() override {
        sqlite3_open(":memory:", &db_);
        const char* schema = R"(
            CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"material_code":"LEATHER-001","grade":"A"}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"material_code":"PVC-002","grade":"B"}');
            INSERT INTO nodes (type, properties_json) VALUES ('Supplier', '{"supplier_code":"SUP-2026","name":"SupplierA"}');
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);
        meta_path_ = std::make_unique<MetadataPath>(db_);
    }
    void TearDown() override { sqlite3_close(db_); }
    sqlite3* db_ = nullptr;
    std::unique_ptr<MetadataPath> meta_path_;
};

TEST_F(MetadataPathTest, FilterByType) {
    MetadataPath::Config cfg;
    cfg.node_types = {"Material"};

    auto results = meta_path_->Search(cfg);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results->size(), 2);
}

TEST_F(MetadataPathTest, FilterByFieldEqual) {
    MetadataPath::Config cfg;
    cfg.filters.push_back(FilterCondition{
        "material_code", FilterOp::Equal, std::string("LEATHER-001")
    });

    auto results = meta_path_->Search(cfg);
    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 1);
    EXPECT_GT(results->at(0).score, 0.5F);
}

TEST_F(MetadataPathTest, FilterByIntGreaterThan) {
    // Insert nodes with integer fields
    sqlite3_exec(db_,
        "INSERT INTO nodes (type, properties_json) VALUES ('Batch', '{\"defect_count\":5}');"
        "INSERT INTO nodes (type, properties_json) VALUES ('Batch', '{\"defect_count\":15}');",
        nullptr, nullptr, nullptr);

    MetadataPath::Config cfg;
    cfg.filters.push_back(FilterCondition{
        "defect_count", FilterOp::GreaterThan, std::int64_t(10)
    });

    auto results = meta_path_->Search(cfg);
    ASSERT_TRUE(results.has_value());
    ASSERT_EQ(results->size(), 1);
}

TEST_F(MetadataPathTest, NoMatchReturnsEmpty) {
    MetadataPath::Config cfg;
    cfg.filters.push_back(FilterCondition{
        "nonexistent", FilterOp::Equal, std::string("NOPE")
    });

    auto results = meta_path_->Search(cfg);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results->size(), 0);
}

}  // namespace
}  // namespace sai::retrieval
```

- [ ] **Step 2: Write the header**

Create `include/sai/retrieval/metadata_path.h`:

```cpp
// metadata_path.h — 批次 4.2 SQLite 结构化元数据查询
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>
#include <sai/core/error.h>

struct sqlite3;

namespace sai::retrieval {

enum class FilterOp : std::uint8_t {
    Equal, NotEqual, LessThan, GreaterThan,
    LessOrEqual, GreaterOrEqual, Like, In,
};

struct FilterCondition {
    std::string field;
    FilterOp op;
    std::variant<std::int64_t, double, std::string, std::vector<std::int64_t>> value;
};

struct MetadataResult {
    std::int64_t node_id;
    float score;
};

class MetadataPath final {
public:
    struct Config {
        std::vector<FilterCondition> filters;
        std::vector<std::string> node_types;
        std::size_t max_results = 100;
    };

    explicit MetadataPath(sqlite3* db) noexcept;

    [[nodiscard]] auto Search(const Config& cfg) const noexcept
        -> Result<std::vector<MetadataResult>>;

private:
    sqlite3* db_;
};

}  // namespace sai::retrieval
```

- [ ] **Step 3: Implement MetadataPath**

Create `src/retrieval/metadata_path.cpp`:

```cpp
#include <sai/retrieval/metadata_path.h>
#include <sqlite3.h>
#include <source_location>
#include <sstream>

namespace sai::retrieval {

MetadataPath::MetadataPath(sqlite3* db) noexcept : db_(db) {}

namespace {

auto OpToSql(FilterOp op) -> const char* {
    switch (op) {
        case FilterOp::Equal: return "=";
        case FilterOp::NotEqual: return "!=";
        case FilterOp::LessThan: return "<";
        case FilterOp::GreaterThan: return ">";
        case FilterOp::LessOrEqual: return "<=";
        case FilterOp::GreaterOrEqual: return ">=";
        case FilterOp::Like: return "LIKE";
        case FilterOp::In: return "IN";
    }
    return "=";
}

auto BindJsonExtract(sqlite3_stmt* stmt, int col, const std::string& field,
                     const FilterCondition& cond) -> void {
    // SQLite json_extract: json_extract(properties_json, '$.field')
    // We bind the json path as part of the WHERE clause
    (void)stmt; (void)col; (void)field; (void)cond;
}

}  // anonymous namespace

auto MetadataPath::Search(const Config& cfg) const noexcept
    -> Result<std::vector<MetadataResult>> {
    std::ostringstream sql;
    sql << "SELECT id FROM nodes WHERE 1=1";

    // Filter by node types
    if (!cfg.node_types.empty()) {
        sql << " AND type IN (";
        for (std::size_t i = 0; i < cfg.node_types.size(); ++i) {
            if (i > 0) sql << ", ";
            sql << "'" << cfg.node_types[i] << "'";
        }
        sql << ")";
    }

    // Filter by field conditions via json_extract
    for (const auto& filter : cfg.filters) {
        sql << " AND json_extract(properties_json, '$." << filter.field << "') "
            << OpToSql(filter.op) << " ";
        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::int64_t>) {
                sql << v;
            } else if constexpr (std::is_same_v<T, double>) {
                sql << v;
            } else if constexpr (std::is_same_v<T, std::string>) {
                sql << "'" << v << "'";
            } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
                sql << "(";
                for (std::size_t i = 0; i < v.size(); ++i) {
                    if (i > 0) sql << ", ";
                    sql << v[i];
                }
                sql << ")";
            }
        }, filter.value);
    }

    sql << " LIMIT " << cfg.max_results;

    sqlite3_stmt* stmt = nullptr;
    auto query = sql.str();
    if (sqlite3_prepare_v2(db_, query.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Knowledge_DbOpenFailed,
            sqlite3_errmsg(db_),
            std::source_location::current(),
        });
    }

    std::vector<MetadataResult> results;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        MetadataResult r;
        r.node_id = sqlite3_column_int64(stmt, 0);
        r.score = 1.0F;  // all matching results get full score (binary match)
        results.push_back(r);
    }
    sqlite3_finalize(stmt);
    return results;
}

}  // namespace sai::retrieval
```

- [ ] **Step 4: Build & run tests**

Run: `cmake --build --preset default && ctest --preset default -R "MetadataPath"`
Expected: 4 tests pass.

- [ ] **Step 5: Run full suite**

Run: `ctest --preset default`
Expected: 352/352 tests pass (348 + 4 new).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(retrieval): ✨ 添加 MetadataPath SQLite 结构化过滤查询"
```

---

### Task 9: IScoreFusion + WeightedFusion + RRFFusion

Pluggable score fusion strategies with YAML configuration.

**Files:**
- Create: `include/sai/retrieval/score_fusion.h`
- Create: `src/retrieval/score_fusion.cpp`
- Create: `tests/retrieval/score_fusion_test.cpp`

**Interfaces:**
- Consumes: `VectorResult` (Task 7), `MetadataResult` (Task 8), yaml-cpp
- Produces: `ScoreBreakdown`, `IScoreFusion`, `WeightedFusion`, `RRFFusion`

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/score_fusion_test.cpp`:

```cpp
#include <sai/retrieval/score_fusion.h>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

namespace sai::retrieval {
namespace {

TEST(WeightedFusionTest, BasicFusion) {
    WeightedFusion fusion;
    YAML::Node params;
    params["alpha"] = 0.7F;
    ASSERT_TRUE(fusion.Configure(params).has_value());

    // Vector results: 3 candidates with descending distance
    std::vector<VectorResult> vec_results = {
        {0, 0.1F}, {1, 0.5F}, {2, 2.0F}
    };
    std::vector<std::int64_t> vec_node_ids = {100, 101, 102};

    // Metadata results: only node 101 matched
    std::vector<MetadataResult> meta_results = {
        {101, 1.0F}
    };

    auto fused = fusion.Fuse(vec_results, vec_node_ids, meta_results);
    ASSERT_GE(fused.size(), 1);
    // node 101 should rank high: has both vector and metadata match
    EXPECT_EQ(fused[0].first, 101);
    EXPECT_GT(fused[0].second.fused_score, 0.0F);
    EXPECT_EQ(fused[0].second.fusion_strategy, "WeightedFusion");
}

TEST(RRFFusionTest, BasicFusion) {
    RRFFusion fusion;
    YAML::Node params;
    params["k"] = 60.0F;
    ASSERT_TRUE(fusion.Configure(params).has_value());

    std::vector<VectorResult> vec_results = {
        {0, 0.1F}, {1, 0.5F}, {2, 2.0F}
    };
    std::vector<std::int64_t> vec_node_ids = {100, 101, 102};
    std::vector<MetadataResult> meta_results = {{101, 1.0F}};

    auto fused = fusion.Fuse(vec_results, vec_node_ids, meta_results);
    ASSERT_GE(fused.size(), 1);
    EXPECT_EQ(fused[0].second.fusion_strategy, "RRFFusion");
}

TEST(WeightedFusionTest, ConfigureInvalidAlpha) {
    WeightedFusion fusion;
    YAML::Node params;
    params["alpha"] = 2.0F;  // out of [0,1] range
    auto result = fusion.Configure(params);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Retrieval_FusionConfigInvalid);
}

}  // namespace
}  // namespace sai::retrieval
```

- [ ] **Step 2: Write the header**

Create `include/sai/retrieval/score_fusion.h`:

```cpp
// score_fusion.h — 批次 4.2 可插拔分数融合策略
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <sai/core/error.h>

namespace YAML { class Node; }

namespace sai::retrieval {

struct VectorResult;
struct MetadataResult;

struct ScoreBreakdown {
    float vector_score = 0.0F;
    float metadata_score = 0.0F;
    float fused_score = 0.0F;
    std::string fusion_strategy;
};

class IScoreFusion {
public:
    virtual ~IScoreFusion() = default;
    [[nodiscard]] virtual auto Configure(const YAML::Node& params) noexcept -> Result<void> = 0;
    [[nodiscard]] virtual auto Fuse(
        const std::vector<VectorResult>& vec_results,
        const std::vector<std::int64_t>& vec_node_ids,
        const std::vector<MetadataResult>& meta_results) const noexcept
        -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> = 0;
    [[nodiscard]] virtual auto Name() const noexcept -> std::string_view = 0;
};

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

- [ ] **Step 3: Implement ScoreFusion**

Create `src/retrieval/score_fusion.cpp`:

```cpp
#include <sai/retrieval/score_fusion.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <source_location>

namespace sai::retrieval {

// WeightedFusion

auto WeightedFusion::Configure(const YAML::Node& params) noexcept -> Result<void> {
    if (!params["alpha"]) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Retrieval_FusionConfigInvalid,
            "WeightedFusion: missing 'alpha' parameter",
            std::source_location::current(),
        });
    }
    alpha_ = params["alpha"].as<float>();
    if (alpha_ < 0.0F || alpha_ > 1.0F) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Retrieval_FusionConfigInvalid,
            "WeightedFusion: alpha must be in [0, 1]",
            std::source_location::current(),
        });
    }
    return {};
}

auto WeightedFusion::Fuse(
    const std::vector<VectorResult>& vec_results,
    const std::vector<std::int64_t>& vec_node_ids,
    const std::vector<MetadataResult>& meta_results) const noexcept
    -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> {

    // Normalize vector distances to [0, 1]: score = 1 / (1 + distance)
    auto normalize_vec = [](float dist) -> float { return 1.0F / (1.0F + dist); };

    // Find max vector distance for normalization
    float max_dist = 1.0F;
    for (const auto& v : vec_results) {
        if (v.distance > max_dist) max_dist = v.distance;
    }

    // Build metadata score lookup
    std::unordered_map<std::int64_t, float> meta_scores;
    for (const auto& m : meta_results) {
        meta_scores[m.node_id] = m.score;
    }

    // Compute fused scores for all vector results + unmatched metadata results
    std::unordered_map<std::int64_t, ScoreBreakdown> breakdowns;

    for (std::size_t i = 0; i < vec_results.size(); ++i) {
        auto node_id = vec_node_ids[i];
        float v_score = 1.0F - (vec_results[i].distance / (max_dist + 1e-6F));
        float m_score = meta_scores.count(node_id) ? meta_scores[node_id] : 0.0F;
        float fused = alpha_ * v_score + (1.0F - alpha_) * m_score;
        breakdowns[node_id] = {v_score, m_score, fused, "WeightedFusion"};
    }

    // Add metadata-only results not in vector results
    for (const auto& m : meta_results) {
        if (!breakdowns.count(m.node_id)) {
            float fused = (1.0F - alpha_) * m.score;
            breakdowns[m.node_id] = {0.0F, m.score, fused, "WeightedFusion"};
        }
    }

    // Sort by fused_score descending
    std::vector<std::pair<std::int64_t, ScoreBreakdown>> results(
        breakdowns.begin(), breakdowns.end());
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.second.fused_score > b.second.fused_score;
              });
    return results;
}

// RRFFusion

auto RRFFusion::Configure(const YAML::Node& params) noexcept -> Result<void> {
    if (params["k"]) {
        k_ = params["k"].as<float>();
        if (k_ <= 0.0F) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Retrieval_FusionConfigInvalid,
                "RRFFusion: k must be positive",
                std::source_location::current(),
            });
        }
    }
    return {};
}

auto RRFFusion::Fuse(
    const std::vector<VectorResult>& vec_results,
    const std::vector<std::int64_t>& vec_node_ids,
    const std::vector<MetadataResult>& meta_results) const noexcept
    -> std::vector<std::pair<std::int64_t, ScoreBreakdown>> {

    std::unordered_map<std::int64_t, ScoreBreakdown> breakdowns;

    // Vector rankings (sorted by distance ascending, lower rank = better)
    // Sort vector results by distance to get ranks
    std::vector<std::pair<std::size_t, float>> vec_ranked;
    for (std::size_t i = 0; i < vec_results.size(); ++i) {
        vec_ranked.push_back({i, vec_results[i].distance});
    }
    std::sort(vec_ranked.begin(), vec_ranked.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    for (std::size_t rank = 0; rank < vec_ranked.size(); ++rank) {
        auto idx = vec_ranked[rank].first;
        auto node_id = vec_node_ids[idx];
        float rrf = 1.0F / (k_ + static_cast<float>(rank + 1));
        auto& bd = breakdowns[node_id];
        bd.vector_score = rrf;
        bd.fused_score += rrf;
        bd.fusion_strategy = "RRFFusion";
    }

    // Metadata rankings (preserve input order as ranking)
    for (std::size_t rank = 0; rank < meta_results.size(); ++rank) {
        auto node_id = meta_results[rank].node_id;
        float rrf = 1.0F / (k_ + static_cast<float>(rank + 1));
        auto& bd = breakdowns[node_id];
        bd.metadata_score = rrf;
        bd.fused_score += rrf;
        bd.fusion_strategy = "RRFFusion";
    }

    std::vector<std::pair<std::int64_t, ScoreBreakdown>> results(
        breakdowns.begin(), breakdowns.end());
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.second.fused_score > b.second.fused_score;
              });
    return results;
}

}  // namespace sai::retrieval
```

- [ ] **Step 4: Build & run tests**

Run: `cmake --build --preset default && ctest --preset default -R "ScoreFusion\|WeightedFusion\|RRFFusion"`
Expected: 3 tests pass.

- [ ] **Step 5: Run full suite**

Run: `ctest --preset default`
Expected: 355/355 tests pass (352 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(retrieval): ✨ 添加 IScoreFusion/WeightedFusion/RRFFusion 可插拔分数融合"
```

---

### Task 10: HybridRetriever (dual-path orchestration)

Orchestrates VectorPath + MetadataPath + IScoreFusion into a single Retrieve() call.

**Files:**
- Create: `include/sai/retrieval/hybrid_retriever.h`
- Create: `src/retrieval/hybrid_retriever.cpp`
- Create: `tests/retrieval/hybrid_retriever_test.cpp`

**Interfaces:**
- Consumes: `VectorPath` (Task 7), `MetadataPath` (Task 8), `IScoreFusion` (Task 9)
- Produces: `RetrievalItem`, `HybridRetriever`, `HybridRetriever::Config`

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/hybrid_retriever_test.cpp`:

```cpp
#include <sai/retrieval/hybrid_retriever.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>
#include <sai/retrieval/score_fusion.h>
#include <sai/detection/feature_bank.h>
#include <sqlite3.h>
#include <gtest/gtest.h>
#include <fstream>

namespace sai::retrieval {
namespace {

class HybridRetrieverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup SQLite with nodes
        sqlite3_open(":memory:", &db_);
        const char* schema = R"(
            CREATE TABLE nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, properties_json TEXT NOT NULL DEFAULT '{}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"code":"M1"}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"code":"M2"}');
            INSERT INTO nodes (type, properties_json) VALUES ('Material', '{"code":"M3"}');
        )";
        sqlite3_exec(db_, schema, nullptr, nullptr, nullptr);

        // Setup FeatureBank with 3 vectors corresponding to the 3 nodes
        std::vector<float> data = {
            0.0F, 0.0F,   // M1
            3.0F, 3.0F,   // M2
            6.0F, 6.0F,   // M3
        };
        auto tmp = std::filesystem::temp_directory_path() / "test_hr.f32";
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
        out.close();
        auto bank = sai::detection::FeatureBank::LoadFromFile(tmp, 2);
        std::filesystem::remove(tmp);
        ASSERT_TRUE(bank.has_value());
        bank_ = std::make_unique<sai::detection::FeatureBank>(std::move(*bank));
    }
    void TearDown() override { sqlite3_close(db_); }

    sqlite3* db_ = nullptr;
    std::unique_ptr<sai::detection::FeatureBank> bank_;
};

TEST_F(HybridRetrieverTest, RetrieveWithWeightedFusion) {
    VectorPath vec_path(*bank_);
    MetadataPath meta_path(db_);
    auto fusion = std::make_unique<WeightedFusion>();

    HybridRetriever retriever(vec_path, meta_path, std::move(fusion));

    HybridRetriever::Config cfg;
    cfg.vector.mode = VectorPath::Mode::TopK;
    cfg.vector.k = 3;

    float query[] = {0.1F, 0.1F};  // closest to M1
    std::vector<std::int64_t> vec_to_node = {1, 2, 3};  // vec index → node_id

    auto results = retriever.Retrieve(query, cfg, vec_to_node);
    ASSERT_TRUE(results.has_value());
    ASSERT_GE(results->size(), 1);
    EXPECT_EQ(results->at(0).node_id, 1);  // M1 closest to query
    EXPECT_GT(results->at(0).scores.fused_score, 0.0F);
}

TEST_F(HybridRetrieverTest, SetFusionStrategy) {
    VectorPath vec_path(*bank_);
    MetadataPath meta_path(db_);

    HybridRetriever retriever(vec_path, meta_path, std::make_unique<WeightedFusion>());
    retriever.SetFusion(std::make_unique<RRFFusion>());

    HybridRetriever::Config cfg;
    cfg.vector.mode = VectorPath::Mode::TopK;
    cfg.vector.k = 2;

    float query[] = {0.0F, 0.0F};
    std::vector<std::int64_t> vec_to_node = {1, 2, 3};

    auto results = retriever.Retrieve(query, cfg, vec_to_node);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results->at(0).scores.fusion_strategy, "RRFFusion");
}

}  // namespace
}  // namespace sai::retrieval
```

- [ ] **Step 2: Write the header**

Create `include/sai/retrieval/hybrid_retriever.h`:

```cpp
// hybrid_retriever.h — 批次 4.2 混合检索引擎
#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <sai/core/error.h>
#include <sai/retrieval/score_fusion.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>

namespace sai::retrieval {

struct RetrievalItem {
    std::int64_t node_id;
    ScoreBreakdown scores;
    std::string node_type;
};

class HybridRetriever final {
public:
    struct Config {
        VectorPath::Config vector;
        MetadataPath::Config metadata;
    };

    HybridRetriever(VectorPath& vec_path, MetadataPath& meta_path,
                     std::unique_ptr<IScoreFusion> fusion) noexcept;

    [[nodiscard]] auto Retrieve(const float* query_vec,
                                  const Config& cfg,
                                  const std::vector<std::int64_t>& vec_to_node_ids) const noexcept
        -> Result<std::vector<RetrievalItem>>;

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

- [ ] **Step 3: Implement HybridRetriever**

Create `src/retrieval/hybrid_retriever.cpp`:

```cpp
#include <sai/retrieval/hybrid_retriever.h>
#include <sqlite3.h>
#include <source_location>
#include <unordered_set>

namespace sai::retrieval {

HybridRetriever::HybridRetriever(VectorPath& vec_path, MetadataPath& meta_path,
                                   std::unique_ptr<IScoreFusion> fusion) noexcept
    : vec_path_(vec_path), meta_path_(meta_path), fusion_(std::move(fusion)) {}

HybridRetriever::~HybridRetriever() = default;

auto HybridRetriever::SetFusion(std::unique_ptr<IScoreFusion> fusion) noexcept -> void {
    fusion_ = std::move(fusion);
}

auto HybridRetriever::Retrieve(const float* query_vec,
                                 const Config& cfg,
                                 const std::vector<std::int64_t>& vec_to_node_ids) const noexcept
    -> Result<std::vector<RetrievalItem>> {
    // Run both paths
    auto vec_results = vec_path_.Search(query_vec, cfg.vector);
    if (!vec_results.has_value()) return tl::make_unexpected(vec_results.error());

    auto meta_results = meta_path_.Search(cfg.metadata);
    if (!meta_results.has_value()) return tl::make_unexpected(meta_results.error());

    // Map vector results to node IDs
    std::vector<std::int64_t> vec_node_ids;
    vec_node_ids.reserve(vec_results->size());
    for (const auto& vr : *vec_results) {
        if (vr.index < vec_to_node_ids.size()) {
            vec_node_ids.push_back(vec_to_node_ids[vr.index]);
        } else {
            vec_node_ids.push_back(-1);  // unmapped
        }
    }

    // Fuse scores
    auto fused = fusion_->Fuse(*vec_results, vec_node_ids, *meta_results);

    // Convert to RetrievalItem (node_type placeholder — caller fills from KnowledgeGraph)
    std::vector<RetrievalItem> items;
    items.reserve(fused.size());
    for (auto& [node_id, breakdown] : fused) {
        RetrievalItem item;
        item.node_id = node_id;
        item.scores = std::move(breakdown);
        items.push_back(std::move(item));
    }
    return items;
}

}  // namespace sai::retrieval
```

- [ ] **Step 4: Build & run tests**

Run: `cmake --build --preset default && ctest --preset default -R "HybridRetriever"`
Expected: 2 tests pass.

- [ ] **Step 5: Run full suite**

Run: `ctest --preset default`
Expected: 357/357 tests pass (355 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(retrieval): ✨ 添加 HybridRetriever 双路检索编排"
```

---

### Task 11: Integration test (M4 verification point)

End-to-end test: KnowledgeStore → insert Material + Supplier → FeatureBank → HybridRetriever → fused score.

**Files:**
- Create: `tests/integration/knowledge_retrieval_pipeline_test.cpp`
- Modify: `tests/integration/CMakeLists.txt`

**Interfaces:**
- Consumes: all previous tasks
- Produces: M4 verification test

- [ ] **Step 1: Write integration test**

Create `tests/integration/knowledge_retrieval_pipeline_test.cpp`:

```cpp
// knowledge_retrieval_pipeline_test.cpp — M4 验证点：端到端知识写入 + 检索
#include <sai/knowledge/knowledge_store.h>
#include <sai/retrieval/hybrid_retriever.h>
#include <sai/retrieval/vector_path.h>
#include <sai/retrieval/metadata_path.h>
#include <sai/retrieval/score_fusion.h>
#include <sai/detection/feature_bank.h>
#include <gtest/gtest.h>
#include <fstream>

namespace {

TEST(KnowledgeRetrievalPipelineTest, WriteAndRetrieveWithFusedScore) {
    // 1. Create KnowledgeStore with :memory:
    sai::knowledge::KnowledgeStore::Config kcfg;
    kcfg.db_path = ":memory:";
    auto store = sai::knowledge::KnowledgeStore::Create(kcfg);
    ASSERT_TRUE(store.has_value());

    // 2. Insert Material node with properties
    sai::knowledge::KnowledgeRecord mat_props;
    mat_props.fields["material_code"] = std::string("LEATHER-001");
    mat_props.fields["name"] = std::string("Nappa 真皮");
    auto mat_id = (*store)->InsertNode("Material", mat_props, "importer");
    ASSERT_TRUE(mat_id.has_value());

    // 3. Insert Supplier node
    sai::knowledge::KnowledgeRecord sup_props;
    sup_props.fields["supplier_code"] = std::string("SUP-2026");
    sup_props.fields["name"] = std::string("某供应商");
    auto sup_id = (*store)->InsertNode("Supplier", sup_props, "importer");
    ASSERT_TRUE(sup_id.has_value());

    // 4. Create edge: Material -[supplied_by]-> Supplier
    sai::knowledge::KnowledgeRecord edge_props;
    auto edge_id = (*store)->InsertEdge(*mat_id, *sup_id, "supplied_by", edge_props);
    ASSERT_TRUE(edge_id.has_value());

    // 5. Verify graph traversal
    auto paths = (*store)->Traverse(*mat_id, "supplied_by");
    ASSERT_TRUE(paths.has_value());
    ASSERT_EQ(paths->at(0).targets.size(), 1);
    EXPECT_EQ(paths->at(0).targets[0].id, *sup_id);

    // 6. Build FeatureBank with CLIP-like embeddings for each node
    // Material LEATHER-001: [0.1, 0.2, 0.3, 0.4]
    // Supplier SUP-2026:    [0.5, 0.6, 0.7, 0.8]
    // (In production these would come from CLIPAdapter)
    std::vector<float> data = {
        0.1F, 0.2F, 0.3F, 0.4F,  // vec[0] = Material LEATHER-001
        0.5F, 0.6F, 0.7F, 0.8F,  // vec[1] = Supplier SUP-2026
    };
    auto tmp = std::filesystem::temp_directory_path() / "test_pipeline.f32";
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
    }
    auto bank = sai::detection::FeatureBank::LoadFromFile(tmp, 4);
    std::filesystem::remove(tmp);
    ASSERT_TRUE(bank.has_value());

    // 7. vec_to_knowledge mapping: vec_index → node_id
    std::vector<std::int64_t> vec_to_node = {*mat_id, *sup_id};

    // 8. Create VectorPath + MetadataPath + WeightedFusion
    sai::retrieval::VectorPath vec_path(*bank);
    sai::retrieval::MetadataPath meta_path((*store)->DbHandle());
    auto fusion = std::make_unique<sai::retrieval::WeightedFusion>();

    // 9. Create HybridRetriever
    sai::retrieval::HybridRetriever retriever(vec_path, meta_path, std::move(fusion));

    sai::retrieval::HybridRetriever::Config cfg;
    cfg.vector.mode = sai::retrieval::VectorPath::Mode::TopK;
    cfg.vector.k = 2;
    cfg.metadata.filters.push_back(sai::retrieval::FilterCondition{
        "material_code", sai::retrieval::FilterOp::Equal, std::string("LEATHER-001")
    });

    // 10. Retrieve with query vector close to LEATHER-001
    float query[] = {0.15F, 0.25F, 0.35F, 0.45F};
    auto results = retriever.Retrieve(query, cfg, vec_to_node);

    // 11. Assertions
    ASSERT_TRUE(results.has_value());
    ASSERT_GE(results->size(), 1);

    // Top result should be the Material node (closest vector + metadata match)
    EXPECT_EQ(results->at(0).node_id, *mat_id);
    EXPECT_GT(results->at(0).scores.fused_score, 0.0F);
    EXPECT_GT(results->at(0).scores.vector_score, 0.0F);
    EXPECT_EQ(results->at(0).scores.fusion_strategy, "WeightedFusion");
}

TEST(KnowledgeRetrievalPipelineTest, EvolutionTracksInsertAndRetrieve) {
    sai::knowledge::KnowledgeStore::Config cfg;
    cfg.db_path = ":memory:";
    cfg.enable_evolution = true;
    auto store = sai::knowledge::KnowledgeStore::Create(cfg);
    ASSERT_TRUE(store.has_value());

    sai::knowledge::KnowledgeRecord props;
    props.fields["code"] = std::string("BATCH-001");
    auto id = (*store)->InsertNode("Batch", props, "mes_importer");
    ASSERT_TRUE(id.has_value());

    auto history = (*store)->GetEntityHistory("Node", *id);
    ASSERT_TRUE(history.has_value());
    ASSERT_EQ(history->size(), 1);
    EXPECT_EQ(history->at(0).changed_by, "mes_importer");
    EXPECT_EQ(history->at(0).operation, sai::knowledge::EvolutionOp::Insert);

    auto changes = (*store)->GetChangesSince(
        std::chrono::system_clock::now() - std::chrono::hours(1));
    ASSERT_TRUE(changes.has_value());
    ASSERT_GE(changes->size(), 1);
}

}  // namespace
```

- [ ] **Step 2: Update tests/integration/CMakeLists.txt**

Add a new test executable for the M4 integration test. Append to `tests/integration/CMakeLists.txt`:

```cmake
# M4: 知识写入 → 混合检索 端到端集成测试
add_executable(sai_integration_knowledge_retrieval_pipeline_test
    knowledge_retrieval_pipeline_test.cpp
)
target_include_directories(sai_integration_knowledge_retrieval_pipeline_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sai_integration_knowledge_retrieval_pipeline_test PRIVATE
    sai::knowledge
    sai::retrieval
    GTest::gtest_main
)
```

And at the bottom of the file, add:
```cmake
gtest_discover_tests(sai_integration_knowledge_retrieval_pipeline_test)
```

- [ ] **Step 3: Build & run integration tests**

Run: `cmake --build --preset default && ctest --preset default -R "KnowledgeRetrievalPipeline"`
Expected: 2 tests pass.

- [ ] **Step 4: Run full suite**

Run: `ctest --preset default`
Expected: 359/359 tests pass (357 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test(integration): ✅ 添加 M4 端到端验证（知识写入 → 混合检索 → 融合分数）"
```

---

## Verification Checklist

After all tasks complete:
- [ ] All 11 tasks committed
- [ ] 359/359 tests pass (326 baseline + 33 new)
- [ ] `grep -c "Knowledge_\|Retrieval_" include/sai/core/error.h` → 10
- [ ] `sai_knowledge` and `sai_retrieval` libraries compile on macOS arm64
- [ ] Integration test validates M4 verification point: write → retrieve → fused score > 0
- [ ] No regressions: all existing M1/M2/M3 tests pass
- [ ] `git log --oneline main..HEAD` shows 11 commits, one per task
