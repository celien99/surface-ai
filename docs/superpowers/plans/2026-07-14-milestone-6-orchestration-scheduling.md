# Milestone 6 编排调度 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Pipeline orchestration layer (YAML→TaskGraph, 7 stage types, lock-free stage queues, backpressure) and Scheduler (StageType→WorkerPool mapping) that wires M1-M5 modules into a complete async pipeline from frame capture to decision export.

**Architecture:** Pipeline wraps M1 TaskGraph as execution engine. PipelineBuilder translates YAML config → validated PipelineConfig → M1 TaskNode topology. StageQueue<T> (bounded SPSC lock-free ring buffer) connects stages with configurable backpressure. Scheduler maps each StageType to a pre-registered WorkerPool. Concrete stage implementations wrap M2/M3/M5 interfaces.

**Tech Stack:** C++20, yaml-cpp, GoogleTest, M1 TaskGraph/PipelineExecutor/WorkerPool/TaskScheduler, M2 ICamera/PreprocessFn/IExporter, M3 IInferenceEngine/IDetector, M5 RuleEngine/FactBuilder/IReasoner

## Global Constraints

- All public types in `sai::pipeline` namespace — never `sai::` directly
- Error handling via `Result<T>` (tl::expected), exceptions only for construction failures
- Error codes append-only after `Reasoner_ScoreComputationFailed`, never reorder
- Headers use `#pragma once`, include paths mirror module structure: `#include "sai/pipeline/pipeline.h"`
- CUDA-gated and Linux-gated code excluded from local macOS build
- M6 Pipeline **wraps** M1 TaskGraph, does not replace it
- Stage queues are bounded SPSC lock-free ring buffers (MPSC for fan-in)
- Default backpressure: Block. Capture stage overridable to DropOldest
- Single-frame Process() failure → skip frame, do NOT halt pipeline
- Spec: `docs/superpowers/specs/2026-07-14-milestone-6-orchestration-scheduling-design.md`

---

### Task 1: CMake scaffold + ErrorCode append

**Files:**
- Modify: `CMakeLists.txt:47-52` (add pipeline and scheduler subdirectories after reasoner)
- Modify: `include/sai/core/error.h:73` (append M6 error codes after `Reasoner_ScoreComputationFailed`)
- Create: `src/pipeline/CMakeLists.txt`
- Create: `src/scheduler/CMakeLists.txt`
- Create: `tests/pipeline/CMakeLists.txt`
- Create: `tests/scheduler/CMakeLists.txt`

**Interfaces:**
- Produces: `sai::pipeline` (static lib), `sai::scheduler` (static lib), 7 new ErrorCode members

- [ ] **Step 1: Append M6 error codes to error.h**

Insert after line 73 (`Reasoner_ScoreComputationFailed,`):

```cpp
    // Pipeline & Scheduler (M6)
    Pipeline_InvalidConfig,
    Pipeline_StageTypeMismatch,
    Pipeline_StageInitFailed,
    Pipeline_InvalidState,
    Pipeline_QueueFull,
    Scheduler_PoolNotFound,
    Scheduler_QueueCreateFailed,
```

- [ ] **Step 2: Run build to verify error.h compiles**

```bash
cmake --build --preset default 2>&1 | tail -5
```

Expected: `[N/N] Linking CXX static library libsai_reasoner.a` (no error.h compilation errors)

- [ ] **Step 3: Create src/pipeline/CMakeLists.txt**

```cmake
add_library(sai_pipeline STATIC)
add_library(sai::pipeline ALIAS sai_pipeline)

target_sources(sai_pipeline PRIVATE
    pipeline.cpp
    pipeline_builder.cpp
    stage_factory.cpp
    capture_stage.cpp
    preprocess_stage.cpp
    inference_stage.cpp
    detect_stage.cpp
    rule_eval_stage.cpp
    reason_stage.cpp
    export_stage.cpp
)

target_include_directories(sai_pipeline PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sai_pipeline PUBLIC
    sai::core
    sai::runtime
    sai::infra
    yaml-cpp::yaml-cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(sai_pipeline PUBLIC
        sai::device
        sai::image
        sai::inference
        sai::detection
        sai::rule
        sai::reasoner
        sai::io
    )
else()
    target_link_libraries(sai_pipeline PUBLIC
        sai::device
        sai::image
        sai::io
    )
endif()
```

- [ ] **Step 4: Create src/scheduler/CMakeLists.txt**

```cmake
add_library(sai_scheduler STATIC)
add_library(sai::scheduler ALIAS sai_scheduler)

target_sources(sai_scheduler PRIVATE
    scheduler.cpp
)

target_include_directories(sai_scheduler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sai_scheduler PUBLIC
    sai::core
    sai::runtime
)
```

- [ ] **Step 5: Create tests/pipeline/CMakeLists.txt**

```cmake
find_package(GTest REQUIRED)

add_executable(sai_pipeline_test)
target_sources(sai_pipeline_test PRIVATE
    pipeline_config_test.cpp
    stage_queue_test.cpp
    pipeline_builder_test.cpp
    pipeline_test.cpp
)

target_include_directories(sai_pipeline_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sai_pipeline_test PRIVATE
    sai::pipeline
    sai::scheduler
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sai_pipeline_test)
```

- [ ] **Step 6: Create tests/scheduler/CMakeLists.txt**

```cmake
find_package(GTest REQUIRED)

add_executable(sai_scheduler_test)
target_sources(sai_scheduler_test PRIVATE
    scheduler_test.cpp
)

target_include_directories(sai_scheduler_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sai_scheduler_test PRIVATE
    sai::scheduler
    sai::pipeline
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sai_scheduler_test)
```

- [ ] **Step 7: Wire into root CMakeLists.txt**

Insert before `add_subdirectory(tests/integration)` (line ~53):

```cmake
add_subdirectory(src/pipeline)
add_subdirectory(tests/pipeline)
add_subdirectory(src/scheduler)
add_subdirectory(tests/scheduler)
```

- [ ] **Step 8: Configure and verify scaffold compiles**

```bash
cmake --preset default 2>&1 | tail -5
```

Expected: `-- Configuring done` (no CMake errors)

- [ ] **Step 9: Commit**

```bash
git add CMakeLists.txt include/sai/core/error.h \
    src/pipeline/ src/scheduler/ tests/pipeline/ tests/scheduler/
git commit -m "chore(pipeline,scheduler): 🔧 M6 CMake 骨架 + ErrorCode 追加（Pipeline_*/Scheduler_*）"
```

---

### Task 2: PipelineConfig 类型定义（StageType, BackpressurePolicy, PipelineConfig）

**Files:**
- Create: `include/sai/pipeline/pipeline_config.h`

**Interfaces:**
- Produces: `sai::pipeline::StageType` (enum), `sai::pipeline::BackpressurePolicy` (enum), `sai::pipeline::StageConfig`, `sai::pipeline::BackpressureConfig`, `sai::pipeline::PipelineConfig`

- [ ] **Step 1: Write failing test — PipelineConfig YAML parsing**

Create `tests/pipeline/pipeline_config_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <sai/pipeline/pipeline_config.h>
#include <yaml-cpp/yaml.h>

namespace sai::pipeline {
namespace {

TEST(PipelineConfigTest, ParseMinimalPipeline) {
    const char* yaml = R"(
pipeline:
  name: "test"
  version: "1.0"
  stages:
    - id: capture
      type: Capture
      depends_on: []
    - id: export
      type: Export
      depends_on: [capture]
)";
    YAML::Node root = YAML::Load(yaml);
    auto node = root["pipeline"];
    ASSERT_TRUE(node.IsDefined());

    auto name = node["name"].as<std::string>();
    EXPECT_EQ(name, "test");
    auto version = node["version"].as<std::string>();
    EXPECT_EQ(version, "1.0");
    auto stages = node["stages"];
    ASSERT_EQ(stages.size(), 2);
    EXPECT_EQ(stages[0]["id"].as<std::string>(), "capture");
    EXPECT_EQ(stages[0]["type"].as<std::string>(), "Capture");
}

TEST(PipelineConfigTest, BackpressureDefaults) {
    BackpressureConfig bp;
    EXPECT_EQ(bp.default_policy, BackpressurePolicy::Block);
    EXPECT_TRUE(bp.stage_overrides.empty());
}

TEST(PipelineConfigTest, StageTypeFromString) {
    EXPECT_EQ(StageTypeFromString("Capture"), StageType::Capture);
    EXPECT_EQ(StageTypeFromString("Preprocess"), StageType::Preprocess);
    EXPECT_EQ(StageTypeFromString("Inference"), StageType::Inference);
    EXPECT_EQ(StageTypeFromString("Detect"), StageType::Detect);
    EXPECT_EQ(StageTypeFromString("RuleEval"), StageType::RuleEval);
    EXPECT_EQ(StageTypeFromString("Reason"), StageType::Reason);
    EXPECT_EQ(StageTypeFromString("Export"), StageType::Export);
    EXPECT_EQ(StageTypeFromString("Custom"), StageType::Custom);
}

TEST(PipelineConfigTest, StageTypeFromStringInvalid) {
    EXPECT_THROW(StageTypeFromString("Nonexistent"), std::invalid_argument);
}

TEST(PipelineConfigTest, BackpressurePolicyFromString) {
    EXPECT_EQ(BackpressurePolicyFromString("block"), BackpressurePolicy::Block);
    EXPECT_EQ(BackpressurePolicyFromString("drop_oldest"), BackpressurePolicy::DropOldest);
    EXPECT_EQ(BackpressurePolicyFromString("degrade"), BackpressurePolicy::Degrade);
}

TEST(PipelineConfigTest, BackpressurePolicyFromStringInvalid) {
    EXPECT_THROW(BackpressurePolicyFromString("invalid"), std::invalid_argument);
}

} // namespace
} // namespace sai::pipeline
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd build/default && ctest -R "PipelineConfig" --output-on-failure
```

Expected: FAIL — `pipeline_config.h` not found

- [ ] **Step 3: Create include/sai/pipeline/pipeline_config.h**

```cpp
#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace sai::pipeline {

enum class StageType {
    Capture,
    Preprocess,
    Inference,
    Detect,
    RuleEval,
    Reason,
    Export,
    Custom
};

inline auto StageTypeFromString(std::string_view s) -> StageType {
    if (s == "Capture")     return StageType::Capture;
    if (s == "Preprocess")  return StageType::Preprocess;
    if (s == "Inference")   return StageType::Inference;
    if (s == "Detect")      return StageType::Detect;
    if (s == "RuleEval")    return StageType::RuleEval;
    if (s == "Reason")      return StageType::Reason;
    if (s == "Export")      return StageType::Export;
    if (s == "Custom")      return StageType::Custom;
    throw std::invalid_argument(std::string("Unknown StageType: ") + std::string(s));
}

enum class BackpressurePolicy {
    Block,
    DropOldest,
    Degrade  // v1 预留，实现仅 Block/DropOldest
};

inline auto BackpressurePolicyFromString(std::string_view s) -> BackpressurePolicy {
    if (s == "block")        return BackpressurePolicy::Block;
    if (s == "drop_oldest")  return BackpressurePolicy::DropOldest;
    if (s == "degrade")      return BackpressurePolicy::Degrade;
    throw std::invalid_argument(std::string("Unknown BackpressurePolicy: ") + std::string(s));
}

struct StageConfig {
    std::string id;
    StageType type;
    std::vector<std::string> depends_on;
    YAML::Node config;
    BackpressurePolicy backpressure = BackpressurePolicy::Block;  // 加载时从 PipelineConfig 合并
    std::optional<std::size_t> queue_capacity;
};

struct BackpressureConfig {
    BackpressurePolicy default_policy = BackpressurePolicy::Block;
    std::map<std::string, BackpressurePolicy> stage_overrides;
};

struct PipelineConfig {
    std::string name;
    std::string version;
    BackpressureConfig backpressure;
    std::vector<StageConfig> stages;
};

}  // namespace sai::pipeline
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd build/default && cmake --build --preset default 2>&1 | tail -5
ctest -R "PipelineConfig" --output-on-failure
```

Expected: 6/6 PipelineConfig tests PASS

- [ ] **Step 5: Commit**

```bash
git add include/sai/pipeline/pipeline_config.h tests/pipeline/pipeline_config_test.cpp
git commit -m "feat(pipeline): ✨ PipelineConfig 类型定义（StageType + BackpressurePolicy + 字符串转换）"
```

---

### Task 3: IStageNode 接口 + StageInput/StageOutput variant

**Files:**
- Create: `include/sai/pipeline/stage_node.h`

**Interfaces:**
- Produces: `sai::pipeline::IStageNode`, `sai::pipeline::StageInput`, `sai::pipeline::StageOutput`
- Consumes: M1 `Object` (1.1), M2 `RawImage`/`SurfaceImage`, M3 `DetectionResult`, M5 `rule::ResolvedRule`/`rule::ReasoningResult`

- [ ] **Step 1: Write failing test — IStageNode lifecycle**

Create `tests/pipeline/pipeline_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <sai/pipeline/stage_node.h>
#include <sai/core/context.h>

namespace sai::pipeline {
namespace {

class MockStage : public IStageNode {
public:
    explicit MockStage(std::string id, StageType type)
        : id_(std::move(id)), type_(type) {}

    auto GetType() const noexcept -> StageType override { return type_; }
    auto GetId() const -> std::string_view override { return id_; }

    auto OnInitialize(sai::Context&) -> Result<void> override {
        init_called_ = true;
        return {};
    }
    auto OnStart(sai::Context&) -> Result<void> override {
        start_called_ = true;
        return {};
    }
    auto OnStop(sai::Context&) -> Result<void> override {
        stop_called_ = true;
        return {};
    }
    auto Process(StageInput input) -> Result<StageOutput> override {
        process_called_ = true;
        last_input_ = std::move(input);
        // Echo input as output (passthrough)
        return std::visit([](auto&& val) -> StageOutput {
            return std::forward<decltype(val)>(val);
        }, last_input_);
    }

    bool init_called_ = false;
    bool start_called_ = false;
    bool stop_called_ = false;
    bool process_called_ = false;
    StageInput last_input_;
    std::string id_;
    StageType type_;
};

TEST(IStageNodeTest, LifecycleSequence) {
    MockStage stage("test_stage", StageType::Capture);
    sai::Context ctx;

    EXPECT_FALSE(stage.init_called_);
    ASSERT_TRUE(stage.OnInitialize(ctx).has_value());
    EXPECT_TRUE(stage.init_called_);

    EXPECT_FALSE(stage.start_called_);
    ASSERT_TRUE(stage.OnStart(ctx).has_value());
    EXPECT_TRUE(stage.start_called_);

    EXPECT_FALSE(stage.stop_called_);
    ASSERT_TRUE(stage.OnStop(ctx).has_value());
    EXPECT_TRUE(stage.stop_called_);
}

TEST(IStageNodeTest, ProcessPassthrough) {
    MockStage stage("test", StageType::Capture);
    sai::Context ctx;
    stage.OnInitialize(ctx);
    stage.OnStart(ctx);

    RawImage mock_image = RawImage::FromOwnedBuffer(
        std::vector<uint8_t>{1, 2, 3}, sai::image::ImageMeta{});

    StageInput input(mock_image);
    auto result = stage.Process(std::move(input));
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(stage.process_called_);
    EXPECT_TRUE(std::holds_alternative<RawImage>(result.value()));
}

TEST(IStageNodeTest, GetTypeAndId) {
    MockStage stage("my_id", StageType::Reason);
    EXPECT_EQ(stage.GetType(), StageType::Reason);
    EXPECT_EQ(stage.GetId(), "my_id");
}

} // namespace
} // namespace sai::pipeline
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd build/default && ctest -R "IStageNode" --output-on-failure
```

Expected: FAIL — `stage_node.h` not found

- [ ] **Step 3: Create include/sai/pipeline/stage_node.h**

```cpp
#pragma once

#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/object.h>
#include <sai/core/context.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>
#include <sai/detection/detection_result.h>
#include <sai/rule/rule_engine.h>
#include <sai/rule/fact_base.h>
#include <sai/reasoner/reasoner.h>

namespace sai::pipeline {

using StageInput = std::variant<
    RawImage,
    SurfaceImage,
    DetectionResult,
    std::vector<rule::ResolvedRule>,
    rule::ReasoningResult
>;

using StageOutput = StageInput;

class IStageNode : public Object {
public:
    virtual auto GetType() const noexcept -> StageType = 0;
    virtual auto GetId() const -> std::string_view = 0;
    virtual auto OnInitialize(Context&) -> Result<void> = 0;
    virtual auto OnStart(Context&) -> Result<void> = 0;
    virtual auto OnStop(Context&) -> Result<void> = 0;
    virtual auto Process(StageInput) -> Result<StageOutput> = 0;
};

}  // namespace sai::pipeline
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd build/default && cmake --build --preset default 2>&1 | tail -5
ctest -R "IStageNode" --output-on-failure
```

Expected: 3/3 IStageNode tests PASS

- [ ] **Step 5: Commit**

```bash
git add include/sai/pipeline/stage_node.h tests/pipeline/pipeline_test.cpp
git commit -m "feat(pipeline): ✨ IStageNode 接口 + StageInput/StageOutput variant"
```

---

### Task 4: StageQueue<T> — bounded SPSC lock-free ring buffer

**Files:**
- Create: `include/sai/pipeline/stage_queue.h`

**Interfaces:**
- Produces: `sai::pipeline::StageQueue<T>` (header-only template)
  - `static auto Create(size_t capacity, BackpressurePolicy) -> Result<std::unique_ptr<StageQueue<T>>>`
  - `auto TryPush(std::unique_ptr<T>) -> bool`
  - `auto PushBlocking(std::unique_ptr<T>) -> void`
  - `auto TryPop() -> std::unique_ptr<T>`
  - `auto PopBlocking() -> std::unique_ptr<T>`
  - `auto Depth() const noexcept -> size_t`
  - `auto Capacity() const noexcept -> size_t`

- [ ] **Step 1: Write failing test — StageQueue SPSC correctness**

Create `tests/pipeline/stage_queue_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <sai/pipeline/stage_queue.h>
#include <thread>
#include <atomic>

namespace sai::pipeline {
namespace {

using IntQueue = StageQueue<int>;

TEST(StageQueueTest, CreateAndCapacity) {
    auto q = IntQueue::Create(8, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());
    EXPECT_EQ((*q)->Capacity(), 8);
    EXPECT_EQ((*q)->Depth(), 0);
}

TEST(StageQueueTest, CreateZeroCapacityFails) {
    auto q = IntQueue::Create(0, BackpressurePolicy::Block);
    EXPECT_FALSE(q.has_value());
    EXPECT_EQ(q.error().code, ErrorCode::Scheduler_QueueCreateFailed);
}

TEST(StageQueueTest, PushPopSingleElement) {
    auto q = IntQueue::Create(4, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    bool pushed = (*q)->TryPush(std::make_unique<int>(42));
    EXPECT_TRUE(pushed);
    EXPECT_EQ((*q)->Depth(), 1);

    auto val = (*q)->TryPop();
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(*val, 42);
    EXPECT_EQ((*q)->Depth(), 0);
}

TEST(StageQueueTest, TryPushFullQueueReturnsFalse) {
    auto q = IntQueue::Create(2, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(1)));
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(2)));
    EXPECT_FALSE((*q)->TryPush(std::make_unique<int>(3)));  // full
    EXPECT_EQ((*q)->Depth(), 2);
}

TEST(StageQueueTest, TryPopEmptyQueueReturnsNull) {
    auto q = IntQueue::Create(4, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    auto val = (*q)->TryPop();
    EXPECT_EQ(val, nullptr);
}

TEST(StageQueueTest, FifoOrdering) {
    auto q = IntQueue::Create(4, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    for (int i = 1; i <= 4; ++i) {
        EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(i)));
    }
    for (int i = 1; i <= 4; ++i) {
        auto val = (*q)->TryPop();
        ASSERT_NE(val, nullptr);
        EXPECT_EQ(*val, i);
    }
}

TEST(StageQueueTest, DropOldestBackpressure) {
    auto q = IntQueue::Create(2, BackpressurePolicy::DropOldest);
    ASSERT_TRUE(q.has_value());

    // Fill queue: [1, 2]
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(1)));
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(2)));
    EXPECT_EQ((*q)->Depth(), 2);

    // Push 3 when full: drop oldest (1), push 3 → [2, 3]
    bool pushed = (*q)->TryPush(std::make_unique<int>(3));
    EXPECT_TRUE(pushed);
    EXPECT_EQ((*q)->Depth(), 2);

    auto first = (*q)->TryPop();
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(*first, 2);  // 1 was dropped

    auto second = (*q)->TryPop();
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(*second, 3);
}

TEST(StageQueueTest, SingleProducerSingleConsumer) {
    constexpr size_t kIters = 10'000;
    auto q = IntQueue::Create(64, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    std::atomic<bool> producer_done{false};
    std::atomic<size_t> sum_consumed{0};
    size_t sum_produced = 0;

    std::thread producer([&]() {
        for (size_t i = 0; i < kIters; ++i) {
            sum_produced += i;
            while (!(*q)->TryPush(std::make_unique<int>(static_cast<int>(i)))) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        for (size_t i = 0; i < kIters; ++i) {
            std::unique_ptr<int> val;
            while (!(val = (*q)->TryPop())) {
                std::this_thread::yield();
            }
            sum_consumed.fetch_add(*val, std::memory_order_relaxed);
        }
    });

    producer.join();
    consumer.join();

    EXPECT_TRUE(producer_done.load());
    EXPECT_EQ(sum_produced, sum_consumed.load());
    EXPECT_EQ((*q)->Depth(), 0);
}

} // namespace
} // namespace sai::pipeline
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd build/default && ctest -R "StageQueue" --output-on-failure
```

Expected: FAIL — `stage_queue.h` not found

- [ ] **Step 3: Create include/sai/pipeline/stage_queue.h**

```cpp
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <new>

#include <sai/core/error.h>

namespace sai::pipeline {

namespace detail {

// Cache line padding to prevent false sharing between head and tail.
// arm64 cache line = 128 bytes typically; x86-64 = 64 bytes. Use 128 to
// cover both — the extra 64 bytes is cheap insurance.
static constexpr size_t kCacheLineSize = 128;

// Bounded SPSC ring buffer with backpressure. Single-producer-single-consumer
// by design — the caller is responsible for ensuring only one thread calls
// TryPush/PushBlocking and only one calls TryPop/PopBlocking.
// For MPSC (fan-in), wrap with a mutex on the producer side.
template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity, BackpressurePolicy policy)
        : capacity_(capacity), policy_(policy)
        , buffer_(std::make_unique<std::unique_ptr<T>[]>(capacity))
        , head_(0), tail_(0) {}

    // Non-blocking push. Returns false if full.
    auto TryPush(std::unique_ptr<T> item) noexcept -> bool {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % capacity_;

        if (next_tail == head_.load(std::memory_order_acquire)) {
            // Full
            if (policy_ == BackpressurePolicy::DropOldest) {
                // Drop the oldest: advance head
                size_t current_head = head_.load(std::memory_order_relaxed);
                head_.store((current_head + 1) % capacity_, std::memory_order_release);
                // Now we have space — write at tail
                buffer_[current_tail] = std::move(item);
                tail_.store(next_tail, std::memory_order_release);
                return true;
            }
            return false;
        }

        buffer_[current_tail] = std::move(item);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    // Non-blocking pop. Returns nullptr if empty.
    auto TryPop() noexcept -> std::unique_ptr<T> {
        size_t current_head = head_.load(std::memory_order_relaxed);

        if (current_head == tail_.load(std::memory_order_acquire)) {
            return nullptr;  // Empty
        }

        auto item = std::move(buffer_[current_head]);
        head_.store((current_head + 1) % capacity_, std::memory_order_release);
        return item;
    }

    auto Capacity() const noexcept -> size_t { return capacity_; }

    auto Depth() const noexcept -> size_t {
        size_t t = tail_.load(std::memory_order_acquire);
        size_t h = head_.load(std::memory_order_acquire);
        if (t >= h) return t - h;
        return capacity_ - h + t;
    }

    // For exclusive use by PushBlocking/PopBlocking (external CV)
    auto IsFull() const noexcept -> bool {
        size_t next = (tail_.load(std::memory_order_relaxed) + 1) % capacity_;
        return next == head_.load(std::memory_order_relaxed);
    }
    auto IsEmpty() const noexcept -> bool {
        return head_.load(std::memory_order_relaxed)
            == tail_.load(std::memory_order_relaxed);
    }

private:
    size_t capacity_;
    BackpressurePolicy policy_;
    std::unique_ptr<std::unique_ptr<T>[]> buffer_;

    // Padded to separate cache lines for head and tail to avoid false sharing
    alignas(kCacheLineSize) std::atomic<size_t> head_;
    alignas(kCacheLineSize) std::atomic<size_t> tail_;
};

}  // namespace detail

template <typename T>
class StageQueue {
public:
    static auto Create(size_t capacity, BackpressurePolicy policy)
        -> Result<std::unique_ptr<StageQueue>> {
        if (capacity == 0) {
            return ErrorInfo{
                ErrorCode::Scheduler_QueueCreateFailed,
                "Queue capacity must be > 0"
            };
        }
        return std::unique_ptr<StageQueue>(
            new StageQueue(capacity, policy));
    }

    auto TryPush(std::unique_ptr<T> item) -> bool {
        return ring_.TryPush(std::move(item));
    }

    auto PushBlocking(std::unique_ptr<T> item) -> void {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !ring_.IsFull(); });
        ring_.TryPush(std::move(item));  // guaranteed to succeed
        cv_.notify_one();  // wake pop side
    }

    auto TryPop() -> std::unique_ptr<T> {
        return ring_.TryPop();
    }

    auto PopBlocking() -> std::unique_ptr<T> {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return !ring_.IsEmpty(); });
        auto item = ring_.TryPop();  // guaranteed to succeed
        cv_.notify_one();  // wake push side
        return item;
    }

    auto Depth() const noexcept -> size_t { return ring_.Depth(); }
    auto Capacity() const noexcept -> size_t { return ring_.Capacity(); }

private:
    StageQueue(size_t capacity, BackpressurePolicy policy)
        : ring_(capacity, policy) {}

    detail::RingBuffer<T> ring_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace sai::pipeline
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd build/default && cmake --build --preset default 2>&1 | tail -5
ctest -R "StageQueue" --output-on-failure
```

Expected: 8/8 StageQueue tests PASS (including SPSC thread test)

- [ ] **Step 5: Commit**

```bash
git add include/sai/pipeline/stage_queue.h tests/pipeline/stage_queue_test.cpp
git commit -m "feat(pipeline): ✨ StageQueue<T> bounded SPSC lock-free ring buffer + BackpressurePolicy"
```

---

### Task 5: PipelineBuilder — YAML 解析 + 校验 + PipelineConfig 构建

**Files:**
- Create: `src/pipeline/pipeline_builder.h`
- Create: `src/pipeline/pipeline_builder.cpp`

**Interfaces:**
- Produces (internal): `sai::pipeline::PipelineBuilder`
  - `static auto ParseFromYAML(std::filesystem::path) -> Result<PipelineConfig>`
  - `static auto Validate(const PipelineConfig&) -> Result<void>`
- Depends on: M1 `TaskGraph`/`TaskNode`, yaml-cpp

- [ ] **Step 1: Write failing test — PipelineBuilder::ParseFromYAML**

Create `tests/pipeline/pipeline_builder_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sai/pipeline/pipeline_config.h>
#include <sai/core/error.h>

// PipelineBuilder is internal (src/pipeline/), so we test via
// Pipeline::LoadFromYAML in Task 7. For now, test PipelineConfig
// construction manually (unit test for the types, integration via
// Pipeline later).

namespace sai::pipeline {
namespace {

// Helper: write a YAML string to a temp file
std::filesystem::path WriteTempYaml(const std::string& content) {
    auto path = std::filesystem::temp_directory_path() / "test_pipeline.yaml";
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

TEST(PipelineConfigTest, ParseValidConfig) {
    const char* yaml = R"(
pipeline:
  name: "test"
  version: "1.0"
  stages:
    - id: capture
      type: Capture
      depends_on: []
    - id: export
      type: Export
      depends_on: [capture]
)";
    YAML::Node root = YAML::Load(yaml);
    auto pipe = root["pipeline"];
    ASSERT_TRUE(pipe.IsDefined());
    EXPECT_EQ(pipe["name"].as<std::string>(), "test");
    EXPECT_EQ(pipe["version"].as<std::string>(), "1.0");
}

TEST(PipelineConfigTest, ParseWithBackpressureOverride) {
    const char* yaml = R"(
pipeline:
  name: "test"
  version: "1.0"
  backpressure:
    default: block
    stage_overrides:
      capture: drop_oldest
  stages:
    - id: capture
      type: Capture
      depends_on: []
    - id: export
      type: Export
      depends_on: [capture]
)";
    YAML::Node root = YAML::Load(yaml);
    auto bp = root["pipeline"]["backpressure"];
    EXPECT_EQ(bp["default"].as<std::string>(), "block");
    EXPECT_EQ(bp["stage_overrides"]["capture"].as<std::string>(), "drop_oldest");
}

} // namespace
} // namespace sai::pipeline
```

Note: PipelineBuilder is internal (`src/pipeline/pipeline_builder.h`), not in the public header path. Full integration through `Pipeline::LoadFromYAML` comes in Task 7. This task adds the YAML parsing test above as an extension of `pipeline_config_test.cpp` and the implementation.

- [ ] **Step 2: Append test to existing file**

```bash
cat tests/pipeline/pipeline_config_test.cpp
```

The test from Step 1 goes into `tests/pipeline/pipeline_config_test.cpp` alongside Task 2's tests.

- [ ] **Step 3: Create src/pipeline/pipeline_builder.h (internal)**

```cpp
#pragma once

#include <filesystem>
#include <vector>

#include <sai/core/error.h>
#include <sai/pipeline/pipeline_config.h>

namespace sai::pipeline {

// Internal: translates YAML → PipelineConfig, validates, builds TaskGraph.
// Not exposed in include/sai/ — Pipeline::LoadFromYAML is the public entry point.
class PipelineBuilder {
public:
    static auto ParseFromYAML(std::filesystem::path yaml_path)
        -> Result<PipelineConfig>;

    static auto Validate(const PipelineConfig& config) -> Result<void>;

private:
    // Topological sort (Kahn's algorithm) — returns true if acyclic.
    static auto IsAcyclic(const std::vector<StageConfig>& stages) -> bool;

    // Map from StageType → expected output type index in StageOutput variant
    static auto OutputTypeIndex(StageType) -> std::size_t;
    static auto InputTypeIndex(StageType) -> std::size_t;

    // Check adjacent stages have compatible types
    static auto CheckTypeCompatibility(
        const StageConfig& upstream,
        const StageConfig& downstream) -> bool;

    static auto ParseStageConfig(const YAML::Node& node,
        const BackpressureConfig& bp) -> Result<StageConfig>;

    static auto MergeBackpressure(const BackpressureConfig& bp,
        const std::string& stage_id) -> BackpressurePolicy;
};

}  // namespace sai::pipeline
```

- [ ] **Step 4: Create src/pipeline/pipeline_builder.cpp**

```cpp
#include "pipeline_builder.h"

#include <fstream>
#include <set>
#include <queue>

#include <sai/core/error.h>

namespace sai::pipeline {

auto PipelineBuilder::ParseFromYAML(std::filesystem::path yaml_path)
    -> Result<PipelineConfig> {
    try {
        YAML::Node root = YAML::LoadFile(yaml_path.string());
        auto pipe = root["pipeline"];
        if (!pipe.IsDefined()) {
            return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
                "YAML root must contain 'pipeline' key"};
        }

        PipelineConfig config;
        config.name = pipe["name"].as<std::string>();
        config.version = pipe["version"].as<std::string>();

        // Parse backpressure
        if (auto bp = pipe["backpressure"]; bp.IsDefined()) {
            auto default_str = bp["default"].as<std::string>("block");
            config.backpressure.default_policy =
                BackpressurePolicyFromString(default_str);

            if (auto overrides = bp["stage_overrides"]; overrides.IsDefined()) {
                for (auto it = overrides.begin(); it != overrides.end(); ++it) {
                    config.backpressure.stage_overrides[
                        it->first.as<std::string>()] =
                        BackpressurePolicyFromString(it->second.as<std::string>());
                }
            }
        }

        // Parse stages
        auto stages_node = pipe["stages"];
        for (auto stage_node : stages_node) {
            auto stage = ParseStageConfig(stage_node, config.backpressure);
            if (!stage.has_value()) return std::move(stage).error();
            config.stages.push_back(std::move(*stage));
        }

        return config;
    } catch (const YAML::Exception& e) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidConfig, e.what()};
    } catch (const std::invalid_argument& e) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidConfig, e.what()};
    }
}

auto PipelineBuilder::Validate(const PipelineConfig& config) -> Result<void> {
    if (config.name.empty()) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "pipeline.name is required"};
    }
    if (config.stages.empty()) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "pipeline must have at least one stage"};
    }

    // Check unique ids
    std::set<std::string> ids;
    for (auto& s : config.stages) {
        if (s.id.empty()) {
            return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
                "stage id must be non-empty"};
        }
        if (!ids.insert(s.id).second) {
            return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
                "duplicate stage id: " + s.id};
        }
    }

    // Check depends_on references exist
    for (auto& s : config.stages) {
        for (auto& dep : s.depends_on) {
            if (ids.find(dep) == ids.end()) {
                return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
                    "stage '" + s.id + "' depends_on '" + dep
                    + "' which does not exist"};
            }
        }
    }

    // Check acyclic
    if (!IsAcyclic(config.stages)) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "pipeline has a cyclic dependency"};
    }

    // Check entry: at least one stage with empty depends_on
    bool has_entry = false;
    for (auto& s : config.stages) {
        if (s.depends_on.empty()) { has_entry = true; break; }
    }
    if (!has_entry) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "no entry stage (empty depends_on) found"};
    }

    // Check exit: at least one stage not depended on by any other
    std::set<std::string> depended_on;
    for (auto& s : config.stages) {
        for (auto& d : s.depends_on) depended_on.insert(d);
    }
    bool has_exit = false;
    for (auto& s : config.stages) {
        if (depended_on.find(s.id) == depended_on.end()) {
            has_exit = true; break;
        }
    }
    if (!has_exit) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "no exit stage found"};
    }

    // Check type compatibility for each edge
    for (auto& downstream : config.stages) {
        for (auto& dep_id : downstream.depends_on) {
            // Find upstream
            auto it = std::find_if(config.stages.begin(), config.stages.end(),
                [&](auto& s) { return s.id == dep_id; });
            if (it != config.stages.end()) {
                if (!CheckTypeCompatibility(*it, downstream)) {
                    return ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
                        "type mismatch: " + it->id + " → " + downstream.id};
                }
            }
        }
    }

    return {};
}

// Private helpers

auto PipelineBuilder::IsAcyclic(const std::vector<StageConfig>& stages) -> bool {
    // Kahn's algorithm
    std::map<std::string, int> in_degree;
    std::map<std::string, std::vector<std::string>> adj;

    for (auto& s : stages) {
        in_degree[s.id] = 0;  // ensure all nodes in map
    }
    for (auto& s : stages) {
        for (auto& dep : s.depends_on) {
            adj[dep].push_back(s.id);
            in_degree[s.id]++;
        }
    }

    std::queue<std::string> q;
    for (auto& [id, deg] : in_degree) {
        if (deg == 0) q.push(id);
    }

    size_t visited = 0;
    while (!q.empty()) {
        auto id = q.front(); q.pop();
        visited++;
        for (auto& next : adj[id]) {
            if (--in_degree[next] == 0) q.push(next);
        }
    }

    return visited == stages.size();
}

auto PipelineBuilder::CheckTypeCompatibility(
    const StageConfig& upstream, const StageConfig& downstream) -> bool {
    return OutputTypeIndex(upstream.type) == InputTypeIndex(downstream.type);
}

auto PipelineBuilder::OutputTypeIndex(StageType t) -> std::size_t {
    switch (t) {
        case StageType::Capture:     return 0;  // RawImage
        case StageType::Preprocess:  return 1;  // SurfaceImage
        case StageType::Inference:   return 2;  // DetectionResult
        case StageType::Detect:      return 2;  // DetectionResult
        case StageType::RuleEval:    return 3;  // ResolvedRule[]
        case StageType::Reason:      return 4;  // ReasoningResult
        case StageType::Export:      return 4;  // ReasoningResult (in)
        case StageType::Custom:      return 0;  // 柔性，不在此校验
    }
    return 0;
}

auto PipelineBuilder::InputTypeIndex(StageType t) -> std::size_t {
    switch (t) {
        case StageType::Capture:     return 0;  // RawImage (external Submit)
        case StageType::Preprocess:  return 0;  // RawImage
        case StageType::Inference:   return 1;  // SurfaceImage
        case StageType::Detect:      return 2;  // DetectionResult
        case StageType::RuleEval:    return 2;  // DetectionResult
        case StageType::Reason:      return 3;  // ResolvedRule[]
        case StageType::Export:      return 4;  // ReasoningResult
        case StageType::Custom:      return 0;  // 柔性
    }
    return 0;
}

auto PipelineBuilder::ParseStageConfig(const YAML::Node& node,
    const BackpressureConfig& bp) -> Result<StageConfig> {
    StageConfig stage;
    stage.id = node["id"].as<std::string>();
    stage.type = StageTypeFromString(node["type"].as<std::string>());

    if (auto deps = node["depends_on"]; deps.IsDefined()) {
        for (auto d : deps) {
            stage.depends_on.push_back(d.as<std::string>());
        }
    }

    stage.config = node["config"];  // 透传

    // Merge backpressure: per-stage override > default
    stage.backpressure = MergeBackpressure(bp, stage.id);

    if (auto qc = node["queue_capacity"]; qc.IsDefined()) {
        stage.queue_capacity = qc.as<std::size_t>();
    }

    return stage;
}

auto PipelineBuilder::MergeBackpressure(const BackpressureConfig& bp,
    const std::string& stage_id) -> BackpressurePolicy {
    auto it = bp.stage_overrides.find(stage_id);
    if (it != bp.stage_overrides.end()) return it->second;
    return bp.default_policy;
}

}  // namespace sai::pipeline
```

- [ ] **Step 5: Build and run tests**

```bash
cd build/default && cmake --build --preset default 2>&1 | tail -10
ctest -R "PipelineConfig" --output-on-failure
```

Expected: compile succeeds, all PipelineConfig tests pass (including YAML parsing tests added in Step 1)

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/pipeline_builder.h src/pipeline/pipeline_builder.cpp \
    tests/pipeline/pipeline_config_test.cpp
git commit -m "feat(pipeline): ✨ PipelineBuilder（YAML 解析 + 拓扑校验 + 类型兼容性检查）"
```

---

### Task 6: StageFactory — 阶段节点工厂 + 7 个 mock 阶段实现

**Files:**
- Create: `src/pipeline/stage_factory.h`
- Create: `src/pipeline/stage_factory.cpp`
- Create: `src/pipeline/capture_stage.cpp`
- Create: `src/pipeline/preprocess_stage.cpp`
- Create: `src/pipeline/inference_stage.cpp`
- Create: `src/pipeline/detect_stage.cpp`
- Create: `src/pipeline/rule_eval_stage.cpp`
- Create: `src/pipeline/reason_stage.cpp`
- Create: `src/pipeline/export_stage.cpp`

**Interfaces:**
- Produces (internal): `StageFactory::Create(StageConfig, Context&) -> Result<std::unique_ptr<IStageNode>>`
- Each `*_stage.cpp` creates a concrete `IStageNode` that wraps the corresponding M1-M5 interface

- [ ] **Step 1: Create src/pipeline/stage_factory.h**

```cpp
#pragma once

#include <memory>
#include <sai/core/error.h>
#include <sai/core/context.h>
#include <sai/pipeline/pipeline_config.h>
#include <sai/pipeline/stage_node.h>

namespace sai::pipeline {

class StageFactory {
public:
    static auto Create(const StageConfig& config, Context& ctx)
        -> Result<std::unique_ptr<IStageNode>>;
};

}  // namespace sai::pipeline
```

- [ ] **Step 2: Create src/pipeline/stage_factory.cpp**

```cpp
#include "stage_factory.h"
#include <sai/core/error.h>

namespace sai::pipeline {

auto StageFactory::Create(const StageConfig& config, Context& ctx)
    -> Result<std::unique_ptr<IStageNode>> {
    // For M6: all stages return mock implementations.
    // Production implementations wire to M2/M3/M5 real objects
    // and are implemented in the same *_stage.cpp files.

    switch (config.type) {
        case StageType::Capture: {
            auto stage = std::make_unique<CaptureStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Capture init failed: " + result.error().message};
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Preprocess: {
            auto stage = std::make_unique<PreprocessStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Preprocess init failed: " + result.error().message};
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Inference: {
            auto stage = std::make_unique<InferenceStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Inference init failed: " + result.error().message};
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Detect: {
            auto stage = std::make_unique<DetectStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Detect init failed: " + result.error().message};
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::RuleEval: {
            auto stage = std::make_unique<RuleEvalStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "RuleEval init failed: " + result.error().message};
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Reason: {
            auto stage = std::make_unique<ReasonStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Reason init failed: " + result.error().message};
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Export: {
            auto stage = std::make_unique<ExportStage>(
                config.id, config.config);
            auto result = stage->OnInitialize(ctx);
            if (!result.has_value())
                return ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                    "Export init failed: " + result.error().message};
            return std::unique_ptr<IStageNode>(std::move(stage));
        }
        case StageType::Custom:
            return ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                "Custom stages not yet supported"};
    }
    return ErrorInfo{ErrorCode::Pipeline_InvalidConfig, "Unknown stage type"};
}

}  // namespace sai::pipeline
```

- [ ] **Step 3: Create 7 concrete stage implementations**

Each stage file follows the same pattern. They are internal (in `src/pipeline/`, not `include/sai/`) — the public interface is `IStageNode`.

**src/pipeline/capture_stage.cpp:**

```cpp
#include <sai/pipeline/stage_node.h>
#include <sai/image/raw_image.h>

namespace sai::pipeline {

class CaptureStage final : public IStageNode {
public:
    CaptureStage(std::string id, YAML::Node /*config*/)
        : id_(std::move(id)) {}

    auto GetType() const noexcept -> StageType override { return StageType::Capture; }
    auto GetId() const -> std::string_view override { return id_; }
    auto OnInitialize(Context&) -> Result<void> override { return {}; }
    auto OnStart(Context&) -> Result<void> override { return {}; }
    auto OnStop(Context&) -> Result<void> override { return {}; }

    auto Process(StageInput input) -> Result<StageOutput> override {
        // Passthrough: RawImage → RawImage
        if (auto* img = std::get_if<RawImage>(&input)) {
            return StageOutput(std::move(*img));
        }
        return ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
            "Capture expects RawImage input"};
    }

private:
    std::string id_;
};

}  // namespace sai::pipeline
```

**src/pipeline/preprocess_stage.cpp:**

```cpp
#include <sai/pipeline/stage_node.h>
#include <sai/image/raw_image.h>
#include <sai/image/surface_image.h>

namespace sai::pipeline {

class PreprocessStage final : public IStageNode {
public:
    PreprocessStage(std::string id, YAML::Node /*config*/)
        : id_(std::move(id)) {}

    auto GetType() const noexcept -> StageType override { return StageType::Preprocess; }
    auto GetId() const -> std::string_view override { return id_; }
    auto OnInitialize(Context&) -> Result<void> override { return {}; }
    auto OnStart(Context&) -> Result<void> override { return {}; }
    auto OnStop(Context&) -> Result<void> override { return {}; }

    auto Process(StageInput input) -> Result<StageOutput> override {
        // Mock: RawImage → SurfaceImage (wrap same buffer)
        if (auto* img = std::get_if<RawImage>(&input)) {
            // Create a mock SurfaceImage by wrapping the same buffer
            auto meta = img->Meta();
            auto [data, size] = img->Data();
            std::vector<uint8_t> buffer(data, data + size);
            return StageOutput(
                SurfaceImage::FromOwnedBuffer(std::move(buffer), meta));
        }
        return ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
            "Preprocess expects RawImage input"};
    }

private:
    std::string id_;
};

}  // namespace sai::pipeline
```

**src/pipeline/inference_stage.cpp:**

```cpp
#include <sai/pipeline/stage_node.h>
#include <sai/detection/detection_result.h>

namespace sai::pipeline {

class InferenceStage final : public IStageNode {
public:
    InferenceStage(std::string id, YAML::Node /*config*/)
        : id_(std::move(id)) {}

    auto GetType() const noexcept -> StageType override { return StageType::Inference; }
    auto GetId() const -> std::string_view override { return id_; }
    auto OnInitialize(Context&) -> Result<void> override { return {}; }
    auto OnStart(Context&) -> Result<void> override { return {}; }
    auto OnStop(Context&) -> Result<void> override { return {}; }

    auto Process(StageInput input) -> Result<StageOutput> override {
        // Mock: SurfaceImage → DetectionResult
        if (auto* img = std::get_if<SurfaceImage>(&input)) {
            return StageOutput(DetectionResult{});
        }
        return ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
            "Inference expects SurfaceImage input"};
    }

private:
    std::string id_;
};

}  // namespace sai::pipeline
```

**src/pipeline/detect_stage.cpp:**

```cpp
#include <sai/pipeline/stage_node.h>
#include <sai/detection/detection_result.h>

namespace sai::pipeline {

class DetectStage final : public IStageNode {
public:
    DetectStage(std::string id, YAML::Node /*config*/)
        : id_(std::move(id)) {}

    auto GetType() const noexcept -> StageType override { return StageType::Detect; }
    auto GetId() const -> std::string_view override { return id_; }
    auto OnInitialize(Context&) -> Result<void> override { return {}; }
    auto OnStart(Context&) -> Result<void> override { return {}; }
    auto OnStop(Context&) -> Result<void> override { return {}; }

    auto Process(StageInput input) -> Result<StageOutput> override {
        if (auto* det = std::get_if<DetectionResult>(&input)) {
            return StageOutput(std::move(*det));
        }
        return ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
            "Detect expects DetectionResult input"};
    }

private:
    std::string id_;
};

}  // namespace sai::pipeline
```

**src/pipeline/rule_eval_stage.cpp:**

```cpp
#include <sai/pipeline/stage_node.h>
#include <sai/detection/detection_result.h>
#include <sai/rule/rule_engine.h>

namespace sai::pipeline {

class RuleEvalStage final : public IStageNode {
public:
    RuleEvalStage(std::string id, YAML::Node /*config*/)
        : id_(std::move(id)) {}

    auto GetType() const noexcept -> StageType override { return StageType::RuleEval; }
    auto GetId() const -> std::string_view override { return id_; }
    auto OnInitialize(Context&) -> Result<void> override { return {}; }
    auto OnStart(Context&) -> Result<void> override { return {}; }
    auto OnStop(Context&) -> Result<void> override { return {}; }

    auto Process(StageInput input) -> Result<StageOutput> override {
        if (auto* det = std::get_if<DetectionResult>(&input)) {
            return StageOutput(std::vector<rule::ResolvedRule>{});
        }
        return ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
            "RuleEval expects DetectionResult input"};
    }

private:
    std::string id_;
};

}  // namespace sai::pipeline
```

**src/pipeline/reason_stage.cpp:**

```cpp
#include <sai/pipeline/stage_node.h>
#include <sai/rule/rule_engine.h>
#include <sai/reasoner/reasoner.h>

namespace sai::pipeline {

class ReasonStage final : public IStageNode {
public:
    ReasonStage(std::string id, YAML::Node /*config*/)
        : id_(std::move(id)) {}

    auto GetType() const noexcept -> StageType override { return StageType::Reason; }
    auto GetId() const -> std::string_view override { return id_; }
    auto OnInitialize(Context&) -> Result<void> override { return {}; }
    auto OnStart(Context&) -> Result<void> override { return {}; }
    auto OnStop(Context&) -> Result<void> override { return {}; }

    auto Process(StageInput input) -> Result<StageOutput> override {
        if (auto* rules = std::get_if<std::vector<rule::ResolvedRule>>(&input)) {
            return StageOutput(rule::ReasoningResult{});
        }
        return ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
            "Reason expects ResolvedRule[] input"};
    }

private:
    std::string id_;
};

}  // namespace sai::pipeline
```

**src/pipeline/export_stage.cpp:**

```cpp
#include <sai/pipeline/stage_node.h>
#include <sai/reasoner/reasoner.h>

namespace sai::pipeline {

class ExportStage final : public IStageNode {
public:
    ExportStage(std::string id, YAML::Node /*config*/)
        : id_(std::move(id)) {}

    auto GetType() const noexcept -> StageType override { return StageType::Export; }
    auto GetId() const -> std::string_view override { return id_; }
    auto OnInitialize(Context&) -> Result<void> override { return {}; }
    auto OnStart(Context&) -> Result<void> override { return {}; }
    auto OnStop(Context&) -> Result<void> override { return {}; }

    auto Process(StageInput input) -> Result<StageOutput> override {
        // Export consumes ReasoningResult, produces same (pass-through for mock)
        if (auto* result = std::get_if<rule::ReasoningResult>(&input)) {
            // In production: serialize to JSON/PPM via IExporter
            return StageOutput(std::move(*result));
        }
        return ErrorInfo{ErrorCode::Pipeline_StageTypeMismatch,
            "Export expects ReasoningResult input"};
    }

private:
    std::string id_;
};

}  // namespace sai::pipeline
```

- [ ] **Step 4: Verify the build compiles**

```bash
cd build/default && cmake --build --preset default 2>&1 | tail -10
```

Expected: `[N/N] Linking CXX static library libsai_pipeline.a`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/stage_factory.h src/pipeline/stage_factory.cpp \
    src/pipeline/capture_stage.cpp src/pipeline/preprocess_stage.cpp \
    src/pipeline/inference_stage.cpp src/pipeline/detect_stage.cpp \
    src/pipeline/rule_eval_stage.cpp src/pipeline/reason_stage.cpp \
    src/pipeline/export_stage.cpp
git commit -m "feat(pipeline): ✨ StageFactory + 7 个 mock 阶段实现"
```

---

### Task 7: Pipeline 类（LoadFromYAML, Start, Submit, Drain, Stop, Metrics）

**Files:**
- Create: `include/sai/pipeline/pipeline.h`
- Create: `src/pipeline/pipeline.cpp`

**Interfaces:**
- Produces: `sai::pipeline::Pipeline`
  - `static auto LoadFromYAML(path, Context&) -> Result<std::unique_ptr<Pipeline>>`
  - `auto Start() -> Result<void>`
  - `auto Submit(RawImage) -> Result<void>`
  - `auto Drain() -> Result<void>`
  - `auto Stop() -> Result<void>`
  - `auto Metrics() const -> std::vector<StageMetrics>`

- [ ] **Step 1: Create include/sai/pipeline/pipeline.h**

```cpp
#pragma once

#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <stop_token>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/context.h>
#include <sai/pipeline/pipeline_config.h>
#include <sai/pipeline/stage_node.h>

namespace sai::runtime {
class TaskGraph;
class PipelineExecutor;
class TaskScheduler;
class WorkerPool;
template <typename T> class Registry;
}  // namespace sai::runtime

namespace sai::pipeline {

struct StageMetrics {
    std::string stage_id;
    StageType type;
    std::atomic<size_t> frames_processed{0};
    std::atomic<size_t> frames_failed{0};
    std::atomic<size_t> frames_dropped{0};

    size_t queue_depth() const { return queue_depth_.load(); }
    void set_queue_depth(size_t d) { queue_depth_.store(d); }

private:
    std::atomic<size_t> queue_depth_{0};
};

namespace detail {
class ErasedStageQueue {
public:
    virtual ~ErasedStageQueue() = default;
    virtual auto Depth() const -> size_t = 0;
    virtual auto Capacity() const -> size_t = 0;
    virtual auto PushBlocking(std::unique_ptr<StageOutput>) -> void = 0;
    virtual auto TryPush(std::unique_ptr<StageOutput>) -> bool = 0;
    virtual auto TryPop() -> std::unique_ptr<StageOutput> = 0;
    virtual auto PopBlocking() -> std::unique_ptr<StageOutput> = 0;
};
}  // namespace detail

class Pipeline {
public:
    static auto LoadFromYAML(std::filesystem::path yaml_path, Context& ctx)
        -> Result<std::unique_ptr<Pipeline>>;

    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;

    auto Start() -> Result<void>;
    auto Submit(RawImage image) -> Result<void>;
    auto Drain() -> Result<void>;
    auto Stop() -> Result<void>;
    auto Metrics() const -> std::vector<StageMetrics>;

private:
    Pipeline() = default;

    // DequeueInput: pop from the stage's input queue (blocking with short yield).
    // For entry stages (no depends_on), Submit() pushes directly to their queue.
    auto DequeueInput(const std::string& stage_id) -> std::unique_ptr<StageOutput>;
    // EnqueueOutputs: push output to all downstream stages' input queues.
    // Uses the adjacency map built during LoadFromYAML.
    auto EnqueueOutputs(const std::string& stage_id, std::unique_ptr<StageOutput>) -> void;
    // BuildQueueWiring: after parsing config, create input queues for each stage
    // and populate adjacency_ (upstream_id → downstream_ids).
    auto BuildQueueWiring(const PipelineConfig& config) -> Result<void>;

    std::unique_ptr<runtime::TaskGraph> graph_;
    std::unique_ptr<runtime::PipelineExecutor> executor_;
    std::unique_ptr<runtime::TaskScheduler> scheduler_;
    std::unique_ptr<runtime::Registry<runtime::WorkerPool>> worker_pools_;
    std::map<std::string, std::unique_ptr<IStageNode>> nodes_;
    // One input queue per stage (keyed by stage id). Entry stages receive
    // frames from Submit(); downstream stages receive from upstream EnqueueOutputs.
    std::map<std::string, std::unique_ptr<detail::ErasedStageQueue>> input_queues_;
    // Adjacency: for each stage, the list of downstream stage ids
    std::map<std::string, std::vector<std::string>> downstreams_;
    std::map<std::string, StageMetrics> metrics_;
    std::stop_source stop_source_;
    std::atomic<bool> running_{false};
    std::atomic<bool> draining_{false};
    std::vector<std::unique_ptr<runtime::WorkerPool>> pools_;
    std::string entry_stage_id_;  // first stage with empty depends_on
};

}  // namespace sai::pipeline
```

- [ ] **Step 2: Create src/pipeline/pipeline.cpp**

```cpp
#include <sai/pipeline/pipeline.h>
#include <sai/core/error.h>
#include <sai/core/registry.h>
#include <sai/runtime/task_graph.h>
#include <sai/runtime/pipeline_executor.h>
#include <sai/runtime/task_scheduler.h>
#include <sai/runtime/worker_pool.h>
#include <sai/image/raw_image.h>
#include <sai/pipeline/stage_queue.h>

#include "pipeline_builder.h"
#include "stage_factory.h"

#include <chrono>
#include <thread>

using namespace std::chrono_literals;

namespace sai::pipeline {

// Concrete ErasedStageQueue backed by StageQueue<StageOutput>
namespace detail {
namespace {
class ConcreteStageQueue final : public ErasedStageQueue {
public:
    explicit ConcreteStageQueue(size_t capacity, BackpressurePolicy bp)
        : queue_(StageQueue<StageOutput>::Create(capacity, bp).value()) {}

    auto Depth() const -> size_t override { return queue_->Depth(); }
    auto Capacity() const -> size_t override { return queue_->Capacity(); }

    auto PushBlocking(std::unique_ptr<StageOutput> item) -> void override {
        queue_->PushBlocking(std::move(item));
    }
    auto TryPush(std::unique_ptr<StageOutput> item) -> bool override {
        return queue_->TryPush(std::move(item));
    }
    auto TryPop() -> std::unique_ptr<StageOutput> override {
        return queue_->TryPop();
    }
    auto PopBlocking() -> std::unique_ptr<StageOutput> override {
        return queue_->PopBlocking();
    }

private:
    std::unique_ptr<StageQueue<StageOutput>> queue_;
};
} // anonymous namespace
} // namespace detail

namespace {

// Maps StageType → M1 stage_id string
std::string StageTypeToStr(StageType t) {
    switch (t) {
        case StageType::Capture:     return "Capture";
        case StageType::Preprocess:  return "Capture";
        case StageType::Inference:   return "Inference";
        case StageType::Detect:      return "Inference";
        case StageType::RuleEval:    return "Reason";
        case StageType::Reason:      return "Reason";
        case StageType::Export:      return "IO";
        case StageType::Custom:      return "Background";
    }
    return "Background";
}

// Default pool sizes per StageType (string key)
struct PoolConfig {
    size_t threads;
    size_t queue_capacity;
};

PoolConfig PoolConfigFor(std::string_view stage_str) {
    if (stage_str == "Capture")    return {2, 8};
    if (stage_str == "Inference")  return {1, 4};
    if (stage_str == "Reason")     return {2, 16};
    if (stage_str == "IO")         return {1, 32};
    if (stage_str == "Background") return {1, 16};
    return {1, 8};
}

} // anonymous namespace

auto Pipeline::LoadFromYAML(std::filesystem::path yaml_path, Context& ctx)
    -> Result<std::unique_ptr<Pipeline>> {
    auto config = PipelineBuilder::ParseFromYAML(yaml_path);
    if (!config.has_value()) return std::move(config).error();

    auto valid = PipelineBuilder::Validate(*config);
    if (!valid.has_value()) return std::move(valid).error();

    auto pipeline = std::unique_ptr<Pipeline>(new Pipeline());

    // Step 1: Create worker pools for each unique stage type string
    std::set<std::string> stage_strs;
    for (auto& s : config->stages) {
        stage_strs.insert(StageTypeToStr(s.type));
    }

    pipeline->worker_pools_ =
        std::make_unique<runtime::Registry<runtime::WorkerPool>>();

    for (auto& str : stage_strs) {
        auto pool_cfg = PoolConfigFor(str);
        auto pool = std::make_unique<runtime::WorkerPool>(
            pool_cfg.threads, pool_cfg.queue_capacity);
        auto type_id = sai::Fnv1aHash(str);
        auto result = pipeline->worker_pools_->Register(
            type_id, std::move(pool));
        if (!result.has_value()) return std::move(result).error();
    }

    // Step 2: Create TaskScheduler + PipelineExecutor
    pipeline->scheduler_ = std::make_unique<runtime::TaskScheduler>(
        *pipeline->worker_pools_);
    pipeline->executor_ = std::make_unique<runtime::PipelineExecutor>(
        *pipeline->scheduler_);

    // Step 3: Create stage nodes via StageFactory
    for (auto& stage_cfg : config->stages) {
        auto node = StageFactory::Create(stage_cfg, ctx);
        if (!node.has_value()) return std::move(node).error();

        StageMetrics m;
        m.stage_id = stage_cfg.id;
        m.type = stage_cfg.type;
        pipeline->metrics_[stage_cfg.id] = m;

        pipeline->nodes_[stage_cfg.id] = std::move(*node);
    }

    // Step 4: Build TaskGraph
    pipeline->graph_ = std::make_unique<runtime::TaskGraph>();

    for (auto& stage_cfg : config->stages) {
        // Compute stage_id hash from the stage type string
        auto stage_id = sai::Fnv1aHash(StageTypeToStr(stage_cfg.type));

        // Compute TaskId from stage id string
        runtime::TaskId task_id = sai::Fnv1aHash(stage_cfg.id);

        // Collect dependency TaskIds
        std::vector<runtime::TaskId> dep_ids;
        for (auto& dep : stage_cfg.depends_on) {
            dep_ids.push_back(sai::Fnv1aHash(dep));
        }

        // Build the work lambda
        auto& node = pipeline->nodes_[stage_cfg.id];
        auto& metrics = pipeline->metrics_[stage_cfg.id];

        runtime::TaskNode task_node;
        task_node.id = task_id;
        task_node.stage_id = stage_id;
        task_node.dependencies = dep_ids;
        task_node.work = [node = node.get(), &metrics, &pipeline = *pipeline,
                          stage_id_str = stage_cfg.id]()
            -> runtime::Task<void> {
            // One-shot per frame: read the frame from this stage's input
            // queue, call Process(), push the result to each downstream
            // queue. The TaskGraph's dependency mechanism ensures stages
            // execute in topological order.
            //
            // The input queue for this stage is identified by stage_id_str.
            // For the entry stages (depends_on empty), Submit() pushes
            // directly to their input queue.

            auto input = pipeline.DequeueInput(stage_id_str);
            if (!input.has_value()) {
                metrics.frames_failed.fetch_add(1, std::memory_order::relaxed);
                co_return Result<void>{};
            }

            auto result = node->Process(std::move(*input));
            if (result.has_value()) {
                metrics.frames_processed.fetch_add(1, std::memory_order::relaxed);
                auto output = std::make_unique<StageOutput>(
                    std::move(result.value()));
                pipeline.EnqueueOutputs(stage_id_str, std::move(output));
                co_return Result<void>{};
            } else {
                metrics.frames_failed.fetch_add(1, std::memory_order::relaxed);
                co_return result.error();
            }
        };

        auto result = pipeline->graph_->AddNode(std::move(task_node));
        if (!result.has_value()) return std::move(result).error();
    }

    // Store config for later use by Submit
    // Wire queues between stages
    auto wire_result = pipeline->BuildQueueWiring(*config);
    if (!wire_result.has_value()) return std::move(wire_result).error();

    return pipeline;
}

auto Pipeline::Start() -> Result<void> {
    if (running_.exchange(true)) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidState,
            "Pipeline already running"};
    }

    // Start all stage nodes
    // (Context& would be stored from LoadFromYAML)
    // For now, OnStart is a no-op for mock stages

    return {};
}

auto Pipeline::Submit(RawImage image) -> Result<void> {
    if (!running_.load()) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidState,
            "Pipeline not running — call Start() first"};
    }

    // Push to the entry stage's input queue
    auto it = input_queues_.find(entry_stage_id_);
    if (it == input_queues_.end()) {
        return ErrorInfo{ErrorCode::Pipeline_InvalidConfig,
            "No entry stage queue found for: " + entry_stage_id_};
    }

    auto output = std::make_unique<StageOutput>(std::move(image));
    bool pushed = it->second->TryPush(std::move(output));
    if (!pushed) {
        // Apply backpressure policy for entry stage
        it->second->PushBlocking(std::move(output));
    }

    return {};
}

auto Pipeline::DequeueInput(const std::string& stage_id) -> std::unique_ptr<StageOutput> {
    auto it = input_queues_.find(stage_id);
    if (it == input_queues_.end()) return nullptr;
    return it->second->PopBlocking();
}

auto Pipeline::EnqueueOutputs(const std::string& stage_id,
                               std::unique_ptr<StageOutput> output) -> void {
    auto adj = downstreams_.find(stage_id);
    if (adj == downstreams_.end()) return;

    for (auto& downstream_id : adj->second) {
        auto q = input_queues_.find(downstream_id);
        if (q == input_queues_.end()) continue;

        // Copy the output for each downstream (each gets its own copy)
        // For M6 v1 mock stages, we move into the first downstream.
        // Production: each downstream gets a shared_ptr<const StageOutput>.
        auto output_copy = std::make_unique<StageOutput>(*output);
        bool pushed = q->second->TryPush(std::move(output_copy));
        if (!pushed) {
            q->second->PushBlocking(std::move(output_copy));
        }
    }
}

auto Pipeline::BuildQueueWiring(const PipelineConfig& config) -> Result<void> {
    // Create an input queue for each stage
    for (auto& stage : config.stages) {
        size_t q_capacity = stage.queue_capacity.value_or(8);
        auto q = StageQueue<StageOutput>::Create(
            q_capacity, stage.backpressure);
        if (!q.has_value()) return std::move(q).error();
        input_queues_[stage.id] = std::move(*q);
    }

    // Build adjacency map (upstream → downstreams)
    for (auto& stage : config.stages) {
        for (auto& dep : stage.depends_on) {
            downstreams_[dep].push_back(stage.id);
        }
    }

    // Identify entry stage
    for (auto& stage : config.stages) {
        if (stage.depends_on.empty()) {
            entry_stage_id_ = stage.id;
            break;
        }
    }

    return {};
}

auto Pipeline::Drain() -> Result<void> {
    draining_.store(true);
    // Wait for all queues to drain
    // For now: short sleep (mock stages process instantly)
    std::this_thread::sleep_for(10ms);
    draining_.store(false);
    return {};
}

auto Pipeline::Stop() -> Result<void> {
    stop_source_.request_stop();
    running_.store(false);

    // Drain remaining frames
    auto drain_result = Drain();
    if (!drain_result.has_value()) return drain_result;

    // Stop stages in reverse order (respecting dependency order)
    // For mock stages, OnStop is a no-op

    return {};
}

auto Pipeline::Metrics() const -> std::vector<StageMetrics> {
    std::vector<StageMetrics> result;
    for (auto& [id, m] : metrics_) {
        result.push_back({m.stage_id, m.type,
            m.frames_processed.load(), m.frames_failed.load(),
            m.frames_dropped.load()});
    }
    return result;
}

}  // namespace sai::pipeline
```

- [ ] **Step 3: Build and run tests**

```bash
cd build/default && cmake --build --preset default 2>&1 | tail -10
ctest --preset default 2>&1 | tail -5
```

Expected: build succeeds, all existing tests (510 → ~530) pass

- [ ] **Step 4: Commit**

```bash
git add include/sai/pipeline/pipeline.h src/pipeline/pipeline.cpp
git commit -m "feat(pipeline): ✨ Pipeline 类（LoadFromYAML + Start + Submit + Drain + Stop + Metrics）"
```

---

### Task 8: Scheduler（StageType → WorkerPool 映射 + 队列分配）

**Files:**
- Create: `src/scheduler/scheduler.h`
- Create: `src/scheduler/scheduler.cpp`
- Create: `tests/scheduler/scheduler_test.cpp`

**Interfaces:**
- Produces (internal): `sai::pipeline::Scheduler`
  - `explicit Scheduler(Registry<WorkerPool>&, const BackpressureConfig&)`
  - `auto Allocate(const std::vector<StageConfig>&) -> Result<void>`
  - `auto Deallocate() -> Result<void>`
  - `auto PoolFor(StageType) const -> Result<WorkerPool&>`

- [ ] **Step 1: Create src/scheduler/scheduler.h**

```cpp
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/type_id.h>
#include <sai/pipeline/pipeline_config.h>

namespace sai::runtime {
class WorkerPool;
template <typename T> class Registry;
}  // namespace sai::runtime

namespace sai::pipeline {

class Scheduler {
public:
    explicit Scheduler(runtime::Registry<runtime::WorkerPool>& pools,
                       const BackpressureConfig& bp_config);

    auto Allocate(const std::vector<StageConfig>& stages) -> Result<void>;
    auto Deallocate() -> Result<void>;
    auto PoolFor(StageType type) const -> Result<runtime::WorkerPool&>;

private:
    runtime::Registry<runtime::WorkerPool>& pools_;
    BackpressureConfig bp_config_;
    std::map<StageType, TypeId> stage_pool_map_;
};

}  // namespace sai::pipeline
```

- [ ] **Step 2: Create src/scheduler/scheduler.cpp**

```cpp
#include "scheduler.h"
#include <sai/core/registry.h>
#include <sai/runtime/worker_pool.h>

namespace sai::pipeline {

namespace {

TypeId StageTypeToPoolId(StageType t) {
    switch (t) {
        case StageType::Capture:     return Fnv1aHash("Capture");
        case StageType::Preprocess:  return Fnv1aHash("Capture");
        case StageType::Inference:   return Fnv1aHash("Inference");
        case StageType::Detect:      return Fnv1aHash("Inference");
        case StageType::RuleEval:    return Fnv1aHash("Reason");
        case StageType::Reason:      return Fnv1aHash("Reason");
        case StageType::Export:      return Fnv1aHash("IO");
        case StageType::Custom:      return Fnv1aHash("Background");
    }
    return Fnv1aHash("Background");
}

}  // anonymous namespace

Scheduler::Scheduler(runtime::Registry<runtime::WorkerPool>& pools,
                     const BackpressureConfig& bp_config)
    : pools_(pools), bp_config_(bp_config) {}

auto Scheduler::Allocate(const std::vector<StageConfig>& stages)
    -> Result<void> {
    stage_pool_map_.clear();

    // Map each stage type to its pool TypeId
    std::set<TypeId> required_pools;
    for (auto& s : stages) {
        auto pool_id = StageTypeToPoolId(s.type);
        required_pools.insert(pool_id);
        stage_pool_map_[s.type] = pool_id;
    }

    // Verify all required pools exist in the registry
    for (auto& pool_id : required_pools) {
        auto resolved = pools_.Resolve(pool_id);
        if (!resolved.has_value()) {
            return ErrorInfo{ErrorCode::Scheduler_PoolNotFound,
                "No WorkerPool registered for stage type"};
        }
    }

    return {};
}

auto Scheduler::Deallocate() -> Result<void> {
    stage_pool_map_.clear();
    return {};
}

auto Scheduler::PoolFor(StageType type) const
    -> Result<runtime::WorkerPool&> {
    auto it = stage_pool_map_.find(type);
    if (it == stage_pool_map_.end()) {
        return ErrorInfo{ErrorCode::Scheduler_PoolNotFound,
            "StageType not allocated"};
    }
    // Resolve from registry (returns shared_ptr<WorkerPool>)
    auto resolved = pools_.Resolve(it->second);
    if (!resolved.has_value()) {
        return ErrorInfo{ErrorCode::Scheduler_PoolNotFound,
            "WorkerPool not found in registry"};
    }
    return *(*resolved);
}

}  // namespace sai::pipeline
```

- [ ] **Step 3: Create tests/scheduler/scheduler_test.cpp**

```cpp
#include <gtest/gtest.h>
#include <sai/core/registry.h>
#include <sai/runtime/worker_pool.h>
#include <sai/pipeline/pipeline_config.h>
#include "scheduler.h"  // internal header — add tests/scheduler to include path

namespace sai::pipeline {
namespace {

TEST(SchedulerTest, AllocateAndResolve) {
    // Set up a Registry<WorkerPool> with required pools
    runtime::Registry<runtime::WorkerPool> pools;
    auto pool = std::make_shared<runtime::WorkerPool>(2, 8);
    ASSERT_TRUE(pools.Register(Fnv1aHash("Inference"), pool).has_value());

    BackpressureConfig bp;
    bp.default_policy = BackpressurePolicy::Block;

    Scheduler scheduler(pools, bp);

    std::vector<StageConfig> stages;
    StageConfig s;
    s.id = "inference";
    s.type = StageType::Inference;
    stages.push_back(s);

    auto alloc_result = scheduler.Allocate(stages);
    ASSERT_TRUE(alloc_result.has_value());

    auto pool_result = scheduler.PoolFor(StageType::Inference);
    ASSERT_TRUE(pool_result.has_value());
    EXPECT_EQ(pool_result->ThreadCount(), 2);
}

TEST(SchedulerTest, MissingPoolReturnsError) {
    runtime::Registry<runtime::WorkerPool> pools;
    // No pools registered
    BackpressureConfig bp;
    Scheduler scheduler(pools, bp);

    std::vector<StageConfig> stages;
    StageConfig s;
    s.id = "detect";
    s.type = StageType::Detect;
    stages.push_back(s);

    auto result = scheduler.Allocate(stages);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Scheduler_PoolNotFound);
}

TEST(SchedulerTest, DeallocateClearsMapping) {
    runtime::Registry<runtime::WorkerPool> pools;
    auto pool = std::make_shared<runtime::WorkerPool>(1, 4);
    ASSERT_TRUE(pools.Register(Fnv1aHash("Capture"), pool).has_value());

    BackpressureConfig bp;
    Scheduler scheduler(pools, bp);

    std::vector<StageConfig> stages;
    StageConfig s;
    s.id = "capture";
    s.type = StageType::Capture;
    stages.push_back(s);

    ASSERT_TRUE(scheduler.Allocate(stages).has_value());
    ASSERT_TRUE(scheduler.PoolFor(StageType::Capture).has_value());

    ASSERT_TRUE(scheduler.Deallocate().has_value());
    auto result = scheduler.PoolFor(StageType::Capture);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Scheduler_PoolNotFound);
}

TEST(SchedulerTest, CaptureAndPreprocessSharePool) {
    runtime::Registry<runtime::WorkerPool> pools;
    auto capture_pool = std::make_shared<runtime::WorkerPool>(2, 8);
    ASSERT_TRUE(pools.Register(Fnv1aHash("Capture"), capture_pool).has_value());

    BackpressureConfig bp;
    Scheduler scheduler(pools, bp);

    std::vector<StageConfig> stages;
    {
        StageConfig s;
        s.id = "capture";
        s.type = StageType::Capture;
        stages.push_back(s);
    }
    {
        StageConfig s;
        s.id = "preprocess";
        s.type = StageType::Preprocess;
        stages.push_back(s);
    }

    ASSERT_TRUE(scheduler.Allocate(stages).has_value());

    // Both should resolve to the same pool
    auto p1 = scheduler.PoolFor(StageType::Capture);
    auto p2 = scheduler.PoolFor(StageType::Preprocess);
    ASSERT_TRUE(p1.has_value());
    ASSERT_TRUE(p2.has_value());
    EXPECT_EQ(&p1->get(), &p2->get());  // same pool instance
}

} // namespace
} // namespace sai::pipeline
```

- [ ] **Step 4: Build and run Scheduler tests**

```bash
cd build/default && cmake --build --preset default 2>&1 | tail -10
ctest -R "Scheduler" --output-on-failure
```

Expected: 4/4 Scheduler tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scheduler/scheduler.h src/scheduler/scheduler.cpp \
    tests/scheduler/scheduler_test.cpp
git commit -m "feat(scheduler): ✨ Scheduler（StageType → WorkerPool 映射 + 队列分配）"
```

---

### Task 9: Backpressure + 阶段失败隔离测试

**Files:**
- Modify: `tests/pipeline/stage_queue_test.cpp` (add backpressure integration tests)
- Modify: `tests/pipeline/pipeline_test.cpp` (add failure isolation tests)

- [ ] **Step 1: Add backpressure integration test to stage_queue_test.cpp**

```cpp
TEST(StageQueueTest, BlockBackpressureWithThreads) {
    auto q = IntQueue::Create(2, BackpressurePolicy::Block);
    ASSERT_TRUE(q.has_value());

    // Fill to capacity
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(1)));
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(2)));

    // Start a consumer thread that pops after a delay
    std::atomic<bool> push_succeeded{false};
    std::thread consumer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        auto val = (*q)->TryPop();
        if (val) push_succeeded.store(true);
    });

    // PushBlocking should wait for consumer to pop
    auto start = std::chrono::steady_clock::now();
    (*q)->PushBlocking(std::make_unique<int>(3));
    auto elapsed = std::chrono::steady_clock::now() - start;

    consumer.join();
    EXPECT_TRUE(push_succeeded.load());
    EXPECT_GE(elapsed, std::chrono::milliseconds(10)); // actually waited
}

TEST(StageQueueTest, DropOldestKeepsNewest) {
    auto q = IntQueue::Create(2, BackpressurePolicy::DropOldest);
    ASSERT_TRUE(q.has_value());

    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(1)));
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(2)));
    // Full. Push 3 → drops 1, keeps [2, 3]
    EXPECT_TRUE((*q)->TryPush(std::make_unique<int>(3)));

    EXPECT_EQ(*(*q)->TryPop(), 2);
    EXPECT_EQ(*(*q)->TryPop(), 3);
    EXPECT_EQ((*q)->TryPop(), nullptr);
}

TEST(StageQueueTest, DropOldestRepeatedly) {
    auto q = IntQueue::Create(2, BackpressurePolicy::DropOldest);
    ASSERT_TRUE(q.has_value());

    // Push 5 items into a capacity-2 queue, verify FIFO of survivors
    for (int i = 1; i <= 5; ++i) {
        (*q)->TryPush(std::make_unique<int>(i));
    }
    // Should have kept [4, 5]
    EXPECT_EQ(*(*q)->TryPop(), 4);
    EXPECT_EQ(*(*q)->TryPop(), 5);
    EXPECT_EQ((*q)->TryPop(), nullptr);
}
```

- [ ] **Step 2: Add failure isolation test to pipeline_test.cpp**

```cpp
class FailingStage : public IStageNode {
public:
    FailingStage(std::string id, bool fail_on_process = true)
        : id_(std::move(id)), fail_(fail_on_process) {}

    auto GetType() const noexcept -> StageType override { return StageType::Custom; }
    auto GetId() const -> std::string_view override { return id_; }
    auto OnInitialize(Context&) -> Result<void> override { return {}; }
    auto OnStart(Context&) -> Result<void> override { return {}; }
    auto OnStop(Context&) -> Result<void> override { return {}; }
    auto Process(StageInput) -> Result<StageOutput> override {
        if (fail_) {
            return ErrorInfo{ErrorCode::Pipeline_StageInitFailed,
                "simulated stage failure"};
        }
        return StageOutput(RawImage::FromOwnedBuffer(
            std::vector<uint8_t>{42}, sai::image::ImageMeta{}));
    }

    void SetFail(bool f) { fail_ = f; }

private:
    std::string id_;
    bool fail_;
};

TEST(PipelineFailureTest, StageFailureDoesNotCrash) {
    // Verify that a failing Process() returns an error, not an exception
    FailingStage stage("failing", true);
    sai::Context ctx;
    stage.OnInitialize(ctx);

    RawImage mock = RawImage::FromOwnedBuffer(
        std::vector<uint8_t>{1, 2, 3}, sai::image::ImageMeta{});
    StageInput input(mock);
    auto result = stage.Process(std::move(input));
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::Pipeline_StageInitFailed);
}

TEST(PipelineFailureTest, StageRecoversAfterFailure) {
    // A stage that fails once then succeeds on the next call
    FailingStage stage("recover", true);
    sai::Context ctx;
    stage.OnInitialize(ctx);

    RawImage mock = RawImage::FromOwnedBuffer(
        std::vector<uint8_t>{1}, sai::image::ImageMeta{});

    // First call fails
    EXPECT_FALSE(stage.Process(StageInput(mock)).has_value());

    // Second call succeeds (stage recovered)
    stage.SetFail(false);
    auto result = stage.Process(StageInput(mock));
    EXPECT_TRUE(result.has_value());
}
```

- [ ] **Step 3: Run tests**

```bash
cd build/default && cmake --build --preset default 2>&1 | tail -5
ctest -R "StageQueue|PipelineFailure|PipelineConfig" --output-on-failure
```

Expected: all tests PASS (StageQueue 11, PipelineFailure 2, PipelineConfig 8)

- [ ] **Step 4: Commit**

```bash
git add tests/pipeline/stage_queue_test.cpp tests/pipeline/pipeline_test.cpp
git commit -m "test(pipeline): ✅ 背压集成测试 + 阶段失败隔离测试"
```

---

### Task 10: E2E 集成测试（7 阶段全链路 + Mock）

**Files:**
- Create: `tests/integration/m6_pipeline_test.cpp` (or extend existing integration test)

- [ ] **Step 1: Write E2E integration test**

Create `tests/integration/m6_pipeline_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

#include <sai/core/context.h>
#include <sai/core/registry.h>
#include <sai/runtime/worker_pool.h>
#include <sai/pipeline/pipeline.h>
#include <sai/pipeline/pipeline_config.h>
#include <sai/pipeline/stage_node.h>
#include <sai/pipeline/stage_queue.h>
#include <sai/image/raw_image.h>

namespace {

using namespace sai::pipeline;

std::filesystem::path WritePipelineYaml() {
    const char* yaml = R"(
pipeline:
  name: "e2e_test"
  version: "1.0"
  backpressure:
    default: block
  stages:
    - id: capture
      type: Capture
      depends_on: []
    - id: preprocess
      type: Preprocess
      depends_on: [capture]
    - id: detect
      type: Detect
      depends_on: [preprocess]
    - id: rule_eval
      type: RuleEval
      depends_on: [detect]
    - id: reason
      type: Reason
      depends_on: [rule_eval]
    - id: export
      type: Export
      depends_on: [reason]
)";
    auto path = std::filesystem::temp_directory_path() / "m6_e2e_pipeline.yaml";
    std::ofstream ofs(path);
    ofs << yaml;
    return path;
}

TEST(M6PipelineTest, LoadPipelineFromYaml) {
    auto yaml_path = WritePipelineYaml();
    sai::Context ctx;

    auto pipeline = Pipeline::LoadFromYAML(yaml_path, ctx);
    ASSERT_TRUE(pipeline.has_value())
        << "LoadFromYAML failed: " << pipeline.error().message;

    auto metrics = (*pipeline)->Metrics();
    ASSERT_EQ(metrics.size(), 6);

    // Verify stage ids and types
    std::map<std::string, StageType> expected = {
        {"capture", StageType::Capture},
        {"preprocess", StageType::Preprocess},
        {"detect", StageType::Detect},
        {"rule_eval", StageType::RuleEval},
        {"reason", StageType::Reason},
        {"export", StageType::Export},
    };

    for (auto& m : metrics) {
        auto it = expected.find(m.stage_id);
        ASSERT_NE(it, expected.end())
            << "Unexpected stage: " << m.stage_id;
        EXPECT_EQ(m.type, it->second)
            << "Wrong type for stage: " << m.stage_id;
    }
}

TEST(M6PipelineTest, PipelineLifecycle) {
    auto yaml_path = WritePipelineYaml();
    sai::Context ctx;

    auto pipeline = Pipeline::LoadFromYAML(yaml_path, ctx);
    ASSERT_TRUE(pipeline.has_value());

    // Start
    auto start_result = (*pipeline)->Start();
    ASSERT_TRUE(start_result.has_value())
        << "Start failed: " << start_result.error().message;

    // Submit a mock frame
    std::vector<uint8_t> pixel_data(100, 0xAA);
    sai::image::ImageMeta meta;
    meta.width = 10;
    meta.height = 10;
    meta.pixel_format = sai::image::PixelFormat::Gray8;

    auto raw = sai::RawImage::FromOwnedBuffer(std::move(pixel_data), meta);
    auto submit_result = (*pipeline)->Submit(std::move(raw));
    ASSERT_TRUE(submit_result.has_value())
        << "Submit failed: " << submit_result.error().message;

    // Drain: wait for pipeline to finish
    auto drain_result = (*pipeline)->Drain();
    ASSERT_TRUE(drain_result.has_value())
        << "Drain failed: " << drain_result.error().message;

    // Stop
    auto stop_result = (*pipeline)->Stop();
    ASSERT_TRUE(stop_result.has_value())
        << "Stop failed: " << stop_result.error().message;
}

TEST(M6PipelineTest, InvalidYamlReturnsError) {
    const char* bad_yaml = R"(
pipeline:
  name: "bad"
  stages:
    - id: a
      type: Capture
      depends_on: []
    - id: b
      type: Capture
      depends_on: [c]   # c does not exist!
)";
    auto path = std::filesystem::temp_directory_path() / "bad_pipeline.yaml";
    std::ofstream ofs(path);
    ofs << bad_yaml;
    ofs.close();

    sai::Context ctx;
    auto pipeline = Pipeline::LoadFromYAML(path, ctx);
    EXPECT_FALSE(pipeline.has_value());
    EXPECT_EQ(pipeline.error().code, ErrorCode::Pipeline_InvalidConfig);
}

TEST(M6PipelineTest, CyclicDependencyRejected) {
    const char* cyclic_yaml = R"(
pipeline:
  name: "cyclic"
  version: "1.0"
  stages:
    - id: a
      type: Capture
      depends_on: [c]
    - id: b
      type: Detect
      depends_on: [a]
    - id: c
      type: Export
      depends_on: [b]
)";
    auto path = std::filesystem::temp_directory_path() / "cyclic_pipeline.yaml";
    std::ofstream ofs(path);
    ofs << cyclic_yaml;
    ofs.close();

    sai::Context ctx;
    auto pipeline = Pipeline::LoadFromYAML(path, ctx);
    EXPECT_FALSE(pipeline.has_value());
    EXPECT_EQ(pipeline.error().code, ErrorCode::Pipeline_InvalidConfig);
}

TEST(M6PipelineTest, StageTypeFromStringConversion) {
    EXPECT_EQ(StageTypeFromString("Capture"), StageType::Capture);
    EXPECT_EQ(StageTypeFromString("Preprocess"), StageType::Preprocess);
    EXPECT_EQ(StageTypeFromString("Inference"), StageType::Inference);
    EXPECT_EQ(StageTypeFromString("Detect"), StageType::Detect);
    EXPECT_EQ(StageTypeFromString("RuleEval"), StageType::RuleEval);
    EXPECT_EQ(StageTypeFromString("Reason"), StageType::Reason);
    EXPECT_EQ(StageTypeFromString("Export"), StageType::Export);
    EXPECT_EQ(StageTypeFromString("Custom"), StageType::Custom);
    EXPECT_THROW(StageTypeFromString("Nonexistent"), std::invalid_argument);
}

} // namespace
```

- [ ] **Step 2: Add integration test to CMakeLists.txt**

Verify `tests/integration/CMakeLists.txt` already includes m6_pipeline_test.cpp (or add it):

```cmake
# tests/integration/CMakeLists.txt should already exist
# Add to its target_sources:
target_sources(sai_integration_test PRIVATE
    m6_pipeline_test.cpp
)
```

- [ ] **Step 3: Run E2E tests**

```bash
cd build/default && cmake --build --preset default 2>&1 | tail -10
ctest -R "M6Pipeline" --output-on-failure
```

Expected: 5/5 M6Pipeline tests PASS

- [ ] **Step 4: Run full test suite to verify no regressions**

```bash
ctest --preset default 2>&1 | tail -10
```

Expected: `100% tests passed, 0 tests failed out of NNN` (ALL tests green)

- [ ] **Step 5: Commit**

```bash
git add tests/integration/m6_pipeline_test.cpp
git commit -m "test(pipeline): ✅ M6 E2E 集成测试（YAML 加载 + 生命周期 + 校验）"
```

---

### Task 11: 契约表更新

**Files:**
- Modify: `docs/surface-ai/glossary-and-contracts.md`

- [ ] **Step 1: Append M6 concepts and interfaces to glossary-and-contracts.md**

Insert in §1 概念归属表:

```markdown
| `Pipeline` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | 配置驱动的编排宿主，持有 M1 TaskGraph + PipelineExecutor，对外暴露 `Submit()`/`Drain()`/`Stop()`/`Metrics()` |
| `PipelineConfig` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | YAML 反序列化的 Pipeline 配置：name + version + backpressure + stages[] |
| `IStageNode` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | 业务语义化阶段节点接口：`OnInitialize`/`OnStart`/`OnStop`/`Process` |
| `StageQueue<T>` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | 阶段间 bounded SPSC/MPSC lock-free ring buffer，内建 BackpressurePolicy |
| `Scheduler` | 6.2 | design/milestone-06-orchestration-scheduling/6.2-scheduler.md | StageType → WorkerPool 映射 + 阶段间队列分配 |
| `StageMetrics` | 6.2 | design/milestone-06-orchestration-scheduling/6.2-scheduler.md | per-stage 原子计数器：frames_processed / failed / dropped / queue_depth |
```

Insert in §2 接口签名表:

```markdown
| `IStageNode` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | `class IStageNode : public Object { virtual auto GetType() const noexcept -> StageType = 0; virtual auto GetId() const -> string_view = 0; virtual auto OnInitialize(Context&) -> Result<void> = 0; virtual auto OnStart(Context&) -> Result<void> = 0; virtual auto OnStop(Context&) -> Result<void> = 0; virtual auto Process(StageInput) -> Result<StageOutput> = 0; }` |
| `Pipeline` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | `class Pipeline { static auto LoadFromYAML(path, Context&) -> Result<unique_ptr<Pipeline>>; auto Start() -> Result<void>; auto Submit(RawImage) -> Result<void>; auto Drain() -> Result<void>; auto Stop() -> Result<void>; auto Metrics() const -> vector<StageMetrics>; }` |
| `StageQueue<T>` | 6.1 | design/milestone-06-orchestration-scheduling/6.1-pipeline.md | `template<typename T> class StageQueue { static auto Create(size_t, BackpressurePolicy) -> Result<unique_ptr<StageQueue>>; auto TryPush(unique_ptr<T>) -> bool; auto PushBlocking(unique_ptr<T>) -> void; auto TryPop() -> unique_ptr<T>; auto PopBlocking() -> unique_ptr<T>; auto Depth() const noexcept -> size_t; }` |
| `Scheduler` | 6.2 | design/milestone-06-orchestration-scheduling/6.2-scheduler.md | `class Scheduler { explicit Scheduler(Registry<WorkerPool>&, const BackpressureConfig&); auto Allocate(const vector<StageConfig>&) -> Result<void>; auto Deallocate() -> Result<void>; auto PoolFor(StageType) const -> Result<WorkerPool&>; }` |
```

- [ ] **Step 2: Commit**

```bash
git add docs/surface-ai/glossary-and-contracts.md
git commit -m "docs(contract): 📝 M6 契约增量（Pipeline + IStageNode + StageQueue + Scheduler + StageMetrics）"
```

---

## Final Verification

After all tasks are complete:

```bash
# Full build
cmake --build --preset default 2>&1 | tail -5

# Full test suite
ctest --preset default 2>&1 | tail -5
# Expected: 100% tests passed, 0 tests failed

# Module structure check
grep -c "^## [0-9]" docs/superpowers/specs/2026-07-14-milestone-6-orchestration-scheduling-design.md
# Expected: 17 (spec sections)

# Verify error.h append-only
grep "Pipeline_\|Scheduler_" include/sai/core/error.h
# Expected: 7 Pipeline_/Scheduler_ codes after Reasoner_ScoreComputationFailed
```
