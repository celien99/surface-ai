# Milestone 1 Code Scaffold (1.1 + 1.2 subset) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the frozen 1.1 (Core Foundation) and 1.2 (Core Lifecycle) design docs into a real, locally compilable and testable C++20 CMake project — the first executable code in this repository.

**Architecture:** A single `sai_core` static library under `src/core/`, headers under `include/sai/core/`, GoogleTest/gmock unit tests under `tests/core/`. Dependencies (`tl-expected`, `gtest`) are pulled via vcpkg manifest mode. No CUDA/TensorRT/FAISS code — this scaffold only covers the two CUDA-free design docs.

**Tech Stack:** C++20, CMake + CMakePresets.json, vcpkg (manifest mode), `tl::expected` (aliased `sai::Result<T>`), GoogleTest + gmock.

## Global Constraints

- C++20 required (`CMAKE_CXX_STANDARD 20`, `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_CXX_EXTENSIONS OFF`).
- `vcpkg.json` manifest declares exactly two dependencies: `tl-expected`, `gtest`. Do not add `fmt`/`yaml-cpp`/`spdlog` — those belong to batch 1.6, out of scope here.
- No CUDA/TensorRT code of any kind in this scaffold (dev machine is macOS arm64, not the target platform; per the approved design doc there is no conditional-compilation escape hatch — GPU-dependent parts of 1.4/1.5 are simply not implemented yet).
- Every class name, method signature, and namespace must match the "4. Interfaces" section of `docs/surface-ai/design/milestone-01-foundation/1.1-core-foundation.md` and `1.2-core-lifecycle.md` verbatim. If a documented signature turns out not to compile, stop and report the discrepancy — do not silently change it.
- Template methods (`TypeRegistry::Register<T>`, `Registry<TInterface>::Register`/`Resolve`, `Context::Register<T>`/`Resolve<T>`) are header-only. Non-template methods (`TypeRegistry::Instance`/`Resolve`, `Context::RegisterModule`/`Initialize`/`Start`/`Stop`/`CurrentState`) live in `.cpp` files.
- Coding style: avoid over-defensive checks for states that can't occur, avoid multi-level nested `if`, prefer early return, prefer `tl::expected`'s `and_then`/`or_else`/`map` chaining over nested `has_value()` checks.
- Verification is real: every task ends with an actual `cmake --build` + `ctest` run, not static review.

## File Structure Overview

```
surface-ai/
├── CMakeLists.txt
├── CMakePresets.json
├── vcpkg.json
├── .gitignore
├── include/sai/core/
│   ├── type_id.h
│   ├── error.h
│   ├── object.h
│   ├── resource.h
│   ├── reflectable.h
│   ├── type_registry.h
│   ├── lifecycle.h
│   ├── module.h
│   ├── service.h
│   ├── factory.h
│   ├── registry.h
│   └── context.h
├── src/core/
│   ├── CMakeLists.txt
│   ├── type_registry.cpp
│   └── context.cpp
└── tests/core/
    ├── CMakeLists.txt
    ├── smoke_test.cpp              # deleted again in Task 2 once real tests exist
    ├── object_resource_test.cpp
    ├── type_registry_test.cpp
    ├── registry_factory_test.cpp
    └── context_lifecycle_test.cpp
```

## Execution Order

```
Task 1 (vcpkg + CMake scaffold, smoke test)
  └─> Task 2 (Object / Resource)
        └─> Task 3 (TypeId / ErrorCode / Result<T> / IReflectable / TypeRegistry)
              └─> Task 4 (Registry<TInterface> / Factory<TInterface>)
                    └─> Task 5 (LifecycleState / IModule / IService / Context)
```

Task 2 has no dependency on Task 3+ (headers only, no `Result<T>` usage), but is kept second to match the design docs' own ordering and because Task 1's smoke test is replaced by real tests as soon as something exists to test.

---

## Task 1: vcpkg + CMake scaffold

**Files:**
- Create: `CMakeLists.txt`
- Create: `CMakePresets.json`
- Create: `vcpkg.json`
- Create: `.gitignore`
- Create: `tests/core/CMakeLists.txt`
- Create: `tests/core/smoke_test.cpp`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: a working `cmake --preset default` configure/build/test pipeline that later tasks extend. No `sai` types yet.

- [ ] **Step 1: Bootstrap vcpkg (one-time local environment setup)**

Run:
```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=~/vcpkg
```
Expected: `bootstrap-vcpkg.sh` ends with "vcpkg package management program is now ready." `VCPKG_ROOT` must be exported in every shell session used to configure this project (add it to `~/.zshrc` for persistence).

- [ ] **Step 2: Create `vcpkg.json`**

```json
{
  "name": "surface-ai",
  "version": "0.1.0",
  "dependencies": [
    "tl-expected",
    "gtest"
  ]
}
```

- [ ] **Step 3: Create top-level `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.21)
project(surface_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_subdirectory(tests/core)
```

`add_subdirectory(src/core)` is intentionally absent — it is added in Task 3 when the first `.cpp` file exists.

- [ ] **Step 4: Create `CMakePresets.json`**

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "default",
      "generator": "Unix Makefiles",
      "binaryDir": "${sourceDir}/build/default",
      "toolchainFile": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "default",
      "configurePreset": "default"
    }
  ],
  "testPresets": [
    {
      "name": "default",
      "configurePreset": "default",
      "output": { "outputOnFailure": true }
    }
  ]
}
```

- [ ] **Step 5: Create `.gitignore`**

```
build/
```

- [ ] **Step 6: Create `tests/core/CMakeLists.txt`**

```cmake
find_package(GTest CONFIG REQUIRED)

add_executable(sai_core_smoke_test smoke_test.cpp)
target_link_libraries(sai_core_smoke_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_core_smoke_test)
```

- [ ] **Step 7: Create `tests/core/smoke_test.cpp`**

```cpp
#include <gtest/gtest.h>

TEST(Smoke, ToolchainIsWorking) {
    EXPECT_EQ(1 + 1, 2);
}
```

- [ ] **Step 8: Configure, build, and run**

Run:
```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```
Expected: configure succeeds (vcpkg builds `tl-expected` and `gtest` ports — first run may take several minutes), build succeeds, `ctest` reports `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 9: Commit**

```bash
git add CMakeLists.txt CMakePresets.json vcpkg.json .gitignore tests/core/CMakeLists.txt tests/core/smoke_test.cpp
git commit -m "build: bootstrap vcpkg + CMake scaffold with a smoke test"
```

---

## Task 2: Object / Resource

**Files:**
- Create: `include/sai/core/object.h`
- Create: `include/sai/core/resource.h`
- Create: `tests/core/object_resource_test.cpp`
- Modify: `tests/core/CMakeLists.txt`
- Delete: `tests/core/smoke_test.cpp`

**Interfaces:**
- Consumes: nothing (no dependency on `Result<T>` or any other batch construct)
- Produces: `class sai::Object` (non-copyable, non-movable, protected default constructor), `class sai::Resource` (non-copyable, movable, `IsValid()`/`Release()` pure virtual)

- [ ] **Step 1: Write the failing test**

Create `tests/core/object_resource_test.cpp`:

```cpp
#include <sai/core/object.h>
#include <sai/core/resource.h>

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace {

class ConcreteObject final : public sai::Object {
public:
    ConcreteObject() = default;
};

static_assert(!std::is_copy_constructible_v<ConcreteObject>);
static_assert(!std::is_copy_assignable_v<ConcreteObject>);
static_assert(!std::is_move_constructible_v<ConcreteObject>);
static_assert(!std::is_move_assignable_v<ConcreteObject>);

class FakeResource final : public sai::Resource {
public:
    explicit FakeResource(int handle) : handle_(handle) {}

    [[nodiscard]] bool IsValid() const noexcept override { return handle_ != kInvalidHandle; }

    void Release() noexcept override { handle_ = kInvalidHandle; }

    [[nodiscard]] int Handle() const noexcept { return handle_; }

private:
    static constexpr int kInvalidHandle = -1;
    int handle_ = kInvalidHandle;
};

static_assert(!std::is_copy_constructible_v<FakeResource>);
static_assert(!std::is_copy_assignable_v<FakeResource>);

}  // namespace

TEST(ObjectTest, DerivedTypeIsConstructibleButNeverCopyableOrMovable) {
    ConcreteObject object;
    SUCCEED();
}

TEST(ResourceTest, StartsValidAndReleaseInvalidatesIt) {
    FakeResource resource(42);
    EXPECT_TRUE(resource.IsValid());

    resource.Release();

    EXPECT_FALSE(resource.IsValid());
}

TEST(ResourceTest, MoveTransfersOwnership) {
    FakeResource source(7);

    FakeResource moved(std::move(source));

    EXPECT_TRUE(moved.IsValid());
    EXPECT_EQ(moved.Handle(), 7);
}
```

Modify `tests/core/CMakeLists.txt` (replace its entire contents — the smoke test is retired now that real tests exist):

```cmake
find_package(GTest CONFIG REQUIRED)

add_executable(sai_core_object_resource_test object_resource_test.cpp)
target_include_directories(sai_core_object_resource_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_core_object_resource_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_core_object_resource_test)
```

Delete `tests/core/smoke_test.cpp`.

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset default`
Expected: FAIL — `fatal error: 'sai/core/object.h' file not found`

- [ ] **Step 3: Create `include/sai/core/object.h`**

```cpp
#pragma once

namespace sai {

class Object {
public:
    virtual ~Object() = default;

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&) = delete;
    Object& operator=(Object&&) = delete;

protected:
    Object() = default;
};

}  // namespace sai
```

- [ ] **Step 4: Create `include/sai/core/resource.h`**

```cpp
#pragma once

namespace sai {

class Resource {
public:
    Resource() noexcept = default;
    virtual ~Resource() noexcept = default;

    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&&) noexcept = default;
    Resource& operator=(Resource&&) noexcept = default;

    [[nodiscard]] virtual bool IsValid() const noexcept = 0;
    virtual void Release() noexcept = 0;
};

}  // namespace sai
```

- [ ] **Step 5: Run to verify it passes**

Run:
```bash
cmake --build --preset default
ctest --preset default
```
Expected: build succeeds, `ctest` reports `100% tests passed, 0 tests failed out of 3`.

- [ ] **Step 6: Commit**

```bash
git add include/sai/core/object.h include/sai/core/resource.h tests/core/object_resource_test.cpp tests/core/CMakeLists.txt
git rm tests/core/smoke_test.cpp
git commit -m "feat: add Object and Resource base classes"
```

---

## Task 3: TypeId / ErrorCode / Result / IReflectable / TypeRegistry

**Files:**
- Create: `include/sai/core/type_id.h`
- Create: `include/sai/core/error.h`
- Create: `include/sai/core/reflectable.h`
- Create: `include/sai/core/type_registry.h`
- Create: `src/core/CMakeLists.txt`
- Create: `src/core/type_registry.cpp`
- Create: `tests/core/type_registry_test.cpp`
- Modify: `CMakeLists.txt` (add `add_subdirectory(src/core)`)
- Modify: `tests/core/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing from Task 2 (independent of `Object`/`Resource`)
- Produces:
  - `using sai::TypeId = std::uint64_t;` and `SAI_DECLARE_TYPE_ID(QualifiedName)` macro
  - `enum class sai::ErrorCode`, `struct sai::ErrorInfo { code; message; source_location; }`, `template<typename T> using sai::Result = tl::expected<T, ErrorInfo>;`
  - `class sai::IReflectable { virtual auto TypeId() const noexcept -> sai::TypeId = 0; };`
  - `class sai::TypeRegistry` with `Instance()`, `Register<T>()`, `Resolve(TypeId) const`
  - `sai::TypeInfo { id; name; }` and the `sai::Reflectable` concept
  - Later tasks (4, 5) rely on: `sai::Result<T>`, `sai::ErrorCode::Core_TypeAlreadyRegistered`, `sai::ErrorCode::Core_TypeNotFound`, `sai::TypeId`, `sai::Reflectable`

- [ ] **Step 1: Write the failing test**

Create `tests/core/type_registry_test.cpp`:

```cpp
#include <sai/core/type_registry.h>

#include <string>

#include <gtest/gtest.h>

namespace {

class NewlyRegisteredType final : public sai::IReflectable {
public:
    SAI_DECLARE_TYPE_ID(sai::test::type_registry_test::NewlyRegisteredType)
};

class DuplicateRegisteredType final : public sai::IReflectable {
public:
    SAI_DECLARE_TYPE_ID(sai::test::type_registry_test::DuplicateRegisteredType)
};

class ChainQueriedType final : public sai::IReflectable {
public:
    SAI_DECLARE_TYPE_ID(sai::test::type_registry_test::ChainQueriedType)
};

}  // namespace

TEST(TypeRegistryTest, RegisterSucceedsForNewType) {
    auto result = sai::TypeRegistry::Instance().Register<NewlyRegisteredType>();

    EXPECT_TRUE(result.has_value());
}

TEST(TypeRegistryTest, DuplicateRegisterReturnsTypeAlreadyRegistered) {
    ASSERT_TRUE(sai::TypeRegistry::Instance().Register<DuplicateRegisteredType>().has_value());

    auto second = sai::TypeRegistry::Instance().Register<DuplicateRegisteredType>();

    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, sai::ErrorCode::Core_TypeAlreadyRegistered);
}

TEST(TypeRegistryTest, ResolveUnknownTypeReturnsTypeNotFound) {
    constexpr sai::TypeId kUnknownId = 0xDEADBEEFULL;

    auto result = sai::TypeRegistry::Instance().Resolve(kUnknownId);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Core_TypeNotFound);
}

TEST(TypeRegistryTest, ResolveChainsWithAndThen) {
    ASSERT_TRUE(sai::TypeRegistry::Instance().Register<ChainQueriedType>().has_value());

    auto described = sai::TypeRegistry::Instance()
        .Resolve(ChainQueriedType::kStaticTypeId)
        .and_then([](const sai::TypeInfo& info) -> sai::Result<std::string> {
            return std::string(info.name);
        });

    ASSERT_TRUE(described.has_value());
    EXPECT_EQ(*described, "sai::test::type_registry_test::ChainQueriedType");
}
```

Modify `tests/core/CMakeLists.txt` (append to the existing file):

```cmake

add_executable(sai_core_type_registry_test type_registry_test.cpp)
target_link_libraries(sai_core_type_registry_test PRIVATE sai::core GTest::gtest_main)
gtest_discover_tests(sai_core_type_registry_test)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset default`
Expected: FAIL — `fatal error: 'sai/core/type_registry.h' file not found` (and `sai::core` target does not exist yet)

- [ ] **Step 3: Create `include/sai/core/type_id.h`**

```cpp
#pragma once

#include <cstdint>
#include <string_view>

namespace sai {

using TypeId = std::uint64_t;

namespace detail {
constexpr TypeId Fnv1aHash(std::string_view name) noexcept {
    TypeId hash = 14695981039346656037ULL;  // FNV-1a 64-bit offset basis
    for (char ch : name) {
        hash ^= static_cast<TypeId>(static_cast<unsigned char>(ch));
        hash *= 1099511628211ULL;  // FNV-1a 64-bit prime
    }
    return hash;
}
}  // namespace detail

}  // namespace sai

#define SAI_DECLARE_TYPE_ID(QualifiedName)                                \
    static constexpr std::string_view kStaticTypeName = #QualifiedName;   \
    static constexpr ::sai::TypeId kStaticTypeId =                        \
        ::sai::detail::Fnv1aHash(kStaticTypeName);                        \
    [[nodiscard]] auto TypeId() const noexcept -> ::sai::TypeId override { \
        return kStaticTypeId;                                             \
    }
```

- [ ] **Step 4: Create `include/sai/core/error.h`**

```cpp
#pragma once

#include <cstdint>
#include <source_location>
#include <string>

#include <tl/expected.hpp>

namespace sai {

enum class ErrorCode : std::uint32_t {
    Core_Unknown = 0,
    Core_ConstructionFailed,
    Core_TypeAlreadyRegistered,
    Core_TypeNotFound,
    // Lifecycle_* is added by Task 5 (Context) when it needs its own error code.
    // The full cross-module error code taxonomy is completed by batch 1.6,
    // out of scope for this scaffold.
};

struct ErrorInfo {
    ErrorCode code;
    std::string message;
    std::source_location source_location;
};

template <typename T>
using Result = tl::expected<T, ErrorInfo>;

}  // namespace sai
```

- [ ] **Step 5: Create `include/sai/core/reflectable.h`**

```cpp
#pragma once

#include <sai/core/type_id.h>

namespace sai {

class IReflectable {
public:
    virtual ~IReflectable() = default;

    [[nodiscard]] virtual auto TypeId() const noexcept -> sai::TypeId = 0;
};

}  // namespace sai
```

- [ ] **Step 6: Create `include/sai/core/type_registry.h`**

```cpp
#pragma once

#include <concepts>
#include <shared_mutex>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include <sai/core/error.h>
#include <sai/core/reflectable.h>
#include <sai/core/type_id.h>

namespace sai {

struct TypeInfo {
    TypeId id;
    std::string_view name;
};

template <typename T>
concept Reflectable = std::is_base_of_v<IReflectable, T> &&
    requires {
        { T::kStaticTypeId } -> std::convertible_to<TypeId>;
        { T::kStaticTypeName } -> std::convertible_to<std::string_view>;
    };

class TypeRegistry {
public:
    static auto Instance() noexcept -> TypeRegistry&;

    template <Reflectable T>
    auto Register() -> Result<void>;

    [[nodiscard]] auto Resolve(TypeId id) const -> Result<TypeInfo>;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<TypeId, TypeInfo> entries_;
};

template <Reflectable T>
auto TypeRegistry::Register() -> Result<void> {
    std::unique_lock lock(mutex_);
    auto [it, inserted] = entries_.try_emplace(
        T::kStaticTypeId, TypeInfo{T::kStaticTypeId, T::kStaticTypeName});
    if (!inserted) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeAlreadyRegistered,
            "type already registered",
            std::source_location::current(),
        });
    }
    return {};
}

}  // namespace sai
```

- [ ] **Step 7: Create `src/core/type_registry.cpp`**

```cpp
#include <sai/core/type_registry.h>

namespace sai {

auto TypeRegistry::Instance() noexcept -> TypeRegistry& {
    static TypeRegistry instance;
    return instance;
}

auto TypeRegistry::Resolve(TypeId id) const -> Result<TypeInfo> {
    std::shared_lock lock(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeNotFound,
            "type not found",
            std::source_location::current(),
        });
    }
    return it->second;
}

}  // namespace sai
```

- [ ] **Step 8: Create `src/core/CMakeLists.txt`**

```cmake
find_package(tl-expected CONFIG REQUIRED)

add_library(sai_core STATIC
    type_registry.cpp
)
add_library(sai::core ALIAS sai_core)

target_include_directories(sai_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_core PUBLIC tl::expected)
target_compile_features(sai_core PUBLIC cxx_std_20)
```

- [ ] **Step 9: Modify top-level `CMakeLists.txt`**

Add `add_subdirectory(src/core)` before `add_subdirectory(tests/core)`:

```cmake
enable_testing()

add_subdirectory(src/core)
add_subdirectory(tests/core)
```

- [ ] **Step 10: Run to verify it passes**

Run:
```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```
Expected: configure succeeds (finds `tl-expected` port), build succeeds, `ctest` reports `100% tests passed, 0 tests failed out of 7`.

- [ ] **Step 11: Commit**

```bash
git add include/sai/core/type_id.h include/sai/core/error.h include/sai/core/reflectable.h include/sai/core/type_registry.h src/core/CMakeLists.txt src/core/type_registry.cpp tests/core/type_registry_test.cpp CMakeLists.txt tests/core/CMakeLists.txt
git commit -m "feat: add TypeId, ErrorCode/Result, IReflectable, TypeRegistry"
```

---

## Task 4: Registry\<TInterface\> / Factory\<TInterface\>

**Files:**
- Create: `include/sai/core/registry.h`
- Create: `include/sai/core/factory.h`
- Create: `tests/core/registry_factory_test.cpp`
- Modify: `tests/core/CMakeLists.txt`

**Interfaces:**
- Consumes: `sai::Result<T>`, `sai::ErrorCode::Core_TypeAlreadyRegistered`, `sai::ErrorCode::Core_TypeNotFound`, `sai::TypeId` (all from Task 3)
- Produces:
  - `template<typename TInterface> class sai::Registry { auto Register(TypeId, shared_ptr<TInterface>) -> Result<void>; auto Resolve(TypeId) const -> Result<shared_ptr<TInterface>>; };`
  - `template<typename TInterface> class sai::Factory { virtual auto Create() -> Result<unique_ptr<TInterface>> = 0; };`
  - Task 5 relies on: `sai::Registry<IService>` as `Context`'s internal service table, with these exact method signatures.

- [ ] **Step 1: Write the failing test**

Create `tests/core/registry_factory_test.cpp`:

```cpp
#include <sai/core/factory.h>
#include <sai/core/registry.h>

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

class IWidget {
public:
    virtual ~IWidget() = default;
    virtual auto Name() const -> std::string = 0;
};

class ConcreteWidget final : public IWidget {
public:
    explicit ConcreteWidget(std::string name) : name_(std::move(name)) {}
    auto Name() const -> std::string override { return name_; }

private:
    std::string name_;
};

class MockWidgetFactory final : public sai::Factory<IWidget> {
public:
    MOCK_METHOD(sai::Result<std::unique_ptr<IWidget>>, Create, (), (override));
};

}  // namespace

TEST(RegistryTest, RegisterThenResolveReturnsSameInstance) {
    sai::Registry<IWidget> registry;
    auto widget = std::make_shared<ConcreteWidget>("camera");
    constexpr sai::TypeId kWidgetId = 1;

    ASSERT_TRUE(registry.Register(kWidgetId, widget).has_value());

    auto resolved = registry.Resolve(kWidgetId);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(*resolved, widget);
}

TEST(RegistryTest, DuplicateRegisterReturnsTypeAlreadyRegistered) {
    sai::Registry<IWidget> registry;
    constexpr sai::TypeId kWidgetId = 2;
    ASSERT_TRUE(registry.Register(kWidgetId, std::make_shared<ConcreteWidget>("a")).has_value());

    auto second = registry.Register(kWidgetId, std::make_shared<ConcreteWidget>("b"));

    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, sai::ErrorCode::Core_TypeAlreadyRegistered);
}

TEST(RegistryTest, ResolveUnregisteredTypeReturnsTypeNotFound) {
    sai::Registry<IWidget> registry;

    auto result = registry.Resolve(999);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Core_TypeNotFound);
}

TEST(FactoryTest, CreateDelegatesToOverriddenImplementation) {
    MockWidgetFactory factory;
    EXPECT_CALL(factory, Create()).WillOnce(::testing::Invoke([] {
        return sai::Result<std::unique_ptr<IWidget>>(std::make_unique<ConcreteWidget>("mocked"));
    }));

    auto result = factory.Create();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)->Name(), "mocked");
}
```

Modify `tests/core/CMakeLists.txt` (append):

```cmake

find_package(GTest CONFIG REQUIRED)

add_executable(sai_core_registry_factory_test registry_factory_test.cpp)
target_link_libraries(sai_core_registry_factory_test PRIVATE sai::core GTest::gmock_main)
gtest_discover_tests(sai_core_registry_factory_test)
```

(The `find_package(GTest CONFIG REQUIRED)` line is redundant with the one already at the top of the file — CMake tolerates this, but you may remove the duplicate if editing the whole file.)

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset default`
Expected: FAIL — `fatal error: 'sai/core/registry.h' file not found`

- [ ] **Step 3: Create `include/sai/core/registry.h`**

```cpp
#pragma once

#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include <sai/core/error.h>
#include <sai/core/type_id.h>

namespace sai {

template <typename TInterface>
class Registry {
public:
    auto Register(TypeId id, std::shared_ptr<TInterface> instance) -> Result<void>;

    [[nodiscard]] auto Resolve(TypeId id) const -> Result<std::shared_ptr<TInterface>>;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<TypeId, std::shared_ptr<TInterface>> entries_;
};

template <typename TInterface>
auto Registry<TInterface>::Register(TypeId id, std::shared_ptr<TInterface> instance)
    -> Result<void> {
    std::unique_lock lock(mutex_);
    auto [it, inserted] = entries_.try_emplace(id, std::move(instance));
    if (!inserted) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeAlreadyRegistered,
            "type already registered",
            std::source_location::current(),
        });
    }
    return {};
}

template <typename TInterface>
auto Registry<TInterface>::Resolve(TypeId id) const -> Result<std::shared_ptr<TInterface>> {
    std::shared_lock lock(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Core_TypeNotFound,
            "type not found",
            std::source_location::current(),
        });
    }
    return it->second;
}

}  // namespace sai
```

- [ ] **Step 4: Create `include/sai/core/factory.h`**

```cpp
#pragma once

#include <memory>

#include <sai/core/error.h>

namespace sai {

template <typename TInterface>
class Factory {
public:
    virtual ~Factory() = default;

    [[nodiscard]] virtual auto Create() -> Result<std::unique_ptr<TInterface>> = 0;
};

}  // namespace sai
```

- [ ] **Step 5: Run to verify it passes**

Run:
```bash
cmake --build --preset default
ctest --preset default
```
Expected: build succeeds, `ctest` reports `100% tests passed, 0 tests failed out of 11`.

- [ ] **Step 6: Commit**

```bash
git add include/sai/core/registry.h include/sai/core/factory.h tests/core/registry_factory_test.cpp tests/core/CMakeLists.txt
git commit -m "feat: add Registry<TInterface> and Factory<TInterface>"
```

---

## Task 5: LifecycleState / IModule / IService / Context

**Files:**
- Create: `include/sai/core/lifecycle.h`
- Create: `include/sai/core/module.h`
- Create: `include/sai/core/service.h`
- Create: `include/sai/core/context.h`
- Create: `src/core/context.cpp`
- Create: `tests/core/context_lifecycle_test.cpp`
- Modify: `include/sai/core/error.h` (add `Lifecycle_RegisterAfterAssembly`)
- Modify: `src/core/CMakeLists.txt` (add `context.cpp` to `sai_core` sources)
- Modify: `tests/core/CMakeLists.txt`

**Interfaces:**
- Consumes: `sai::Object` (Task 2); `sai::Result<T>`, `sai::IReflectable`, `sai::Reflectable` concept (Task 3); `sai::Registry<TInterface>` (Task 4)
- Produces: `enum class sai::LifecycleState`, `class sai::IModule : public Object`, `class sai::IService : public Object, public IReflectable`, `class sai::Context` with `RegisterModule`, `Register<T>`, `Resolve<T>`, `Initialize`, `Start`, `Stop`, `CurrentState`. This is the last task in this scaffold — nothing downstream in this plan consumes it.

- [ ] **Step 1: Write the failing test**

Create `tests/core/context_lifecycle_test.cpp`:

```cpp
#include <sai/core/context.h>

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

class RecordingModule final : public sai::IModule {
public:
    RecordingModule(std::string name, std::vector<std::string>& initialize_log,
                     std::vector<std::string>& stop_log)
        : name_(std::move(name)), initialize_log_(initialize_log), stop_log_(stop_log) {}

    auto OnInitialize(sai::Context& /*context*/) -> sai::Result<void> override {
        initialize_log_.push_back(name_);
        return {};
    }

    auto OnStart(sai::Context& /*context*/) -> sai::Result<void> override { return {}; }

    auto OnStop(sai::Context& /*context*/) -> sai::Result<void> override {
        stop_log_.push_back(name_);
        return {};
    }

private:
    std::string name_;
    std::vector<std::string>& initialize_log_;
    std::vector<std::string>& stop_log_;
};

class ProbeService final : public sai::IService {
public:
    SAI_DECLARE_TYPE_ID(sai::test::context_lifecycle_test::ProbeService)
};

}  // namespace

TEST(ContextLifecycleTest, FullAssemblyWalksStatesInOrder) {
    sai::Context context;
    std::vector<std::string> initialize_log;
    std::vector<std::string> stop_log;

    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Created);

    ASSERT_TRUE(context
        .RegisterModule(std::make_unique<RecordingModule>("only", initialize_log, stop_log))
        .has_value());

    ASSERT_TRUE(context.Initialize().has_value());
    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Initialized);

    ASSERT_TRUE(context.Start().has_value());
    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Running);

    ASSERT_TRUE(context.Stop().has_value());
    EXPECT_EQ(context.CurrentState(), sai::LifecycleState::Stopped);
}

TEST(ContextLifecycleTest, OnStopRunsInReverseRegistrationOrder) {
    sai::Context context;
    std::vector<std::string> initialize_log;
    std::vector<std::string> stop_log;

    ASSERT_TRUE(context
        .RegisterModule(std::make_unique<RecordingModule>("first", initialize_log, stop_log))
        .has_value());
    ASSERT_TRUE(context
        .RegisterModule(std::make_unique<RecordingModule>("second", initialize_log, stop_log))
        .has_value());

    ASSERT_TRUE(context.Initialize().has_value());
    ASSERT_TRUE(context.Start().has_value());
    ASSERT_TRUE(context.Stop().has_value());

    EXPECT_EQ(initialize_log, (std::vector<std::string>{"first", "second"}));
    EXPECT_EQ(stop_log, (std::vector<std::string>{"second", "first"}));
}

TEST(ContextLifecycleTest, RegisterAfterAssemblyIsRejectedWhileRunning) {
    sai::Context context;
    ASSERT_TRUE(context.Initialize().has_value());
    ASSERT_TRUE(context.Start().has_value());
    ASSERT_EQ(context.CurrentState(), sai::LifecycleState::Running);

    auto result = context.Register<ProbeService>(std::make_shared<ProbeService>());

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Lifecycle_RegisterAfterAssembly);
}
```

Modify `tests/core/CMakeLists.txt` (append):

```cmake

add_executable(sai_core_context_lifecycle_test context_lifecycle_test.cpp)
target_link_libraries(sai_core_context_lifecycle_test PRIVATE sai::core GTest::gtest_main)
gtest_discover_tests(sai_core_context_lifecycle_test)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset default`
Expected: FAIL — `fatal error: 'sai/core/context.h' file not found`

- [ ] **Step 3: Create `include/sai/core/lifecycle.h`**

```cpp
#pragma once

namespace sai {

// Destroyed is the conceptual terminal state used to make the lifecycle
// diagram complete. Context::state_ never holds this value at runtime —
// there is no post-destruction state to query — so no API sets it explicitly.
enum class LifecycleState {
    Created,
    Initialized,
    Running,
    Stopped,
    Destroyed,
};

}  // namespace sai
```

- [ ] **Step 4: Create `include/sai/core/module.h`**

```cpp
#pragma once

#include <sai/core/error.h>
#include <sai/core/object.h>

namespace sai {

class Context;

class IModule : public Object {
public:
    virtual ~IModule() = default;

    virtual auto OnInitialize(Context& context) -> Result<void> = 0;
    virtual auto OnStart(Context& context) -> Result<void> = 0;
    virtual auto OnStop(Context& context) -> Result<void> = 0;
};

}  // namespace sai
```

- [ ] **Step 5: Create `include/sai/core/service.h`**

```cpp
#pragma once

#include <sai/core/object.h>
#include <sai/core/reflectable.h>

namespace sai {

class IService : public Object, public IReflectable {
public:
    virtual ~IService() = default;
};

}  // namespace sai
```

- [ ] **Step 6: Modify `include/sai/core/error.h`**

Add `Lifecycle_RegisterAfterAssembly` to the `ErrorCode` enum, replacing the trailing comment:

```cpp
enum class ErrorCode : std::uint32_t {
    Core_Unknown = 0,
    Core_ConstructionFailed,
    Core_TypeAlreadyRegistered,
    Core_TypeNotFound,
    Lifecycle_RegisterAfterAssembly,
    // The full cross-module error code taxonomy is completed by batch 1.6,
    // out of scope for this scaffold.
};
```

- [ ] **Step 7: Create `include/sai/core/context.h`**

```cpp
#pragma once

#include <concepts>
#include <memory>
#include <vector>

#include <sai/core/error.h>
#include <sai/core/lifecycle.h>
#include <sai/core/module.h>
#include <sai/core/registry.h>
#include <sai/core/service.h>
#include <sai/core/type_registry.h>

namespace sai {

class Context {
public:
    Context() noexcept = default;
    ~Context() = default;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    auto RegisterModule(std::unique_ptr<IModule> module) -> Result<void>;

    template <Reflectable T>
        requires std::derived_from<T, IService>
    auto Register(std::shared_ptr<T> instance) -> Result<void>;

    template <Reflectable T>
        requires std::derived_from<T, IService>
    [[nodiscard]] auto Resolve() const -> Result<std::shared_ptr<T>>;

    auto Initialize() -> Result<void>;
    auto Start() -> Result<void>;
    auto Stop() -> Result<void>;

    [[nodiscard]] auto CurrentState() const noexcept -> LifecycleState;

private:
    LifecycleState state_ = LifecycleState::Created;
    std::vector<std::unique_ptr<IModule>> modules_;
    Registry<IService> service_registry_;
};

template <Reflectable T>
    requires std::derived_from<T, IService>
auto Context::Register(std::shared_ptr<T> instance) -> Result<void> {
    if (state_ != LifecycleState::Created) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Lifecycle_RegisterAfterAssembly,
            "cannot register a service after assembly",
            std::source_location::current(),
        });
    }
    return service_registry_.Register(T::kStaticTypeId, std::move(instance));
}

template <Reflectable T>
    requires std::derived_from<T, IService>
auto Context::Resolve() const -> Result<std::shared_ptr<T>> {
    return service_registry_.Resolve(T::kStaticTypeId)
        .map([](const std::shared_ptr<IService>& service) {
            return std::static_pointer_cast<T>(service);
        });
}

}  // namespace sai
```

- [ ] **Step 8: Create `src/core/context.cpp`**

```cpp
#include <sai/core/context.h>

namespace sai {

auto Context::RegisterModule(std::unique_ptr<IModule> module) -> Result<void> {
    if (state_ != LifecycleState::Created) {
        return tl::make_unexpected(ErrorInfo{
            ErrorCode::Lifecycle_RegisterAfterAssembly,
            "cannot register a module after assembly",
            std::source_location::current(),
        });
    }
    modules_.push_back(std::move(module));
    return {};
}

auto Context::Initialize() -> Result<void> {
    for (auto& module : modules_) {
        auto result = module->OnInitialize(*this);
        if (!result) {
            return result;
        }
    }
    state_ = LifecycleState::Initialized;
    return {};
}

auto Context::Start() -> Result<void> {
    for (auto& module : modules_) {
        auto result = module->OnStart(*this);
        if (!result) {
            return result;
        }
    }
    state_ = LifecycleState::Running;
    return {};
}

auto Context::Stop() -> Result<void> {
    Result<void> first_error{};
    for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) {
        auto result = (*it)->OnStop(*this);
        if (!result && first_error) {
            first_error = std::move(result);
        }
    }
    state_ = LifecycleState::Stopped;
    return first_error;
}

auto Context::CurrentState() const noexcept -> LifecycleState {
    return state_;
}

}  // namespace sai
```

- [ ] **Step 9: Modify `src/core/CMakeLists.txt`**

```cmake
find_package(tl-expected CONFIG REQUIRED)

add_library(sai_core STATIC
    type_registry.cpp
    context.cpp
)
add_library(sai::core ALIAS sai_core)

target_include_directories(sai_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_core PUBLIC tl::expected)
target_compile_features(sai_core PUBLIC cxx_std_20)
```

- [ ] **Step 10: Run to verify it passes**

Run:
```bash
cmake --build --preset default
ctest --preset default
```
Expected: build succeeds, `ctest` reports `100% tests passed, 0 tests failed out of 14`.

- [ ] **Step 11: Commit**

```bash
git add include/sai/core/lifecycle.h include/sai/core/module.h include/sai/core/service.h include/sai/core/context.h include/sai/core/error.h src/core/context.cpp src/core/CMakeLists.txt tests/core/context_lifecycle_test.cpp tests/core/CMakeLists.txt
git commit -m "feat: add LifecycleState, IModule, IService, and Context"
```

---
