# 里程碑 1 代码阶段（1.1 + 1.2 子集）设计文档

> Status: Approved
> Date: 2026-07-08
> Source: `docs/surface-ai/design/milestone-01-foundation/1.1-core-foundation.md`、`1.2-core-lifecycle.md`
> Purpose: 把已定稿的 1.1（Core 基础）与 1.2（Core 生命周期与装配）两份设计文档转成真实的、可在本机编译运行的 C++20 CMake 项目骨架，作为里程碑 1 代码阶段的第一个可验证产出。

---

## 1. 背景与范围决策

里程碑 1 的 6 份设计文档（1.1/1.2/1.3/1.5/1.4/1.6）已全部定稿并通过评审（含最终全分支评审），冻结为后续代码实现的接口基线。用户在文档阶段结束后明确要求"看得见、能跑起来"的代码，而不是继续产出文档。

本轮代码阶段的范围与验证策略经过以下确认：

1. **CUDA/平台策略**：当前开发环境是 macOS arm64，没有 CUDA/nvcc，架构也非目标平台（Ubuntu x64 + NVIDIA GPU）的 x64。用户明确选择**不做条件编译，直接写目标平台代码，本机不验证** GPU 相关部分。这意味着 1.4（Runtime）与 1.5（Memory）两份文档中依赖 CUDA API（`cudaMalloc`/`cudaHostAlloc`/`cudaStream`/`cudaMemcpyAsync` 等）的接口，本轮不实现。
2. **代码范围**：先做 **1.1 + 1.2 子集**，不一次性覆盖全部 6 份文档。这两份文档不依赖 CUDA，接口数量适中，能最快产出一个真实可编译的项目骨架。1.3/1.5/1.4/1.6 留给后续批次。
3. **验证方式**：**本机实际编译运行**，要求真实的 `cmake configure` → `cmake build` → `ctest` 输出作为完成证据，不是人工静态审查。
4. **依赖获取方式**：现在在本机安装 vcpkg（clone + bootstrap），用 `vcpkg.json` manifest 模式拉取 `tl-expected` 和 `gtest`，与 spec 锁定的 vcpkg 依赖管理决策保持一致，不用 FetchContent 或 Homebrew 作为临时替代方案。

## 2. 目录结构

```
surface-ai/
├── CMakeLists.txt              # 顶层，project()、C++20、启用测试、子目录
├── CMakePresets.json           # vcpkg toolchain 集成
├── vcpkg.json                  # manifest 模式依赖：tl-expected、gtest
├── include/sai/core/
│   ├── type_id.h                # TypeId 别名 + Fnv1aHash + SAI_DECLARE_TYPE_ID 宏
│   ├── error.h                  # ErrorCode / ErrorInfo / Result<T>
│   ├── object.h                 # Object
│   ├── resource.h                # Resource
│   ├── reflectable.h             # IReflectable
│   ├── type_registry.h           # TypeInfo / Reflectable concept / TypeRegistry
│   ├── lifecycle.h                # LifecycleState
│   ├── module.h                  # IModule
│   ├── service.h                  # IService
│   ├── factory.h                  # Factory<TInterface>（纯头文件，模板全内联）
│   ├── registry.h                 # Registry<TInterface>（纯头文件，模板全内联）
│   └── context.h                  # Context
├── src/core/
│   ├── CMakeLists.txt            # add_library(sai_core ...)
│   ├── type_registry.cpp          # TypeRegistry::Instance/Resolve 非模板部分实现
│   └── context.cpp                # Context 非模板方法实现
└── tests/core/
    ├── CMakeLists.txt
    ├── type_registry_test.cpp
    ├── object_resource_test.cpp
    ├── registry_factory_test.cpp
    └── context_lifecycle_test.cpp
```

**模板与非模板方法的拆分原则**：`TypeRegistry::Register<T>()`、`Context::Register<T>()`/`Resolve<T>()` 是模板方法，必须内联在头文件里（C++ 模板的常规约束）。非模板方法（`TypeRegistry::Instance/Resolve`、`Context::Initialize/Start/Stop/RegisterModule/CurrentState`）拆到对应 `.cpp` 里实现，避免头文件膨胀。`Registry<TInterface>` 和 `Factory<TInterface>` 整个类都是模板，全内联在头文件，没有对应 `.cpp`。

## 3. CMake 组织

- 顶层 `project(surface_ai LANGUAGES CXX)`，`CMAKE_CXX_STANDARD 20`
- `sai_core` 是一个 library target（本轮先用 `STATIC`；1.3 插件体系需要动态库跨 `.so` 边界共享符号时再评估改 `SHARED`，留给未来批次决策，不在本轮引入）
- 提供 `sai::core` ALIAS target，符合现代 CMake consumer 用法习惯
- `vcpkg.json` manifest 只声明 `tl-expected` 和 `gtest` 两个依赖（`fmt`/`yaml-cpp`/`spdlog` 属于 1.6 批次，本轮不引入）
- `CMakePresets.json` 提供一个 `default` preset，指向 vcpkg toolchain 文件，本机装好 vcpkg 后 `cmake --preset default` 直接可用

## 4. 测试覆盖（GoogleTest + gmock）

- `type_registry_test.cpp`：注册成功、重复注册返回 `Core_TypeAlreadyRegistered`、查询不存在返回 `Core_TypeNotFound`、`and_then` 链式查询
- `object_resource_test.cpp`：`Object` 派生类禁止拷贝/移动的编译期断言（`static_assert(!std::is_copy_constructible_v<...>)`）；`Resource` 派生类的移动语义与 `IsValid`/`Release` 契约
- `registry_factory_test.cpp`：`Registry<TInterface>` 的注册/查询/重复/未找到；`Factory<TInterface>` 用 gmock 构造一个 mock 工厂验证 `Create()` 契约
- `context_lifecycle_test.cpp`：完整装配流程（`RegisterModule→Initialize→Start→Stop`）状态迁移验证；`OnStop` 逆序调用验证（两个 mock 模块，用共享 vector 记录调用顺序）；`Running` 状态下 `Register` 返回 `Lifecycle_RegisterAfterAssembly`

## 5. 验证方式

实际执行：
1. `git clone` vcpkg + `./bootstrap-vcpkg.sh`（本机首次安装）
2. `cmake --preset default`
3. `cmake --build --preset default`
4. `ctest --preset default` （或等效的 `ctest` 调用）

把真实终端输出作为完成证据。vcpkg 首次拉取+编译 `gtest`/`tl-expected` 端口可能耗时数分钟，这是一次性成本。

## 6. 与设计文档的对应关系

所有类名、方法签名、命名空间严格照抄 1.1/1.2 文档"4. Interfaces"章节的代码块，不做任何"顺手改进"。编码风格遵循此前确认的偏好：避免过度防御性代码、避免多层嵌套/多层 if 语句，倾向递归思想，错误处理走 `tl::expected` 的链式 `and_then`/`or_else` 风格（这与两份文档本身在"3. Design"和"14. Anti Pattern"章节里定下的风格一致，不是额外要求）。

如果实现过程中发现文档签名存在编译不通过的问题（类似文档阶段 Task 2 评审中发现的 `TypeId()`/`TypeId` 命名冲突那类），必须在实现报告中明确记录偏差原因，不静默修改文档，也不静默改变签名而不说明。

## 7. 本轮不覆盖的内容

- 1.3（插件体系）、1.5（Memory）、1.4（Runtime）、1.6（横切关注点）四份文档对应的代码，留给后续批次
- CUDA/TensorRT/FAISS 相关的任何实现
- `SHARED` 库/动态插件加载相关的构建配置调整
- CI/CD 配置（GitHub Actions 等），本轮只要求本机可编译可测试
