# Repository Guidelines

## Project Structure & Module Organization

Surface AI is a C++20 industrial vision framework. Public interfaces live in `include/sai/`; implementations are grouped by domain in `src/`, including `detection`, `inference`, `pipeline`, and `image`. The main application is `apps/seat-aoi`, with YAML configuration under `apps/seat-aoi/resources/`. Tests mirror modules in `tests/`; cross-module scenarios belong in `tests/integration/`. Assets live in `resources/`, deployment files in `deploy/`, utilities in `tools/` and `scripts/`, and external code in `third_party/`.

## Build, Test, and Development Commands

The supported target is Ubuntu 22.04 with GCC 12, vcpkg, and optional CUDA. Set `VCPKG_ROOT` before configuring.

```bash
cmake --preset linux                 # Configure a Debug build
cmake --build --preset linux -j      # Build Debug targets
ctest --preset linux                 # Run all tests with failures shown
cmake --preset linux-release         # Configure optimized Release build
cmake --build --preset linux-release -j
ctest --preset linux -R PatchCore    # Run matching tests only
docker compose build                 # Build the deployment image
```

Build artifacts belong in `build/` and must not be committed.

## Coding Style & Naming Conventions

Use four-space indentation and existing C++20 patterns. Keep headers in `include/sai/<module>/` and implementations in matching `src/<module>/` paths. Use `PascalCase` for types and public methods, `snake_case` for variables and files, and `kPascalCase` for constants. Prefer RAII, smart pointers, `std::span`, and explicit ownership. Keep hot paths flat and avoid redundant allocations. CMake enables `-Wall -Wextra`; resolve new warnings.

## Testing Guidelines

Tests use GoogleTest and files are named `*_test.cpp`. Add unit tests beside the affected module and integration tests only for cross-module behavior. Test names should describe observable behavior, for example `FeatureBankTest.GreedySelectionUsesSmallestIndexOnTie`. Run a narrow filter first, then the full preset before submitting.

## Commit & Pull Request Guidelines

History follows Conventional Commits such as `fix(detection): ...`, `perf(coreset): ...`, and `docs(plan): ...`. Keep commits focused and summaries imperative. Pull requests should explain motivation, behavior, performance or configuration impact, and verification. Link issues; attach screenshots for QML changes and benchmark data for performance work.

## Algorithm and Configuration Safety

Do not commit model binaries, credentials, generated coresets, or machine-specific paths. Preserve PatchCore's exact greedy farthest-point semantics unless a design change is explicitly approved: start at index `0`, choose the global farthest candidate, use the smallest index on ties, and stop for duplicate vectors. CUDA changes must retain a correct CPU path and appropriate CMake capability guards.

## Git 提交规范

约定式提交 + Gitmoji。格式：`<type>(<scope>): <emoji> <中文描述>`，emoji 与描述间**必须有一个空格**，直接用字符不用 `:code:`。允许的 type/emoji（**严禁自造**）：

- `feat`: ✨ 新功能
- `fix`: 🐛 修复 Bug
- `chore`: 🔧 构建/工具/依赖/日常
- `refactor`: ♻️ 重构（无新功能无修复）
- `docs`: 📝 仅文档/注释
- `style`: 💄 不影响含义的格式
- `perf`: ⚡ 性能优化
- `test`: ✅ 测试
- `ci`: 💚 CI/CD 配置

示例：`fix(cpp): 🐛 修复 frame_by_index 使用 std::map 字母序导致取错帧`
