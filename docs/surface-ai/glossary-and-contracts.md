# Surface AI Framework —— 术语与接口契约表

> 本文档是跨批次的活文档。每个设计批次完成后，必须在此追加自己的契约增量，
> 不得修改其他批次已提交的行（若确需修改，必须在 PR 描述中说明原因并经评审）。

## 1. 概念归属表

| 概念名称 | 归属批次 | 定义所在文档 | 说明 |
|---|---|---|---|
| `Object` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | 框架内所有一等对象的公共基类语义，禁止拷贝与移动 |
| `Result<T>` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `tl::expected<T, ErrorInfo>` 的框架别名，框架统一错误返回通道 |
| `Resource` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | RAII 资源句柄公共基类语义，允许移动、禁止拷贝 |
| `TypeId` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | 编译期 `constexpr` 字符串哈希生成的运行时类型标识，不依赖 RTTI |

## 2. 核心接口签名表

| 接口名称 | 归属批次 | 定义所在文档 | 签名摘要 |
|---|---|---|---|
| `Object` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `class Object`（虚析构，拷贝/移动均禁用，构造函数受保护） |
| `IReflectable` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `class IReflectable { virtual auto TypeId() const noexcept -> sai::TypeId = 0; }` |
| `TypeRegistry` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `class TypeRegistry { static auto Instance() noexcept -> TypeRegistry&; template<Reflectable T> auto Register() -> Result<void>; auto Resolve(TypeId) const -> Result<TypeInfo>; }` |
| `Result<T>` | 1.1 | design/milestone-01-foundation/1.1-core-foundation.md | `template<typename T> using Result = tl::expected<T, ErrorInfo>;`（`ErrorInfo` 含 `code`/`message`/`source_location`） |
