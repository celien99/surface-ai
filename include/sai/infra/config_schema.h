#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <sai/core/error.h>

namespace sai::infra {

namespace detail {

// 按点分路径（例如 "capture.camera_count"）从 root 逐级下钻，返回目标节点。
// 路径任一层不存在、或中途遇到非 map 节点时，返回一个未定义节点（IsDefined()
// 为 false）。既服务 ConfigSchema::Validate 的"必需字段缺失"判定，也服务
// ConfigStore::Get<T> 的键查找，两处共用同一套路径行走语义。
[[nodiscard]] auto ResolveFieldPath(const YAML::Node& root,
                                    std::string_view field_path) -> YAML::Node;

}  // namespace detail

// 单个字段的校验闭包：接收该字段对应的 YAML 节点，返回校验结果；字段不存在这一
// 情况由 ConfigSchema::Validate 自身处理（区分"必需字段缺失"与"字段值不合法"两类
// 失败，均映射到 Infra_ConfigValidationFailed，具体原因写入 ErrorInfo::message）。
using FieldValidator = std::function<Result<void>(const YAML::Node&)>;

// JSON-Schema 等价的手写字段校验规则集合；本身不解析 YAML 文本（由 ConfigStore
// 使用 yaml-cpp 完成），只对已解析出的节点树按注册的规则逐条校验。
class ConfigSchema final {
public:
    // field_path 采用点分路径（例如 "capture.camera_count"），校验时按路径逐级
    // 下钻到对应节点；路径任一层不存在视为该必需字段缺失。
    auto RequireField(std::string field_path, FieldValidator validator)
        -> ConfigSchema&;

    // 按注册顺序逐条校验，遇到第一个失败立即返回（不收集全部错误一次性展示）。
    [[nodiscard]] auto Validate(const YAML::Node& root) const -> Result<void>;

private:
    std::vector<std::pair<std::string, FieldValidator>> rules_;
};

}  // namespace sai::infra
