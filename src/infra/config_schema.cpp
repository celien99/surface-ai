#include <sai/infra/config_schema.h>

#include <source_location>
#include <string>
#include <string_view>

namespace sai::infra {

namespace detail {

namespace {

// Walk the remaining dot-separated path recursively (design prefers recursion
// over nested loops for tree descent). A non-map node or a missing key at any
// level short-circuits to an undefined node.
auto Descend(const YAML::Node& node, std::string_view remaining) -> YAML::Node {
    const auto dot = remaining.find('.');
    const std::string_view head = remaining.substr(0, dot);
    if (!node.IsMap()) {
        return YAML::Node(YAML::NodeType::Undefined);
    }
    const YAML::Node child = node[std::string(head)];
    if (!child.IsDefined()) {
        return YAML::Node(YAML::NodeType::Undefined);
    }
    if (dot == std::string_view::npos) {
        return child;
    }
    return Descend(child, remaining.substr(dot + 1));
}

}  // namespace

auto ResolveFieldPath(const YAML::Node& root, std::string_view field_path)
    -> YAML::Node {
    if (field_path.empty()) {
        return YAML::Node(YAML::NodeType::Undefined);
    }
    return Descend(root, field_path);
}

}  // namespace detail

auto ConfigSchema::RequireField(std::string field_path, FieldValidator validator)
    -> ConfigSchema& {
    rules_.emplace_back(std::move(field_path), std::move(validator));
    return *this;
}

auto ConfigSchema::Validate(const YAML::Node& root) const -> Result<void> {
    for (const auto& [path, validator] : rules_) {
        const YAML::Node node = detail::ResolveFieldPath(root, path);
        if (!node.IsDefined()) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Infra_ConfigValidationFailed,
                "required config field missing: " + path,
                std::source_location::current(),
            });
        }
        auto validated = validator(node);
        if (!validated) {
            return tl::make_unexpected(ErrorInfo{
                ErrorCode::Infra_ConfigValidationFailed,
                "config field '" + path + "' failed validation: " +
                    validated.error().message,
                std::source_location::current(),
            });
        }
    }
    return {};
}

}  // namespace sai::infra
