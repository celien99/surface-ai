#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sai::rule {

using Scalar = std::variant<double, std::string, bool>;

class Value {
public:
    enum class Kind { Null, Scalar, List };

    static auto Null() -> Value;
    static auto Of(double v) -> Value;
    static auto Of(std::string v) -> Value;
    static auto Of(bool v) -> Value;
    static auto OfList(std::vector<Value> v) -> Value;

    auto GetKind() const noexcept -> Kind;
    auto IsNull() const -> bool;

    auto AsDouble() const -> std::optional<double>;
    auto AsString() const -> std::optional<std::string_view>;
    auto AsBool() const -> std::optional<bool>;
    auto AsList() const -> const std::vector<Value>*;

    auto operator+(const Value& rhs) const -> Value;
    auto operator-(const Value& rhs) const -> Value;
    auto operator*(const Value& rhs) const -> Value;
    auto operator/(const Value& rhs) const -> Value;

    auto operator==(const Value& rhs) const -> bool;
    auto operator<(const Value& rhs) const -> bool;
    auto operator>(const Value& rhs) const -> bool;

private:
    Kind kind_{Kind::Null};
    std::optional<Scalar> scalar_;
    std::optional<std::vector<Value>> list_;
};

}  // namespace sai::rule
