#include "sai/rule/value.h"
#include <cmath>
#include <stdexcept>

namespace sai::rule {

auto Value::Null() -> Value { return Value{}; }

auto Value::Of(double v) -> Value {
    Value val;
    val.kind_ = Kind::Scalar;
    val.scalar_ = Scalar{v};
    return val;
}

auto Value::Of(std::string v) -> Value {
    Value val;
    val.kind_ = Kind::Scalar;
    val.scalar_ = Scalar{std::move(v)};
    return val;
}

auto Value::Of(bool v) -> Value {
    Value val;
    val.kind_ = Kind::Scalar;
    val.scalar_ = Scalar{v};
    return val;
}

auto Value::Of(std::int64_t v) -> Value {
    Value val;
    val.kind_ = Kind::Scalar;
    val.scalar_ = Scalar{static_cast<double>(v)};
    return val;
}

auto Value::OfList(std::vector<Value> v) -> Value {
    Value val;
    val.kind_ = Kind::List;
    val.list_ = std::move(v);
    return val;
}

auto Value::GetKind() const noexcept -> Kind { return kind_; }
auto Value::IsNull() const -> bool { return kind_ == Kind::Null; }

auto Value::AsDouble() const -> std::optional<double> {
    if (!scalar_.has_value()) return std::nullopt;
    return std::get_if<double>(&*scalar_) ? std::optional{std::get<double>(*scalar_)} : std::nullopt;
}

auto Value::AsString() const -> std::optional<std::string_view> {
    if (!scalar_.has_value()) return std::nullopt;
    auto* s = std::get_if<std::string>(&*scalar_);
    return s ? std::optional{std::string_view{*s}} : std::nullopt;
}

auto Value::AsBool() const -> std::optional<bool> {
    if (!scalar_.has_value()) return std::nullopt;
    auto* b = std::get_if<bool>(&*scalar_);
    return b ? std::optional{*b} : std::nullopt;
}

auto Value::AsList() const -> const std::vector<Value>* {
    if (!list_.has_value()) return nullptr;
    return &*list_;
}

namespace {
auto AsNumeric(const Value& v) -> std::optional<double> {
    if (auto d = v.AsDouble()) return d;
    if (auto b = v.AsBool()) return *b ? 1.0 : 0.0;
    return std::nullopt;
}
}  // namespace

auto Value::operator+(const Value& rhs) const -> Value {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return Null();
    return Of(*a + *b);
}

auto Value::operator-(const Value& rhs) const -> Value {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return Null();
    return Of(*a - *b);
}

auto Value::operator*(const Value& rhs) const -> Value {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return Null();
    return Of(*a * *b);
}

auto Value::operator/(const Value& rhs) const -> Value {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b || *b == 0.0) return Null();
    return Of(*a / *b);
}

auto Value::operator==(const Value& rhs) const -> bool {
    if (kind_ != rhs.kind_) return false;
    if (kind_ == Kind::Null) return true;
    if (kind_ == Kind::Scalar) return scalar_ == rhs.scalar_;
    // List comparison: size + element-wise
    if (!list_ || !rhs.list_) return false;
    if (list_->size() != rhs.list_->size()) return false;
    for (size_t i = 0; i < list_->size(); ++i) {
        if (!((*list_)[i] == (*rhs.list_)[i])) return false;
    }
    return true;
}

auto Value::operator<(const Value& rhs) const -> bool {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return false;
    return *a < *b;
}

auto Value::operator>(const Value& rhs) const -> bool {
    auto a = AsNumeric(*this);
    auto b = AsNumeric(rhs);
    if (!a || !b) return false;
    return *a > *b;
}

}  // namespace sai::rule
