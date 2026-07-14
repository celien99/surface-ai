#include <sai/rule/value.h>
#include <gtest/gtest.h>

namespace sai::rule {
namespace {

TEST(ValueTest, NullByDefault) {
    Value v;
    EXPECT_EQ(v.GetKind(), Value::Kind::Null);
    EXPECT_TRUE(v.IsNull());
}

TEST(ValueTest, OfDoubleRoundTrips) {
    Value v = Value::Of(3.14);
    EXPECT_EQ(v.GetKind(), Value::Kind::Scalar);
    EXPECT_FALSE(v.IsNull());
    auto d = v.AsDouble();
    ASSERT_TRUE(d.has_value());
    EXPECT_DOUBLE_EQ(*d, 3.14);
}

TEST(ValueTest, OfStringRoundTrips) {
    Value v = Value::Of(std::string("hello"));
    EXPECT_EQ(v.GetKind(), Value::Kind::Scalar);
    auto s = v.AsString();
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(*s, "hello");
}

TEST(ValueTest, OfBoolRoundTrips) {
    Value v = Value::Of(true);
    auto b = v.AsBool();
    ASSERT_TRUE(b.has_value());
    EXPECT_TRUE(*b);
}

TEST(ValueTest, OfListRoundTrips) {
    auto v = Value::OfList({Value::Of(1.0), Value::Of(2.0), Value::Of(3.0)});
    EXPECT_EQ(v.GetKind(), Value::Kind::List);
    const auto* lst = v.AsList();
    ASSERT_NE(lst, nullptr);
    EXPECT_EQ(lst->size(), 3);
}

TEST(ValueTest, TypeMismatchReturnsNullopt) {
    Value v = Value::Of(42.0);
    EXPECT_FALSE(v.AsString().has_value());
    EXPECT_FALSE(v.AsBool().has_value());
    EXPECT_EQ(v.AsList(), nullptr);
}

TEST(ValueTest, AddDoubles) {
    Value a = Value::Of(3.0);
    Value b = Value::Of(2.0);
    Value c = a + b;
    auto d = c.AsDouble();
    ASSERT_TRUE(d.has_value());
    EXPECT_DOUBLE_EQ(*d, 5.0);
}

TEST(ValueTest, AddWithNullReturnsNull) {
    Value a = Value::Null();
    Value b = Value::Of(2.0);
    Value c = a + b;
    EXPECT_TRUE(c.IsNull());
}

TEST(ValueTest, MulDoubles) {
    Value a = Value::Of(3.0);
    Value b = Value::Of(2.0);
    Value c = a * b;
    auto d = c.AsDouble();
    ASSERT_TRUE(d.has_value());
    EXPECT_DOUBLE_EQ(*d, 6.0);
}

TEST(ValueTest, DivByZeroReturnsNull) {
    Value a = Value::Of(5.0);
    Value b = Value::Of(0.0);
    Value c = a / b;
    EXPECT_TRUE(c.IsNull());
}

TEST(ValueTest, StringEquality) {
    Value a = Value::Of(std::string("abc"));
    Value b = Value::Of(std::string("abc"));
    Value c = Value::Of(std::string("xyz"));
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(ValueTest, NumericComparison) {
    Value a = Value::Of(5.0);
    Value b = Value::Of(3.0);
    EXPECT_TRUE(b < a);
    EXPECT_TRUE(a > b);
}

TEST(ValueTest, CrossTypeComparisonReturnsFalse) {
    Value a = Value::Of(5.0);
    Value b = Value::Of(std::string("hello"));
    EXPECT_FALSE(a == b);
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a > b);
}

}  // namespace
}  // namespace sai::rule
