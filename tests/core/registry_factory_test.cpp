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
