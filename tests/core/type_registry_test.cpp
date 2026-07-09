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
