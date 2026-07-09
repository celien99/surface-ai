#include <sai/core/object.h>
#include <sai/core/resource.h>

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace {

class ConcreteObject final : public sai::Object {
public:
    ConcreteObject() = default;
};

static_assert(!std::is_copy_constructible_v<ConcreteObject>);
static_assert(!std::is_copy_assignable_v<ConcreteObject>);
static_assert(!std::is_move_constructible_v<ConcreteObject>);
static_assert(!std::is_move_assignable_v<ConcreteObject>);

class FakeResource final : public sai::Resource {
public:
    explicit FakeResource(int handle) : handle_(handle) {}

    [[nodiscard]] bool IsValid() const noexcept override { return handle_ != kInvalidHandle; }

    void Release() noexcept override { handle_ = kInvalidHandle; }

    [[nodiscard]] int Handle() const noexcept { return handle_; }

private:
    static constexpr int kInvalidHandle = -1;
    int handle_ = kInvalidHandle;
};

static_assert(!std::is_copy_constructible_v<FakeResource>);
static_assert(!std::is_copy_assignable_v<FakeResource>);

}  // namespace

TEST(ObjectTest, DerivedTypeIsConstructibleButNeverCopyableOrMovable) {
    ConcreteObject object;
    SUCCEED();
}

TEST(ResourceTest, StartsValidAndReleaseInvalidatesIt) {
    FakeResource resource(42);
    EXPECT_TRUE(resource.IsValid());

    resource.Release();

    EXPECT_FALSE(resource.IsValid());
}

TEST(ResourceTest, MoveTransfersOwnership) {
    FakeResource source(7);

    FakeResource moved(std::move(source));

    EXPECT_TRUE(moved.IsValid());
    EXPECT_EQ(moved.Handle(), 7);
}
