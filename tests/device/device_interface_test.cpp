#include <memory>

#include <sai/device/device.h>

#include "fake_camera.h"
#include "fake_light_controller.h"

#include <gtest/gtest.h>

// ============================================================================
// Rect
// ============================================================================

TEST(Rect, Area) {
    sai::device::Rect r{0, 0, 100, 200};
    EXPECT_EQ(r.Area(), 20000U);
    EXPECT_FALSE(r.IsEmpty());
}

TEST(Rect, IsEmptyZeroWidth) {
    sai::device::Rect r{0, 0, 0, 200};
    EXPECT_TRUE(r.IsEmpty());
}

TEST(Rect, IsEmptyZeroHeight) {
    sai::device::Rect r{0, 0, 100, 0};
    EXPECT_TRUE(r.IsEmpty());
}

TEST(Rect, DefaultIsEmpty) {
    sai::device::Rect r{};
    EXPECT_EQ(r.Area(), 0U);
    EXPECT_TRUE(r.IsEmpty());
}

// ============================================================================
// IDevice::State enum values
// ============================================================================

TEST(DeviceState, EnumValues) {
    using State = sai::device::IDevice::State;
    EXPECT_NE(static_cast<int>(State::Disconnected), static_cast<int>(State::Connected));
    EXPECT_NE(static_cast<int>(State::Connected), static_cast<int>(State::Acquiring));
    EXPECT_NE(static_cast<int>(State::Acquiring), static_cast<int>(State::Error));
}

// ============================================================================
// FakeCamera state machine
// ============================================================================

class FakeCameraTest : public ::testing::Test {
protected:
    void SetUp() override { camera_ = std::make_unique<sai::test::FakeCamera>(); }
    std::unique_ptr<sai::test::FakeCamera> camera_;
};

TEST_F(FakeCameraTest, InitialStateIsDisconnected) {
    EXPECT_EQ(camera_->CurrentState(), sai::device::IDevice::State::Disconnected);
    EXPECT_FALSE(camera_->IsConnected());
}

TEST_F(FakeCameraTest, ConnectTransitionsToConnected) {
    auto result = camera_->Connect();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(camera_->CurrentState(), sai::device::IDevice::State::Connected);
    EXPECT_TRUE(camera_->IsConnected());
}

TEST_F(FakeCameraTest, ConnectTwiceFails) {
    ASSERT_TRUE(camera_->Connect().has_value());
    auto result = camera_->Connect();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Device_ConnectionFailed);
}

TEST_F(FakeCameraTest, StartAcquisitionBeforeConnectFails) {
    auto result = camera_->StartAcquisition();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Device_NotConnected);
}

TEST_F(FakeCameraTest, StartAcquisitionTransitionsToAcquiring) {
    ASSERT_TRUE(camera_->Connect().has_value());
    auto result = camera_->StartAcquisition();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(camera_->CurrentState(), sai::device::IDevice::State::Acquiring);
}

TEST_F(FakeCameraTest, StartAcquisitionTwiceFails) {
    ASSERT_TRUE(camera_->Connect().has_value());
    ASSERT_TRUE(camera_->StartAcquisition().has_value());
    auto result = camera_->StartAcquisition();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Device_AcquisitionInProgress);
}

TEST_F(FakeCameraTest, StopAcquisitionTransitionsBackToConnected) {
    ASSERT_TRUE(camera_->Connect().has_value());
    ASSERT_TRUE(camera_->StartAcquisition().has_value());
    auto result = camera_->StopAcquisition();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(camera_->CurrentState(), sai::device::IDevice::State::Connected);
}

TEST_F(FakeCameraTest, StopAcquisitionWhenNotAcquiringFails) {
    ASSERT_TRUE(camera_->Connect().has_value());
    auto result = camera_->StopAcquisition();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Device_NotConnected);
}

TEST_F(FakeCameraTest, DisconnectFromConnected) {
    ASSERT_TRUE(camera_->Connect().has_value());
    auto result = camera_->Disconnect();
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(camera_->CurrentState(), sai::device::IDevice::State::Disconnected);
    EXPECT_FALSE(camera_->IsConnected());
}

TEST_F(FakeCameraTest, DisconnectWhileAcquiringFails) {
    ASSERT_TRUE(camera_->Connect().has_value());
    ASSERT_TRUE(camera_->StartAcquisition().has_value());
    auto result = camera_->Disconnect();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Device_AcquisitionInProgress);
}

TEST_F(FakeCameraTest, FullStateMachineCycle) {
    // Disconnected -> Connected -> Acquiring -> Connected -> Disconnected
    ASSERT_TRUE(camera_->Connect().has_value());
    ASSERT_TRUE(camera_->StartAcquisition().has_value());
    ASSERT_TRUE(camera_->StopAcquisition().has_value());
    ASSERT_TRUE(camera_->Disconnect().has_value());
    EXPECT_EQ(camera_->CurrentState(), sai::device::IDevice::State::Disconnected);
}

TEST_F(FakeCameraTest, SerialNumberIsNonEmpty) {
    EXPECT_FALSE(camera_->SerialNumber().empty());
}

// ============================================================================
// FakeCamera camera-specific settings
// ============================================================================

TEST_F(FakeCameraTest, SetTriggerMode) {
    auto result = camera_->SetTriggerMode(sai::device::ICamera::TriggerMode::Hardware);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(camera_->GetTriggerMode(), sai::device::ICamera::TriggerMode::Hardware);
}

TEST_F(FakeCameraTest, SetExposureTime) {
    auto result = camera_->SetExposureTime(std::chrono::microseconds(500));
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(camera_->GetExposureTime(), std::chrono::microseconds(500));
}

TEST_F(FakeCameraTest, SetGain) {
    auto result = camera_->SetGain(3.5f);
    EXPECT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(camera_->GetGain(), 3.5f);
}

TEST_F(FakeCameraTest, SetROI) {
    sai::device::Rect roi{10, 20, 640, 480};
    auto result = camera_->SetROI(roi);
    EXPECT_TRUE(result.has_value());
    auto actual = camera_->GetROI();
    EXPECT_EQ(actual.x, 10U);
    EXPECT_EQ(actual.y, 20U);
    EXPECT_EQ(actual.width, 640U);
    EXPECT_EQ(actual.height, 480U);
}

TEST_F(FakeCameraTest, RegisterFrameCallbackRequiresConnection) {
    // RawImage is incomplete until Task 4, so we cannot build a callback that
    // takes one by value; an empty FrameCallback is enough to exercise the
    // state-gate (must be connected first). The fake only stores the callback;
    // invoking it with a real RawImage lands in Task 10's integration test.
    auto result = camera_->RegisterFrameCallback(sai::device::ICamera::FrameCallback{});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, sai::ErrorCode::Device_NotConnected);
}

TEST_F(FakeCameraTest, RegisterFrameCallbackAfterConnect) {
    ASSERT_TRUE(camera_->Connect().has_value());
    auto result = camera_->RegisterFrameCallback(sai::device::ICamera::FrameCallback{});
    EXPECT_TRUE(result.has_value());
}

// ============================================================================
// FakeLightController
// ============================================================================

class FakeLightControllerTest : public ::testing::Test {
protected:
    void SetUp() override { light_ = std::make_unique<sai::test::FakeLightController>(4); }
    std::unique_ptr<sai::test::FakeLightController> light_;
};

TEST_F(FakeLightControllerTest, DefaultChannelCount) {
    EXPECT_EQ(light_->ChannelCount(), 4);
}

TEST_F(FakeLightControllerTest, InitialState) {
    EXPECT_EQ(light_->CurrentState(), sai::device::IDevice::State::Disconnected);
    EXPECT_FALSE(light_->IsConnected());
}

TEST_F(FakeLightControllerTest, ConnectDisconnect) {
    ASSERT_TRUE(light_->Connect().has_value());
    EXPECT_TRUE(light_->IsConnected());
    ASSERT_TRUE(light_->Disconnect().has_value());
    EXPECT_FALSE(light_->IsConnected());
}

TEST_F(FakeLightControllerTest, SetAndGetIntensity) {
    auto set_result = light_->SetIntensity(0, 0.75f);
    ASSERT_TRUE(set_result.has_value());
    auto get_result = light_->GetIntensity(0);
    ASSERT_TRUE(get_result.has_value());
    EXPECT_FLOAT_EQ(*get_result, 0.75f);
}

TEST_F(FakeLightControllerTest, OutOfRangeChannel) {
    auto result = light_->SetIntensity(99, 1.0f);
    EXPECT_FALSE(result.has_value());
}

TEST_F(FakeLightControllerTest, EnableAndDisableChannel) {
    ASSERT_TRUE(light_->Enable(2).has_value());
    ASSERT_TRUE(light_->Disable(2).has_value());
}

TEST_F(FakeLightControllerTest, SetStrobeMode) {
    auto result = light_->SetStrobeMode(1, sai::device::ILightController::StrobeMode::OnTrigger);
    EXPECT_TRUE(result.has_value());
}

TEST_F(FakeLightControllerTest, SetStrobeModeOutOfRange) {
    auto result = light_->SetStrobeMode(-1, sai::device::ILightController::StrobeMode::Continuous);
    EXPECT_FALSE(result.has_value());
}

TEST_F(FakeLightControllerTest, SerialNumber) {
    EXPECT_FALSE(light_->SerialNumber().empty());
}
