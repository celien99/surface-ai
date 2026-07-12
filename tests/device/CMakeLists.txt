find_package(GTest CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_executable(sai_device_ring_buffer_test ring_buffer_test.cpp)
target_link_libraries(sai_device_ring_buffer_test PRIVATE sai::device GTest::gtest_main Threads::Threads)

include(GoogleTest)
gtest_discover_tests(sai_device_ring_buffer_test)

add_executable(sai_device_interface_test
    device_interface_test.cpp
    fake_camera.cpp
    fake_light_controller.cpp
)
target_link_libraries(sai_device_interface_test PRIVATE sai::device sai::image GTest::gtest_main)
gtest_discover_tests(sai_device_interface_test)
