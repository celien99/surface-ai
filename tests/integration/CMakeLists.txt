find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(sai_integration_acquisition_to_export_test
    acquisition_to_export_test.cpp
    ${CMAKE_SOURCE_DIR}/tests/device/fake_camera.cpp
)
target_link_libraries(sai_integration_acquisition_to_export_test PRIVATE
    sai::device
    sai::image
    sai::io
    sai::core
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)
target_include_directories(sai_integration_acquisition_to_export_test PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/device
)

include(GoogleTest)
gtest_discover_tests(sai_integration_acquisition_to_export_test)
