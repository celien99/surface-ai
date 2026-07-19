find_package(GTest CONFIG REQUIRED)

add_executable(sai_inference_mock_engine_test mock_engine_test.cpp mock_engine.cpp)
target_include_directories(sai_inference_mock_engine_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_inference_mock_engine_test PRIVATE sai::core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_inference_mock_engine_test)
