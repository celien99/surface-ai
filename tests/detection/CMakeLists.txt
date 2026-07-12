find_package(GTest CONFIG REQUIRED)

add_executable(sai_detection_types_test detection_types_test.cpp)
target_include_directories(sai_detection_types_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_detection_types_test PRIVATE sai::detection GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_detection_types_test)
