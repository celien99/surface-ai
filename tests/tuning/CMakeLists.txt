find_package(GTest CONFIG REQUIRED)

add_executable(sai_tuning_test tuning_test.cpp)
target_include_directories(sai_tuning_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_tuning_test PRIVATE sai::tuning GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_tuning_test)
