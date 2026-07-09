find_package(GTest CONFIG REQUIRED)

add_executable(sai_core_smoke_test smoke_test.cpp)
target_link_libraries(sai_core_smoke_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_core_smoke_test)
