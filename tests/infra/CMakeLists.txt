find_package(GTest CONFIG REQUIRED)

add_executable(sai_infra_logger_test logger_test.cpp)
target_link_libraries(sai_infra_logger_test
    PRIVATE sai::infra GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sai_infra_logger_test)
