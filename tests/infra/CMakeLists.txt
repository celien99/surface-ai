find_package(GTest CONFIG REQUIRED)

add_executable(sai_infra_logger_test logger_test.cpp)
target_link_libraries(sai_infra_logger_test
    PRIVATE sai::infra GTest::gtest_main
)

add_executable(sai_infra_config_store_test config_store_test.cpp)
target_link_libraries(sai_infra_config_store_test
    PRIVATE sai::infra GTest::gtest_main
)

add_executable(sai_infra_daily_and_size_sink_test daily_and_size_sink_test.cpp)
target_link_libraries(sai_infra_daily_and_size_sink_test
    PRIVATE sai::infra GTest::gtest_main
)

add_executable(sai_infra_logger_dropped_count_test logger_dropped_count_test.cpp)
target_link_libraries(sai_infra_logger_dropped_count_test
    PRIVATE sai::infra GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sai_infra_logger_test)
gtest_discover_tests(sai_infra_logger_dropped_count_test)
gtest_discover_tests(sai_infra_config_store_test)
gtest_discover_tests(sai_infra_daily_and_size_sink_test)
