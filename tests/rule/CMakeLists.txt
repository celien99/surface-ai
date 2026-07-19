find_package(GTest CONFIG REQUIRED)

set(SAI_RULE_TEST_SOURCES
    value_test.cpp
    expression_test.cpp
    fact_base_test.cpp
    fact_builder_test.cpp
    rule_engine_test.cpp
    conflict_resolver_test.cpp
    integration_test.cpp
)

add_executable(sai_rule_tests ${SAI_RULE_TEST_SOURCES})
target_include_directories(sai_rule_tests PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_compile_definitions(sai_rule_tests PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(sai_rule_tests PRIVATE sai::rule GTest::gtest GTest::gtest_main)
target_compile_features(sai_rule_tests PRIVATE cxx_std_20)
include(GoogleTest)
gtest_discover_tests(sai_rule_tests)
