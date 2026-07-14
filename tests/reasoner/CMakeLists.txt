find_package(GTest CONFIG REQUIRED)

set(SAI_REASONER_TEST_SOURCES
    decision_tree_test.cpp
    score_calculator_test.cpp
    reasoner_test.cpp
    integration_test.cpp
)

add_executable(sai_reasoner_tests ${SAI_REASONER_TEST_SOURCES})
target_include_directories(sai_reasoner_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_reasoner_tests PRIVATE sai::reasoner GTest::gtest GTest::gtest_main)
target_compile_features(sai_reasoner_tests PRIVATE cxx_std_20)
include(GoogleTest)
gtest_discover_tests(sai_reasoner_tests)
