find_package(GTest CONFIG REQUIRED)

set(SAI_KNOWLEDGE_TEST_SOURCES
    knowledge_graph_test.cpp
    knowledge_evolution_test.cpp
    knowledge_snapshot_test.cpp
    knowledge_store_test.cpp
)

add_executable(sai_knowledge_tests ${SAI_KNOWLEDGE_TEST_SOURCES})
target_include_directories(sai_knowledge_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_knowledge_tests PRIVATE sai::knowledge GTest::gtest GTest::gtest_main)
target_compile_features(sai_knowledge_tests PRIVATE cxx_std_20)
include(GoogleTest)
gtest_discover_tests(sai_knowledge_tests)
