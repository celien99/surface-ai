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

# Task 9: M3 推理管线端到端集成测试
add_executable(sai_integration_inference_pipeline_test
    inference_pipeline_test.cpp
    ${CMAKE_SOURCE_DIR}/tests/inference/mock_engine.cpp
)
target_include_directories(sai_integration_inference_pipeline_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sai_integration_inference_pipeline_test PRIVATE
    sai::detection
    sai::image
    GTest::gtest_main
)

# M4: 知识写入 → 混合检索 端到端集成测试
add_executable(sai_integration_knowledge_retrieval_pipeline_test
    knowledge_retrieval_pipeline_test.cpp
)
target_include_directories(sai_integration_knowledge_retrieval_pipeline_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sai_integration_knowledge_retrieval_pipeline_test PRIVATE
    sai::knowledge
    sai::retrieval
    GTest::gtest_main
)

# M5: Rule → Reasoner 端到端集成测试
add_executable(sai_integration_rule_reasoner_pipeline_test
    rule_reasoner_pipeline_test.cpp
)
target_include_directories(sai_integration_rule_reasoner_pipeline_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sai_integration_rule_reasoner_pipeline_test PRIVATE
    sai::reasoner
    sai::rule
    sai::knowledge
    sai::retrieval
    sai::detection
    GTest::gtest_main
)
# M6: Pipeline E2E 集成测试
add_executable(sai_integration_m6_e2e_pipeline_test
    m6_e2e_pipeline_test.cpp
)
target_include_directories(sai_integration_m6_e2e_pipeline_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sai_integration_m6_e2e_pipeline_test PRIVATE
    sai::pipeline
    GTest::gtest_main
)
# P1: End-to-end regression test (full pipeline)
add_executable(sai_integration_end_to_end_regression_test
    end_to_end_regression_test.cpp
)
target_include_directories(sai_integration_end_to_end_regression_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sai_integration_end_to_end_regression_test PRIVATE
    sai::detection
    sai::embedding
    sai::image
    sai::io
    sai::knowledge
    sai::rule
    sai::reasoner
    sai::pipeline
    GTest::gtest_main
)
# Task 6: KnowledgeEvolution integration
add_executable(sai_coreset_evolution_integration_test
    coreset_evolution_integration_test.cpp)
target_include_directories(sai_coreset_evolution_integration_test
    PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_coreset_evolution_integration_test
    PRIVATE sai::detection sai::knowledge GTest::gtest_main)

# Task 5: Bayesian Auto-Tuning end-to-end integration
add_executable(sai_tuning_integration_test
    tuning_integration_test.cpp)
target_include_directories(sai_tuning_integration_test
    PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_tuning_integration_test
    PRIVATE sai::tuning sai::knowledge sai::reasoner GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_integration_acquisition_to_export_test)
gtest_discover_tests(sai_integration_inference_pipeline_test)
gtest_discover_tests(sai_integration_knowledge_retrieval_pipeline_test)
gtest_discover_tests(sai_integration_rule_reasoner_pipeline_test)
gtest_discover_tests(sai_integration_m6_e2e_pipeline_test)
gtest_discover_tests(sai_integration_end_to_end_regression_test)
gtest_discover_tests(sai_coreset_evolution_integration_test)
gtest_discover_tests(sai_tuning_integration_test)
