find_package(GTest CONFIG REQUIRED)

set(SAI_RETRIEVAL_TEST_SOURCES
    vector_path_test.cpp
    metadata_path_test.cpp
    score_fusion_test.cpp
    hybrid_retriever_test.cpp
)

add_executable(sai_retrieval_tests ${SAI_RETRIEVAL_TEST_SOURCES})
target_include_directories(sai_retrieval_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_retrieval_tests PRIVATE sai::retrieval GTest::gtest GTest::gtest_main)
target_compile_features(sai_retrieval_tests PRIVATE cxx_std_20)
include(GoogleTest)
gtest_discover_tests(sai_retrieval_tests)
