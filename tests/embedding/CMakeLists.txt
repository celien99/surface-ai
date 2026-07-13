find_package(GTest CONFIG REQUIRED)

add_executable(sai_embedding_test embedding_test.cpp)
target_include_directories(sai_embedding_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_embedding_test PRIVATE sai::embedding GTest::gtest_main)

add_executable(sai_dimension_reducer_test dimension_reducer_test.cpp)
target_include_directories(sai_dimension_reducer_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_dimension_reducer_test PRIVATE sai::embedding GTest::gtest_main)

add_executable(sai_feature_cache_test feature_cache_test.cpp)
target_include_directories(sai_feature_cache_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_feature_cache_test PRIVATE sai::embedding GTest::gtest_main)

add_executable(sai_embedder_test embedder_test.cpp)
target_include_directories(sai_embedder_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_embedder_test PRIVATE sai::embedding sai::inference sai::image GTest::gtest_main)

add_executable(sai_pca_scoring_test pca_scoring_test.cpp)
target_include_directories(sai_pca_scoring_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_pca_scoring_test PRIVATE sai::embedding GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_embedding_test)
gtest_discover_tests(sai_dimension_reducer_test)
gtest_discover_tests(sai_feature_cache_test)
gtest_discover_tests(sai_embedder_test)
gtest_discover_tests(sai_pca_scoring_test)
