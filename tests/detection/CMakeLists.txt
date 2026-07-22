find_package(GTest CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

find_package(faiss CONFIG REQUIRED)

add_executable(sai_detection_types_test detection_types_test.cpp)
target_include_directories(sai_detection_types_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_detection_types_test PRIVATE sai::detection GTest::gtest_main)

add_executable(sai_patch_core_test patch_core_test.cpp)
target_include_directories(sai_patch_core_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_patch_core_test PRIVATE sai::detection GTest::gtest_main)

add_executable(sai_pca_detector_test pca_detector_test.cpp)
target_include_directories(sai_pca_detector_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_pca_detector_test PRIVATE sai::detection GTest::gtest_main)

add_executable(sai_coreset_evolution_test coreset_evolution_test.cpp)
target_include_directories(sai_coreset_evolution_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_coreset_evolution_test PRIVATE sai::detection GTest::gtest_main)

add_executable(sai_feature_bank_test feature_bank_test.cpp)
target_include_directories(sai_feature_bank_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_feature_bank_test PRIVATE sai::detection GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_detection_types_test)
gtest_discover_tests(sai_patch_core_test)
gtest_discover_tests(sai_pca_detector_test)
gtest_discover_tests(sai_coreset_evolution_test)
gtest_discover_tests(sai_feature_bank_test)
