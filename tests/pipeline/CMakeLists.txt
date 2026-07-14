find_package(GTest REQUIRED)

add_executable(sai_pipeline_test)
target_sources(sai_pipeline_test PRIVATE
    pipeline_config_test.cpp
    stage_queue_test.cpp
    pipeline_builder_test.cpp
    pipeline_test.cpp
)

target_include_directories(sai_pipeline_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sai_pipeline_test PRIVATE
    sai::pipeline
    sai::scheduler
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sai_pipeline_test)
