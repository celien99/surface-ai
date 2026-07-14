find_package(GTest REQUIRED)

add_executable(sai_scheduler_test)
target_sources(sai_scheduler_test PRIVATE
    scheduler_test.cpp
)

target_include_directories(sai_scheduler_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(sai_scheduler_test PRIVATE
    sai::scheduler
    sai::pipeline
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sai_scheduler_test)
