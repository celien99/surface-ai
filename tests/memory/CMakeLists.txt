find_package(GTest CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_executable(sai_memory_arena_allocator_test arena_allocator_test.cpp)
target_link_libraries(sai_memory_arena_allocator_test PRIVATE sai::memory GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_memory_arena_allocator_test)

add_executable(sai_memory_pooled_ptr_test pooled_ptr_test.cpp host_test_pool.cpp)
target_link_libraries(sai_memory_pooled_ptr_test PRIVATE sai::memory GTest::gtest_main)
gtest_discover_tests(sai_memory_pooled_ptr_test)

add_executable(sai_memory_host_test_pool_concurrency_test
    host_test_pool_concurrency_test.cpp
    host_test_pool.cpp
)
target_link_libraries(sai_memory_host_test_pool_concurrency_test
    PRIVATE sai::memory GTest::gtest_main Threads::Threads
)
gtest_discover_tests(sai_memory_host_test_pool_concurrency_test)
