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

# These two targets compile host_test_pool.cpp directly, whose host_test_pool.h
# static_asserts std::atomic<TaggedHead> (16 bytes) is always lock-free. That
# needs -mcx16 (CMPXCHG16B) on x86-64; the flag is invalid on arm64, so guard it.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(sai_memory_pooled_ptr_test PRIVATE -mcx16)
    target_compile_options(sai_memory_host_test_pool_concurrency_test PRIVATE -mcx16)
endif()
