find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

# image_types_test compiles host_test_pool.cpp directly, whose host_test_pool.h
# static_asserts std::atomic<TaggedHead> (16 bytes) is always lock-free. Gate on
# the same CMake configure-time check used by src/memory and tests/memory.
if(SAI_MEMORY_HAS_LOCKFREE_16)
    add_executable(sai_image_types_test
        image_types_test.cpp
        ${CMAKE_SOURCE_DIR}/tests/memory/host_test_pool.cpp
    )
    target_link_libraries(sai_image_types_test PRIVATE sai::image GTest::gtest_main)
    target_include_directories(sai_image_types_test PRIVATE ${CMAKE_SOURCE_DIR}/tests/memory)

    # -mcx16: required for CMPXCHG16B on x86-64 (16-byte lock-free atomics).
    # Invalid on arm64, so guard by arch.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(sai_image_types_test PRIVATE -mcx16)
    endif()

    gtest_discover_tests(sai_image_types_test)
else()
    message(WARNING "Lock-free atomics NOT available — "
                    "skipping sai_image_types_test (uses host_test_pool.h)")
endif()

add_executable(sai_image_roi_test roi_test.cpp)
target_link_libraries(sai_image_roi_test PRIVATE sai::image GTest::gtest_main)

add_executable(sai_image_preprocess_test preprocess_test.cpp)
target_link_libraries(sai_image_preprocess_test PRIVATE sai::image GTest::gtest_main)

gtest_discover_tests(sai_image_roi_test)
gtest_discover_tests(sai_image_preprocess_test)
