find_package(GTest CONFIG REQUIRED)

add_executable(sai_image_types_test
    image_types_test.cpp
    ${CMAKE_SOURCE_DIR}/tests/memory/host_test_pool.cpp
)
target_link_libraries(sai_image_types_test PRIVATE sai::image GTest::gtest_main)
target_include_directories(sai_image_types_test PRIVATE ${CMAKE_SOURCE_DIR}/tests/memory)

add_executable(sai_image_roi_test roi_test.cpp)
target_link_libraries(sai_image_roi_test PRIVATE sai::image GTest::gtest_main)

add_executable(sai_image_preprocess_test preprocess_test.cpp)
target_link_libraries(sai_image_preprocess_test PRIVATE sai::image GTest::gtest_main)

# image_types_test compiles host_test_pool.cpp directly, whose host_test_pool.h
# static_asserts std::atomic<TaggedHead> (16 bytes) is always lock-free. That needs
# -mcx16 (CMPXCHG16B) on x86-64; the flag is invalid on arm64, so guard it by arch.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(sai_image_types_test PRIVATE -mcx16)
endif()

include(GoogleTest)
gtest_discover_tests(sai_image_types_test)
gtest_discover_tests(sai_image_roi_test)
gtest_discover_tests(sai_image_preprocess_test)
