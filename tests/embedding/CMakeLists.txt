find_package(GTest CONFIG REQUIRED)

add_executable(sai_embedding_test embedding_test.cpp)
target_include_directories(sai_embedding_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_embedding_test PRIVATE sai::embedding GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_embedding_test)
