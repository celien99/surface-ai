find_package(GTest CONFIG REQUIRED)

add_executable(sai_io_json_exporter_test json_exporter_test.cpp)
target_link_libraries(sai_io_json_exporter_test PRIVATE sai::io GTest::gtest_main)

add_executable(sai_io_basic_importer_test basic_importer_test.cpp)
target_link_libraries(sai_io_basic_importer_test PRIVATE sai::io GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_io_json_exporter_test)
gtest_discover_tests(sai_io_basic_importer_test)
