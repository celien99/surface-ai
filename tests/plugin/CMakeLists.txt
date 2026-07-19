find_package(GTest CONFIG REQUIRED)

add_executable(sai_plugin_manifest_validators_test manifest_validators_test.cpp)
target_link_libraries(sai_plugin_manifest_validators_test PRIVATE sai::plugin GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_plugin_manifest_validators_test)

# The fixture plugin is a real SHARED library, dlopen'd by plugin_manager_test
# at test time — deliberately not linked into the test executable below (that
# would resolve its symbols statically and defeat the point of this test
# strategy).
add_subdirectory(fixture_plugin)

add_executable(sai_plugin_plugin_manager_test plugin_manager_test.cpp)
target_link_libraries(sai_plugin_plugin_manager_test PRIVATE sai::plugin GTest::gtest_main)
target_compile_definitions(sai_plugin_plugin_manager_test PRIVATE
    FIXTURE_PLUGIN_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixture_plugin"
    FIXTURE_PLUGIN_LIBRARY_FILE="${FIXTURE_PLUGIN_LIBRARY_FILE}"
)
# fixture_plugin is not linked above, so CMake would not otherwise know the
# test executable must wait for it to finish building before it can dlopen it.
add_dependencies(sai_plugin_plugin_manager_test fixture_plugin)
gtest_discover_tests(sai_plugin_plugin_manager_test)
