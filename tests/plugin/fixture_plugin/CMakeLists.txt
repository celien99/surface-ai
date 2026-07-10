# A tiny real shared library exercised by plugin_manager_test.cpp via a
# genuine dlopen — never linked directly into any test executable. Whatever
# extension the platform produces (.dylib here, .so on the target platform)
# is fine; nothing in this file or the test hardcodes it, see
# FIXTURE_PLUGIN_LIBRARY_FILE below.
add_library(fixture_plugin SHARED
    fixture_plugin.cpp
)

target_include_directories(fixture_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fixture_plugin PRIVATE sai::plugin)
target_compile_features(fixture_plugin PRIVATE cxx_std_20)

# Exposes the exact absolute path CMake will build the shared library at
# (correct on every platform/config, no prefix/suffix guessing) to whichever
# target links against this property, so a test can compose a plugin.yaml
# whose library_path field is guaranteed to resolve to the real artifact.
set(FIXTURE_PLUGIN_LIBRARY_FILE "$<TARGET_FILE:fixture_plugin>" PARENT_SCOPE)
