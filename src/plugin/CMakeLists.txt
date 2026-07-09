find_package(yaml-cpp CONFIG REQUIRED)

add_library(sai_plugin STATIC
    manifest.cpp
    version_manager.cpp
    capability_manager.cpp
    license_manager.cpp
)
add_library(sai::plugin ALIAS sai_plugin)

target_include_directories(sai_plugin PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_plugin PUBLIC sai::core yaml-cpp::yaml-cpp)
target_compile_features(sai_plugin PUBLIC cxx_std_20)
