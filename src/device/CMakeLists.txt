add_library(sai_device INTERFACE)
add_library(sai::device ALIAS sai_device)

target_include_directories(sai_device INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_device INTERFACE sai::plugin)
target_compile_features(sai_device INTERFACE cxx_std_20)
