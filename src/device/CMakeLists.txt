add_library(sai_device STATIC
    fake_camera.cpp
)
add_library(sai::device ALIAS sai_device)

target_include_directories(sai_device PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_device PUBLIC
    sai::plugin
    sai::image
)
target_compile_features(sai_device PUBLIC cxx_std_20)
