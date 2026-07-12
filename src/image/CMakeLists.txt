add_library(sai_image STATIC
    image.cpp
    raw_image.cpp
    surface_image.cpp
    roi.cpp
)
add_library(sai::image ALIAS sai_image)

target_include_directories(sai_image PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_image PUBLIC sai::core sai::memory sai::device)
target_compile_features(sai_image PUBLIC cxx_std_20)
