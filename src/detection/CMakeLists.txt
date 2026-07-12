find_package(tl-expected CONFIG REQUIRED)

add_library(sai_detection STATIC
    detection_result.cpp
)
add_library(sai::detection ALIAS sai_detection)

target_include_directories(sai_detection PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_detection PUBLIC sai::core)
target_compile_features(sai_detection PUBLIC cxx_std_20)
