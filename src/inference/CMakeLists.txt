find_package(tl-expected CONFIG REQUIRED)

add_library(sai_inference STATIC
    mock_engine.cpp
)
add_library(sai::inference ALIAS sai_inference)

target_include_directories(sai_inference PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_inference PUBLIC sai::core)
target_compile_features(sai_inference PUBLIC cxx_std_20)
