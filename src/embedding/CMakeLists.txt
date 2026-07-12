find_package(tl-expected CONFIG REQUIRED)

add_library(sai_embedding STATIC
    embedding.cpp
)
add_library(sai::embedding ALIAS sai_embedding)

target_include_directories(sai_embedding PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_embedding PUBLIC sai::core sai::memory)
target_compile_features(sai_embedding PUBLIC cxx_std_20)
