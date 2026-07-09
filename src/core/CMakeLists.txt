find_package(tl-expected CONFIG REQUIRED)

add_library(sai_core STATIC
    type_registry.cpp
)
add_library(sai::core ALIAS sai_core)

target_include_directories(sai_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_core PUBLIC tl::expected)
target_compile_features(sai_core PUBLIC cxx_std_20)
