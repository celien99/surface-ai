add_library(sai_memory STATIC
    arena_allocator.cpp
)
add_library(sai::memory ALIAS sai_memory)

target_include_directories(sai_memory PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_memory PUBLIC sai::core)
target_compile_features(sai_memory PUBLIC cxx_std_20)
