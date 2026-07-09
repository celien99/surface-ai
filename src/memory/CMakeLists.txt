# GpuPool/PinnedPool (gpu_pool.cpp/pinned_pool.cpp) wrap real CUDA Runtime API
# calls (cudaMalloc/cudaFree/cudaHostAlloc/cudaFreeHost) for the target
# platform (Ubuntu x64 + NVIDIA GPU). There is no dev-machine fallback
# implementation — on hosts without the CUDA Toolkit (e.g. this macOS arm64
# dev machine), these two files are simply absent from the source list below
# and are never compiled.
find_package(CUDAToolkit QUIET)

set(SAI_MEMORY_SOURCES arena_allocator.cpp)
if(CUDAToolkit_FOUND)
    list(APPEND SAI_MEMORY_SOURCES gpu_pool.cpp pinned_pool.cpp)
endif()

add_library(sai_memory STATIC ${SAI_MEMORY_SOURCES})
add_library(sai::memory ALIAS sai_memory)

target_include_directories(sai_memory PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_memory PUBLIC sai::core)
if(CUDAToolkit_FOUND)
    target_link_libraries(sai_memory PUBLIC CUDA::cudart)
endif()
target_compile_features(sai_memory PUBLIC cxx_std_20)
