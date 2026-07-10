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

# gpu_pool.h/pinned_pool.h static_assert std::atomic<TaggedHead> (16 bytes) is
# always lock-free. On x86-64, __atomic_always_lock_free(16) is only true with
# the CMPXCHG16B instruction, which GCC/Clang emit only under -mcx16. The flag is
# invalid on arm64 (which has native 16-byte LSE atomics), so guard it by arch.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(sai_memory PRIVATE -mcx16)
endif()
