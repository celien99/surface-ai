# GpuPool/PinnedPool (gpu_pool.cpp/pinned_pool.cpp) wrap real CUDA Runtime API
# calls (cudaMalloc/cudaFree/cudaHostAlloc/cudaFreeHost) for the target
# platform (Ubuntu x64 + NVIDIA GPU). Without the CUDA Toolkit these two files
# are absent from the source list below and are never compiled.
find_package(CUDAToolkit QUIET)

set(SAI_MEMORY_SOURCES arena_allocator.cpp)
if(CUDAToolkit_FOUND)
    # gpu_pool.h/pinned_pool.h static_assert on 16-byte lock-free atomics.
    # On x86-64, CMPXCHG16B (-mcx16) is required for lock-free 16-byte CAS.
    # GCC's std::atomic<T>::is_always_lock_free is a libstdc++ ABI constant
    # that reports false for 16-byte types even when -mcx16 is active, because
    # the value was baked in when libstdc++ was built (Ubuntu packages use the
    # baseline x86-64 target which lacks CMPXCHG16B). The compiler intrinsic
    # macro __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 correctly reflects whether the
    # compiler can emit lock-free 16-byte CAS with the current flags, so we
    # check that instead. See 1.5-memory.md §12.
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-mcx16 -std=c++20")
    check_cxx_source_compiles(
        "#if !defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
        #error \"no 16-byte lock-free CAS\"
        #endif
        int main() { return 0; }"
        SAI_MEMORY_HAS_LOCKFREE_16
    )
    unset(CMAKE_REQUIRED_FLAGS)

    if(SAI_MEMORY_HAS_LOCKFREE_16)
        list(APPEND SAI_MEMORY_SOURCES gpu_pool.cpp pinned_pool.cpp)
        message(STATUS "16-byte lock-free atomics available — enabling GpuPool/PinnedPool")
    else()
        message(WARNING "16-byte lock-free atomics NOT available on this compiler "
                        "(-mcx16 applied but __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 "
                        "is not defined). GpuPool/PinnedPool are disabled. "
                        "Compile with -mcx16 or -march=x86-64-v2+.")
    endif()
endif()

add_library(sai_memory STATIC ${SAI_MEMORY_SOURCES})
add_library(sai::memory ALIAS sai_memory)

target_include_directories(sai_memory PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_memory PUBLIC sai::core)
if(CUDAToolkit_FOUND AND SAI_MEMORY_HAS_LOCKFREE_16)
    target_link_libraries(sai_memory PUBLIC CUDA::cudart)
endif()
target_compile_features(sai_memory PUBLIC cxx_std_20)
