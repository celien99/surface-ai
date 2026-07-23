# gpu_stream_queue.cpp wraps real CUDA Runtime API calls (cudaStreamCreate/
# cudaMemcpyAsync/cudaStreamAddCallback/cudaStreamDestroy) for the target
# platform (Ubuntu x64 + NVIDIA GPU). Without the CUDA Toolkit this file is
# absent from the source list below and is never compiled. It also depends on
# batch 1.5's gated PinnedPool symbol, so sai::memory is only linked when CUDA
# is present too.
find_package(Threads REQUIRED)
find_package(CUDAToolkit QUIET)

set(SAI_RUNTIME_SOURCES
    worker_pool.cpp
    task_scheduler.cpp
    pipeline_executor.cpp
    task_graph.cpp
)
if(SAI_CUDA_ENABLED)
    # gpu_stream_queue.h includes <sai/memory/pinned_pool.h> which carries a
    # static_assert on 16-byte lock-free atomics (see src/memory/CMakeLists.txt).
    if(SAI_MEMORY_HAS_LOCKFREE_16)
        list(APPEND SAI_RUNTIME_SOURCES gpu_stream_queue.cpp)
        message(STATUS "CUDA + lock-free atomics — enabling GpuStreamQueue")
    else()
        message(WARNING "Lock-free atomics NOT available — GpuStreamQueue disabled")
    endif()
endif()

add_library(sai_runtime STATIC ${SAI_RUNTIME_SOURCES})
add_library(sai::runtime ALIAS sai_runtime)

target_include_directories(sai_runtime PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_runtime PUBLIC sai::core Threads::Threads)
if(SAI_CUDA_ENABLED AND SAI_MEMORY_HAS_LOCKFREE_16)
    target_link_libraries(sai_runtime PUBLIC sai::memory CUDA::cudart)
endif()
target_compile_features(sai_runtime PUBLIC cxx_std_20)
