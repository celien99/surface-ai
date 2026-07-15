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
if(CUDAToolkit_FOUND)
    list(APPEND SAI_RUNTIME_SOURCES gpu_stream_queue.cpp)
endif()

add_library(sai_runtime STATIC ${SAI_RUNTIME_SOURCES})
add_library(sai::runtime ALIAS sai_runtime)

target_include_directories(sai_runtime PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_runtime PUBLIC sai::core Threads::Threads)
if(CUDAToolkit_FOUND)
    target_link_libraries(sai_runtime PUBLIC sai::memory CUDA::cudart)
endif()
target_compile_features(sai_runtime PUBLIC cxx_std_20)
