# gpu_image.cpp 和 gpu_preprocess.cpp 包装真实的 CUDA Runtime API 调用
# (cudaMalloc/cudaFree/cudaMemcpy)，仅用于目标平台 (Ubuntu x64 + NVIDIA GPU)。
# 在没有 CUDA Toolkit 的宿主机上，这两个文件不在源文件列表中，不会被编译——
# 见下方的 if(SAI_CUDA_ENABLED) 门控逻辑。
find_package(CUDAToolkit QUIET)

set(SAI_IMAGE_SOURCES
    image.cpp
    raw_image.cpp
    surface_image.cpp
    roi.cpp
    preprocess.cpp
)
if(SAI_CUDA_ENABLED)
    # gpu_image.cpp includes <sai/memory/gpu_pool.h>, gpu_preprocess.cpp includes
    # <sai/memory/pinned_pool.h> — both headers carry a static_assert on 16-byte
    # lock-free atomics (see src/memory/CMakeLists.txt). Gate on the same check.
    if(SAI_MEMORY_HAS_LOCKFREE_16)
        list(APPEND SAI_IMAGE_SOURCES gpu_image.cpp gpu_preprocess.cpp)
        message(STATUS "Lock-free atomics + CUDA — enabling GpuImage/GpuPreprocess")
    else()
        message(WARNING "Lock-free atomics NOT available — "
                        "GpuImage/GpuPreprocess disabled")
    endif()
endif()

add_library(sai_image STATIC ${SAI_IMAGE_SOURCES})
add_library(sai::image ALIAS sai_image)

target_include_directories(sai_image PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_image PUBLIC sai::core sai::memory)
if(SAI_CUDA_ENABLED AND SAI_MEMORY_HAS_LOCKFREE_16)
    target_link_libraries(sai_image PUBLIC sai::runtime CUDA::cudart)
endif()
target_compile_features(sai_image PUBLIC cxx_std_20)
