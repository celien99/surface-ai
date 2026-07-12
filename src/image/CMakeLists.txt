# gpu_image.cpp 和 gpu_preprocess.cpp 包装真实的 CUDA Runtime API 调用
# (cudaMalloc/cudaFree/cudaMemcpy)，仅用于目标平台 (Ubuntu x64 + NVIDIA GPU)。
# 在没有 CUDA Toolkit 的宿主机 (如 macOS arm64 开发机) 上，这两个文件不在
# 源文件列表中，不会被编译——见下方的 if(CUDAToolkit_FOUND) 门控逻辑。
find_package(CUDAToolkit QUIET)

set(SAI_IMAGE_SOURCES
    image.cpp
    raw_image.cpp
    surface_image.cpp
    roi.cpp
    preprocess.cpp
)
if(CUDAToolkit_FOUND)
    list(APPEND SAI_IMAGE_SOURCES gpu_image.cpp gpu_preprocess.cpp)
endif()

add_library(sai_image STATIC ${SAI_IMAGE_SOURCES})
add_library(sai::image ALIAS sai_image)

target_include_directories(sai_image PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_image PUBLIC sai::core sai::memory sai::device)
if(CUDAToolkit_FOUND)
    target_link_libraries(sai_image PUBLIC sai::runtime CUDA::cudart)
endif()
target_compile_features(sai_image PUBLIC cxx_std_20)

# gpu_pool.h/pinned_pool.h static_assert std::atomic<TaggedHead> (16 bytes) 是
# 始终 lock-free。在 x86-64 上 __atomic_always_lock_free(16) 仅在 CMPXCHG16B 指令
# 下为 true，该指令需 -mcx16。该标志在 arm64 上无效（arm64 有原生的 16 字节 LSE 原子操作），
# 故按架构守卫——与 src/memory/CMakeLists.txt 的模式完全一致。
if(CUDAToolkit_FOUND AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(sai_image PRIVATE -mcx16)
endif()
