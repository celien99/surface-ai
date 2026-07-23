# TensorRT engines/adapters; DINO uses a CUDA translation unit for preprocessing.
# 包含真实的 CUDA Runtime API 和 TensorRT API 调用，仅用于目标平台
# (Ubuntu x64 + NVIDIA GPU)。在没有 CUDA Toolkit 或 TensorRT SDK 的宿主机上，
# 这些文件不在源文件列表中，不会被编译——见下方的门控逻辑。
find_package(tl-expected CONFIG REQUIRED)
find_package(CUDAToolkit QUIET)

set(SAI_INFERENCE_SOURCES
    multi_layer_aggregator.cpp
    saliency_mask.cpp
)
if(SAI_CUDA_ENABLED)
    # TensorRT SDK is separate from CUDA Toolkit — check for its header before
    # compiling any CUDA-gated sources. On systems with CUDA but without
    # TensorRT (e.g. WSL with nvidia-smi but no TensorRT installed), the
    # previous CUDAToolkit_FOUND-only gate would produce a hard compilation
    # error (NvInfer.h not found).
    find_path(NVINFER_INCLUDE_DIR NvInfer.h
        PATHS /usr/local/cuda/include
              /usr/include/x86_64-linux-gnu
              /opt/tensorrt/include
              /usr/local/tensorrt/include
              ${CUDAToolkit_INCLUDE_DIRS}
    )
    if(NVINFER_INCLUDE_DIR)
        list(APPEND SAI_INFERENCE_SOURCES
            tensorrt_engine.cpp
            dino_v3_adapter.cu
            sam2_adapter.cpp
            sam2_segmenter.cpp
            clip_adapter.cpp
        )
        message(STATUS "TensorRT found — enabling GPU inference engines")
    elseif(NOT NVINFER_INCLUDE_DIR)
        message(WARNING "TensorRT NvInfer.h not found — "
                        "TensorRtEngine and adapters disabled. "
                        "Install TensorRT SDK (or libnvinfer-dev) to enable GPU inference.")
    else()
        message(WARNING "CUDA compiler not found — TensorRT adapters disabled")
    endif()
endif()

add_library(sai_inference STATIC ${SAI_INFERENCE_SOURCES})
add_library(sai::inference ALIAS sai_inference)

target_include_directories(sai_inference PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_inference PUBLIC sai::core)
if(SAI_CUDA_ENABLED AND NVINFER_INCLUDE_DIR)
    # dino_v3_adapter.cu includes ErrorInfo, which uses std::source_location.
    # Keep the CUDA frontend in C++20 without asking CMake 3.21 to resolve the
    # unavailable cuda_std_20 feature mapping.
    sai_enable_cuda_cxx20(sai_inference)
    target_link_libraries(sai_inference PUBLIC CUDA::cudart)
    target_include_directories(sai_inference PRIVATE ${NVINFER_INCLUDE_DIR})

    # TensorRT SDK
    find_library(NVINFER_LIB nvinfer
        PATHS /usr/lib/x86_64-linux-gnu /opt/tensorrt/lib /usr/local/tensorrt/lib
    )
    find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
        PATHS /usr/lib/x86_64-linux-gnu /opt/tensorrt/lib /usr/local/tensorrt/lib
    )
    if(NVINFER_LIB)
        target_link_libraries(sai_inference PUBLIC ${NVINFER_LIB})
    else()
        message(WARNING "TensorRT nvinfer library not found — "
                        "TensorRtEngine will not link. "
                        "Install TensorRT SDK or set NVINFER_LIB manually.")
    endif()
    if(NVINFER_PLUGIN_LIB)
        target_link_libraries(sai_inference PUBLIC ${NVINFER_PLUGIN_LIB})
    endif()
endif()
target_compile_features(sai_inference PUBLIC cxx_std_20)
