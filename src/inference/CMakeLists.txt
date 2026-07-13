# tensorrt_engine.cpp / dino_v3_adapter.cpp / sam2_adapter.cpp / clip_adapter.cpp
# 包含真实的 CUDA Runtime API 和 TensorRT API 调用，仅用于目标平台
# (Ubuntu x64 + NVIDIA GPU)。在没有 CUDA Toolkit 的宿主机 (如 macOS arm64
# 开发机) 上，这些文件不在源文件列表中，不会被编译——见下方的
# if(CUDAToolkit_FOUND) 门控逻辑。
find_package(tl-expected CONFIG REQUIRED)
find_package(CUDAToolkit QUIET)

set(SAI_INFERENCE_SOURCES
    mock_engine.cpp
    multi_layer_aggregator.cpp
    saliency_mask.cpp
)
if(CUDAToolkit_FOUND)
    list(APPEND SAI_INFERENCE_SOURCES
        tensorrt_engine.cpp
        dino_v3_adapter.cpp
        sam2_adapter.cpp
        clip_adapter.cpp
    )
endif()

add_library(sai_inference STATIC ${SAI_INFERENCE_SOURCES})
add_library(sai::inference ALIAS sai_inference)

target_include_directories(sai_inference PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_inference PUBLIC sai::core)
if(CUDAToolkit_FOUND)
    target_link_libraries(sai_inference PUBLIC CUDA::cudart)

    # TensorRT SDK——通过 find_library 定位 nvinfer / nvinfer_plugin。
    # 常见安装路径：/usr/lib/x86_64-linux-gnu, /opt/tensorrt/lib,
    # /usr/local/tensorrt/lib。
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
