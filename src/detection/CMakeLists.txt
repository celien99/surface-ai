find_package(tl-expected CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

find_package(faiss CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

# 主检测库源文件
set(SAI_DETECTION_SOURCES
    detection_result.cpp
    feature_bank.cpp
    patch_core.cpp
    post_process_utils.cpp
    specular_filter.cpp
    pca_detector.cpp
    normality_profile.cpp
    candidate_buffer.cpp
    multi_signal_consensus.cpp
    coreset_evolution.cpp
)

# CUDA greedy FPS only needs the CUDA runtime. GPU search additionally needs
# the FAISS GPU clone API, so keep the two capabilities independently gated.
find_package(CUDAToolkit QUIET)
if(SAI_CUDA_ENABLED)
    list(APPEND SAI_DETECTION_SOURCES greedy_coreset_cuda.cu)

    get_target_property(SAI_FAISS_INCLUDE_DIRS faiss INTERFACE_INCLUDE_DIRECTORIES)
    find_path(SAI_FAISS_GPU_CLONER_INCLUDE_DIR
        NAMES faiss/gpu/GpuCloner.h
        PATHS ${SAI_FAISS_INCLUDE_DIRS}
        NO_DEFAULT_PATH)
    find_path(SAI_FAISS_GPU_RESOURCES_INCLUDE_DIR
        NAMES faiss/gpu/StandardGpuResources.h
        PATHS ${SAI_FAISS_INCLUDE_DIRS}
        NO_DEFAULT_PATH)
    set(SAI_FAISS_GPU_FOUND FALSE)
    if(SAI_FAISS_GPU_CLONER_INCLUDE_DIR AND SAI_FAISS_GPU_RESOURCES_INCLUDE_DIR)
        set(SAI_FAISS_GPU_FOUND TRUE)
    endif()
    if(SAI_FAISS_GPU_FOUND)
        message(STATUS "CUDA + FAISS GPU found — enabling GPU FeatureBank")
    else()
        message(WARNING "CUDA found but the configured faiss target has no GPU clone API — "
                        "GPU FeatureBank disabled; CUDA greedy coreset remains enabled.")
    endif()
endif()

add_library(sai_detection STATIC ${SAI_DETECTION_SOURCES})
add_library(sai::detection ALIAS sai_detection)

target_include_directories(sai_detection PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_detection PUBLIC sai::core sai::embedding sai::infra faiss OpenMP::OpenMP_CXX yaml-cpp::yaml-cpp
                      PRIVATE sai::knowledge)
if(SAI_CUDA_ENABLED)
    target_link_libraries(sai_detection PUBLIC CUDA::cudart)
    target_compile_definitions(sai_detection PUBLIC SAI_CUDA_ENABLED)
endif()
if(SAI_CUDA_ENABLED AND SAI_FAISS_GPU_FOUND)
    target_include_directories(sai_detection PRIVATE
        ${SAI_FAISS_GPU_CLONER_INCLUDE_DIR}
        ${SAI_FAISS_GPU_RESOURCES_INCLUDE_DIR})
    target_compile_definitions(sai_detection PUBLIC SAI_FAISS_GPU_ENABLED)
endif()
target_compile_features(sai_detection PUBLIC cxx_std_20)
