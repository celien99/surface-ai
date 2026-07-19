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

# CUDA 门控：仅在 CUDAToolkit 可用且 FAISS GPU 头文件存在时编译 GPU FeatureBank
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    include(CheckIncludeFileCXX)
    check_include_file_cxx("faiss/gpu/StandardGpuResources.h" SAI_FAISS_GPU_FOUND)
    if(SAI_FAISS_GPU_FOUND)
        message(STATUS "CUDA + FAISS GPU found — enabling GPU FeatureBank")
        list(APPEND SAI_DETECTION_SOURCES feature_bank_cuda.cpp)
    else()
        message(WARNING "CUDA found but FAISS GPU headers missing — GPU FeatureBank disabled. "
                        "Enable faiss[gpu] feature in vcpkg.json to enable.")
    endif()
endif()

add_library(sai_detection STATIC ${SAI_DETECTION_SOURCES})
add_library(sai::detection ALIAS sai_detection)

target_include_directories(sai_detection PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_detection PUBLIC sai::core sai::embedding sai::infra faiss OpenMP::OpenMP_CXX yaml-cpp::yaml-cpp
                      PRIVATE sai::knowledge)
if(CUDAToolkit_FOUND AND SAI_FAISS_GPU_FOUND)
    target_link_libraries(sai_detection PUBLIC CUDA::cudart)
    target_compile_definitions(sai_detection PUBLIC SAI_CUDA_ENABLED SAI_FAISS_GPU_ENABLED)
endif()
target_compile_features(sai_detection PUBLIC cxx_std_20)
