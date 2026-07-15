find_package(tl-expected CONFIG REQUIRED)

# On macOS with Apple Clang, CMake's FindOpenMP cannot detect the correct flag
# combination for Homebrew libomp. We create a synthetic imported target that
# matches what FAISS exports (OpenMP::OpenMP_CXX).
if(APPLE AND NOT OpenMP_FOUND)
    set(OPENMP_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
    set(OPENMP_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    if(EXISTS "${OPENMP_LIBRARY}")
        add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            IMPORTED_LOCATION "${OPENMP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OPENMP_INCLUDE_DIR}"
            INTERFACE_COMPILE_OPTIONS "-Xclang;-fopenmp"
        )
        set(OpenMP_FOUND TRUE)
        message(STATUS "OpenMP: using synthetic target (Homebrew libomp)")
    endif()
endif()

if(NOT OpenMP_FOUND)
    find_package(OpenMP REQUIRED)
endif()

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
    normality_scorer.cpp
    novelty_filter.cpp
    candidate_buffer.cpp
    multi_signal_consensus.cpp
)

# CUDA 门控：仅在 CUDAToolkit 可用时编译 GPU FeatureBank
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found — enabling FAISS GPU FeatureBank")
    list(APPEND SAI_DETECTION_SOURCES feature_bank_cuda.cpp)
endif()

add_library(sai_detection STATIC ${SAI_DETECTION_SOURCES})
add_library(sai::detection ALIAS sai_detection)

target_include_directories(sai_detection PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_detection PUBLIC sai::core sai::embedding faiss OpenMP::OpenMP_CXX yaml-cpp::yaml-cpp)
if(CUDAToolkit_FOUND)
    target_link_libraries(sai_detection PUBLIC CUDA::cudart)
    target_compile_definitions(sai_detection PUBLIC SAI_CUDA_ENABLED)
endif()
target_compile_features(sai_detection PUBLIC cxx_std_20)
