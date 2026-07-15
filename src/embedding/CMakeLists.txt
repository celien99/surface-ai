find_package(tl-expected CONFIG REQUIRED)
find_package(CUDAToolkit QUIET)

set(SAI_EMBEDDING_SOURCES
    embedding.cpp
    dimension_reducer.cpp
    pca_fitting.cpp
    pca_scoring.cpp
    pca_serialization.cpp
    feature_cache.cpp
    patch_embedder.cpp
    global_embedder.cpp
    simple_patch_embedder.cpp
)

# CUDA gate: enable GPU inference paths for PatchEmbedder and GlobalEmbedder
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found — enabling GPU embedder paths")
    target_compile_definitions(sai_embedding PUBLIC SAI_CUDA_ENABLED)
    list(APPEND SAI_EMBEDDING_SOURCES
        patch_embedder_cuda.cpp
        global_embedder_cuda.cpp
    )
endif()

add_library(sai_embedding STATIC ${SAI_EMBEDDING_SOURCES})
add_library(sai::embedding ALIAS sai_embedding)

target_include_directories(sai_embedding PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_embedding PUBLIC sai::core sai::memory sai::inference)
if(CUDAToolkit_FOUND)
    target_link_libraries(sai_embedding PUBLIC CUDA::cudart)
endif()
target_compile_features(sai_embedding PUBLIC cxx_std_20)
