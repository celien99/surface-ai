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

# embedding_cuda.cpp includes <sai/memory/gpu_pool.h> which carries a
# static_assert on 16-byte lock-free atomics (see src/memory/CMakeLists.txt).
# Gate CUDA embedder paths on BOTH CUDA and lock-free support.
if(SAI_CUDA_ENABLED)
    if(SAI_MEMORY_HAS_LOCKFREE_16)
        message(STATUS "CUDA + lock-free atomics — enabling GPU embedder paths")
        list(APPEND SAI_EMBEDDING_SOURCES
            patch_embedder_cuda.cpp
            global_embedder_cuda.cpp
            embedding_cuda.cpp
        )
    else()
        message(WARNING "CUDA found but lock-free atomics NOT available — "
                        "GPU embedder paths disabled")
    endif()
endif()

add_library(sai_embedding STATIC ${SAI_EMBEDDING_SOURCES})
add_library(sai::embedding ALIAS sai_embedding)

target_include_directories(sai_embedding PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_embedding PUBLIC sai::core sai::memory sai::inference)
if(SAI_CUDA_ENABLED AND SAI_MEMORY_HAS_LOCKFREE_16)
    target_compile_definitions(sai_embedding PUBLIC SAI_CUDA_ENABLED)
    target_link_libraries(sai_embedding PUBLIC CUDA::cudart)
endif()
target_compile_features(sai_embedding PUBLIC cxx_std_20)
