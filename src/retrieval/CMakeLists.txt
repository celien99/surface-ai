find_package(faiss CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

# On macOS with Apple Clang, synthetic OpenMP target (copied from src/detection/CMakeLists.txt)
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
    endif()
endif()
if(NOT OpenMP_FOUND)
    find_package(OpenMP REQUIRED)
endif()

set(SAI_RETRIEVAL_SOURCES
    vector_path.cpp
    metadata_path.cpp
    score_fusion.cpp
    hybrid_retriever.cpp
)

add_library(sai_retrieval STATIC ${SAI_RETRIEVAL_SOURCES})
add_library(sai::retrieval ALIAS sai_retrieval)
target_include_directories(sai_retrieval PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_retrieval PUBLIC sai::core sai::knowledge sai::detection faiss OpenMP::OpenMP_CXX yaml-cpp::yaml-cpp)
target_compile_features(sai_retrieval PUBLIC cxx_std_20)
