find_package(faiss CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

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
