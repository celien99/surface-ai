find_package(yaml-cpp CONFIG REQUIRED)

set(SAI_REASONER_SOURCES
    decision_tree.cpp
    tree_walker.cpp
    score_calculator.cpp
    trace_recorder.cpp
    evidence_collector.cpp
    reasoner.cpp
)

add_library(sai_reasoner STATIC ${SAI_REASONER_SOURCES})
add_library(sai::reasoner ALIAS sai_reasoner)
target_include_directories(sai_reasoner PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_reasoner PUBLIC sai::core sai::rule yaml-cpp::yaml-cpp)
target_compile_features(sai_reasoner PUBLIC cxx_std_20)
