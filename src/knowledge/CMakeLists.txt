find_package(SQLite3 REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

set(SAI_KNOWLEDGE_SOURCES
    knowledge_record.cpp
    knowledge_graph.cpp
    knowledge_evolution.cpp
    knowledge_snapshot.cpp
    knowledge_store.cpp
)

add_library(sai_knowledge STATIC ${SAI_KNOWLEDGE_SOURCES})
add_library(sai::knowledge ALIAS sai_knowledge)
target_include_directories(sai_knowledge PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_knowledge PUBLIC sai::core sai::infra SQLite::SQLite3 nlohmann_json::nlohmann_json)
target_compile_features(sai_knowledge PUBLIC cxx_std_20)
