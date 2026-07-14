find_package(yaml-cpp CONFIG REQUIRED)

set(SAI_RULE_SOURCES
    value.cpp
    lexer.cpp
    parser.cpp
    expression.cpp
    fact_base.cpp
    fact_builder.cpp
    rule_engine.cpp
    conflict_resolver.cpp
)

add_library(sai_rule STATIC ${SAI_RULE_SOURCES})
add_library(sai::rule ALIAS sai_rule)
target_include_directories(sai_rule PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_rule PUBLIC sai::core sai::knowledge sai::retrieval sai::detection yaml-cpp::yaml-cpp)
target_compile_features(sai_rule PUBLIC cxx_std_20)
