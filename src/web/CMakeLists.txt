find_package(nlohmann_json CONFIG REQUIRED)

add_library(sai_web STATIC http_server.cpp)
add_library(sai::web ALIAS sai_web)

target_include_directories(sai_web
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/httplib
)
target_link_libraries(sai_web PUBLIC
    sai::core
    sai::pipeline
    sai::reasoner
    nlohmann_json::nlohmann_json
)
target_compile_features(sai_web PUBLIC cxx_std_20)
