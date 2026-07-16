find_package(nlohmann_json CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_library(sai_io STATIC json_exporter.cpp basic_importer.cpp coreset_manifest.cpp alert_sender.cpp)
add_library(sai::io ALIAS sai_io)

target_include_directories(sai_io PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(sai_io PRIVATE ${CMAKE_SOURCE_DIR}/third_party/stb)
target_link_libraries(sai_io PUBLIC sai::core sai::image sai::plugin nlohmann_json::nlohmann_json yaml-cpp::yaml-cpp)
target_compile_features(sai_io PUBLIC cxx_std_20)
