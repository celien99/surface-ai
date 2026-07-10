find_package(spdlog CONFIG REQUIRED)

add_library(sai_infra STATIC
    logger.cpp
)
add_library(sai::infra ALIAS sai_infra)

target_include_directories(sai_infra PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_infra PUBLIC sai::core spdlog::spdlog)
target_compile_features(sai_infra PUBLIC cxx_std_20)
