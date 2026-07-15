find_package(spdlog CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_library(sai_infra STATIC
    logger.cpp
    config_schema.cpp
    config_store.cpp
)
# config_store_inotify.cpp defines ConfigStore::EnableHotReload against the Linux
# inotify API (<sys/inotify.h>) for the target platform (Ubuntu x64).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(sai_infra PRIVATE config_store_inotify.cpp)
endif()

add_library(sai::infra ALIAS sai_infra)

target_include_directories(sai_infra PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_infra PUBLIC sai::core spdlog::spdlog yaml-cpp::yaml-cpp)
target_compile_features(sai_infra PUBLIC cxx_std_20)
