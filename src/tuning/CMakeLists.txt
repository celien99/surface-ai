find_package(yaml-cpp CONFIG REQUIRED)

set(SAI_TUNING_SOURCES
    tuning_space.cpp
    tuning_objective.cpp
)

add_library(sai_tuning STATIC ${SAI_TUNING_SOURCES})
add_library(sai::tuning ALIAS sai_tuning)

target_include_directories(sai_tuning PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_tuning PUBLIC sai::core sai::knowledge yaml-cpp::yaml-cpp)
target_compile_features(sai_tuning PUBLIC cxx_std_20)
