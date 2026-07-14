add_library(sai_scheduler STATIC)
add_library(sai::scheduler ALIAS sai_scheduler)

target_sources(sai_scheduler PRIVATE
    scheduler.cpp
)

target_include_directories(sai_scheduler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sai_scheduler PUBLIC
    sai::core
    sai::runtime
)
