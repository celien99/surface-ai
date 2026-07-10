find_package(Threads REQUIRED)

add_library(sai_runtime STATIC
    worker_pool.cpp
    task_scheduler.cpp
)
add_library(sai::runtime ALIAS sai_runtime)

target_include_directories(sai_runtime PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_runtime PUBLIC sai::core Threads::Threads)
target_compile_features(sai_runtime PUBLIC cxx_std_20)
