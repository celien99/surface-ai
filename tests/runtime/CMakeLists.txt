find_package(GTest CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_executable(sai_runtime_task_worker_scheduler_test task_worker_scheduler_test.cpp)
target_link_libraries(sai_runtime_task_worker_scheduler_test
    PRIVATE sai::runtime GTest::gtest_main Threads::Threads
)

add_executable(sai_runtime_task_graph_executor_test task_graph_executor_test.cpp)
target_link_libraries(sai_runtime_task_graph_executor_test
    PRIVATE sai::runtime GTest::gtest_main Threads::Threads
)

include(GoogleTest)
gtest_discover_tests(sai_runtime_task_worker_scheduler_test)
gtest_discover_tests(sai_runtime_task_graph_executor_test)
