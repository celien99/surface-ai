find_package(GTest REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Gui)

add_executable(sai_visualization_test)
target_sources(sai_visualization_test PRIVATE
    pipeline_viewmodel_test.cpp
    inspection_viewmodel_test.cpp
    frame_provider_test.cpp
)

target_include_directories(sai_visualization_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sai_visualization_test PRIVATE
    sai::visualization
    Qt6::Gui
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sai_visualization_test)
