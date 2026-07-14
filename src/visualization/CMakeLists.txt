find_package(Qt6 REQUIRED COMPONENTS Quick QuickControls2 Charts)

add_library(sai_visualization STATIC)
add_library(sai::visualization ALIAS sai_visualization)

target_sources(sai_visualization PRIVATE
    placeholder.cpp
    # ViewModels added in subsequent tasks
)

target_link_libraries(sai_visualization PUBLIC
    Qt6::Quick
    Qt6::QuickControls2
    Qt6::Charts
    sai::pipeline
    sai::rule
    sai::reasoner
    sai::detection
    sai::image
    sai::infra
    sai::core
)

target_include_directories(sai_visualization PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
