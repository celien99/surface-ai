find_package(yaml-cpp CONFIG REQUIRED)

add_library(sai_pipeline STATIC)
add_library(sai::pipeline ALIAS sai_pipeline)

target_sources(sai_pipeline PRIVATE
    pipeline.cpp
    pipeline_builder.cpp
    stage_factory.cpp
    capture_stage.cpp
    preprocess_stage.cpp
    inference_stage.cpp
    detect_stage.cpp
    rule_eval_stage.cpp
    reason_stage.cpp
    export_stage.cpp
    inspection_recorder.cpp
    scheduler.cpp
)

target_include_directories(sai_pipeline PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sai_pipeline PUBLIC
    sai::core
    sai::runtime
    sai::infra
    sai::device
    sai::image
    sai::io
    sai::inference
    sai::embedding
    sai::detection
    sai::rule
    sai::reasoner
    sai::knowledge
    sai::retrieval
    yaml-cpp::yaml-cpp
)
