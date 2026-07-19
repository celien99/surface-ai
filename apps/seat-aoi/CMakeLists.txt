# seat_aoi: Linux + GPU only (CUDA/TensorRT + lock-free atomics required)
# app_builder.cpp and coreset_builder.cpp include <sai/memory/gpu_pool.h> which
# carries a static_assert on 16-byte lock-free atomics (see src/memory/CMakeLists.txt).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CUDAToolkit_FOUND AND
   SAI_MEMORY_HAS_LOCKFREE_16)
    find_package(Qt6 REQUIRED COMPONENTS Quick QuickControls2 Charts)
    find_package(nlohmann_json CONFIG REQUIRED)

    add_executable(seat_aoi
        main.cpp
        app_builder.cpp
        cli_args.cpp
        coreset_builder.cpp
        gui_runner.cpp
        headless_runner.cpp
        knowledge_seed.cpp
        tuning_wiring.cpp
        qml.qrc
    )

    # Allow main.cpp to include internal pipeline stage headers for DI wiring
    target_include_directories(seat_aoi PRIVATE ${CMAKE_SOURCE_DIR}/src)

    target_link_libraries(seat_aoi PRIVATE
        sai::visualization
        sai::pipeline
        sai::tuning
        sai::rule
        sai::reasoner
        sai::detection
        sai::inference
        sai::embedding
        sai::knowledge
        sai::retrieval
        sai::device
        sai::image
        sai::io
        sai::infra
        sai::core
        sai::memory
        sai::plugin
        sai::runtime
        nlohmann_json::nlohmann_json
    )
endif()
