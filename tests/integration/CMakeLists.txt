find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# On macOS with Apple Clang, CMake's FindOpenMP cannot detect the correct flag
# combination for Homebrew libomp. Ensure OpenMP::OpenMP_CXX target exists
# (required transitively through sai::detection → faiss).
if(APPLE AND NOT OpenMP_FOUND)
    set(OPENMP_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
    set(OPENMP_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    if(EXISTS "${OPENMP_LIBRARY}")
        add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            IMPORTED_LOCATION "${OPENMP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OPENMP_INCLUDE_DIR}"
            INTERFACE_COMPILE_OPTIONS "-Xclang;-fopenmp"
        )
        set(OpenMP_FOUND TRUE)
    endif()
endif()

add_executable(sai_integration_acquisition_to_export_test
    acquisition_to_export_test.cpp
    ${CMAKE_SOURCE_DIR}/tests/device/fake_camera.cpp
)
target_link_libraries(sai_integration_acquisition_to_export_test PRIVATE
    sai::device
    sai::image
    sai::io
    sai::core
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)
target_include_directories(sai_integration_acquisition_to_export_test PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/device
)

# Task 9: M3 推理管线端到端集成测试
add_executable(sai_integration_inference_pipeline_test
    inference_pipeline_test.cpp
)
target_include_directories(sai_integration_inference_pipeline_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sai_integration_inference_pipeline_test PRIVATE
    sai::detection
    sai::image
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(sai_integration_acquisition_to_export_test)
gtest_discover_tests(sai_integration_inference_pipeline_test)
