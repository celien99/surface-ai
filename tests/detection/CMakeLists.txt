find_package(GTest CONFIG REQUIRED)

# On macOS with Apple Clang, ensure OpenMP::OpenMP_CXX target is available
# (required by FAISS's exported CMake config transitively through sai::detection).
if(APPLE AND NOT OpenMP_FOUND)
    set(OPENMP_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    if(EXISTS "${OPENMP_LIBRARY}")
        add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            IMPORTED_LOCATION "${OPENMP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "/opt/homebrew/opt/libomp/include"
            INTERFACE_COMPILE_OPTIONS "-Xclang;-fopenmp"
        )
        set(OpenMP_FOUND TRUE)
    endif()
endif()

if(NOT OpenMP_FOUND)
    find_package(OpenMP REQUIRED)
endif()

find_package(faiss CONFIG REQUIRED)

add_executable(sai_detection_types_test detection_types_test.cpp)
target_include_directories(sai_detection_types_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_detection_types_test PRIVATE sai::detection GTest::gtest_main)

add_executable(sai_patch_core_test patch_core_test.cpp)
target_include_directories(sai_patch_core_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_patch_core_test PRIVATE sai::detection GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sai_detection_types_test)
gtest_discover_tests(sai_patch_core_test)
