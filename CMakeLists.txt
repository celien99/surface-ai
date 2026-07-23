cmake_minimum_required(VERSION 3.21)
project(surface_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(CUDAToolkit QUIET)
set(SAI_CUDA_ENABLED OFF)
if(CUDAToolkit_FOUND)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "CUDA Toolkit was found, but no CUDA compiler is available. "
            "Install/configure nvcc with CUDA C++20 support.")
    endif()

    enable_language(CUDA)
    set(SAI_CUDA_ENABLED ON)
endif()

# CMake 3.21 does not know the CUDA C++20 flag for NVCC, despite NVCC accepting
# it. Apply the flag directly to CUDA sources instead of using CUDA_STANDARD,
# whose feature resolution fails during generation on that CMake version.
function(sai_enable_cuda_cxx20 target)
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>")
endfunction()

# GCC on x86-64 requires -latomic for 16-byte std::atomic operations even when
# -mcx16 is active (libstdc++ ABI limitation).  Link it globally so that any
# target using TaggedHead (gpu_pool.h, pinned_pool.h, host_test_pool.h) resolves
# __atomic_store_16 / __atomic_load_16 / __atomic_compare_exchange_16.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    link_libraries(-latomic)
endif()

# -mcx16: enable CMPXCHG16B instruction on x86-64, required for 16-byte lock-free
# atomics used by gpu_pool.h / pinned_pool.h / host_test_pool.h.  Applied globally
# so that __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 is defined in every translation unit
# that includes these headers (not just the ones that were manually annotated).
# Invalid on arm64 (which has native 16-byte LSE atomics), so guard by arch.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mcx16>")
endif()

# yaml-cpp is used by many sub-libraries (retrieval, rule, reasoner, pipeline,
# tuning, visualization, web).  find_package at root level ensures the vcpkg
# imported target yaml-cpp::yaml-cpp is registered before any subdirectory
# add_library call that links against it.
find_package(yaml-cpp CONFIG REQUIRED)

# Warning flags: -Wall -Wextra catch switch-enum gaps, unused, sign-mismatch, etc.
# -Wpedantic is omitted for now (Qt/QML + third-party headers generate noise).
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")

# Optimization: Release builds get LTO + march=native for cross-TU inlining.
# Debug builds keep defaults (-Og/-O0) for fast iteration.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto -march=native")

enable_testing()

add_subdirectory(src/core)
add_subdirectory(tests/core)

add_subdirectory(src/memory)
add_subdirectory(tests/memory)

add_subdirectory(src/plugin)
add_subdirectory(tests/plugin)

add_subdirectory(src/runtime)
add_subdirectory(tests/runtime)

add_subdirectory(src/infra)
add_subdirectory(tests/infra)

add_subdirectory(src/image)
add_subdirectory(tests/image)

add_subdirectory(src/device)
add_subdirectory(tests/device)

add_subdirectory(src/io)
add_subdirectory(tests/io)

add_subdirectory(src/inference)
add_subdirectory(tests/inference)

add_subdirectory(src/embedding)
add_subdirectory(tests/embedding)

add_subdirectory(src/detection)
add_subdirectory(tests/detection)

add_subdirectory(src/knowledge)
add_subdirectory(tests/knowledge)
add_subdirectory(src/retrieval)
add_subdirectory(tests/retrieval)

add_subdirectory(src/rule)
add_subdirectory(tests/rule)
add_subdirectory(src/reasoner)
add_subdirectory(tests/reasoner)

add_subdirectory(src/pipeline)
add_subdirectory(tests/pipeline)
add_subdirectory(tests/scheduler)

add_subdirectory(src/tuning)
add_subdirectory(tests/tuning)

add_subdirectory(src/visualization)
add_subdirectory(src/web)
add_subdirectory(apps/seat-aoi)
add_subdirectory(tests/visualization)

add_subdirectory(tests/integration)
