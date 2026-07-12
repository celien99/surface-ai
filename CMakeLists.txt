cmake_minimum_required(VERSION 3.21)
project(surface_ai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

add_subdirectory(src/device)
add_subdirectory(tests/device)

add_subdirectory(src/image)
add_subdirectory(tests/image)

add_subdirectory(src/io)
add_subdirectory(tests/io)

add_subdirectory(src/inference)
add_subdirectory(tests/inference)

add_subdirectory(src/embedding)
add_subdirectory(tests/embedding)

add_subdirectory(src/detection)
add_subdirectory(tests/detection)

add_subdirectory(tests/integration)
