# GenICam camera: requires aravis (Linux, GigE Vision / USB3 Vision)
find_package(aravis QUIET)

set(SAI_DEVICE_SOURCES
    fake_camera.cpp
)

if(aravis_FOUND)
    message(STATUS "Aravis found — enabling GenICamCamera")
    list(APPEND SAI_DEVICE_SOURCES genicam_camera.cpp)
    set(SAI_HAS_GENICAM TRUE)
endif()

add_library(sai_device STATIC ${SAI_DEVICE_SOURCES})
add_library(sai::device ALIAS sai_device)

target_include_directories(sai_device PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sai_device PUBLIC
    sai::plugin
    sai::image
)
if(aravis_FOUND)
    target_link_libraries(sai_device PUBLIC aravis)
    target_compile_definitions(sai_device PUBLIC SAI_HAS_GENICAM)
endif()
target_compile_features(sai_device PUBLIC cxx_std_20)
