cmake_minimum_required(VERSION 3.18)
project(PathtracerSpectralRealtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

find_package(CUDAToolkit REQUIRED)

if(MSVC)
    message(STATUS "Compiling with MSVC")
else()
    message(FATAL_ERROR "On Windows, CUDA requires MSVC as the host compiler.")
endif()

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/scene_setup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_io.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vec3.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ray.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/material.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hittable.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/camera.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/spectrum.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/scene_setup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/image_io.h
)

add_executable(PathtracerSpectralRealtime ${SOURCES} ${HEADERS})

target_include_directories(PathtracerSpectralRealtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(PathtracerSpectralRealtime PRIVATE
    CUDA::cudart
)

if(MSVC)
    target_compile_options(PathtracerSpectralRealtime PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /MP>
    )
else()
    target_compile_options(PathtracerSpectralRealtime PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    )
endif()

set_target_properties(PathtracerSpectralRealtime PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
)

if(WIN32)
    add_custom_command(
        TARGET PathtracerSpectralRealtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:CUDA::cudart>
        $<TARGET_FILE_DIR:PathtracerSpectralRealtime>
    )
endif()
