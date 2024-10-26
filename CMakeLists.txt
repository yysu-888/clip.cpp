cmake_minimum_required(VERSION 3.0)
project(clip LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

option(GGML_METAL "ggml: use metal" OFF)
option(GGML_CUDA "ggml: use cuda" OFF)
option(CLIP_DEBUG "clip enable debug" OFF)

if(GGML_METAL)
    if(APPLE)
        add_definitions(-DGGML_USE_METAL)
    else()
        message(FATAL_ERROR "FATAL: not mac  platform are not support matal.")
    endif()
endif()


if(GGML_CUDA)
    add_definitions(-DGGML_USE_CUDA)
endif()

if(CLIP_DEBUG)
    add_definitions(-DCLIP_DEBUG)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(-Wno-deprecated-declarations -Wunused-value)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/wd4996)  # Disable deprecation warning
endif()

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP_CXX_FOUND Found")
else()
    message(FATAL_ERROR "OpenMP_CXX_FOUND Not Found!")
endif()

FILE(GLOB SRC_LST *.cpp *.hpp)

add_executable(clip ${SRC_LST})
target_link_libraries(clip PUBLIC
    ggml
    OpenMP::OpenMP_CXX
    utf8proc
)

target_include_directories(clip PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/stb_image
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc
    )
    
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/ggml)



