cmake_minimum_required(VERSION 3.16)
project(rk3588_python_bindings LANGUAGES CXX)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_PC_PYTHON "Build worker-pc Python binding" ON)
option(BUILD_RK3588_PYTHON "Build rk3588 worker Python binding" OFF)
option(RK3588_ENGINE_SMOKE_TEST "Build rk3588 Python module without worker_core for import smoke test" OFF)
set(RK3588_PYTHON_INCLUDE_DIR "" CACHE PATH "Target Python include dir for rk3588 binding")
set(RK3588_PYTHON_CONFIG_INCLUDE_DIR "" CACHE PATH "Target Python config include dir for rk3588 binding")
set(RK3588_PYTHON_LIBRARY "" CACHE FILEPATH "Target Python library for rk3588 binding")

if(BUILD_RK3588_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

if(BUILD_PC_PYTHON)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../worker-pc ${CMAKE_CURRENT_BINARY_DIR}/worker-pc-build)
endif()

if(BUILD_RK3588_PYTHON AND NOT RK3588_ENGINE_SMOKE_TEST)
    if(NOT RK3588_PYTHON_INCLUDE_DIR)
        message(FATAL_ERROR "RK3588_PYTHON_INCLUDE_DIR is required when BUILD_RK3588_PYTHON=ON")
    endif()
    set(BUILD_WORKER_APPS OFF CACHE BOOL "Build worker example apps" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../worker ${CMAKE_CURRENT_BINARY_DIR}/rk3588-worker-build)
    set_target_properties(worker_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(BUILD_PC_PYTHON)
    pybind11_add_module(pc_engine MODULE
        src/pc_engine_module.cpp
    )

    target_include_directories(pc_engine PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../worker-pc/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    )

    target_link_libraries(pc_engine PRIVATE
        worker_pc_core
    )

    set(PC_ENGINE_COMPILE_OPTIONS -O3)
    check_cxx_compiler_flag("-march=native" PC_ENGINE_HAS_MARCH_NATIVE)
    if(PC_ENGINE_HAS_MARCH_NATIVE)
        list(APPEND PC_ENGINE_COMPILE_OPTIONS -march=native)
    endif()
    target_compile_options(pc_engine PRIVATE ${PC_ENGINE_COMPILE_OPTIONS})

    set_target_properties(pc_engine PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/runtime
    )
endif()

if(BUILD_RK3588_PYTHON)
    add_library(rk3588_engine MODULE
        src/rk3588_engine_module.cpp
    )

    target_include_directories(rk3588_engine PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../worker/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
        ${RK3588_PYTHON_INCLUDE_DIR}
        ${RK3588_PYTHON_CONFIG_INCLUDE_DIR}
        ${pybind11_INCLUDE_DIRS}
    )

    target_link_libraries(rk3588_engine PRIVATE pybind11::headers)
    if(NOT RK3588_ENGINE_SMOKE_TEST)
        target_link_libraries(rk3588_engine PRIVATE worker_core)
    endif()
    if(RK3588_PYTHON_LIBRARY)
        target_link_libraries(rk3588_engine PRIVATE ${RK3588_PYTHON_LIBRARY})
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(rk3588_engine PRIVATE -O3 -march=armv8.2-a+fp16)
    else()
        target_compile_options(rk3588_engine PRIVATE -O3)
    endif()
    if(RK3588_ENGINE_SMOKE_TEST)
        target_compile_definitions(rk3588_engine PRIVATE RK3588_ENGINE_SMOKE_TEST=1)
    endif()

    set_target_properties(rk3588_engine PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/runtime
        PREFIX ""
        SUFFIX ".so"
    )
endif()
