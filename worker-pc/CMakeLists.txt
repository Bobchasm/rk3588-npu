cmake_minimum_required(VERSION 3.16)
project(worker_pc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_WORKER_PC_APPS "Build worker-pc example apps" ON)
option(BUILD_WORKER_PC_RPC "Build worker-pc RPC server" ON)
option(BUILD_WORKER_PC_GPU_STUB "Enable GPU backend abstraction stub" ON)
option(BUILD_WORKER_PC_CUDA "Enable CUDA linear backend when toolkit is available" ON)

include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
)

set(WORKER_PC_OPS_SOURCES
    src/ops/op_linear.cpp
    src/ops/op_rmsnorm.cpp
    src/ops/op_softmax.cpp
    src/ops/op_cast.cpp
    src/ops/op_embedding.cpp
    src/ops/op_attention.cpp
)

set(WORKER_PC_BACKEND_SOURCES
    src/backend/device_type.cpp
    src/backend/cpu_linear.cpp
    src/backend/gpu_linear.cpp
    src/backend/gpu_kv_cache.cpp
)

set(WORKER_PC_MODEL_SOURCES
    src/model/kv_cache.cpp
    src/model/weight_loader.cpp
    src/model/qwen2_model.cpp
)

set(WORKER_PC_API_SOURCES
    src/api/llm_engine.cpp
)

set(WORKER_PC_NETWORK_SOURCES
    src/network/rpc_server.cpp
    src/network/worker_service.cpp
)

add_library(worker_pc_core STATIC
    ${WORKER_PC_OPS_SOURCES}
    ${WORKER_PC_BACKEND_SOURCES}
    ${WORKER_PC_MODEL_SOURCES}
    ${WORKER_PC_API_SOURCES}
)

target_include_directories(worker_pc_core PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
)

target_compile_definitions(worker_pc_core PUBLIC
    $<$<BOOL:${BUILD_WORKER_PC_GPU_STUB}>:WORKER_PC_ENABLE_GPU_STUB=1>
)

set(WORKER_PC_CUDA_ENABLED OFF)
if(BUILD_WORKER_PC_CUDA)
    set(_worker_pc_cuda_roots
        /usr/local/cuda
        /usr/local/cuda-12.6
        /usr/local/cuda-12
    )
    find_path(WORKER_PC_CUDA_INCLUDE_DIR cuda_runtime_api.h
        HINTS ${_worker_pc_cuda_roots}
        PATH_SUFFIXES include
    )
    find_library(WORKER_PC_CUDART_LIB cudart
        HINTS ${_worker_pc_cuda_roots}
        PATH_SUFFIXES lib64 targets/x86_64-linux/lib
    )
    find_library(WORKER_PC_CUBLAS_LIB cublas
        HINTS ${_worker_pc_cuda_roots}
        PATH_SUFFIXES lib64 targets/x86_64-linux/lib
    )
    find_program(WORKER_PC_NVCC nvcc
        HINTS /usr/local/cuda/bin /usr/local/cuda-12.6/bin /usr/local/cuda-12/bin
    )
    if(WORKER_PC_CUDA_INCLUDE_DIR AND WORKER_PC_CUDART_LIB AND WORKER_PC_CUBLAS_LIB AND WORKER_PC_NVCC)
        set(CMAKE_CUDA_COMPILER "${WORKER_PC_NVCC}")
        enable_language(CUDA)
        set(WORKER_PC_CUDA_ENABLED ON)
        target_sources(worker_pc_core PRIVATE src/ops/op_attention_cuda.cu)
        target_include_directories(worker_pc_core PUBLIC ${WORKER_PC_CUDA_INCLUDE_DIR})
        target_compile_definitions(worker_pc_core PUBLIC WORKER_PC_ENABLE_CUDA=1)
        target_link_libraries(worker_pc_core PUBLIC ${WORKER_PC_CUDART_LIB} ${WORKER_PC_CUBLAS_LIB})
        set_target_properties(worker_pc_core PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "89"
        )
        message(STATUS "worker-pc CUDA backend enabled")
    else()
        message(STATUS "worker-pc CUDA backend disabled: toolkit not fully found")
    endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    target_compile_options(worker_pc_core PRIVATE -O3 -ffast-math -funroll-loops -march=native)
else()
    target_compile_options(worker_pc_core PRIVATE -O3 -ffast-math -funroll-loops)
endif()

target_link_libraries(worker_pc_core PUBLIC m pthread)

if(BUILD_WORKER_PC_APPS)
    add_executable(qwen2_pc_demo src/main.cpp)
    target_link_libraries(qwen2_pc_demo PRIVATE worker_pc_core)

    add_executable(qwen2_pc_chat src/chat_main.cpp)
    target_link_libraries(qwen2_pc_chat PRIVATE worker_pc_core)

    add_executable(worker_pc_gpu_probe src/gpu_probe.cpp)
    target_link_libraries(worker_pc_gpu_probe PRIVATE worker_pc_core)
endif()

if(BUILD_WORKER_PC_RPC)
    add_executable(worker_pc_rpc_server
        src/worker_rpc_server.cpp
        ${WORKER_PC_NETWORK_SOURCES}
    )
    target_link_libraries(worker_pc_rpc_server PRIVATE worker_pc_core pthread)
endif()
