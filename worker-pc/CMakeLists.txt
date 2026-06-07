cmake_minimum_required(VERSION 3.16)
project(worker_pc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_WORKER_PC_APPS "Build worker-pc example apps" ON)
option(BUILD_WORKER_PC_RPC "Build worker-pc RPC server" ON)
option(BUILD_WORKER_PC_GPU_STUB "Enable GPU backend abstraction stub" ON)

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

target_compile_definitions(worker_pc_core PRIVATE
    $<$<BOOL:${BUILD_WORKER_PC_GPU_STUB}>:WORKER_PC_ENABLE_GPU_STUB=1>
)

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
endif()

if(BUILD_WORKER_PC_RPC)
    add_executable(worker_pc_rpc_server
        src/worker_rpc_server.cpp
        ${WORKER_PC_NETWORK_SOURCES}
    )
    target_link_libraries(worker_pc_rpc_server PRIVATE worker_pc_core pthread)
endif()
