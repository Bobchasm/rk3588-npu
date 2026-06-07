cmake_minimum_required(VERSION 3.10)
project(qwen2_worker CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_WORKER_APPS "Build worker example apps" ON)
option(BUILD_WITH_RKNN "Link RKNN runtime when building worker core" ON)

# ---------- RKNN SDK 路径 ----------
set(RKNN_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/rknpu2")
set(RKNN_INC      "${RKNN_SDK_ROOT}/runtime/Linux/librknn_api/include")
set(RKNN_LIB      "${RKNN_SDK_ROOT}/runtime/Linux/librknn_api/aarch64")
set(FP16_INC      "${RKNN_SDK_ROOT}/fp16/include")

# ---------- 头文件搜索路径 ----------
# include/ 下按分层组织（core / ops / backend / model / api）
# 源文件里用 "ops/op_rmsnorm.h" 这样的带前缀路径
include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${RKNN_INC}
    ${FP16_INC}
)

# ---------- 按层收集源文件 ----------
set(OPS_SOURCES
    src/ops/op_linear.cpp
    src/ops/op_rmsnorm.cpp
    src/ops/op_softmax.cpp
    src/ops/op_cast.cpp
    src/ops/op_embedding.cpp
    src/ops/op_attention.cpp
)

set(BACKEND_SOURCES
    src/backend/cpu_linear.cpp
    src/backend/npu_linear.cpp
    src/backend/npu_linear_w8.cpp
    src/backend/sharded_npu_linear.cpp
)

set(MODEL_SOURCES
    src/model/weight_loader.cpp
    src/model/kv_cache.cpp
    src/model/qwen2_model.cpp
)

set(API_SOURCES
    src/api/llm_engine.cpp
)

# ---------- 汇总成一个「核心静态库」worker_core ----------
# 给不同入口（qwen2_demo / qwen2_chat / worker_rpc_server / 工具）共享编译
add_library(worker_core STATIC
    ${OPS_SOURCES}
    ${BACKEND_SOURCES}
    ${MODEL_SOURCES}
    ${API_SOURCES}
)

target_include_directories(worker_core PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${RKNN_INC}
    ${FP16_INC}
)

if(BUILD_WITH_RKNN)
    target_link_directories(worker_core PUBLIC ${RKNN_LIB})
    target_link_libraries(worker_core PUBLIC rknnrt m pthread)
else()
    target_link_libraries(worker_core PUBLIC m pthread)
endif()

# aarch64 下启用更激进的优化参数
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(worker_core PRIVATE
        -O3 -ffast-math -funroll-loops -march=armv8.2-a+fp16
    )
endif()

# ---------- 入口 1：qwen2_demo（token id 命令行）----------
if(BUILD_WORKER_APPS)
    add_executable(qwen2_demo src/main.cpp)
    target_link_libraries(qwen2_demo PRIVATE worker_core)

    # ---------- 入口 2：qwen2_chat（REPL，供 Python 包装器驱动）----------
    add_executable(qwen2_chat src/chat_main.cpp)
    target_link_libraries(qwen2_chat PRIVATE worker_core)

    # ---------- 入口 3：worker_rpc_server（RPC 服务版）----------
    add_executable(worker_rpc_server
        src/worker_rpc_server.cpp
        src/network/worker_service.cpp
        src/network/rpc_server.cpp
    )
    target_link_libraries(worker_rpc_server PRIVATE worker_core pthread)
endif()

# ---------- 工具：验证 RKNN 低比特 matmul 支持 ----------
add_executable(probe_matmul_w8 src/tools/probe_matmul_w8.cpp)
target_link_libraries(probe_matmul_w8 PRIVATE worker_core)

# ---------- install ----------
if(BUILD_WORKER_APPS)
    install(TARGETS qwen2_demo qwen2_chat worker_rpc_server
            DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install)
endif()

install(TARGETS probe_matmul_w8
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install)

if(BUILD_WITH_RKNN)
    install(FILES
        ${RKNN_LIB}/librknnrt.so
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install
    )
endif()

install(PROGRAMS
    rate_test.sh
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install
)