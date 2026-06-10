cmake_minimum_required(VERSION 3.10)
project(qwen2_worker CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    src/backend/npu_linear_i4.cpp
    src/backend/npu_linear_w8.cpp
    src/backend/npu_weight_cache.cpp
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
# 给不同入口（qwen2_demo / qwen2_chat / 未来 RPC server 等）共享编译
add_library(worker_core STATIC
    ${OPS_SOURCES}
    ${BACKEND_SOURCES}
    ${MODEL_SOURCES}
    ${API_SOURCES}
)
target_include_directories(worker_core PUBLIC
    include
    ${RKNN_INC}
    ${FP16_INC}
)
target_link_directories(worker_core PUBLIC ${RKNN_LIB})
target_link_libraries(worker_core PUBLIC rknnrt m pthread)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(worker_core PRIVATE
        -O3 -ffast-math -funroll-loops -march=armv8.2-a+fp16
    )
endif()

# ---------- 入口 1：qwen2_demo（token id 命令行，保留旧用法）----------
add_executable(qwen2_demo src/main.cpp)
target_link_libraries(qwen2_demo PRIVATE worker_core)

# ---------- 入口 2：qwen2_chat（REPL，供 Python 包装器驱动）----------
add_executable(qwen2_chat src/chat_main.cpp)
target_link_libraries(qwen2_chat PRIVATE worker_core)

# ---------- 工具：验证 RKNN matmul dtype 支持 ----------
add_executable(probe_matmul_types src/tools/probe_matmul_types.cpp)
target_link_libraries(probe_matmul_types PRIVATE worker_core)

# ---------- 工具：单纯比较 CPU GEMM 和 RKNN NPU matmul ----------
add_executable(probe_gemm_speed src/tools/probe_gemm_speed.cpp)
target_link_libraries(probe_gemm_speed PRIVATE worker_core)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(probe_gemm_speed PRIVATE
        -O3 -ffast-math -funroll-loops -march=armv8.2-a+fp16
    )
endif()

# ---------- install ----------
install(TARGETS qwen2_demo qwen2_chat probe_matmul_types probe_gemm_speed
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install)
install(FILES
    ${RKNN_LIB}/librknnrt.so
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install
)
install(PROGRAMS
    rate_test.sh
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install
)
