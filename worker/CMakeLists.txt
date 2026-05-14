cmake_minimum_required(VERSION 3.10)
project(qwen2_demo CXX)

set(CMAKE_CXX_STANDARD 14)

# ---------- RKNN SDK 路径（相对于本文件）----------
set(RKNN_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/rknpu2")
set(RKNN_INC     "${RKNN_SDK_ROOT}/runtime/Linux/librknn_api/include")
set(RKNN_LIB     "${RKNN_SDK_ROOT}/runtime/Linux/librknn_api/aarch64")
set(FP16_INC     "${RKNN_SDK_ROOT}/fp16/include")

# ---------- 头文件 ----------
include_directories(
    include
    ${RKNN_INC}
    ${FP16_INC}
)

# ---------- 源文件 ----------
set(SOURCES
    src/main.cpp
    src/weight_loader.cpp
    src/cpu_ops.cpp
    src/npu_matmul.cpp
    src/qwen2.cpp
)

add_executable(qwen2_demo ${SOURCES})

# ---------- 链接 RKNN 运行时 ----------
target_link_directories(qwen2_demo PRIVATE ${RKNN_LIB})
target_link_libraries(qwen2_demo
    rknnrt   # librknnrt.so
    m        # libm（数学函数）
    pthread
)

# ---------- 安装：把可执行文件复制到 install/ 目录 ----------
install(TARGETS qwen2_demo DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install)
install(FILES
    ${RKNN_LIB}/librknnrt.so
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install
)
