cmake_minimum_required(VERSION 3.10)
project(rk3588_scheduler CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(RK3588_NPU_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

option(SCHEDULER_USE_WORKER_CORE "Build scheduler with worker_core integration" ON)
if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND NOT DEFINED ENV{FORCE_SCHEDULER_USE_WORKER_CORE})
    set(SCHEDULER_USE_WORKER_CORE OFF CACHE BOOL "Disable worker_core integration on non-aarch64 host" FORCE)
endif()

set(BUILD_WORKER_APPS OFF CACHE BOOL "Disable worker example apps when building scheduler" FORCE)

if(SCHEDULER_USE_WORKER_CORE)
    set(BUILD_WITH_RKNN ON CACHE BOOL "Enable RKNN runtime when building worker core" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../worker ${CMAKE_CURRENT_BINARY_DIR}/worker_build)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
)
if(SCHEDULER_USE_WORKER_CORE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../worker/include)
endif()

add_executable(scheduler_cli
    tools/scheduler_cli.cpp
    src/session_manager.cpp
    src/worker_interface.cpp
    src/coordinator.cpp
    src/request_router.cpp
    src/tokenizer/tokenizer_adapter.cpp
    src/network/rpc_client.cpp
    src/network/rpc_server.cpp
    src/registry/node_registry.cpp
    src/strategy/load_balancer.cpp
    src/strategy/schedule_policy.cpp
)

target_compile_definitions(scheduler_cli PRIVATE
    SCHEDULER_REPO_ROOT="${RK3588_NPU_REPO_ROOT}"
)

if(SCHEDULER_USE_WORKER_CORE)
    target_compile_definitions(scheduler_cli PRIVATE SCHEDULER_USE_WORKER_CORE=1)
    target_link_libraries(scheduler_cli PRIVATE worker_core pthread)
else()
    target_link_libraries(scheduler_cli PRIVATE pthread)
    message(WARNING "Building scheduler_cli without worker_core integration. LocalWorker will be a host-side stub.")
endif()
