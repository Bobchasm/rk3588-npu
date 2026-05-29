cmake_minimum_required(VERSION 3.16)
project(board_queue CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rkq
    src/main.cpp
    src/task_queue.cpp
)

target_include_directories(rkq PRIVATE include)

install(TARGETS rkq RUNTIME DESTINATION .)
