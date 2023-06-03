cmake_minimum_required(VERSION 3.22.1)

project(neuralwindow LANGUAGES CUDA C)

set(TEST_DIR test)
set(SRC_DIR src)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")

add_subdirectory(${SRC_DIR})
enable_testing()
add_subdirectory(${TEST_DIR})