cmake_minimum_required(VERSION 3.22.1)

project(neuralwindow LANGUAGES CUDA C CXX)

set(TEST_DIR test)
set(SRC_DIR src)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

if (DEFINED ENV{CPU_ONLY})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCPU_ONLY=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_ONLY=1")
endif()

list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

add_subdirectory(${SRC_DIR})
enable_testing()
add_subdirectory(${TEST_DIR})