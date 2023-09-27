cmake_minimum_required(VERSION 3.22.1)

if (DEFINED ENV{CPU_ONLY})
    project(neuralwindow LANGUAGES C CXX)
else()
    project(neuralwindow LANGUAGES CUDA C CXX)
endif()

set(TEST_DIR test)
set(SRC_DIR src)
set(EXAMPLE_DIR example)
set(REPORT_DIR lcov)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3 -g3 -gdwarf-2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g3 -gdwarf-2")

# Generate coverage files
set(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
set(GCC_COVERAGE_LINK_FLAGS "-lgcov --coverage")

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

if (DEFINED ENV{CPU_ONLY})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCPU_ONLY")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_ONLY")
endif()

if (DEFINED ENV{DEBUG})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
endif()

if (DEFINED ENV{GRAPH})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGRAPH")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGRAPH")
    find_program(GRAPHVIZ dot)
    if (NOT GRAPHVIZ)
        message(FATAL_ERROR "Could not find Graphviz.")
    endif()
endif()

list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

add_subdirectory(${SRC_DIR})
add_subdirectory(${EXAMPLE_DIR})
enable_testing()
add_subdirectory(${TEST_DIR})

find_program(LCOV lcov)
if (NOT LCOV)
    message(FATAL_ERROR "Could not find lcov.")
endif()

find_program(GENHTML genhtml)
if (NOT GENHTML)
    message(FATAL_ERROR "Could not find genhtml.")
endif()

add_custom_target(report
    COMMAND mkdir -p ${REPORT_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_command(TARGET report POST_BUILD
    COMMAND ${LCOV} --directory ../src/ --capture --output-file lcov.info
    COMMAND ${LCOV} --remove lcov.info -o lcov.info "/usr/include/x86_64-linux-gnu/bits/*" "${PROJECT_SOURCE_DIR}/${TEST_DIR}/*"
    COMMAND ${GENHTML} -o . lcov.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${REPORT_DIR}
)

# Add REPORT_DIR into clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${REPORT_DIR})
