cmake_minimum_required(VERSION 3.2)

project(neuralwindow C)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage -O0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
find_program(GCOV gcov)
find_program(LCOV lcov)
find_program(GENHTML genhtml)

add_custom_target(gcov
  COMMAND mkdir -p Coverage
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Testing
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_command(TARGET gcov POST_BUILD
  COMMAND find ${CMAKE_BINARY_DIR} -type f -name *.c.o -exec ${GCOV} {} "\;"
  COMMAND echo "Coverage reports generated."
  COMMAND echo "You can find them on ${CMAKE_BINARY_DIR}/Coverage"
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/Coverage
  COMMENT "Coverage Reports Generated"
)

add_custom_target(lcov
  COMMAND mkdir -p Coverage
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Testing
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_command(TARGET lcov POST_BUILD
  COMMAND ${LCOV} --directory .. --capture --output-file lcov.info
  COMMAND ${GENHTML} -o . lcov.info
  COMMAND echo "Coverage reports generated."
  COMMAND echo "You can find them on ${CMAKE_BINARY_DIR}/Coverage"
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/Coverage
  COMMENT "Coverage Reports Generated"
)
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES Coverage)
