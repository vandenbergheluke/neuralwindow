set(EXAMPLES mnist)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options=--disable-warnings")
foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} "${EXAMPLE}.c")
    target_link_libraries(${EXAMPLE} ${PROJECT_NAME} m)
endforeach()
