set(EXAMPLES mnist)
set(GRAPHEXAMPLE graph_example)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options=--disable-warnings")
foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} "${EXAMPLE}.c")
    target_link_libraries(${EXAMPLE} ${PROJECT_NAME} m)
endforeach()

find_package(Torch REQUIRED)
foreach(E ${GRAPHEXAMPLE})
    add_executable(${E} "${E}.cc")
    target_link_libraries(${E} ${PROJECT_NAME} check subunit m "${TORCH_LIBRARIES}")
endforeach()