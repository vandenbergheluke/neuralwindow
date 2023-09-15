find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(EXAMPLES mnist)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options=--disable-warnings")
foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} "${EXAMPLE}.cc")
    target_link_libraries(${EXAMPLE} ${PROJECT_NAME} m "${TORCH_LIBRARIES}")
endforeach()
