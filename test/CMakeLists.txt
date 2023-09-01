find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TEST_TENSOR
    test_buffer
    # test_function
    test_map
    test_queue
    # test_autograd
    test_view
)
set(TESTS ${TEST_TENSOR})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options=--disable-warnings")
foreach(TEST ${TESTS})
    if(${TEST} STREQUAL "test_cu_runtime")
        if (NOT DEFINED ENV{CPU_ONLY})
            add_custom_target("compute_sanitizer_${TEST}" COMMAND compute-sanitizer --leak-check full --target-processes all $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
        else()
            continue()
        endif()
    endif()
    if(${TEST} STREQUAL "test_buffer" OR ${TEST} STREQUAL "test_autograd" OR ${TEST} STREQUAL "test_function")
        add_executable(${TEST} "${TEST}.cc")
    else()
        add_executable(${TEST} "${TEST}.c")
    endif()
    target_link_libraries(${TEST} ${PROJECT_NAME} check subunit "${TORCH_LIBRARIES}")
    add_test(NAME ${TEST} COMMAND ${TEST} WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${TEST_DIR})
    add_custom_target("valgrind_${TEST}" COMMAND valgrind --leak-check=full  --show-reachable=yes --error-limit=no --gen-suppressions=all --suppressions=../../suppressions.supp $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
endforeach()
