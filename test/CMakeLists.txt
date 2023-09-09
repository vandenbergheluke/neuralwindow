find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TEST_C
    test_map
    test_queue
    test_view
)

set(TEST_CXX
    test_tensor_reduction
    test_tensor_unary
    test_tensor_binary_elementwise
    # test_buffer_reduction
    test_buffer_unary
    # test_buffer_binary
    # test_buffer_unary_performance
    # test_buffer_binary_performance
)

set(TEST_HELPER
    test_helper
)

set(TESTS ${TEST_C} ${TEST_CXX})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options=--disable-warnings")
foreach(TEST ${TESTS})
    if (NOT DEFINED ENV{CPU_ONLY})
        add_custom_target("compute_sanitizer_${TEST}" COMMAND compute-sanitizer --leak-check full --target-processes all $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
    endif()
    if(${TEST} IN_LIST TEST_CXX)
        add_executable(${TEST} "${TEST}.cc" "${TEST_HELPER}.c")
    elseif(${TEST} IN_LIST TEST_C)
        add_executable(${TEST} "${TEST}.c" "${TEST_HELPER}.c")
    endif()
    target_include_directories(${TEST} PUBLIC ".")
    set_target_properties(${TEST} PROPERTIES PUBLIC_HEADER "${TEST_HELPER}.h")
    target_link_libraries(${TEST} ${PROJECT_NAME} check subunit m "${TORCH_LIBRARIES}")
    add_test(NAME ${TEST} COMMAND ${TEST} WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${TEST_DIR})
    # add_custom_target("valgrind_${TEST}" COMMAND valgrind --leak-check=full  --show-reachable=yes --error-limit=no --gen-suppressions=all --log-file=../suppressions.log --suppressions=../../suppressions.supp $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
    add_custom_target("valgrind_${TEST}" COMMAND valgrind --leak-check=full  --show-reachable=yes --error-limit=no --suppressions=../../suppressions.supp $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
endforeach()
