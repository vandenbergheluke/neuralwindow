
set(TEST_TENSOR test_tensor)
set(TESTS ${TEST_TENSOR})
foreach(TEST ${TESTS})
    add_executable(${TEST} "${TEST}.c")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options=--disable-warnings")
    target_link_libraries(${TEST} ${PROJECT_NAME} check subunit)
    add_test(NAME ${TEST} COMMAND ${TEST} WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${TEST_DIR})
    add_custom_target("compute_sanitizer_${TEST}" COMMAND compute-sanitizer --leak-check full --target-processes all $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
    add_custom_target("valgrind_${TEST}" COMMAND valgrind --leak-check=full $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
endforeach()

