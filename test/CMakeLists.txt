
set(TEST_DEVICE test_device)
set(TESTS ${TEST_DEVICE})
foreach(TEST ${TESTS})
    add_executable(${TEST} "${TEST}.c")
    target_link_libraries(${TEST} ${PROJECT_NAME} check subunit)
    add_test(NAME ${TEST} COMMAND ${TEST} WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${TEST_DIR})
    add_custom_target("compute_sanitizer_${TEST}" COMMAND compute-sanitizer --leak-check full $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
    add_custom_target("valgrind_${TEST}" COMMAND valgrind --leak-check=full $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
endforeach()

