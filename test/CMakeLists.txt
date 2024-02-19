SET(CMAKE_PREFIX_PATH "../libtorch/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(MathGL2 QUIET)
if(MathGL2_FOUND)
    set(MathGL2_LIBRARIES "${MathGL2_LIBRARIES_DIRS}/libmgl.so")
else()
    message("Could not find MathGL2. Omitting performance tests...")
endif()


set(TEST_C
    # test_map
    # test_queue
    # test_view
)

set(TEST_CXX
    # test_tensor_ternary
    # test_tensor_structure
    # test_tensor_reduction
    # test_tensor_unary
    # test_tensor_binary
    # test_autograd
    # test_optimizer
    # test_model_exporter
)

set(TEST_CXX_PERF
    # test_tensor_unary_performance
    # test_tensor_binary_performance
    # test_tensor_reduction_performance
)

set(TEST_HELPER_TORCH
    test_helper_torch
)

set(TEST_HELPER
    test_helper
)

if(MathGL2_FOUND)
    set(TESTS ${TEST_C} ${TEST_CXX} ${TEST_CXX_PERF})
else()
    set(TESTS ${TEST_C} ${TEST_CXX})
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --nvlink-options=--disable-warnings")
foreach(TEST ${TESTS})
    if (NOT DEFINED ENV{CPU_ONLY})
        add_custom_target("compute_sanitizer_${TEST}" COMMAND compute-sanitizer --leak-check full --target-processes all $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
    endif()
    if(${TEST} IN_LIST TEST_CXX OR (${TEST} IN_LIST TEST_CXX_PERF))
        add_executable(${TEST} "${TEST}.cc" "${TEST_HELPER_TORCH}.cc" "${TEST_HELPER}.c")
        set_target_properties(${TEST} PROPERTIES PUBLIC_HEADER "${TEST_HELPER_TORCH}.h ${TEST_HELPER}.h")
    elseif(${TEST} IN_LIST TEST_C)
        add_executable(${TEST} "${TEST}.c" "${TEST_HELPER}.c")
        set_target_properties(${TEST} PROPERTIES PUBLIC_HEADER "${TEST_HELPER}.h")
    endif()
    target_include_directories(${TEST} PUBLIC "." ${MATHGL2_INCLUDE_DIRS})
    set_target_properties(${TEST} PROPERTIES PUBLIC_HEADER "${TEST_HELPER}.h")
    target_link_libraries(${TEST} ${PROJECT_NAME} check subunit m "${TORCH_LIBRARIES}" ${MathGL2_LIBRARIES})
    add_test(NAME ${TEST} COMMAND ${TEST} WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/${TEST_DIR})
    # add_custom_target("valgrind_${TEST}" COMMAND valgrind --leak-check=full  --show-reachable=yes --error-limit=no --gen-suppressions=all --log-file=../suppressions.log --suppressions=../../suppressions.supp $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
    add_custom_target("valgrind_${TEST}" COMMAND valgrind --leak-check=full  --show-reachable=yes --error-limit=no --suppressions=../../suppressions.supp $<TARGET_FILE:${TEST}> DEPENDS ${TEST})
endforeach()
