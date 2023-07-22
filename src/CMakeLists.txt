set(UTIL_DIR "util")
set(TENSOR_DIR "tensor")
set(RUNTIME_DIR "runtime")
set(NN_DIR "nn")
set(MKL_DIR "/usr/include/mkl")

set(
    SOURCE 
    "${TENSOR_DIR}/buffer.c"
    "${TENSOR_DIR}/function.c"
    "${TENSOR_DIR}/tensor.c"
    "${TENSOR_DIR}/view.c"
    "${UTIL_DIR}/datatype.c"
    "${UTIL_DIR}/errors.c"
    "${UTIL_DIR}/map.c"
    "${UTIL_DIR}/measure.c"
    "${UTIL_DIR}/queue.c"
    "${UTIL_DIR}/random.c"
    "${RUNTIME_DIR}/c_runtime.c"
    "${RUNTIME_DIR}/cu_runtime.cu"
    "${RUNTIME_DIR}/mkl_runtime.c"
    "${RUNTIME_DIR}/nw_runtime.c"
    "${RUNTIME_DIR}/openblas_runtime.c"
    "${NN_DIR}/init.c"
    "${NN_DIR}/layer.c"
)

set(
    HEADERS
    "${TENSOR_DIR}/buffer.h"
    "${TENSOR_DIR}/function.h"
    "${TENSOR_DIR}/tensor.h"
    "${TENSOR_DIR}/view.h"
    "${UTIL_DIR}/datatype.h"
    "${UTIL_DIR}/errors.h"
    "${UTIL_DIR}/map.h"
    "${UTIL_DIR}/measure.h"
    "${UTIL_DIR}/queue.h"
    "${UTIL_DIR}/random.h"
    "${RUNTIME_DIR}/c_runtime.h"
    "${RUNTIME_DIR}/cu_runtime.h"
    "${RUNTIME_DIR}/mkl_runtime.h"
    "${RUNTIME_DIR}/nw_runtime.h"
    "${RUNTIME_DIR}/openblas_runtime.h"
    "${NN_DIR}/init.h"
    "${NN_DIR}/layer.h"
)

add_library(${PROJECT_NAME} STATIC ${SOURCE})

target_include_directories(${PROJECT_NAME} PUBLIC ${TENSOR_DIR} ${UTIL_DIR} ${RUNTIME_DIR} ${MKL_DIR} ${NN_DIR} ${INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION 1)
target_link_libraries(${TEST} ${PROJECT_NAME} openblas mkl_core cublas)
install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
