set(INCLUDE_DIR "include")

set(
    SOURCE 
    "buffer.c"
    "cpu_device.c"
    "cpu_operation.c"
    "cuda_device.cu"
    "cuda_operation.cu"
    "datatype.c"
    "device.c"
    "errors.c"
    "measure.c"
    "shape.c"
    "tensor.c"
)

set(
    HEADERS
    "${INCLUDE_DIR}/buffer.h"
    "${INCLUDE_DIR}/cpu_device.h"
    "${INCLUDE_DIR}/cpu_operation.h"
    "${INCLUDE_DIR}/cuda_device.h"
    "${INCLUDE_DIR}/cuda_operation.h"
    "${INCLUDE_DIR}/datatype.h"
    "${INCLUDE_DIR}/device.h"
    "${INCLUDE_DIR}/errors.h"
    "${INCLUDE_DIR}/measure.h"
    # "${INCLUDE_DIR}/operation.h"
    "${INCLUDE_DIR}/shape.h"
    "${INCLUDE_DIR}/stdfloat.h"
    "${INCLUDE_DIR}/tensor.h"
)

add_library(${PROJECT_NAME} STATIC ${SOURCE})

target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION 1)
install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
