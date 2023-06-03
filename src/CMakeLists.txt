set(INCLUDE_DIR "include")

set(
    SOURCE 
    "cpu_operation.c"
    "cuda_operation.cu"
    "device.c"
    "errors.c"
)

set(
    HEADERS
    "${INCLUDE_DIR}/cpu_operation.h"
    "${INCLUDE_DIR}/cuda_operation.h"
    "${INCLUDE_DIR}/device.h"
    "${INCLUDE_DIR}/errors.h"
    "${INCLUDE_DIR}/operation.h"
    "${INCLUDE_DIR}/stdfloat.h"
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
