set(UTIL_DIR "util")
set(TENSOR_DIR "tensor")
set(RUNTIME_DIR "runtime")
set(NN_DIR "nn")
set(MKL_DIR "/usr/include/mkl")

set(
    SOURCE 
    "${TENSOR_DIR}/buffer.c"
    "${TENSOR_DIR}/function.c"
    "${TENSOR_DIR}/operation.c"
    "${TENSOR_DIR}/tensor.c"
    "${TENSOR_DIR}/view.c"
    "${UTIL_DIR}/datatype.c"
    "${UTIL_DIR}/element.c"
    "${UTIL_DIR}/errors.c"
    "${UTIL_DIR}/map.c"
    "${UTIL_DIR}/measure.c"
    "${UTIL_DIR}/queue.c"
    "${UTIL_DIR}/random.c"
    "${UTIL_DIR}/stack.c"
    "${UTIL_DIR}/graph.c"
    "${UTIL_DIR}/sort.c"
    "${RUNTIME_DIR}/mkl_runtime.c"
    "${RUNTIME_DIR}/openblas_runtime.c"
    "${RUNTIME_DIR}/runtime.c"
    "${NN_DIR}/cost.c"
    "${NN_DIR}/init.c"
    "${NN_DIR}/layer.c"
    "${NN_DIR}/metric.c"
    "${NN_DIR}/optimizer.c"
    "${NN_DIR}/train.c"
)

set(
    SOURCE_CUDA
    "${RUNTIME_DIR}/cu_runtime.cu"
)

set(
    HEADERS
    "${TENSOR_DIR}/buffer.h"
    "${TENSOR_DIR}/function.h"
    "${TENSOR_DIR}/operation.h"
    "${TENSOR_DIR}/tensor.h"
    "${TENSOR_DIR}/view.h"
    "${UTIL_DIR}/datatype.h"
    "${UTIL_DIR}/element.h"
    "${UTIL_DIR}/errors.h"
    "${UTIL_DIR}/map.h"
    "${UTIL_DIR}/measure.h"
    "${UTIL_DIR}/queue.h"
    "${UTIL_DIR}/random.h"
    "${UTIL_DIR}/stack.h"
    "${UTIL_DIR}/graph.h"
    "${UTIL_DIR}/sort.h"
    "${RUNTIME_DIR}/mkl_runtime.h"
    "${RUNTIME_DIR}/openblas_runtime.h"
    "${RUNTIME_DIR}/runtime.h"
    "${NN_DIR}/cost.h"
    "${NN_DIR}/init.h"
    "${NN_DIR}/layer.h"
    "${NN_DIR}/metric.h"
    "${NN_DIR}/optimizer.h"
    "${NN_DIR}/train.h"
)

set(
    HEADER_CUDA
    "${RUNTIME_DIR}/cu_runtime.h"
)

set(
    CPU_LIBRARIES
    openblas
    mkl_core
    mkl_rt
    mkl_sequential
)

set(
    GPU_LIBRARIES
    cublas
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(GraphViz REQUIRED)

if (NOT DEFINED ENV{CPU_ONLY})
    add_library(${PROJECT_NAME} STATIC ${SOURCE} ${SOURCE_CUDA})
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CPU_LIBRARIES} ${GPU_LIBRARIES} ${GRAPHVIZ_CDT_LIBRARY} ${GRAPHVIZ_GVC_LIBRARY} ${GRAPHVIZ_CGRAPH_LIBRARY} ${GRAPHVIZ_PATHPLAN_LIBRARY})
    set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION 1)
    set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS} ${CUDA_HEADERS}")
else()
    add_library(${PROJECT_NAME} STATIC ${SOURCE})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CPU_LIBRARIES} ${GRAPHVIZ_CDT_LIBRARY} ${GRAPHVIZ_GVC_LIBRARY} ${GRAPHVIZ_CGRAPH_LIBRARY} ${GRAPHVIZ_PATHPLAN_LIBRARY})
    set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
endif()
target_include_directories(${PROJECT_NAME} PUBLIC ${TENSOR_DIR} ${UTIL_DIR} ${RUNTIME_DIR} ${MKL_DIR} ${NN_DIR} ${INCLUDE_DIR} ${GRAPHVIZ_INCLUDE_DIRS})
install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
