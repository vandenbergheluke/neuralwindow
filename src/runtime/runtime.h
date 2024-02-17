#ifndef RUNTIME_H
#define RUNTIME_H

#include <errors.h>
#include <operation.h>

typedef struct buffer_t buffer_t;
typedef struct storage_t storage_t;

#ifdef __cplusplus
typedef enum runtime_t: int
#else
typedef enum runtime_t
#endif
{
   OPENBLAS_RUNTIME,
   MKL_RUNTIME,
   CU_RUNTIME
} runtime_t;

#ifdef CPU_ONLY
#define RUNTIMES 2
#else
#define RUNTIMES 3
#endif

#define EPSILON 1e-7

nw_error_t *runtime_create_context(runtime_t runtime);
void runtime_destroy_context(runtime_t runtime);
nw_error_t *runtime_malloc(void **data, int64_t n, datatype_t datatype, runtime_t runtime);
void runtime_free(void *data, runtime_t runtime);
void runtime_unary(unary_operation_type_t unary_operation_type, runtime_t runtime, datatype_t datatype, int64_t n, 
                   void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset);
void runtime_binary_elementwise(binary_operation_type_t binary_operation_type, runtime_t runtime, datatype_t datatype, int64_t n,
                                void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_stride, int64_t y_offset,
                                void *z_data, int64_t z_stride, int64_t z_offset);
void runtime_ternary(ternary_operation_type_t ternary_operation_type, runtime_t runtime, datatype_t datatype, int64_t n,
                     void *w_data, int64_t w_stride, int64_t w_offset, void *x_data, int64_t x_stride, int64_t x_offset, 
                     void *y_data, int64_t y_stride, int64_t y_offset, void *z_data, int64_t z_stride, int64_t z_offset);
void runtime_matrix_multiplication(runtime_t runtime, datatype_t datatype, int64_t m, int64_t k, int64_t n, bool_t x_transpose, bool_t y_transpose,
                                   void *x_data, int64_t x_offset, void *y_data, int64_t y_offset, void *z_data, int64_t z_offset);
void runtime_reduction(reduction_operation_type_t reduction_operation_type, runtime_t runtime, datatype_t datatype, int64_t n,
                       void *x_data, int64_t x_stride, int64_t x_offset, void *y_data, int64_t y_offset);
string_t runtime_string(runtime_t runtime);
void runtime_zeroes(void *data, int64_t n, datatype_t datatype);
void runtime_ones(void *data, int64_t n, datatype_t datatype);
void runtime_arange(void *data, datatype_t datatype, void *start, void *stop, void *step);
void runtime_uniform(void *data, int64_t n, datatype_t datatype, void *lower_bound, void *upper_bound);
void runtime_normal(void *data, int64_t n, datatype_t datatype, void *mean, void *standard_deviation);
void runtime_image_to_column(datatype_t datatype, void *x_data, 
                             int64_t batch_size, int64_t channels, int64_t height, int64_t width, 
                             int64_t kernel_size, int64_t output_height, int64_t output_width,
                             int64_t stride, int64_t padding, void *y_data, bool_t inverse, void *padding_value);

#endif