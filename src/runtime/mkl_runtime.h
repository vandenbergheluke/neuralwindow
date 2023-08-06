#ifndef MKL_RUNTIME_H
#define MKL_RUNTIME_H

#include <errors.h>

// TODO: We should choose this based on detected CPU architecture.
// This parameter can make a significant impact on performance.
#define ALIGNMENT 64

error_t *mkl_memory_allocate(void **pp, size_t size);
void mkl_memory_free(void *p);
error_t *mkl_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data);
error_t *mkl_matrix_multiplication(datatype_t datatype, uint32_t m, uint32_t k, uint32_t n, bool_t x_transpose, bool_t y_transpose,  const void *x_data, const void *y_data, void *z_data);
error_t *mkl_summation(datatype_t datatype, uint32_t axis, uint32_t current_dimension, uint32_t x_index, uint32_t *y_index, 
                       uint32_t *x_shape, uint32_t x_rank, uint32_t *x_strides, const void *x_data, void *y_data);

#endif