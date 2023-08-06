#ifndef OPENBLAS_RUNTIME_H
#define OPENBLAS_RUNTIME_H

#include <errors.h>
#include <datatype.h>

error_t *openblas_memory_allocate(void **pp, size_t size);
void openblas_memory_free(void *p);
error_t *openblas_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data);
error_t *openblas_matrix_multiplication(datatype_t datatype, uint32_t m, uint32_t k, uint32_t n, bool_t x_transpose, bool_t y_transpose,  const void *x_data, const void *y_data, void *z_data);
error_t *openblas_summation(datatype_t datatype, uint32_t axis, uint32_t current_dimension, uint32_t x_index, uint32_t *y_index, 
                            uint32_t *x_shape, uint32_t x_rank, uint32_t *x_strides, const void *x_data, void *y_data);
#endif