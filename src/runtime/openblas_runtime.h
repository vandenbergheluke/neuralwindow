#ifndef OPENBLAS_RUNTIME_H
#define OPENBLAS_RUNTIME_H

#include <errors.h>
#include <datatype.h>

error_t *openblas_memory_allocate(void **pp, size_t size);
void openblas_memory_free(void *p);
void openblas_exponential(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, void *y_data, uint32_t y_stride);
void openblas_logarithm(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, void *y_data, uint32_t y_stride);
void openblas_sine(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, void *y_data, uint32_t y_stride);
void openblas_cosine(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, void *y_data, uint32_t y_stride);
void openblas_square_root(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, void *y_data, uint32_t y_stride);
void openblas_reciprocal(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, void *y_data, uint32_t y_stride);
void openblas_copy(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, void *y_data, uint32_t y_stride);
void openblas_addition(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, const void *y_data, uint32_t y_stride, void *z_data, uint32_t z_stride);
void openblas_subtraction(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, const void *y_data, uint32_t y_stride, void *z_data, uint32_t z_stride);
void openblas_multiplication(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, const void *y_data, uint32_t y_stride, void *z_data, uint32_t z_stride);
void openblas_division(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, const void *y_data, uint32_t y_stride, void *z_data, uint32_t z_stride);
void openblas_power(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, const void *y_data, uint32_t y_stride, void *z_data, uint32_t z_stride);
void openblas_matrix_multiplication(datatype_t datatype, uint32_t m, uint32_t k, uint32_t n, bool_t x_transpose, bool_t y_transpose, const void *x_data, const void *y_data, void *z_data);
void openblas_summation(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, void *y_data);
void openblas_maximum(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, void *y_data);

#endif