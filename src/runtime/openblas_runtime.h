#ifndef OPENBLAS_RUNTIME_H
#define OPENBLAS_RUNTIME_H

#include <errors.h>
#include <datatype.h>

nw_error_t *openblas_memory_allocate(void **pp, size_t size);
void openblas_memory_free(void *p);
void openblas_exponential(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void openblas_logarithm(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void openblas_sine(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void openblas_cosine(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void openblas_square_root(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void openblas_reciprocal(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void openblas_copy(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void openblas_negation(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void openblas_rectified_linear(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void openblas_addition(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void openblas_subtraction(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void openblas_multiplication(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void openblas_division(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void openblas_power(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void openblas_compare_equal(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void openblas_compare_greater(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void openblas_matrix_multiplication(datatype_t datatype, uint64_t m, uint64_t k, uint64_t n, bool_t x_transpose, bool_t y_transpose, const void *x_data, uint64_t x_offset, const void *y_data, uint64_t y_offset, void *z_data, uint64_t z_offset);
void openblas_summation(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_offset);
void openblas_maximum(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_offset);

#endif