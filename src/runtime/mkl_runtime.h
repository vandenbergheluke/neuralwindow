#ifndef MKL_RUNTIME_H
#define MKL_RUNTIME_H

#include <errors.h>
#include <datatype.h>

// TODO: We should choose this based on detected CPU architecture.
// This parameter can make a significant impact on performance.
#define ALIGNMENT 64

nw_error_t *mkl_memory_allocate(void **pp, size_t size);
void mkl_memory_free(void *p);
void mkl_exponential(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void mkl_logarithm(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void mkl_sine(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void mkl_cosine(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void mkl_square_root(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void mkl_reciprocal(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void mkl_copy(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void mkl_negation(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void mkl_rectified_linear(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void mkl_addition(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void mkl_subtraction(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void mkl_multiplication(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void mkl_division(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void mkl_power(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void mkl_compare_equal(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void mkl_compare_greater(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void mkl_matrix_multiplication(datatype_t datatype, uint64_t m, uint64_t k, uint64_t n, bool_t x_transpose, bool_t y_transpose, const void *x_data, uint64_t x_offset, const void *y_data, uint64_t y_offset, void *z_data, uint64_t z_offset);
void mkl_summation(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_offset);
void mkl_maximum(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_offset);

#endif
