#ifndef CU_RUNTIME_H
#define CU_RUNTIME_H

#include <errors.h>

nw_error_t *cu_memory_allocate(void **pp, size_t size);
void cu_memory_free(void *p);
nw_error_t *cu_create_context(void);
void cu_destroy_context(void);
void cu_exponential(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void cu_logarithm(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void cu_sine(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void cu_cosine(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void cu_square_root(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void cu_reciprocal(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void cu_copy(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void cu_negation(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void cu_rectified_linear(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_stride, uint64_t y_offset);
void cu_addition(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void cu_subtraction(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void cu_multiplication(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void cu_division(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void cu_power(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void cu_compare_equal(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void cu_compare_greater(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, const void *y_data, uint64_t y_stride, uint64_t y_offset, void *z_data, uint64_t z_stride, uint64_t z_offset);
void cu_matrix_multiplication(datatype_t datatype, uint64_t m, uint64_t k, uint64_t n, bool_t x_transpose, bool_t y_transpose, const void *x_data, uint64_t x_offset, const void *y_data, uint64_t y_offset, void *z_data, uint64_t z_offset);
void cu_summation(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_offset);
void cu_maximum(datatype_t datatype, uint64_t n, const void *x_data, uint64_t x_stride, uint64_t x_offset, void *y_data, uint64_t y_offset);

#endif