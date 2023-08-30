#ifndef BUFFER_H
#define BUFFER_H

#include <view.h>
#include <errors.h>

typedef enum runtime_t
{
   OPENBLAS_RUNTIME,
   MKL_RUNTIME,
   CU_RUNTIME
} runtime_t;

typedef struct buffer_t
{
    view_t *view;
    runtime_t runtime;
    datatype_t datatype;
    void *data;
    size_t size;
    uint32_t n;
    bool_t copy;
} buffer_t;

nw_error_t *buffer_create(buffer_t **buffer, runtime_t runtime, datatype_t datatype, view_t *view, void *data, uint32_t n, bool_t copy);
void buffer_destroy(buffer_t *buffer);

nw_error_t *runtime_malloc(buffer_t *buffer);
void runtime_free(buffer_t *buffer);
string_t runtime_string(runtime_t runtime);
nw_error_t *runtime_create_context(runtime_t runtime);
void runtime_destroy_context(runtime_t runtime);
nw_error_t *runtime_exponential(buffer_t *x, buffer_t *result);
nw_error_t *runtime_logarithm(buffer_t *x, buffer_t *result);
nw_error_t *runtime_sine(buffer_t *x, buffer_t *result);
nw_error_t *runtime_cosine(buffer_t *x, buffer_t *result);
nw_error_t *runtime_square_root(buffer_t *x, buffer_t *result);
nw_error_t *runtime_reciprocal(buffer_t *x, buffer_t *result);
nw_error_t *runtime_copy(buffer_t *x, buffer_t *result);
nw_error_t *runtime_contiguous(buffer_t *x, buffer_t *result);
nw_error_t *runtime_negation(buffer_t *x, buffer_t *result);
nw_error_t *runtime_rectified_linear(buffer_t *x, buffer_t *result);
nw_error_t *runtime_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_subtraction(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_division(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_power(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_compare_equal(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_compare_greater(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_matrix_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_summation(buffer_t *x, buffer_t *result, uint32_t axis);
nw_error_t *runtime_maximum(buffer_t *x, buffer_t *result, uint32_t axis);

#endif