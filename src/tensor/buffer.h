/**@file buffer.h
 * @brief
 *
 */

#ifndef BUFFER_H
#define BUFFER_H

#include <errors.h>

// Forward declarations
typedef struct view_t view_t;

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

typedef struct storage_t
{
    uint64_t reference_count;
    runtime_t runtime;
    datatype_t datatype;
    uint64_t n;
    void *data;
} storage_t;

typedef struct buffer_t
{
    view_t *view;
    storage_t *storage;
} buffer_t;

typedef enum runtime_unary_type_t
{
    RUNTIME_EXPONENTIAL,
    RUNTIME_LOGARITHM,
    RUNTIME_SINE,
    RUNTIME_COSINE,
    RUNTIME_SQUARE_ROOT,
    RUNTIME_RECIPROCAL,
    RUNTIME_CONTIGUOUS,
    RUNTIME_NEGATION,
    RUNTIME_RECTIFIED_LINEAR,
    RUNTIME_SIGMOID,
} runtime_unary_type_t;

typedef enum runtime_binary_elementwise_type_t
{
    RUNTIME_ADDITION,
    RUNTIME_SUBTRACTION,
    RUNTIME_MULTIPLICATION,
    RUNTIME_DIVISION,
    RUNTIME_POWER,
    RUNTIME_COMPARE_EQUAL,
    RUNTIME_COMPARE_GREATER
} runtime_binary_elementwise_type_t;

typedef enum runtime_reduction_type_t
{
    RUNTIME_SUMMATION,
    RUNTIME_MAXIMUM
} runtime_reduction_type_t;

nw_error_t *storage_create(storage_t **storage, runtime_t runtime, datatype_t datatype, uint64_t n, void *data, bool_t copy);
void storage_destroy(storage_t *storage);
nw_error_t *runtime_malloc(storage_t *storage);
void runtime_free(storage_t *storage);

nw_error_t *buffer_create(buffer_t **buffer, view_t *view, storage_t *storage, bool_t copy);
void buffer_destroy(buffer_t *buffer);
nw_error_t *buffer_create_empty(buffer_t **buffer, const uint64_t *shape, const uint64_t *strides,
                                uint64_t rank, runtime_t runtime, datatype_t datatype);
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
nw_error_t *runtime_sigmoid(buffer_t *x, buffer_t *result);
nw_error_t *runtime_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_subtraction(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_division(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_power(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_compare_equal(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_compare_greater(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_matrix_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
nw_error_t *runtime_summation(buffer_t *x, uint64_t *axis, uint64_t length, buffer_t *result, bool_t keep_dimension);
nw_error_t *runtime_maximum(buffer_t *x, uint64_t *axis, uint64_t length, buffer_t *result, bool_t keep_dimension);
nw_error_t *runtime_init_zeroes(buffer_t *buffer);
nw_error_t *runtime_init_ones(buffer_t *buffer);
nw_error_t *runtime_init_arange(buffer_t *buffer, uint64_t start, uint64_t stop, uint64_t step);
#endif
