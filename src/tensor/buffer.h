#ifndef BUFFER_H
#define BUFFER_H

#include <view.h>

typedef enum runtime_t
{
   OPENBLAS_RUNTIME,
   MKL_RUNTIME,
   CU_RUNTIME
} runtime_t;

typedef enum runtime_unary_elementwise_type_t
{
    RUNTIME_LOG
} runtime_binary_elementwise_type_t;

typedef enum runtime_binary_elementwise_type_t
{
    RUNTIME_ADDITION,
    RUNTIME_SUBTRACTION,
    RUNTIME_DIVISION,
    RUNTIME_MULTIPLICATION
} runtime_binary_elementwise_type_t;

typedef enum runtime_reduce_type_t
{
    RUNTIME_SUM,
    RUNTIME_MAX
} runtime_reduce_type_t;

typedef enum runtime_matrix_multiplication_type_t
{
    RUNTIME_MATRIX_MULTIPLICATION
} runtime_matrix_multiplication_type_t;

typedef struct buffer_t
{
    view_t *view;
    runtime_t runtime;
    datatype_t datatype;
    void *data;
} buffer_t;

error_t *buffer_create(buffer_t **buffer, runtime_t runtime, datatype_t datatype, view_t *view, void *data);
void buffer_destroy(buffer_t *buffer);

error_t *runtime_malloc(buffer_t *buffer);
void runtime_free(buffer_t *buffer);
error_t *runtime_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
error_t *runtime_matrix_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
string_t runtime_string(runtime_t runtime);
error_t *runtime_create_context(runtime_t runtime);
void runtime_destroy_context(runtime_t runtime);

#endif