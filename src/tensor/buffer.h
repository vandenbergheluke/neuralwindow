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
    bool_t new;
} buffer_t;

error_t *buffer_create(buffer_t **buffer, runtime_t runtime, datatype_t datatype, view_t *view, void *data, size_t size, bool_t new);
void buffer_destroy(buffer_t *buffer);

error_t *runtime_malloc(buffer_t *buffer);
void runtime_free(buffer_t *buffer);
error_t *runtime_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
error_t *runtime_matrix_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
string_t runtime_string(runtime_t runtime);
error_t *runtime_create_context(runtime_t runtime);
void runtime_destroy_context(runtime_t runtime);

#endif