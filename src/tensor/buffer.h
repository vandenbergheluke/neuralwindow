#ifndef BUFFER_H
#define BUFFER_H

#include <view.h>

typedef enum runtime_t
{
   C_RUNTIME,
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
} buffer_t;

error_t *buffer_create(buffer_t **buffer, runtime_t runtime, datatype_t datatype, view_t *view, void *data);
void buffer_destroy(buffer_t *buffer);

#endif