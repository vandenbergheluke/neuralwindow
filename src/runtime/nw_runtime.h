#ifndef NW_RUNTIME_H
#define NW_RUNTIME_H

#include <errors.h>
#include <buffer.h>

typedef enum runtime_t
{
   C_RUNTIME,
   OPENBLAS_RUNTIME,
   MKL_RUNTIME,
   CU_RUNTIME
} runtime_t;

error_t *nw_malloc(void **pp, size_t size, runtime_t runtime);
error_t *nw_free(void *p, runtime_t runtime);
error_t *nw_copy(const void *src, void *dst, size_t size);
error_t *nw_addition(buffer_t *buffer_x, buffer_t *buffer_y, buffer_t *buffer_z);
string_t runtime_string(runtime_t runtime);

#endif
