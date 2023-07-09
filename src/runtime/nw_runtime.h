#ifndef NW_RUNTIME_H
#define NW_RUNTIME_H

#include <errors.h>

typedef enum runtime_t
{
   C,
   OPENBLAS,
   MKL,
   CU
} runtime_t;

error_t *nw_malloc(void **pp, size_t size, runtime_t runtime);
error_t *nw_free(void *p, runtime_t runtime);
string_t runtime_string(runtime_t runtime);

#endif
