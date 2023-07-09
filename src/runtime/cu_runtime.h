#ifndef CU_RUNTIME_H
#define CU_RUNTIME_H

#include <errors.h>

error_t *cu_malloc(void **pp, size_t size);
error_t *cu_free(void *p);

#endif