#ifndef C_RUNTIME_H
#define C_RUNTIME_H

#include <errors.h>

error_t *c_malloc(void **pp, size_t size);
error_t *c_free(void *p);

#endif