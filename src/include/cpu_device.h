#ifndef CPU_OPERATION_H
#define CPU_OPERATION_H

#include <errors.h>

error_t *cpu_malloc(void **p, size_t size);
error_t *cpu_free(void *p);

#endif