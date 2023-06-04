#ifndef CUDA_OPERATION_H
#define CUDA_OPERATION_H

#include <errors.h>

error_t *cuda_malloc(void **p, size_t size);
error_t *cuda_free(void *p);

#endif