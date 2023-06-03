#ifndef CUDA_OPERATION_H
#define CUDA_OPERATION_H

#include <errors.h>
#include <device.h>

error_t cuda_malloc(buffer_t *buffer_ptr, size_t size);
error_t cuda_free(buffer_t buffer);

#endif