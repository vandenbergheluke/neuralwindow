#ifndef DEVICE_H
#define DEVICE_H

#include <errors.h>

typedef enum device_t
{
    DEVICE_CPU,
    DEVICE_CUDA
} device_t;

typedef void * buffer_t;

error_t memory_allocate(buffer_t *buffer_ptr, size_t size, device_t device);
error_t memory_free(buffer_t buffer, device_t device);

#endif
