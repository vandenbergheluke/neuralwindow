#ifndef DEVICE_H
#define DEVICE_H

#include <errors.h>

typedef enum device_t
{
    DEVICE_CPU,
    DEVICE_CUDA
} device_t;

error_t *memory_allocate(void **p, size_t size, device_t device);
error_t *memory_free(void *p, device_t device);
char *device_string(device_t device);

#endif
