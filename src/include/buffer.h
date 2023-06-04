#ifndef BUFFER_H
#define BUFFER_H

#include <errors.h>
#include <device.h>

typedef void * buffer_t;

error_t *create_buffer(buffer_t *b, size_t size, device_t device);
error_t *destory_buffer(buffer_t b, device_t device);

#endif