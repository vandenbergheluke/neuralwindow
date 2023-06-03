#ifndef CPU_OPERATION_H
#define CPU_OPERATION_H

#include <errors.h>
#include <device.h>
#include <stdlib.h>

error_t cpu_malloc(buffer_t *buffer_ptr, size_t size);
error_t cpu_free(buffer_t buffer);

#endif