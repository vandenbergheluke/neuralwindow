#ifndef NW_RUNTIME_H
#define NW_RUNTIME_H

#include <errors.h>
#include <buffer.h>

error_t *nw_malloc(void **pp, size_t size, runtime_t runtime);
void nw_free(void *p, runtime_t runtime);
error_t *nw_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer);
string_t runtime_string(runtime_t runtime);

#endif
