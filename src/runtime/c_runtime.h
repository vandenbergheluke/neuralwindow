#ifndef C_RUNTIME_H
#define C_RUNTIME_H

#include <errors.h>

error_t *c_malloc(void **pp, size_t size);
void c_free(void *p);
error_t *c_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data);

#endif