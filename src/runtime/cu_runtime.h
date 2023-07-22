#ifndef CU_RUNTIME_H
#define CU_RUNTIME_H

#include <errors.h>

error_t *cu_malloc(void **pp, size_t size);
void cu_free(void *p);
error_t *cu_create_context(void);
void cu_destroy_context(void);
error_t *cu_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data);

#endif