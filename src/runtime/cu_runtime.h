#ifndef CU_RUNTIME_H
#define CU_RUNTIME_H

#include <errors.h>

error_t *cu_malloc(void **pp, size_t size);
error_t *cu_free(void *p);
error_t *cu_copy(const void *in_p, void *out_p, size_t size);
error_t *cu_addition(datatype_t datatype, uint32_t size, const void *in_data_x, const void *in_data_y, void *out_data);

#endif