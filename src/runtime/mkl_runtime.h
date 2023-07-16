#ifndef MKL_RUNTIME_H
#define MKL_RUNTIME_H

#include <errors.h>

error_t *mkl_addition(datatype_t datatype, uint32_t size, const void *in_data_x, const void *in_data_y, void *out_data);

#endif