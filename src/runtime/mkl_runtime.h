#ifndef MKL_RUNTIME_H
#define MKL_RUNTIME_H

#include <errors.h>

error_t *mkl_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data);

#endif