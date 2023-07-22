#ifndef OPENBLAS_RUNTIME_H
#define OPENBLAS_RUNTIME_H

#include <errors.h>
#include <datatype.h>

error_t *openblas_addition(datatype_t datatype, uint32_t size, const void *x_data, const void *y_data, void *z_data);

#endif