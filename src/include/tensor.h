#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>

typedef void * buffer_t;

typedef enum datatype_t
{
    FLOAT32,
    FLOAT64
} datatype_t;

typedef struct shape_t
{
    uint32_t *dimensions;
    uint32_t rank;
    uint32_t *stride;
    uint32_t offset;
} shape_t;

typedef struct tensor_t {
    buffer_t *data;
    datatype_t datatype;
    shape_t shape;
} tensor_t;

tensor_t *construct_tensor(buffer_t data, datatype_t datatype, shape_t shape);
void destroy_tensor(tensor_t *x);

#endif
