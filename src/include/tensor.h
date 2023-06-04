#ifndef TENSOR_H
#define TENSOR_H

#include <device.h>
#include <datatype.h>
#include <stdbool.h>
#include <shape.h>
#include <buffer.h>

typedef struct tensor_t
{
    shape_t shape;
    datatype_t datatype;
    buffer_t buffer;
    device_t device;
    struct tensor_t *gradient;
    // operation_t context;
    bool requires_gradient;
} tensor_t;

error_t *create_tensor(tensor_t **t, device_t device);
error_t *destroy_tensor(tensor_t *t, device_t device);

#endif