#ifndef TENSOR_H
#define TENSOR_H

struct operation_t;
struct tensor_t;

#include <datatype.h>
#include <buffer.h>
#include <operation.h>

typedef struct tensor_t
{
    buffer_t *buffer;
    struct operation_t *context;
    struct tensor_t *gradient;
    bool_t requires_gradient;
} tensor_t;

error_t *create_tensor(tensor_t **tensor, runtime_t runtime);
error_t *destroy_tensor(tensor_t *tensor, runtime_t runtime);

#endif