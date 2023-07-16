#ifndef TENSOR_H
#define TENSOR_H

struct tensor_t;

#include <datatype.h>
#include <buffer.h>
#include <function.h>

typedef struct tensor_t
{
    buffer_t *buffer;
    function_t *context;
    struct tensor_t *gradient;
    bool_t requires_gradient;
} tensor_t;

error_t *create_tensor(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient);
error_t *destroy_tensor(tensor_t *tensor);

#endif