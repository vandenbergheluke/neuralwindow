
#ifndef FUNCTION_H
#define FUNCTION_H

#include <datatype.h>
#include <errors.h>

struct tensor_t;

#include <tensor.h>
#include <operation.h>

typedef struct function_t
{
    operation_type_t operation_type;
    operation_t *operation;
    bool_t requires_gradient;
} function_t;

error_t *create_function(function_t **function, operation_t *operation, operation_type_t operation_type, bool_t requires_gradient);
error_t *destroy_function(function_t *function);
error_t *function_forward(const function_t *function, tensor_t *result);
error_t *function_backward(function_t *function, tensor_t *gradient);

#endif