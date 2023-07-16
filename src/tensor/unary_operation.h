#ifndef UNARY_OPERATION_H
#define UNARY_OPERATION_H

#include <tensor.h>

typedef enum unary_operation_type_t
{
    EXPONENTIAL_OPERATION,
    LOGARITHM_OPERATION,
    SIN_OPERATION,
    SQUARE_ROOT_OPERATION
} unary_operation_type_t;

typedef struct unary_operation_t
{
    tensor_t *x;
    unary_operation_type_t operation_type;
} unary_operation_t;

error_t *create_unary_operation(unary_operation_t **operation, unary_operation_type_t operation_type, tensor_t *x);
error_t *destroy_unary_operation(unary_operation_t *operation);
error_t *unary_operation_forward(unary_operation_t *operation, tensor_t *result);

#endif