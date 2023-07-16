#ifndef BINARY_OPERATION_H
#define BINARY_OPERATION_H

#include <tensor.h>

typedef enum binary_operation_type_t
{
    ADDITION_OPERATION,
    SUBTRACTION_OPERATION,
    MULTIPLICATION_OPERATION,
    DIVISION_OPERATION,
    MATRIX_MULTIPLICATION_OPERATION
} binary_operation_type_t;

typedef struct binary_operation_t
{
    tensor_t *x;
    tensor_t *y;
    binary_operation_type_t operation_type;
} binary_operation_t;

error_t *create_binary_operation(binary_operation_t **operation, binary_operation_type_t operation_type, tensor_t *x, tensor_t *y);
error_t *destroy_binary_operation(binary_operation_t *operation);
error_t *binary_operation_forward(binary_operation_t *operation, tensor_t *result);
error_t *addition_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *z);
error_t *addition_operation_backward(const tensor_t *x, const tensor_t *y, const tensor_t *gradient, tensor_t *gradient_x, tensor_t *gradient_y);

#endif