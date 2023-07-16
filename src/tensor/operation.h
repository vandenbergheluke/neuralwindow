#ifndef OPERATION_H
#define OPERATION_H

#include <unary_operation.h>
#include <binary_operation.h>
#include <reduction_operation.h>
#include <structure_operation.h>

typedef enum operation_type_t
{
    UNARY_OPERATION,
    BINARY_OPERATION,
    REDUCTION_OPERATION,
    STRUCTURE_OPERATION
} operation_type_t;

typedef union operation_t
{
    unary_operation_t *unary_operation;
    binary_operation_t *binary_operation;
    reduction_operation_t *reduction_operation;
    structure_operation_t *structure_operation;
} operation_t;

error_t *create_operation(operation_t **operation, operation_type_t operation_type, void *type_operation);
error_t *destroy_operation(operation_t *operation, operation_type_t operation_type);
error_t *operation_forward(operation_t *operation, operation_type_t operation_type, tensor_t *result);

#endif