
#ifndef FUNCTION_H
#define FUNCTION_H

// Includes
#include <datatype.h>
#include <errors.h>
#include <tensor.h>

typedef struct tensor_t tensor_t;

// Unary Operation
typedef enum unary_operation_type_t
{
    EXPONENTIAL_OPERATION,
    LOGARITHM_OPERATION,
    SIN_OPERATION,
    POWER_OPERATION
} unary_operation_type_t;

typedef struct unary_operation_t
{
    tensor_t *x;
    unary_operation_type_t operation_type;
} unary_operation_t;

error_t *unary_operation_create(unary_operation_t **unary_operation, unary_operation_type_t unary_operation_type, tensor_t *x);
void unary_operation_destroy(unary_operation_t *unary_operation);
error_t *unary_operation_forward(unary_operation_t *unary_operation, tensor_t *result);
error_t *unary_operation_backward(unary_operation_t *unary_operation, tensor_t *gradient);

// Binary Operation
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

error_t *binary_operation_create(binary_operation_t **operation, binary_operation_type_t operation_type, tensor_t *x, tensor_t *y);
void binary_operation_destroy(binary_operation_t *operation);
error_t *binary_operation_forward(const binary_operation_t *operation, tensor_t *result);
error_t *binary_operation_backward(binary_operation_t *binary_operation, tensor_t *gradient);

// Reduction Operation
typedef enum reduction_operation_type_t
{
    SUMMATION_OPERATION,
    MAXIMUM_OPERATION
} reduction_operation_type_t;

typedef struct reduction_operation_t
{
    tensor_t *x;
    uint32_t *axis;
    bool_t keep_dimension;
    reduction_operation_type_t operation_type;
} reduction_operation_t;

error_t *reduction_operation_create(reduction_operation_t **reduction_operation,
                                    reduction_operation_type_t reduction_operation_type,
                                    tensor_t *x, uint32_t *axis, bool_t keep_dimension);
void reduction_operation_destroy(reduction_operation_t *reduction_operation);
error_t *reduction_operation_forward(reduction_operation_t *reduction_operation, tensor_t *result);
error_t *reduction_operation_backward(reduction_operation_t *reduction_operation, tensor_t *gradient);

// Structure Operation
typedef enum structure_operation_type_t
{
    EXPAND_OPERATION,
    PERMUTE_OPERATION,
    RESHAPE_OPERATION,
    PADDING_OPERATION,
    SLICE_OPERATION
} structure_operation_type_t;

typedef struct structure_operation_t
{
    tensor_t *x;
    void *arguments;
    structure_operation_type_t operation_type;
} structure_operation_t;

error_t *structure_operation_create(structure_operation_t **operation, structure_operation_type_t operation_type, tensor_t *x, void *arguments);
void structure_operation_destroy(structure_operation_t *operation);
error_t *structure_operation_forward(structure_operation_t *structure_operation, tensor_t *result);
error_t *structure_operation_backward(structure_operation_t *structure_operation, tensor_t *gradient);

// Operation
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

error_t *operation_create(operation_t **operation, operation_type_t operation_type, void *type_operation);
void operation_destroy(operation_t *operation, operation_type_t operation_type);
error_t *operation_forward(operation_t *operation, operation_type_t operation_type, tensor_t *result);
error_t *operation_backward(operation_t *operation, operation_type_t operation_type, tensor_t *gradient);

// Function
typedef struct function_t
{
    operation_type_t operation_type;
    operation_t *operation;
} function_t;

error_t *function_create(function_t **function, operation_t *operation, operation_type_t operation_type);
void function_destroy(function_t *function);
error_t *function_forward(function_t *function, tensor_t *result);
error_t *function_backward(function_t *function, tensor_t *gradient);

#endif