#ifndef OPERATION_H
#define OPERATION_H

#include <datatype.h>

typedef enum operation_type_t
{
    UNARY_OPERATION,
    BINARY_OPERATION,
    REDUCTION_OPERATION,
    STRUCTURE_OPERATION,
    CREATION_OPERATION,
} operation_type_t;

typedef enum unary_operation_type_t
{
    EXPONENTIAL_OPERATION,
    LOGARITHM_OPERATION,
    SINE_OPERATION,
    COSINE_OPERATION,
    SQUARE_ROOT_OPERATION,
    RECIPROCAL_OPERATION,
    CONTIGUOUS_OPERATION,
    NEGATION_OPERATION,
    RECTIFIED_LINEAR_OPERATION,
    SIGMOID_OPERATION,
    AS_OPERATION,
} unary_operation_type_t;

typedef enum binary_operation_type_t
{
    ADDITION_OPERATION,
    SUBTRACTION_OPERATION,
    MULTIPLICATION_OPERATION,
    DIVISION_OPERATION,
    POWER_OPERATION,
    MATRIX_MULTIPLICATION_OPERATION,
    COMPARE_EQUAL_OPERATION,
    COMPARE_GREATER_OPERATION,
} binary_operation_type_t;

typedef enum reduction_operation_type_t
{
    SUMMATION_OPERATION,
    MAXIMUM_OPERATION,
} reduction_operation_type_t;

typedef enum structure_operation_type_t
{
    EXPAND_OPERATION,
    PERMUTE_OPERATION,
    RESHAPE_OPERATION,
} structure_operation_type_t;

typedef enum creation_operation_type_t
{
    EMPTY_OPERATION,
    ZEROES_OPERATION,
    ONES_OPERATION,
    UNIFORM_OPERATION,
    NORMAL_OPERATION,
    ARANGE_OPERATION,
    FROM_OPERATION,
    COPY_OPERATION,
} creation_operation_type_t;

string_t unary_operation_type_string(unary_operation_type_t unary_operation_type);
string_t binary_operation_type_string(binary_operation_type_t binary_operation_type);
string_t reduction_operation_type_string(reduction_operation_type_t reduction_operation_type);
string_t structure_operation_type_string(structure_operation_type_t structure_operation_type);
string_t creation_operation_type_string(creation_operation_type_t creation_operation_type);
string_t operation_type_string(operation_type_t operation_type);

#endif