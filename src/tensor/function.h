/**
 * @file function.h
 * @brief Mid-level Operations and Automatic Differentiation Engine
 */

#ifndef FUNCTION_H
#define FUNCTION_H

#include <datatype.h>
#include <operation.h>
#include <errors.h>
#include <buffer.h>

typedef struct tensor_t tensor_t;

typedef struct unary_operation_t
{
    tensor_t *x;
    unary_operation_type_t operation_type;
} unary_operation_t;

typedef struct binary_operation_t
{
    tensor_t *x;
    tensor_t *y;
    binary_operation_type_t operation_type;
} binary_operation_t;

typedef struct ternary_operation_t
{
    tensor_t *w;
    tensor_t *x;
    tensor_t *y;
    ternary_operation_type_t operation_type;
} ternary_operation_t;

typedef struct reduction_operation_t
{
    tensor_t *x;
    int64_t *axis;
    int64_t length;
    bool_t keep_dimension;
    reduction_operation_type_t operation_type;
} reduction_operation_t;

typedef struct structure_operation_t
{
    tensor_t *x;
    int64_t *arguments;
    int64_t length;
    structure_operation_type_t operation_type;
} structure_operation_t;

typedef struct creation_operation_t
{
    creation_operation_type_t operation_type;
    int64_t *shape;
    int64_t rank;
    runtime_t runtime;
    datatype_t datatype;
    bool_t requires_gradient;
    bool_t persist;
    void **arguments;
    int64_t length;
    void *data;
} creation_operation_t;

typedef union operation_t
{
    unary_operation_t *unary_operation;
    binary_operation_t *binary_operation;
    ternary_operation_t *ternary_operation;
    reduction_operation_t *reduction_operation;
    structure_operation_t *structure_operation;
    creation_operation_t *creation_operation;
} operation_t;

typedef struct function_t
{
    operation_type_t operation_type;
    operation_t *operation;
} function_t;

nw_error_t *function_create(function_t **function, operation_t *operation, operation_type_t operation_type);
void function_destroy(function_t *function, bool_t destroy_operation);

nw_error_t *apply_operation_unary(unary_operation_type_t unary_operation_type, const tensor_t *x, tensor_t **result);
nw_error_t *apply_operation_binary(binary_operation_type_t binary_operation_type, const tensor_t *x, const tensor_t *y, tensor_t **result);
nw_error_t *apply_operation_ternary(ternary_operation_type_t ternary_operation_type, const tensor_t *w, const tensor_t *x, const tensor_t *y, tensor_t **result);
nw_error_t *apply_operation_reduction(reduction_operation_type_t reduction_operation_type, const tensor_t *x, const int64_t *axis, int64_t length, bool_t keep_dimension, tensor_t **result);
nw_error_t *apply_operation_structure(structure_operation_type_t structure_operation_type, const tensor_t *x, const int64_t *arguments, int64_t length, tensor_t **result);
nw_error_t *apply_operation_creation(creation_operation_type_t creation_operation_type, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype,
                                     bool_t requires_gradient, bool_t persist, const void **arguments, int64_t length, void *data, tensor_t **result);
nw_error_t *apply_backward(tensor_t *result);

#endif
