#ifndef STRUCTURE_OPERATION_H
#define STRUCTURE_OPERATION_H

#include <tensor.h>

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

error_t *create_structure_operation(structure_operation_t **operation, structure_operation_type_t operation_type, tensor_t *x, void *arguments);
error_t *destroy_structure_operation(structure_operation_t *operation);

#endif