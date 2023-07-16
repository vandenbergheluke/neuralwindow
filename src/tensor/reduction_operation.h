#ifndef REDUCTION_OPERATION_H
#define REDUCTION_OPERATION_H

#include <tensor.h>

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

error_t *create_reduction_operation(reduction_operation_t **operation,
                                    reduction_operation_type_t operation_type,
                                    tensor_t *x, uint32_t *axis, bool_t keep_dimension);
error_t *destroy_reduction_operation(reduction_operation_t *operation);
error_t *reduction_operation_forward(reduction_operation_t *operation, tensor_t *result);

#endif