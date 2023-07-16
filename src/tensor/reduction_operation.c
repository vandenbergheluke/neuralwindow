#include <reduction_operation.h>

error_t *create_reduction_operation(reduction_operation_t **operation, reduction_operation_type_t operation_type, tensor_t *x, uint32_t *axis, bool_t keep_dimension)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    error = nw_malloc((void **) operation, sizeof(reduction_operation_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create reduction operation."), error);

    // Initialize
    (*operation)->operation_type = operation_type;
    (*operation)->x = x; 
    (*operation)->axis = axis;
    (*operation)->keep_dimension = keep_dimension;

    return NULL;
}

error_t *destroy_reduction_operation(reduction_operation_t *operation)
{
    error_t *error;
    error = nw_free((void *) operation, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy reduction operation."), error);

    return NULL;
}

error_t *reduction_operation_forward(reduction_operation_t *operation, tensor_t *result)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    switch (operation->operation_type)
    {
    case SUMMATION_OPERATION:
        error = summation_operation_forward(operation->x, operation->axis, result, operation->keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        error = maximum_operation_forward(operation->x, operation->axis, result, operation->keep_dimension);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, create_string("unknown operation type %d", operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
        return ERROR(ERROR_FORWARD, create_string("failed to apply reduction operation."), error);

    return NULL;
}