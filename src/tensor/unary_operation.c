#include <unary_operation.h>

error_t *create_unary_operation(unary_operation_t **operation, unary_operation_type_t operation_type, tensor_t *x)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    error = nw_malloc((void **) operation, sizeof(unary_operation_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create unary operation."), error);

    // Initialize
    (*operation)->operation_type = operation_type;
    (*operation)->x = x; 

    return NULL;
}

error_t *destroy_unary_operation(unary_operation_t *operation)
{
    error_t *error;
    error = nw_free((void *) operation, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy unary operation."), error);

    return NULL;
}

error_t *unary_operation_forward(unary_operation_t *operation, tensor_t *result)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    switch (operation->operation_type)
    {
    case EXPONENTIAL_OPERATION:
        error = exponential_operation_forward(operation->x, result);
        break;
    case LOGARITHM_OPERATION:
        error = logarithm_operation_forward(operation->x, result);
        break;
    case SIN_OPERATION:
        error = sin_operation_forward(operation->x, result);
        break;
    case SQUARE_ROOT_OPERATION:
        error = square_root_operation_forward(operation->x, result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, create_string("unknown operation type %d", operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
        return ERROR(ERROR_FORWARD, create_string("failed to apply unary operation."), error);
    
    return NULL;
}