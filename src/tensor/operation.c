#include <operation.h>

error_t *create_operation(operation_t **operation, operation_type_t operation_type, void *type_operation)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    error = nw_malloc((void **) operation, sizeof(operation_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create operation."), error);

    // Initialize
    switch (operation_type)
    {
    case UNARY_OPERATION:
        (*operation)->unary_operation = (unary_operation_t *) type_operation;
        break;
    case BINARY_OPERATION:
        (*operation)->binary_operation = (binary_operation_t *) type_operation;
        break;
    case REDUCTION_OPERATION:
        (*operation)->reduction_operation = (reduction_operation_t *) type_operation;
        break;
    case STRUCTURE_OPERATION:
        (*operation)->structure_operation = (structure_operation_t *) type_operation;
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, create_string("unknown operation type %d", operation_type), NULL);
        break;
    }

    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create operation."), error);
    
    return NULL;
}

error_t *destroy_operation(operation_t *operation, operation_type_t operation_type)
{
    if (operation == NULL)
        return NULL;

    error_t *error;
    switch (operation_type)
    {
    case UNARY_OPERATION:
        error = destroy_unary_operation(operation->unary_operation);
        break;
    case BINARY_OPERATION:
        error = destroy_binary_operation(operation->binary_operation);
        break;
    case REDUCTION_OPERATION:
        error = destroy_reduction_operation(operation->reduction_operation);
        break;
    case STRUCTURE_OPERATION:
        error = destroy_structure_operation(operation->structure_operation);
        break;
    default:
        break;
    }
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy operation."), error);
    
    error = nw_free((void *) operation, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy operation."), error);

    return NULL;
}

error_t *operation_forward(operation_t *operation, operation_type_t operation_type, tensor_t *result)
{
    CHECK_NULL(operation, "operation");
    CHECK_NULL(result, "result");

    error_t *error;
    switch (operation_type)
    {
    case UNARY_OPERATION:
        error = unary_operation_forward(operation->unary_operation, result);
    case BINARY_OPERATION:
        error = binary_operation_forward(operation->binary_operation, result);
        break;
    case REDUCTION_OPERATION:
        error = reduction_operation_forward(operation->reduction_operation, result);
        break;
    case STRUCTURE_OPERATION:
        error = structure_operation_forward(operation->structure_operation, result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, create_string("unknown operation type %d", operation_type), NULL);
        break;
    }

    if (error != NULL)
        return ERROR(ERROR_FORWARD, create_string("failed to apply operation."), error);

    return NULL;
}

error_t *operation_forward(operation_t *operation, operation_type_t operation_type, tensor_t *gradient)
{
    CHECK_NULL(operation, "operation");
    CHECK_NULL(operation, "operation");
}