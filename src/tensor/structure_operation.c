#include <structure_operation.h>

error_t *create_structure_operation(structure_operation_t **operation, structure_operation_type_t operation_type, tensor_t *x, void *arguments)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    error = nw_malloc((void **) operation, sizeof(structure_operation_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create structure operation."), error);

    // Initialize
    (*operation)->operation_type = operation_type;
    (*operation)->x = x; 
    (*operation)->arguments = arguments;

    return NULL;
}

error_t *destroy_structure_operation(structure_operation_t *operation)
{
    error_t *error;
    error = nw_free((void *) operation, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy structure operation."), error);

    return NULL;
}

error_t *structure_operation_forward(structure_operation_t *operation, tensor_t *result)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    switch (operation->operation_type)
    {
    case EXPAND_OPERATION:
        error = expand_operation_forward(operation->x, operation->arguments, result);
        break;
    case PERMUTE_OPERATION:
        error = permute_operation_forward(operation->x, operation->arguments, result);
        break;
    case RESHAPE_OPERATION:
        error = reshape_operation_forward(operation->x, operation->arguments, result);
        break;
    case PADDING_OPERATION:
        error = padding_operation_forward(operation->x, operation->arguments, result);
        break;
    case SLICE_OPERATION:
        error = slice_operation_forward(operation->x, operation->arguments, result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, create_string("unknown operation type %d", operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
        return ERROR(ERROR_FORWARD, create_string("failed to apply reduction operation."), error);

    return NULL;
}