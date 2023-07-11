#include <operation.h>

error_t *create_operation(operation_t **operation, runtime_t runtime)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    error = nw_malloc((void **) operation, sizeof(operation_t), runtime);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create operation."), error);

    // Initialize
    (*operation)->operands = NULL;
    (*operation)->operation_type = NULL_OPERATION;
    
    return NULL;
}

error_t *destroy_operation(operation_t *operation, runtime_t runtime)
{
    if (operation == NULL)
        return NULL;

    error_t *error;
    for (uint32_t i = 0; i < number_of_operands(operation->operation_type); i++)
    {
        error = destroy_tensor(operation->operands[i], runtime);
        if (error != NULL)
            return ERROR(ERROR_DESTROY, create_string("failed to destroy operation operand."), error);
    }

    error = nw_free(operation->operands, runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy operation->operands."), error);

    error = nw_free(operation, runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy operation."), error);

    return NULL;
}

uint32_t number_of_operands(operation_type_t operation_type)
{
    switch (operation_type)
    {
    case OPERATION_ADDITION:
        return 2; 
    default:
        return 0;
    }
}

error_t *initialize_operation(operation_t *operation, operation_type_t operation_type, runtime_t runtime, tensor_t *tensor, ...)
{
    CHECK_NULL(operation, "operation");

    if (operation->operation_type != NULL_OPERATION || operation->operands != NULL)
        return ERROR(ERROR_INITIALIZATION, create_string("failed to initialize operation. operation already initialized."), NULL);
    uint32_t n = number_of_operands(operation_type);

    error_t *error;
    error = nw_malloc((void **) &operation->operands, n * sizeof(tensor_t), runtime);
    if (error != NULL)
        return ERROR(ERROR_INITIALIZATION, create_string("failed to intialize operation."), error);

    va_list arguments;
    va_start(arguments, tensor);
    
    return NULL;
}