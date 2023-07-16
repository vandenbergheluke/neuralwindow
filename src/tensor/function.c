#include <function.h>

error_t *create_function(function_t **function, operation_t *operation, operation_type_t operation_type, bool_t requires_gradient)
{
    CHECK_NULL(function, "function");

    error_t *error;
    error = nw_malloc((void **) function, sizeof(function_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create function."), error);

    // Initialize
    (*function)->operation = operation;
    (*function)->operation_type = operation_type;
    (*function)->requires_gradient = requires_gradient;
    
    return NULL;
}

error_t *destroy_function(function_t *function)
{
    if (function == NULL)
        return NULL;
    
    error_t *error;
    error = destroy_operation(function->operation, function->operation_type);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy function."), error);
    
    error = nw_free((void *) function, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy function."), error);

    return NULL;
}

error_t *function_forward(const function_t *function, tensor_t *result)
{
    CHECK_NULL(function, "function");
    CHECK_NULL(result, "result");

    error_t *error;
    error = operation_forward(function->operation, function->operation_type, result);
    if (error != NULL)
        return ERROR(ERROR_FORWARD, create_string("failed to apply function."), error);

    result->context = function;
    
    return NULL;
}

error_t *function_backward(function_t *function, tensor_t *gradient)
{
    CHECK_NULL(function, "function");
    CHECK_NULL(gradient, "gradient");

    error_t *error;
    error = operation_backward(function->operation, function->operation_type, gradient);
    if (error != NULL)
        return ERROR(ERROR_BACKWARD, create_string("failed to back propogate function."), error);

    return NULL;
}