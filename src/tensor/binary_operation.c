#include <binary_operation.h>

error_t *create_binary_operation(binary_operation_t **operation, binary_operation_type_t operation_type, tensor_t *x, tensor_t *y)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    error = nw_malloc((void **) operation, sizeof(binary_operation_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create binary operation."), error);

    // Initialize
    (*operation)->operation_type = operation_type;
    (*operation)->x = x; 
    (*operation)->y = y;

    return NULL;
}

error_t *destroy_binary_operation(binary_operation_t *operation)
{
    error_t *error;
    error = nw_free((void *) operation, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy binary operation."), error);

    return NULL;
}

error_t *binary_operation_forward(binary_operation_t *operation, tensor_t *result)
{
    CHECK_NULL(operation, "operation");

    error_t *error;
    switch (operation->operation_type)
    {
    case ADDITION_OPERATION:
        error = addition_operation_forward(operation->x, operation->y, result);
        break;
    case SUBTRACTION_OPERATION:
        error = subtraction_operation_forward(operation->x, operation->y, result);
        break;
    case MULTIPLICATION_OPERATION:
        error = multiplication_operation_forward(operation->x, operation->y, result);
        break;
    case DIVISION_OPERATION:
        error = division_operation_forward(operation->x, operation->y, result);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        error = matrix_multiplication_operation_forward(operation->x, operation->y, result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, create_string("unknown operation type %d", operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
        return ERROR(ERROR_FORWARD, create_string("failed to apply binary operation."), error);
    
    return NULL;
}

error_t *addition_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL(x, "x");
    CHECK_NULL(y, "y");
    CHECK_NULL(z, "z");

    error_t *error = nw_addition(x->buffer, y->buffer, z->buffer);
    if (error != NULL)
        return ERROR(ERROR_FORWARD, create_string("failed to apply addition operation."), error);

    return NULL;
}

error_t *addition_operation_backward(const tensor_t *x, const tensor_t *y, const tensor_t *gradient, tensor_t *gradient_x, tensor_t *gradient_y)
{
    CHECK_NULL(x, "x");
    CHECK_NULL(y, "y");
    CHECK_NULL(gradient, "gradient");
    CHECK_NULL(gradient_x, "gradient_x");
    CHECK_NULL(gradient_y, "gradient_y");

    error_t *error;
    if (x->requires_gradient)
       error = identity(gradient, gradient_x);

    if (error != NULL)
        return ERROR(ERROR_BACKWARD, create_string("failed to apply addition operation."), error);

    if (y->requires_gradient)
        error = identity(gradient, gradient_y);

    if (error != NULL)
        return ERROR(ERROR_BACKWARD, create_string("failed to apply addition operation."), error);

    return NULL;
}