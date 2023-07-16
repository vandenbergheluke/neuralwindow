#include <tensor.h>

error_t *create_tensor(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient)
{
    CHECK_NULL(tensor, "tensor");

    error_t *error;
    error = nw_malloc((void **) tensor, sizeof(tensor_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create tensor."), error);
    
    // Initialize
    (*tensor)->buffer = buffer;
    (*tensor)->context = context;
    (*tensor)->gradient = gradient;
    (*tensor)->requires_gradient = requires_gradient;

    return NULL;
}

error_t *destroy_tensor(tensor_t *tensor)
{
    if (tensor == NULL)
        return NULL;

    error_t *error;
    error = destroy_buffer(tensor->buffer);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy tensor->buffer."), error);

    error = destroy_tensor(tensor->gradient);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy tensor->gradient."), error);

    error = destroy_function(tensor->context);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy tensor->context."), error);

    error = nw_free((void *) tensor, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy tensor."), error);

    return NULL;
}

error_t *identity(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL(x, "x");
    CHECK_NULL(y, "y");

    y->buffer = x->buffer;
    y->context = x->context;
    y->gradient = x->gradient;
    y->requires_gradient = x->requires_gradient;
}

error_t *ones(const uint32_t *shape, uint32_t rank, tensor_t *x)
{
    return NULL;
}

error_t *addition(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL(x, "x");
    CHECK_NULL(y, "y");
    CHECK_NULL(z, "z");

    error_t *error;
    binary_operation_t *binary_operation;
    operation_t *operation;
    function_t *function;

    error = create_binary_operation(&binary_operation, ADDITION_OPERATION, x, y);
    if (error != NULL)
        return ERROR(ERROR_ADDITION, create_string("failed to add tensors."), error);

    error = create_operation(&operation, BINARY_OPERATION, binary_operation);
    if (error != NULL)
        return ERROR(ERROR_ADDITION, create_string("failed to add tensors."), error);

    error = create_function(&function, operation, BINARY_OPERATION, x->requires_gradient || y->requires_gradient);
    if (error != NULL)
        return ERROR(ERROR_ADDITION, create_string("failed to add tensors."), error);

    error = function_forward(function, z);
    if (error != NULL)
        return ERROR(ERROR_ADDITION, create_string("failed to add tensors."), error);
    
    return NULL;
}

error_t *backward(tensor_t *tensor)
{
    CHECK_NULL(tensor, "tensor");
    CHECK_NULL(tensor->buffer, "tensor->buffer");
    CHECK_NULL(tensor->buffer->view, "tensor->buffer->view");

    error_t *error;

    error = create_tensor(&tensor->gradient, NULL, NULL, NULL, false);
    if (error != NULL)
        return ERROR(ERROR_BACKWARD, create_string("backward propogation failed."), error);

    error = ones(tensor->buffer->view->shape, tensor->buffer->view->rank, tensor->gradient);
    if (error != NULL)
        return ERROR(ERROR_BACKWARD, create_string("backward propogation failed."), error);

    error = function_backward(tensor->context, tensor->gradient);
    if (error != NULL)
        return ERROR(ERROR_BACKWARD, create_string("backward propogation failed."), error);

    return NULL;
}

static error_t *_backward(tensor_t *tensor)
{
    CHECK_NULL(tensor, "tensor");

    if (tensor->context != NULL)
    {
        _backward
    }
    
}