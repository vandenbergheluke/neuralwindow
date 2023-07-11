#include <tensor.h>

error_t *create_tensor(tensor_t **tensor, runtime_t runtime)
{
    CHECK_NULL(tensor, "tensor");

    error_t *error;
    error = nw_malloc((void **) tensor, sizeof(tensor_t), runtime);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create tensor."), error);
    
    error = create_buffer(&(*tensor)->buffer, runtime);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create tensor->buffer."), error);

    // Initialize
    (*tensor)->context = NULL;
    (*tensor)->gradient = NULL;
    (*tensor)->requires_gradient = false;

    return NULL;
}

error_t *destroy_tensor(tensor_t *tensor, runtime_t runtime)
{
    if (tensor == NULL)
        return NULL;

    error_t *error;
    error = destroy_buffer(tensor->buffer, runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy tensor->buffer."), error);

    error = destroy_tensor(tensor->gradient, runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy tensor->gradient."), error);

    error = destroy_operation(tensor->context, runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy tensor->context."), error);

    error = nw_free((void *) tensor, runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy tensor."), error);

    return NULL;
}
