#include <tensor.h>

error_t *create_tensor(tensor_t **tensor, runtime_t runtime)
{
    CHECK_NULL_POINTER(tensor, "tensor");

    error_t *error = nw_malloc((void **) tensor, sizeof(tensor_t), runtime);
    if (error != NULL)
    {
        return create_error(ERROR_CREATE, __FILE__, __LINE__, __FUNCTION__, create_string("failed to create tensor."), error);
    }
    
    error = create_buffer(&(*tensor)->buffer, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_CREATE, __FILE__, __LINE__, __FUNCTION__, create_string("failed to create tensor."), error);
    }
    return NULL;
}

error_t *destroy_tensor(tensor_t *tensor, runtime_t runtime)
{
    if (tensor == NULL)
    {
        return NULL;
    }

    error_t *error = destroy_buffer(tensor->buffer, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, create_string("failed to destroy tensor."), error);
    }

    error = destroy_tensor(tensor->gradient, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, create_string("failed to destroy tensor."), error);
    }

    error = nw_free(tensor, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, create_string("failed to destroy tensor."), error);
    }

    return NULL;
}
