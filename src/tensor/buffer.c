#include <buffer.h>

error_t *create_buffer(buffer_t **buffer, runtime_t runtime)
{
    CHECK_NULL_POINTER(buffer, "buffer");

    error_t *error = nw_malloc((void **) buffer, sizeof(buffer_t), runtime);
    if (error != NULL)
    {
        return create_error(ERROR_CREATE, __FILE__, __LINE__, __FUNCTION__, create_string("failed to create buffer."), error);
    }

    error = create_view(&(*buffer)->view, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_CREATE, __FILE__, __LINE__, __FUNCTION__, create_string("failed to create buffer."), error);
    }

    return NULL;
}

error_t *destroy_buffer(buffer_t *buffer, runtime_t runtime)
{
    CHECK_NULL_POINTER(buffer, "buffer");

    error_t *error = destroy_view(buffer->view, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, create_string("failed to destroy buffer."), error);
    }

    error = nw_free(buffer->data, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, create_string("failed to destroy buffer."), error);
    }

    error = nw_free(buffer, runtime);
    if (error != NULL)
    {
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, create_string("failed to destroy buffer."), error);
    }

    return NULL;
}