#include <buffer.h>

error_t *create_buffer(buffer_t *b, size_t size, device_t device)
{
    if (b == NULL)
    {
        message_t message = create_message("received null pointer argument for 'b'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    error_t *error = memory_allocate(b, size, device);

    if (error != NULL)
    {
        message_t message = create_message("failed to create buffer.");
        return create_error(ERROR_CREATE, __FILE__, __LINE__, __FUNCTION__, message, error);
    }

    return NULL;
}

error_t *destory_buffer(buffer_t b, device_t device)
{
    if (b == NULL)
    {
        message_t message = create_message("received null pointer argument for 'b'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    error_t *error = memory_free(b, device);

    if (error != NULL)
    {
        message_t message = create_message("failed to destroy shape.");
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, message, error);
    }

    return NULL;
}