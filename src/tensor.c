#include <tensor.h>

error_t *create_tensor(tensor_t **t, device_t device)
{
    if (t == NULL)
    {
        message_t message = create_message("received null pointer argument for 't'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    error_t *error = memory_allocate((void **) t, sizeof(tensor_t), device);

    if (error != NULL)
    {
        message_t message = create_message("failed to create tensor.");
        return create_error(ERROR_CREATE, __FILE__, __LINE__, __FUNCTION__, message, error);
    }

    return NULL;
}

error_t *destroy_tensor(tensor_t *t, device_t device)
{
    if (t == NULL)
    {
        message_t message = create_message("received null pointer argument for 't'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    error_t *error = memory_free((void *) t, device);

    if (error != NULL)
    {
        message_t message = create_message("failed to destory tensor.");
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, message, error);
    }

    return NULL;
}