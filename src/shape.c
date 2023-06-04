#include <shape.h>

error_t *create_shape(shape_t **s, device_t device)
{
    if (s == NULL)
    {
        message_t message = create_message("received null pointer argument for 's'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    error_t *error = memory_allocate((void **) s, sizeof(shape_t), device);

    if (error != NULL)
    {
        message_t message = create_message("failed to create shape.");
        return create_error(ERROR_CREATE, __FILE__, __LINE__, __FUNCTION__, message, error);
    }

    return NULL;
}

error_t *destroy_shape(shape_t *s, device_t device)
{
    if (s == NULL)
    {
        message_t message = create_message("received null pointer argument for 's'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    error_t *error = memory_free((void *) s, device);

    if (error != NULL)
    {
        message_t message = create_message("failed to destroy shape.");
        return create_error(ERROR_DESTROY, __FILE__, __LINE__, __FUNCTION__, message, error);
    }

    return NULL;
}

bool shape_equal(shape_t x_shape, shape_t y_shape)
{
    if (x_shape.rank != y_shape.rank)
    {
        return false;
    }

    if (x_shape.dimensions == NULL || y_shape.dimensions == NULL)
    {
        return false;
    }

    for (uint32_t d = 0; d < x_shape.rank; d++)
    {
        if (x_shape.dimensions[d] != y_shape.dimensions[d])
        {
            return false;
        }
    }
    return true;
}
