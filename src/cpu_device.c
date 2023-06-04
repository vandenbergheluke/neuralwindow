#include <cpu_operation.h>

error_t *cpu_malloc(void **p, size_t size)
{
    if (p == NULL)
    {
        message_t message = create_message("received null pointer argument for 'p'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    *p = malloc(size);

    if (*p == NULL)
    {
        message_t message = create_message("failed to allocate %zu bytes.", size);
        return create_error(ERROR_MEMORY_ALLOCATION, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    return NULL;
}

error_t *cpu_free(void *p)
{
    if (p == NULL)
    {
        message_t message = create_message("received null pointer argument for 'p'.");
        return create_error(ERROR_NULL_POINTER, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    free(p);

    return NULL;
}

