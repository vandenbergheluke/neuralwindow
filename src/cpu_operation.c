#include <cpu_operation.h>

error_t cpu_malloc(buffer_t *buffer_ptr, size_t size)
{
    if (buffer_ptr == NULL)
    {
        return STATUS_NULL_POINTER;
    }
    *buffer_ptr = (buffer_t) malloc(size);
    if (*buffer_ptr == NULL)
    {
        return STATUS_MEMORY_ALLOCATION_FAILURE;
    }
    return STATUS_SUCCESS;
}

error_t cpu_free(buffer_t buffer)
{
    if (buffer == NULL)
    {
        return STATUS_NULL_POINTER;
    }
    free((void *) buffer);
    return STATUS_SUCCESS;
}