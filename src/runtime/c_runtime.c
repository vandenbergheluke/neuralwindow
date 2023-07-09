#include <c_runtime.h>

error_t *c_malloc(void **pp, size_t size)
{
    CHECK_NULL_POINTER(pp, "pp");

    *pp = malloc(size);

    if (*pp == NULL)
    {
        string_t message = create_string("failed to allocate %zu bytes.", size);
        return create_error(ERROR_MEMORY_ALLOCATION, __FILE__, __LINE__, __FUNCTION__, message, NULL);
    }

    return NULL;
}

error_t *c_free(void *p)
{
    free(p);
    return NULL;
}

