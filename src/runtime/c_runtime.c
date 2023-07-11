#include <c_runtime.h>

error_t *c_malloc(void **pp, size_t size)
{
    CHECK_NULL(pp, "pp");

    *pp = malloc(size);
    if (*pp == NULL)
        return ERROR(ERROR_MEMORY_ALLOCATION, create_string("failed to allocate %zu bytes.", size), NULL);

    return NULL;
}

error_t *c_free(void *p)
{
    free(p);
    return NULL;
}

