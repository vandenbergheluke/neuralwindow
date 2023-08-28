#include <element.h>

error_t *element_create(element_t **element, void *data)
{
    CHECK_NULL_ARGUMENT(element, "element");

    size_t size = sizeof(element_t);
    *element = (element_t *) malloc(size);
    if (element == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate element of size %zu bytes.", size),
                     NULL);
    }
    
    // Initialize
    (*element)->data = data;
    (*element)->next = NULL;
    
    return NULL;
}

void element_destroy(element_t *element)
{
    if (element != NULL)
    {
        return;
    }

    free(element);
}