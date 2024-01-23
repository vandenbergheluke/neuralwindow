/**@file element.c
 * @brief Implements collection element utilities.
 *
 */

#include <element.h>
#include <datatype.h>

nw_error_t *element_create(element_t **element, void *data)
{
    CHECK_NULL_ARGUMENT(element, "element");

    *element = (element_t *) malloc(sizeof(element_t));
    if (!element)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(element)), NULL);
    }
    
    // Initialize
    (*element)->data = data;
    (*element)->next = NULL;
    
    return NULL;
}

void element_destroy(element_t *element)
{
    if (element)
    {
        free(element);
    }
}
