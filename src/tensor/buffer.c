#include <buffer.h>

error_t *buffer_create(buffer_t **buffer, runtime_t runtime, datatype_t datatype, view_t *view, void *data)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(view->shape, "view->shape");
    CHECK_NULL_ARGUMENT(view->strides, "view->strides");

    size_t size = sizeof(buffer_t);
    *buffer = (buffer_t *) malloc(size);
    if (buffer == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate buffer of size %zu bytes.", size),
                     NULL);
    }

    uint32_t number_of_elements = view_size(view);
    uint32_t element_size = datatype_size(datatype);
    size = number_of_elements * element_size;
    error_t *error = nw_malloc(&(*buffer)->data, size, runtime);
    if (error != NULL)
    {
        free(buffer);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate buffer data of size %zu bytes.", size),
                     error);
    }

    (*buffer)->runtime = runtime;
    (*buffer)->datatype = datatype;
    (*buffer)->view = view;
    if (data != NULL)
    {
        memcpy((*buffer)->data, data, size);
    }

    return NULL;
}

void buffer_destroy(buffer_t *buffer)
{
    if (buffer != NULL)
    {
        view_destroy(buffer->view);
        nw_free(buffer->data, buffer->runtime);
        free(buffer);
    }
}