#include <buffer.h>

error_t *create_buffer(buffer_t **buffer, runtime_t runtime, datatype_t datatype, view_t *view, void *data)
{
    CHECK_NULL(buffer, "buffer");

    error_t *error;
    error = nw_malloc((void **) buffer, sizeof(buffer_t), C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create buffer."), error);

    // Initialize
    (*buffer)->runtime = runtime;
    (*buffer)->datatype = datatype;
    (*buffer)->view = view;
    (*buffer)->data = data;

    return NULL;
}

error_t *destroy_buffer(buffer_t *buffer)
{
    if (buffer == NULL)
        return NULL;

    error_t *error;
    error = destroy_view(buffer->view);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy buffer->view."), error);

    error = nw_free(buffer->data, buffer->runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy buffer->data."), error);

    error = nw_free(buffer, C_RUNTIME);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy buffer."), error);

    return NULL;
}