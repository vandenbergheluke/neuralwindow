#include <buffer.h>

error_t *create_buffer(buffer_t **buffer, runtime_t runtime)
{
    CHECK_NULL(buffer, "buffer");

    error_t *error;
    error = nw_malloc((void **) buffer, sizeof(buffer_t), runtime);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create buffer."), error);

    error = create_view(&(*buffer)->view, runtime);
    if (error != NULL)
        return ERROR(ERROR_CREATE, create_string("failed to create buffer->view."), error);

    // Initialize
    (*buffer)->data = NULL;
    (*buffer)->datatype = NONE;
    (*buffer)->runtime = runtime;

    return NULL;
}

error_t *destroy_buffer(buffer_t *buffer, runtime_t runtime)
{
    if (buffer == NULL)
        return NULL;

    error_t *error;
    error = destroy_view(buffer->view, runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy buffer->view."), error);

    error = nw_free(buffer->data, runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy buffer->data."), error);

    error = nw_free(buffer, runtime);
    if (error != NULL)
        return ERROR(ERROR_DESTROY, create_string("failed to destroy buffer."), error);

    return NULL;
}