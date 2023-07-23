#include <buffer.h>
#include <c_runtime.h>
#include <cu_runtime.h>
#include <mkl_runtime.h>
#include <openblas_runtime.h>

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

    (*buffer)->runtime = runtime;
    (*buffer)->datatype = datatype;
    (*buffer)->view = view;

    error_t *error = runtime_malloc(*buffer);
    if (error != NULL)
    {
        free(buffer);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate buffer data of size %zu bytes.", size),
                     error);
    }

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
        runtime_free(buffer);
        view_destroy(buffer->view);
        free(buffer);
    }
}

error_t *runtime_create_context(runtime_t runtime)
{
    error_t *error;
    switch (runtime)
    {
    case C_RUNTIME:
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        error = NULL;
        break;
    case CU_RUNTIME:
        error = cu_create_context();
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.", runtime),
                      NULL);
        break;
    }
    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create context."),
                     error);
    }

    return NULL;
}

void runtime_destroy_context(runtime_t runtime)
{
    switch (runtime)
    {
    case C_RUNTIME:
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        break;
    case CU_RUNTIME:
        cu_destroy_context();
        break;
    default:
        break;
    }
}

error_t *runtime_malloc(buffer_t *buffer)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");

    uint32_t number_of_elements = view_size(buffer->view);
    uint32_t element_size = datatype_size(buffer->datatype);
    size_t size = number_of_elements * element_size;

    error_t *error;
    switch (buffer->runtime)
    {
    case C_RUNTIME:
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        error = c_malloc(&buffer->data, size);
        break;
    case CU_RUNTIME:
        error = cu_malloc(&buffer->data, size);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.", buffer->runtime),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate %zu bytes for runtime %s.",
                                    size, runtime_string(buffer->runtime)),
                     error);
    }
    
    return NULL;
}

void runtime_free(buffer_t *buffer)
{
    if (buffer != NULL)
    {
        switch (buffer->runtime)
        {
        case C_RUNTIME:
        case OPENBLAS_RUNTIME:
        case MKL_RUNTIME:
            c_free(buffer->data);
            break;
        case CU_RUNTIME:
            cu_free(buffer->data);
            break;
        default:
            break;
        }
    }
}

error_t *runtime_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(z_buffer, "z_buffer");

    error_t *error;

    if (x_buffer->datatype != y_buffer->datatype ||
        x_buffer->datatype != z_buffer->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("conflicting datatypes %s + %s = %s.",
                                   datatype_string(x_buffer->datatype),
                                   datatype_string(y_buffer->datatype),
                                   datatype_string(z_buffer->datatype)),
                     NULL);
    }

    if (!view_shape_equal(x_buffer->view, y_buffer->view) ||
        !view_shape_equal(x_buffer->view, z_buffer->view))
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting tensor shapes."),
                     NULL);
    }

    if (!view_is_contiguous(x_buffer->view) ||
        !view_is_contiguous(y_buffer->view) ||
        !view_is_contiguous(z_buffer->view))
    {
        return ERROR(ERROR_CONTIGUOUS,
                     string_create("not all tensors are contiguous."),
                     NULL);
    }

    if (x_buffer->runtime != y_buffer->runtime ||
        x_buffer->runtime != z_buffer->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT,
                     string_create("conflicting runtimes %s + %s = %s.",
                                   runtime_string(x_buffer->runtime),
                                   runtime_string(y_buffer->runtime),
                                   runtime_string(z_buffer->runtime)),
                     NULL);
    }

    switch (z_buffer->runtime)
    {
    case C_RUNTIME:
        error = c_addition(z_buffer->datatype, view_size(z_buffer->view), x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    case OPENBLAS_RUNTIME:
        error = openblas_addition(z_buffer->datatype, view_size(z_buffer->view), x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    case MKL_RUNTIME:
        error = mkl_addition(z_buffer->datatype, view_size(z_buffer->view), x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    case CU_RUNTIME:
        error = cu_addition(z_buffer->datatype, view_size(z_buffer->view), x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.", z_buffer->runtime),
                      NULL);
        break;
    }
    
    if (error != NULL)
    {
        return ERROR(ERROR_ADDITION,
                     string_create("addition operation failed for runtime %s.", 
                                   runtime_string(z_buffer->runtime)),
                     error);
    }

    return NULL;
}

string_t runtime_string(runtime_t runtime)
{
    switch (runtime)
    {
    case C_RUNTIME:
        return "C_RUNTIME";
    case OPENBLAS_RUNTIME:
        return "OPENBLAS_RUNTIME"; 
    case MKL_RUNTIME:
        return "MKL_RUNTIME";
    case CU_RUNTIME:
        return "CU_RUNTIME";
    default:
        return "UNKNOWN_RUNTIME";
    }
}