#include <nw_runtime.h>
#include <c_runtime.h>
#include <cu_runtime.h>
#include <mkl_runtime.h>
#include <openblas_runtime.h>

error_t *nw_malloc(void **pp, size_t size, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(pp, "pp");

    error_t *error;
    switch (runtime)
    {
    case C_RUNTIME:
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        error = c_malloc(pp, size);
        break;
    case CU_RUNTIME:
        error = cu_malloc(pp, size);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.", runtime),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate %zu bytes for runtime %s.", size, runtime_string(runtime)),
                     error);
    }
    
    return NULL;
}

void nw_free(void *p, runtime_t runtime)
{
    if (p != NULL)
    {
        switch (runtime)
        {
        case C_RUNTIME:
        case OPENBLAS_RUNTIME:
        case MKL_RUNTIME:
            c_free(p);
            break;
        case CU_RUNTIME:
            cu_free(p);
            break;
        default:
            break;
        }
    }
}

error_t *nw_copy(const void *src, void *dst, size_t size, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(src, "src");
    CHECK_NULL_ARGUMENT(dst, "dst");

    error_t *error;
    switch (runtime)
    {
    case C_RUNTIME:
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        error = c_copy(src, dst, size);
        break;
    case CU_RUNTIME:
        error = cu_copy(src, dst, size);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.", runtime),
                      NULL);
        break;
    }
    
    if (error != NULL)
    {
        return ERROR(ERROR_COPY,
                     string_create("failed to copy %ld bytes of memory for runtime %s.", size, runtime_string(runtime)),
                     error);
    }

    return NULL;
}

error_t *nw_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
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
                     string_create("addition operation failed for runtime %s.", runtime_string(z_buffer->runtime)),
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