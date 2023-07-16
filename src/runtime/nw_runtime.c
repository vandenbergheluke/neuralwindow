#include <nw_runtime.h>
#include <c_runtime.h>
#include <cu_runtime.h>
#include <mkl_runtime.h>
#include <openblas_runtime.h>

error_t *nw_malloc(void **pp, size_t size, runtime_t runtime)
{
    CHECK_NULL(pp, "pp");

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
        error = ERROR(ERROR_UNKNOWN_RUNTIME, create_string("unknown runtime argument."), NULL);
        break;
    }

    if (error != NULL)
        return ERROR(ERROR_MEMORY_ALLOCATION, create_string("failed to allocate %zu bytes for runtime %s.", size, runtime_string(runtime)), error);
    
    return NULL;
}

error_t *nw_free(void *p, runtime_t runtime)
{
    if (p == NULL)
        return NULL;

    error_t *error;
    switch (runtime)
    {
    case C_RUNTIME:
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        error = c_free(p);
        break;
    case CU_RUNTIME:
        error = cu_free(p);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME, create_string("unknown runtime argument."), NULL);
        break;
    }
    
    if (error != NULL)
        return ERROR(ERROR_MEMORY_ALLOCATION, create_string("failed to free memory for runtime %s.", runtime_string(runtime)), error);

    return NULL;
}

error_t *nw_copy(const void *in_p, void *out_p, size_t size, runtime_t runtime)
{
    CHECK_NULL(in_p, "in_p");
    CHECK_NULL(out_p, "out_p");

    error_t *error;
    switch (runtime)
    {
    case C_RUNTIME:
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        error = c_copy(in_p, out_p, size);
        break;
    case CU_RUNTIME:
        error = cu_copy(in_p, out_p, size);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME, create_string("unknown runtime argument."), NULL);
        break;
    }
    
    if (error != NULL)
        return ERROR(ERROR_COPY, create_string("failed to copy %ld bytes of memory for runtime %s.", size, runtime_string(runtime)), error);

    return NULL;
}

error_t *nw_addition(buffer_t *in_buffer_x, buffer_t *in_buffer_y, buffer_t *out_buffer)
{
    CHECK_NULL(in_buffer_x, "in_buffer_x");
    CHECK_NULL(in_buffer_y, "in_buffer_y");
    CHECK_NULL(out_buffer, "out_buffer");

    error_t *error;

    if (in_buffer_x->datatype != in_buffer_y->datatype ||
        in_buffer_x->datatype != out_buffer->datatype)
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     create_string("conflicting datatypes %s + %s = %s",
                                   datatype_string(in_buffer_x->datatype),
                                   datatype_string(in_buffer_y->datatype),
                                   datatype_string(out_buffer->datatype)), NULL);

    if (!equal_shape(in_buffer_x->view, in_buffer_y->view) ||
        !equal_shape(in_buffer_x->view, out_buffer->view))
        return ERROR(ERROR_SHAPE_CONFLICT, create_string("conflicting tensor shapes"), NULL);

    if (!is_contiguous(in_buffer_x->view) || !is_contiguous(in_buffer_y->view) || !is_contiguous(out_buffer->view))
        return ERROR(ERROR_CONTIGUOUS, create_string("Not all tensors are contiguous"), NULL);

    switch (runtime)
    {
    case C_RUNTIME:
        error = c_addition(out_buffer->datatype, size(out_buffer->view), in_buffer_x->data, in_buffer_y->data, out_buffer->data);
    case OPENBLAS_RUNTIME:
        error = openblas_addition(out_buffer->datatype, size(out_buffer->view), in_buffer_x->data, in_buffer_y->data, out_buffer->data);
    case MKL_RUNTIME:
        error = mkl_addition(out_buffer->datatype, size(out_buffer->view), in_buffer_x->data, in_buffer_y->data, out_buffer->data);
        break;
    case CU_RUNTIME:
        error = cu_addition(out_buffer->datatype, size(out_buffer->view), in_buffer_x->data, in_buffer_y->data, out_buffer->data);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME, create_string("unknown runtime argument."), NULL);
        break;
    }
    
    if (error != NULL)
        return ERROR(ERROR_ADDITION, create_string("addition operation failed for runtime %s.", runtime_string(runtime)), error);

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
        return NULL;
    }
}