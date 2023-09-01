/**@file buffer.c
 * @brief
 *
 */

#include <buffer.h>
#include <view.h>
#include <cu_runtime.h>
#include <mkl_runtime.h>
#include <openblas_runtime.h>
#include <string.h>

nw_error_t *buffer_create(buffer_t **buffer,
                       runtime_t runtime,
                       datatype_t datatype,
                       view_t *view,
                       void *data,
                       uint64_t n,
                       bool_t copy)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(view, "view");

    *buffer = (buffer_t *) malloc(sizeof(buffer_t));
    if (buffer == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate buffer of size %lu bytes.", 
                     (unsigned long) sizeof(buffer_t)), NULL);
    }

    (*buffer)->runtime = runtime;
    (*buffer)->datatype = datatype;
    (*buffer)->view = view;
    (*buffer)->copy = copy;
    (*buffer)->n = n;
    (*buffer)->size = (*buffer)->n * datatype_size((*buffer)->datatype);

    if (copy)
    {
        nw_error_t *error = runtime_malloc(*buffer);
        if (error != NULL)
        {
            free(buffer);
            return ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocate buffer data for runtime %s.",
                         runtime_string(runtime)), error);
        }

        if (data != NULL)
        {
            memcpy((*buffer)->data, data, (*buffer)->size);
        }
    }
    else
    {
        (*buffer)->data = data;
    }

    return NULL;
}

void buffer_destroy(buffer_t *buffer)
{
    if (buffer == NULL)
    {
        return;
    }

    if (buffer->copy)
    {
        runtime_free(buffer);
    }
    view_destroy(buffer->view);
    free(buffer);
}

nw_error_t *runtime_create_context(runtime_t runtime)
{
    nw_error_t *error;
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        error = NULL;
        break;
    case CU_RUNTIME:
        error = cu_create_context();
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.",
                      (int) runtime), NULL);
        break;
    }
    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create context for runtime %s.",
                     runtime_string(runtime)), error);
    }

    return NULL;
}

void runtime_destroy_context(runtime_t runtime)
{
    switch (runtime)
    {
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

nw_error_t *runtime_malloc(buffer_t *buffer)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(buffer->view, "buffer->view");

    if (buffer->size == 0)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("cannot allocate 0 bytes."),
                     NULL);
    }

    nw_error_t *error;
    switch (buffer->runtime)
    {
    case OPENBLAS_RUNTIME:
        error = openblas_memory_allocate(&buffer->data, buffer->size);
        break;
    case MKL_RUNTIME:
        error = mkl_memory_allocate(&buffer->data, buffer->size);
        break;
    case CU_RUNTIME:
        error = cu_memory_allocate(&buffer->data, buffer->size);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.",
                      (int) buffer->runtime), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate %lu bytes for runtime %s.", 
                     (unsigned long) buffer->size, runtime_string(buffer->runtime)),
                     error);
    }
    
    return NULL;
}

void runtime_free(buffer_t *buffer)
{
    if (buffer == NULL)
    {
        return;
    }

    switch (buffer->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_memory_free(buffer->data);
        break;
    case MKL_RUNTIME:
        mkl_memory_free(buffer->data);
        break;
    case CU_RUNTIME:
        cu_memory_free(buffer->data);
        break;
    default:
        break;
    }
}

nw_error_t *runtime_exponential(buffer_t *x, buffer_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->view, "x->view");
    CHECK_NULL_ARGUMENT(x->data, "x->data");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(result->view, "result->view");
    CHECK_NULL_ARGUMENT(result->data, "result->data");

    if (x->datatype != result->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("conflicting datatypes x (%s) and result (%s).",
                     datatype_string(x->datatype), datatype_string(result->datatype)),
                     NULL);
    }

    if (x->runtime != result->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT,
                     string_create("conflicting runtimes x (%s) and result (%s).",
                     runtime_string(x->runtime), runtime_string(result->runtime)),
                     NULL);
    }

    if (x->n != result->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting number of elements in x buffer (%lu) and result buffer (%lu).",
                     (unsigned long) x->n, (unsigned long) result->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, result->view->rank))
    {
        string_t x_shape_string = uint64_array_to_string(x->view->shape, x->view->rank);
        string_t result_shape_string = uint64_array_to_string(result->view->shape, result->view->rank);
        nw_error_t *error = ERROR(ERROR_SHAPE_CONFLICT,
                               string_create("conflicting shapes x %s and result %s.",
                               x_shape_string, result_shape_string), NULL);
        string_destroy(x_shape_string);
        string_destroy(result_shape_string);
        return error;
    }

    switch (x->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_exponential(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case MKL_RUNTIME:
        mkl_exponential(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case CU_RUNTIME:
        cu_exponential(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    default:
        return ERROR(ERROR_UNKNOWN_RUNTIME,
                     string_create("unknown runtime %d.",
                     (int) x->runtime), NULL);
    }

    return NULL;
}

nw_error_t *runtime_logarithm(buffer_t *x, buffer_t *result)
{
    if (x->datatype != result->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("conflicting datatypes x (%s) and result (%s).",
                     datatype_string(x->datatype), datatype_string(result->datatype)),
                     NULL);
    }

    if (x->runtime != result->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT,
                     string_create("conflicting runtimes x (%s) and result (%s).",
                     runtime_string(x->runtime), runtime_string(result->runtime)),
                     NULL);
    }

    if (x->n != result->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting number of elements in x buffer (%lu) and result buffer (%lu).",
                     (unsigned long) x->n, (unsigned long) result->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, result->view->rank))
    {
        string_t x_shape_string = uint64_array_to_string(x->view->shape, x->view->rank);
        string_t result_shape_string = uint64_array_to_string(result->view->shape, result->view->rank);
        nw_error_t *error = ERROR(ERROR_SHAPE_CONFLICT,
                               string_create("conflicting shapes x %s and result %s.",
                               x_shape_string, result_shape_string), NULL);
        string_destroy(x_shape_string);
        string_destroy(result_shape_string);
        return error;
    }

    switch (x->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_logarithm(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case MKL_RUNTIME:
        mkl_logarithm(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case CU_RUNTIME:
        cu_logarithm(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    default:
        return ERROR(ERROR_UNKNOWN_RUNTIME,
                     string_create("unknown runtime %d.",
                     (int) x->runtime), NULL);
    }

    return NULL;
}

nw_error_t *runtime_sine(buffer_t *x, buffer_t *result)
{
    if (x->datatype != result->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("conflicting datatypes x (%s) and result (%s).",
                     datatype_string(x->datatype), datatype_string(result->datatype)),
                     NULL);
    }

    if (x->runtime != result->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT,
                     string_create("conflicting runtimes x (%s) and result (%s).",
                     runtime_string(x->runtime), runtime_string(result->runtime)),
                     NULL);
    }

    if (x->n != result->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting number of elements in x buffer (%lu) and result buffer (%lu).",
                     (unsigned long) x->n, (unsigned long) result->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, result->view->rank))
    {
        string_t x_shape_string = uint64_array_to_string(x->view->shape, x->view->rank);
        string_t result_shape_string = uint64_array_to_string(result->view->shape, result->view->rank);
        nw_error_t *error = ERROR(ERROR_SHAPE_CONFLICT,
                               string_create("conflicting shapes x %s and result %s.",
                               x_shape_string, result_shape_string), NULL);
        string_destroy(x_shape_string);
        string_destroy(result_shape_string);
        return error;
    }

    switch (x->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_sine(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case MKL_RUNTIME:
        mkl_sine(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case CU_RUNTIME:
        cu_sine(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    default:
        return ERROR(ERROR_UNKNOWN_RUNTIME,
                     string_create("unknown runtime %d.",
                     (int) x->runtime), NULL);
    }

    return NULL;
}

nw_error_t *runtime_cosine(buffer_t *x, buffer_t *result)
{
    if (x->datatype != result->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("conflicting datatypes x (%s) and result (%s).",
                     datatype_string(x->datatype), datatype_string(result->datatype)),
                     NULL);
    }

    if (x->runtime != result->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT,
                     string_create("conflicting runtimes x (%s) and result (%s).",
                     runtime_string(x->runtime), runtime_string(result->runtime)),
                     NULL);
    }

    if (x->n != result->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting number of elements in x buffer (%lu) and result buffer (%lu).",
                     (unsigned long) x->n, (unsigned long) result->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, result->view->rank))
    {
        string_t x_shape_string = uint64_array_to_string(x->view->shape, x->view->rank);
        string_t result_shape_string = uint64_array_to_string(result->view->shape, result->view->rank);
        nw_error_t *error = ERROR(ERROR_SHAPE_CONFLICT,
                               string_create("conflicting shapes x %s and result %s.",
                               x_shape_string, result_shape_string), NULL);
        string_destroy(x_shape_string);
        string_destroy(result_shape_string);
        return error;
    }

    switch (x->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_cosine(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case MKL_RUNTIME:
        mkl_cosine(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case CU_RUNTIME:
        cu_cosine(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    default:
        return ERROR(ERROR_UNKNOWN_RUNTIME,
                     string_create("unknown runtime %d.",
                     (int) x->runtime), NULL);
    }

    return NULL;
}

nw_error_t *runtime_square_root(buffer_t *x, buffer_t *result)
{
    if (x->datatype != result->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("conflicting datatypes x (%s) and result (%s).",
                     datatype_string(x->datatype), datatype_string(result->datatype)),
                     NULL);
    }

    if (x->runtime != result->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT,
                     string_create("conflicting runtimes x (%s) and result (%s).",
                     runtime_string(x->runtime), runtime_string(result->runtime)),
                     NULL);
    }

    if (x->n != result->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting number of elements in x buffer (%lu) and result buffer (%lu).",
                     (unsigned long) x->n, (unsigned long) result->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, result->view->rank))
    {
        string_t x_shape_string = uint64_array_to_string(x->view->shape, x->view->rank);
        string_t result_shape_string = uint64_array_to_string(result->view->shape, result->view->rank);
        nw_error_t *error = ERROR(ERROR_SHAPE_CONFLICT,
                               string_create("conflicting shapes x %s and result %s.",
                               x_shape_string, result_shape_string), NULL);
        string_destroy(x_shape_string);
        string_destroy(result_shape_string);
        return error;
    }

    switch (x->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_square_root(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case MKL_RUNTIME:
        mkl_square_root(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case CU_RUNTIME:
        cu_square_root(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    default:
        return ERROR(ERROR_UNKNOWN_RUNTIME,
                     string_create("unknown runtime %d.",
                     (int) x->runtime), NULL);
    }

    return NULL;
}

nw_error_t *runtime_reciprocal(buffer_t *x, buffer_t *result)
{
    if (x->datatype != result->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("conflicting datatypes x (%s) and result (%s).",
                     datatype_string(x->datatype), datatype_string(result->datatype)),
                     NULL);
    }

    if (x->runtime != result->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT,
                     string_create("conflicting runtimes x (%s) and result (%s).",
                     runtime_string(x->runtime), runtime_string(result->runtime)),
                     NULL);
    }

    if (x->n != result->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting number of elements in x buffer (%lu) and result buffer (%lu).",
                     (unsigned long) x->n, (unsigned long) result->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, result->view->rank))
    {
        string_t x_shape_string = uint64_array_to_string(x->view->shape, x->view->rank);
        string_t result_shape_string = uint64_array_to_string(result->view->shape, result->view->rank);
        nw_error_t *error = ERROR(ERROR_SHAPE_CONFLICT,
                               string_create("conflicting shapes x %s and result %s.",
                               x_shape_string, result_shape_string), NULL);
        string_destroy(x_shape_string);
        string_destroy(result_shape_string);
        return error;
    }

    switch (x->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_reciprocal(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case MKL_RUNTIME:
        mkl_reciprocal(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case CU_RUNTIME:
        cu_reciprocal(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    default:
        return ERROR(ERROR_UNKNOWN_RUNTIME,
                     string_create("unknown runtime %d.",
                     (int) x->runtime), NULL);
    }

    return NULL;
}

nw_error_t *runtime_copy(buffer_t *x, buffer_t *result)
{
    if (x->datatype != result->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("conflicting datatypes x (%s) and result (%s).",
                     datatype_string(x->datatype), datatype_string(result->datatype)),
                     NULL);
    }

    if (x->runtime != result->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT,
                     string_create("conflicting runtimes x (%s) and result (%s).",
                     runtime_string(x->runtime), runtime_string(result->runtime)),
                     NULL);
    }

    if (x->n != result->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting number of elements in x buffer (%lu) and result buffer (%lu).",
                     (unsigned long) x->n, (unsigned long) result->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, result->view->rank))
    {
        string_t x_shape_string = uint64_array_to_string(x->view->shape, x->view->rank);
        string_t result_shape_string = uint64_array_to_string(result->view->shape, result->view->rank);
        nw_error_t *error = ERROR(ERROR_SHAPE_CONFLICT,
                               string_create("conflicting shapes x %s and result %s.",
                               x_shape_string, result_shape_string), NULL);
        string_destroy(x_shape_string);
        string_destroy(result_shape_string);
        return error;
    }

    switch (x->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_copy(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case MKL_RUNTIME:
        mkl_copy(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    case CU_RUNTIME:
        cu_copy(x->datatype, x->n, x->data, (uint64_t) 1, x->view->offset, result->data, (uint64_t) 1, result->view->offset);
        break;
    default:
        return ERROR(ERROR_UNKNOWN_RUNTIME,
                     string_create("unknown runtime %d.",
                     (int) x->runtime), NULL);
    }

    return NULL;
}

nw_error_t *runtime_contiguous(buffer_t *x, buffer_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->view, "x->view");
    CHECK_NULL_ARGUMENT(x->view->shape, "x->view->shape");
    CHECK_NULL_ARGUMENT(x->view->strides, "x->view->strides");
    CHECK_NULL_ARGUMENT(x->data, "x->data");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(result->view, "result->view");
    CHECK_NULL_ARGUMENT(result->view->shape, "result->view->shape");
    CHECK_NULL_ARGUMENT(result->view->strides, "result->view->strides");
    CHECK_NULL_ARGUMENT(result->data, "result->data");

    switch (x->view->rank)
    {
    case 1:
        switch (x->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_copy(x->datatype, x->view->shape[0], x->data, x->view->strides[0], x->view->offset, result->data, result->view->strides[0], result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_copy(x->datatype, x->view->shape[0], x->data, x->view->strides[0], x->view->offset, result->data, result->view->strides[0], result->view->offset);
            break;
        case CU_RUNTIME:
            cu_copy(x->datatype, x->view->shape[0], x->data, x->view->strides[0], x->view->offset, result->data, result->view->strides[0], result->view->offset);
            break;
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->datatype), NULL);
        }
        break;
    case 2:
        for (uint64_t i = 0; i < x->view->shape[0]; ++i)
        {
            switch (x->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_copy(x->datatype, x->view->shape[1], x->data, x->view->strides[1], x->view->offset + i * x->view->strides[0], 
                              result->data, result->view->strides[1], result->view->offset + i * result->view->strides[0]);
                break;
            case MKL_RUNTIME:
                mkl_copy(x->datatype, x->view->shape[1], x->data, x->view->strides[1], x->view->offset + i * x->view->strides[0], 
                         result->data, result->view->strides[1], result->view->offset + i * result->view->strides[0]);
                break;
            case CU_RUNTIME:
                cu_copy(x->datatype, x->view->shape[1], x->data, x->view->strides[1], x->view->offset + i * x->view->strides[0], 
                        result->data, result->view->strides[1], result->view->offset + i * result->view->strides[0]);
                break;
            default:
                return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->datatype), NULL);
            }
        }
        break;
    case 3:
        for (uint64_t i = 0; i < x->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < x->view->shape[1]; ++j)
            {
                switch (x->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_copy(x->datatype, x->view->shape[2], x->data, x->view->strides[2], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1], 
                                  result->data, result->view->strides[2], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1]);
                    break;
                case MKL_RUNTIME:
                    mkl_copy(x->datatype, x->view->shape[2], x->data, x->view->strides[2], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1], 
                             result->data, result->view->strides[2], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1]);
                    break;
                case CU_RUNTIME:
                    cu_copy(x->datatype, x->view->shape[2], x->data, x->view->strides[2], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1], 
                            result->data, result->view->strides[2], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1]);
                    break;
                default:
                    return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->datatype), NULL);
                }
            }
        }
        break;
    case 4:
        for (uint64_t i = 0; i < x->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < x->view->shape[1]; ++j)
            {
                for (uint64_t k = 0; k < x->view->shape[2]; ++k)
                {
                    switch (x->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_copy(x->datatype, x->view->shape[3], x->data, x->view->strides[3], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1] + k * x->view->strides[2], 
                                      result->data, result->view->strides[3], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1] + k * result->view->strides[2]);
                        break;
                    case MKL_RUNTIME:
                        mkl_copy(x->datatype, x->view->shape[3], x->data, x->view->strides[3], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1] + k * x->view->strides[2], 
                                 result->data, result->view->strides[3], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1] + k * result->view->strides[2]);
                        break;
                    case CU_RUNTIME:
                        cu_copy(x->datatype, x->view->shape[3], x->data, x->view->strides[3], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1] + k * x->view->strides[2], 
                                result->data, result->view->strides[3], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1] + k * result->view->strides[2]);
                        break;
                    default:
                        return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->datatype), NULL);
                    }
                }
            }
        }
        break;
    default:
        return ERROR(ERROR_RANK_CONFLICT, string_create("only support tensors rank between 1-4."), NULL);
    }

    return NULL;
}

nw_error_t *runtime_negation(buffer_t *x, buffer_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->view, "x->view");
    CHECK_NULL_ARGUMENT(x->data, "x->data");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(result->view, "result->view");
    CHECK_NULL_ARGUMENT(result->data, "result->data");

    if (x->datatype != result->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT, string_create("conflicting datatypes %s and %s.", datatype_string(x->datatype), datatype_string(result->datatype)), NULL);
    }

    if (x->runtime != result->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT, string_create("conflicting runtimes %s and %s.", runtime_string(x->runtime), runtime_string(result->runtime)), NULL);
    }

    if (x->n != result->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT, string_create("conflicting number of elements in buffer %lu and %lu.", x->n, result->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, x->view->rank))
    {
        return ERROR(ERROR_SHAPE_CONFLICT, string_create("conflicting shapes in buffer."), NULL);
    }

    switch (x->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_negation(x->datatype, x->n, x->data, 1, x->view->offset, result->data, 1, result->view->offset);
        break;
    case MKL_RUNTIME:
        mkl_negation(x->datatype, x->n, x->data, 1, x->view->offset, result->data, 1, result->view->offset);
        break;
    case CU_RUNTIME:
        cu_negation(x->datatype, x->n, x->data, 1, x->view->offset, result->data, 1, result->view->offset);
        break;
    default:
        return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->datatype), NULL);
    }

    return NULL;
}

nw_error_t *runtime_rectified_linear(buffer_t *x, buffer_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->view, "x->view");
    CHECK_NULL_ARGUMENT(x->data, "x->data");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(result->view, "result->view");
    CHECK_NULL_ARGUMENT(result->data, "result->data");

    if (x->datatype != result->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT, string_create("conflicting datatypes %s and %s.", datatype_string(x->datatype), datatype_string(result->datatype)), NULL);
    }

    if (x->runtime != result->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT, string_create("conflicting runtimes %s and %s.", runtime_string(x->runtime), runtime_string(result->runtime)), NULL);
    }

    if (x->n != result->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT, string_create("conflicting number of elements in buffer %lu and %lu.", x->n, result->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, x->view->rank))
    {
        return ERROR(ERROR_SHAPE_CONFLICT, string_create("conflicting shapes in buffer."), NULL);
    }

    switch (x->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_rectified_linear(x->datatype, x->n, x->data, 1, x->view->offset, result->data, 1, result->view->offset);
        break;
    case MKL_RUNTIME:
        mkl_rectified_linear(x->datatype, x->n, x->data, 1, x->view->offset, result->data, 1, result->view->offset);
        break;
    case CU_RUNTIME:
        cu_rectified_linear(x->datatype, x->n, x->data, 1, x->view->offset, result->data, 1, result->view->offset);
        break;
    default:
        return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->datatype), NULL);
    }

    return NULL;
}

typedef enum binary_elementwise_t
{
    RUNTIME_ADDITION,
    RUNTIME_SUBTRACTION,
    RUNTIME_MULTIPLICATION,
    RUNTIME_DIVISION,
    RUNTIME_POWER,
    RUNTIME_COMPARE_EQUAL,
    RUNTIME_COMPARE_GREATER
} binary_elementwise_t;

static nw_error_t *runtime_binary_elementwise(binary_elementwise_t binary_elementwise, buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    switch (z_buffer->view->rank)
    {
    case 1:
        switch (binary_elementwise)
        {
        case RUNTIME_ADDITION:
            switch (z_buffer->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_addition(z_buffer->datatype, z_buffer->view->shape[0],
                                  x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_addition(z_buffer->datatype, z_buffer->view->shape[0],
                             x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                             y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                             z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case CU_RUNTIME:
                cu_addition(z_buffer->datatype, z_buffer->view->shape[0],
                            x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                            y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                            z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            default:
                break;
            }
            break;
        case RUNTIME_SUBTRACTION:
            switch (z_buffer->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_subtraction(z_buffer->datatype, z_buffer->view->shape[0],
                                     x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                     y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                     z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_subtraction(z_buffer->datatype, z_buffer->view->shape[0],
                                x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case CU_RUNTIME:
                cu_subtraction(z_buffer->datatype, z_buffer->view->shape[0],
                               x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                               y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                               z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            default:
                break;
            }
            break;
        case RUNTIME_MULTIPLICATION:
            switch (z_buffer->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_multiplication(z_buffer->datatype, z_buffer->view->shape[0],
                                        x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                        y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                        z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_multiplication(z_buffer->datatype, z_buffer->view->shape[0],
                                   x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                   y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                   z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case CU_RUNTIME:
                cu_multiplication(z_buffer->datatype, z_buffer->view->shape[0],
                                  x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            default:
                break;
            }
            break;
        case RUNTIME_DIVISION:
            switch (z_buffer->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_division(z_buffer->datatype, z_buffer->view->shape[0],
                                  x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_division(z_buffer->datatype, z_buffer->view->shape[0],
                             x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                             y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                             z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case CU_RUNTIME:
                cu_division(z_buffer->datatype, z_buffer->view->shape[0],
                            x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                            y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                            z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            default:
                break;
            }
            break;
        case RUNTIME_POWER:
            switch (z_buffer->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_power(z_buffer->datatype, z_buffer->view->shape[0],
                               x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                               y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                               z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_power(z_buffer->datatype, z_buffer->view->shape[0],
                          x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                          y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                          z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case CU_RUNTIME:
                cu_power(z_buffer->datatype, z_buffer->view->shape[0],
                         x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                         y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                         z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            default:
                break;
            }
            break;
        case RUNTIME_COMPARE_EQUAL:
            switch (z_buffer->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_compare_equal(z_buffer->datatype, z_buffer->view->shape[0],
                                       x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                       y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                       z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_compare_equal(z_buffer->datatype, z_buffer->view->shape[0],
                                  x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case CU_RUNTIME:
                cu_compare_equal(z_buffer->datatype, z_buffer->view->shape[0],
                                 x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                 y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                 z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            default:
                break;
            }
            break;
        case RUNTIME_COMPARE_GREATER:
            switch (z_buffer->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_compare_greater(z_buffer->datatype, z_buffer->view->shape[0],
                                       x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                       y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                       z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_compare_greater(z_buffer->datatype, z_buffer->view->shape[0],
                                  x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case CU_RUNTIME:
                cu_compare_greater(z_buffer->datatype, z_buffer->view->shape[0],
                                 x_buffer->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                 y_buffer->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                 z_buffer->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            default:
                break;
            }
            break;
        default:
            break;
        }
        break;
    case 2:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            switch (binary_elementwise)
            {
            case RUNTIME_ADDITION:
                switch (z_buffer->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_addition(z_buffer->datatype, z_buffer->view->shape[1],
                                      x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_addition(z_buffer->datatype, z_buffer->view->shape[1],
                                 x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                 y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                 z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case CU_RUNTIME:
                    cu_addition(z_buffer->datatype, z_buffer->view->shape[1],
                                x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                default:
                    break;
                }
                break;
            case RUNTIME_SUBTRACTION:
                switch (z_buffer->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_subtraction(z_buffer->datatype, z_buffer->view->shape[1],
                                         x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                         y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                         z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_subtraction(z_buffer->datatype, z_buffer->view->shape[1],
                                    x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                    y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                    z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case CU_RUNTIME:
                    cu_subtraction(z_buffer->datatype, z_buffer->view->shape[1],
                                   x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                   y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                   z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                default:
                    break;
                }
                break;
            case RUNTIME_MULTIPLICATION:
                switch (z_buffer->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_multiplication(z_buffer->datatype, z_buffer->view->shape[1],
                                            x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                            y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                            z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_multiplication(z_buffer->datatype, z_buffer->view->shape[1],
                                       x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                       y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                       z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case CU_RUNTIME:
                    cu_multiplication(z_buffer->datatype, z_buffer->view->shape[1],
                                      x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                default:
                    break;
                }
                break;
            case RUNTIME_DIVISION:
                switch (z_buffer->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_division(z_buffer->datatype, z_buffer->view->shape[1],
                                      x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_division(z_buffer->datatype, z_buffer->view->shape[1],
                                 x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                 y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                 z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case CU_RUNTIME:
                    cu_division(z_buffer->datatype, z_buffer->view->shape[1],
                                x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                default:
                    break;
                }
                break;
            case RUNTIME_POWER:
                switch (z_buffer->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_power(z_buffer->datatype, z_buffer->view->shape[1],
                                   x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                   y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                   z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_power(z_buffer->datatype, z_buffer->view->shape[1],
                              x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                              y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                              z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case CU_RUNTIME:
                    cu_power(z_buffer->datatype, z_buffer->view->shape[1],
                             x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                             y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                             z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                default:
                    break;
                }
                break;
            case RUNTIME_COMPARE_EQUAL:
                switch (z_buffer->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_compare_equal(z_buffer->datatype, z_buffer->view->shape[1],
                                           x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                           y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                           z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_compare_equal(z_buffer->datatype, z_buffer->view->shape[1],
                                      x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case CU_RUNTIME:
                    cu_compare_equal(z_buffer->datatype, z_buffer->view->shape[1],
                                     x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                     y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                     z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                default:
                    break;
                }
                break;
            case RUNTIME_COMPARE_GREATER:
                switch (z_buffer->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_compare_greater(z_buffer->datatype, z_buffer->view->shape[1],
                                           x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                           y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                           z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_compare_greater(z_buffer->datatype, z_buffer->view->shape[1],
                                      x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case CU_RUNTIME:
                    cu_compare_greater(z_buffer->datatype, z_buffer->view->shape[1],
                                     x_buffer->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                     y_buffer->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                     z_buffer->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
            break;
        }
        break;
    case 3:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[1]; ++j)
            {
                switch (binary_elementwise)
                {
                case RUNTIME_ADDITION:
                    switch (z_buffer->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_addition(z_buffer->datatype, z_buffer->view->shape[2],
                                          x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_addition(z_buffer->datatype, z_buffer->view->shape[2],
                                     x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                     y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                     z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case CU_RUNTIME:
                        cu_addition(z_buffer->datatype, z_buffer->view->shape[2],
                                    x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                    y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                    z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    default:
                        break;
                    }
                    break;
                case RUNTIME_SUBTRACTION:
                    switch (z_buffer->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_subtraction(z_buffer->datatype, z_buffer->view->shape[2],
                                             x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                             y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                             z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_subtraction(z_buffer->datatype, z_buffer->view->shape[2],
                                        x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                        y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                        z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case CU_RUNTIME:
                        cu_subtraction(z_buffer->datatype, z_buffer->view->shape[2],
                                       x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                       y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                       z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    default:
                        break;
                    }
                    break;
                case RUNTIME_MULTIPLICATION:
                    switch (z_buffer->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_multiplication(z_buffer->datatype, z_buffer->view->shape[2],
                                                x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                                y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                                z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_multiplication(z_buffer->datatype, z_buffer->view->shape[2],
                                           x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                           y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                           z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case CU_RUNTIME:
                        cu_multiplication(z_buffer->datatype, z_buffer->view->shape[2],
                                          x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    default:
                        break;
                    }
                    break;
                case RUNTIME_DIVISION:
                    switch (z_buffer->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_division(z_buffer->datatype, z_buffer->view->shape[2],
                                          x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_division(z_buffer->datatype, z_buffer->view->shape[2],
                                     x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                     y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                     z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case CU_RUNTIME:
                        cu_division(z_buffer->datatype, z_buffer->view->shape[2],
                                    x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                    y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                    z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    default:
                        break;
                    }
                    break;
                case RUNTIME_POWER:
                    switch (z_buffer->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_power(z_buffer->datatype, z_buffer->view->shape[2],
                                       x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                       y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                       z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_power(z_buffer->datatype, z_buffer->view->shape[2],
                                  x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                  y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                  z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case CU_RUNTIME:
                        cu_power(z_buffer->datatype, z_buffer->view->shape[2],
                                 x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                 y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                 z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    default:
                        break;
                    }
                    break;
                case RUNTIME_COMPARE_EQUAL:
                    switch (z_buffer->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_compare_equal(z_buffer->datatype, z_buffer->view->shape[2],
                                               x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                               y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                               z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_compare_equal(z_buffer->datatype, z_buffer->view->shape[2],
                                          x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case CU_RUNTIME:
                        cu_compare_equal(z_buffer->datatype, z_buffer->view->shape[2],
                                         x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                         y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                         z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    default:
                        break;
                    }
                    break;
                case RUNTIME_COMPARE_GREATER:
                    switch (z_buffer->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_compare_greater(z_buffer->datatype, z_buffer->view->shape[2],
                                               x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                               y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                               z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_compare_greater(z_buffer->datatype, z_buffer->view->shape[2],
                                          x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case CU_RUNTIME:
                        cu_compare_greater(z_buffer->datatype, z_buffer->view->shape[2],
                                         x_buffer->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                         y_buffer->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                         z_buffer->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    default:
                        break;
                    }
                    break;
                default:
                    break;
                }
            }
        }
        break;
    case 4:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[1]; ++j)
            {
                for (uint64_t k = 0; k < z_buffer->view->shape[2]; ++k)
                {
                    switch (binary_elementwise)
                    {
                    case RUNTIME_ADDITION:
                        switch (z_buffer->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_addition(z_buffer->datatype, z_buffer->view->shape[3],
                                              x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_addition(z_buffer->datatype, z_buffer->view->shape[3],
                                         x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                         y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                         z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case CU_RUNTIME:
                            cu_addition(z_buffer->datatype, z_buffer->view->shape[3],
                                        x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                        y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                        z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_SUBTRACTION:
                        switch (z_buffer->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_subtraction(z_buffer->datatype, z_buffer->view->shape[3],
                                                 x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                 y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                 z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_subtraction(z_buffer->datatype, z_buffer->view->shape[3],
                                            x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                            y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                            z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case CU_RUNTIME:
                            cu_subtraction(z_buffer->datatype, z_buffer->view->shape[3],
                                           x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                           y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                           z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_MULTIPLICATION:
                        switch (z_buffer->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_multiplication(z_buffer->datatype, z_buffer->view->shape[3],
                                                    x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                    y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                    z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_multiplication(z_buffer->datatype, z_buffer->view->shape[3],
                                               x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                               y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                               z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case CU_RUNTIME:
                            cu_multiplication(z_buffer->datatype, z_buffer->view->shape[3],
                                              x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_DIVISION:
                        switch (z_buffer->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_division(z_buffer->datatype, z_buffer->view->shape[3],
                                              x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_division(z_buffer->datatype, z_buffer->view->shape[3],
                                         x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                         y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                         z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case CU_RUNTIME:
                            cu_division(z_buffer->datatype, z_buffer->view->shape[3],
                                        x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                        y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                        z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_POWER:
                        switch (z_buffer->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_power(z_buffer->datatype, z_buffer->view->shape[3],
                                           x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                           y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                           z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_power(z_buffer->datatype, z_buffer->view->shape[3],
                                      x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                      y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                      z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case CU_RUNTIME:
                            cu_power(z_buffer->datatype, z_buffer->view->shape[3],
                                     x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                     y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                     z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_COMPARE_EQUAL:
                        switch (z_buffer->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_compare_equal(z_buffer->datatype, z_buffer->view->shape[3],
                                                   x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                   y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                   z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_compare_equal(z_buffer->datatype, z_buffer->view->shape[3],
                                              x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case CU_RUNTIME:
                            cu_compare_equal(z_buffer->datatype, z_buffer->view->shape[3],
                                             x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                             y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                             z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_COMPARE_GREATER:
                        switch (z_buffer->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_compare_greater(z_buffer->datatype, z_buffer->view->shape[3],
                                                   x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                   y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                   z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_compare_greater(z_buffer->datatype, z_buffer->view->shape[3],
                                              x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case CU_RUNTIME:
                            cu_compare_greater(z_buffer->datatype, z_buffer->view->shape[3],
                                             x_buffer->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                             y_buffer->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                             z_buffer->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        default:
                            break;
                        }
                        break;
                    default:
                        break;
                    }
                }
            }
        }
        break;
    default:
        break;
    }

    return NULL;
}

nw_error_t *runtime_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_ADDITION, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, string_create("failed to apply binary elementwise operation."), error);
    }

    return NULL;
}

nw_error_t *runtime_subtraction(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_SUBTRACTION, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, string_create("failed to apply binary elementwise operation."), error);
    }

    return NULL;
}

nw_error_t *runtime_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_MULTIPLICATION, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, string_create("failed to apply binary elementwise operation."), error);
    }

    return NULL;
}

nw_error_t *runtime_division(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_DIVISION, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, string_create("failed to apply binary elementwise operation."), error);
    }

    return NULL;
}

nw_error_t *runtime_power(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_POWER, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, string_create("failed to apply binary elementwise operation."), error);
    }

    return NULL;
}

nw_error_t *runtime_compare_equal(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_COMPARE_EQUAL, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, string_create("failed to apply binary elementwise operation."), error);
    }

    return NULL;
}

nw_error_t *runtime_compare_greater(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_COMPARE_GREATER, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, string_create("failed to apply binary elementwise operation."), error);
    }

    return NULL;
}

nw_error_t *runtime_matrix_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{

    if (x_buffer->datatype != y_buffer->datatype || x_buffer->datatype != z_buffer->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT, string_create("conflicting datatypes %s + %s = %s.",
                     datatype_string(x_buffer->datatype), datatype_string(y_buffer->datatype), datatype_string(z_buffer->datatype)), NULL);
    }

    if (x_buffer->view->rank < 2 || y_buffer->view->rank < 2 || z_buffer->view->rank < 2 || 
        x_buffer->view->rank != z_buffer->view->rank || y_buffer->view->rank != z_buffer->view->rank)
    {
        return ERROR(ERROR_RANK_CONFLICT, string_create("ranks %lu + %lu = %lu.", x_buffer->view->rank, y_buffer->view->rank, z_buffer->view->rank), NULL);
    }

    if (!(x_buffer->view->shape[x_buffer->view->rank - 1] == y_buffer->view->shape[y_buffer->view->rank - 2] &&
          z_buffer->view->shape[z_buffer->view->rank - 2] == x_buffer->view->shape[x_buffer->view->rank - 2] &&
          z_buffer->view->shape[z_buffer->view->rank - 1] == y_buffer->view->shape[y_buffer->view->rank - 1]))
    {
        return ERROR(ERROR_SHAPE_CONFLICT, string_create("conflicting tensor shapes."), NULL);
    }

    if (x_buffer->runtime != y_buffer->runtime || x_buffer->runtime != z_buffer->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT, string_create("conflicting runtimes %s + %s = %s.", 
                     runtime_string(x_buffer->runtime), runtime_string(y_buffer->runtime), runtime_string(z_buffer->runtime)), NULL);
    }
    
    uint64_t m = x_buffer->view->shape[x_buffer->view->rank - 2];
    uint64_t k = x_buffer->view->shape[x_buffer->view->rank - 1];
    uint64_t n = y_buffer->view->shape[x_buffer->view->rank - 1];
    bool_t x_transpose = x_buffer->view->shape[x_buffer->view->rank - 1] == x_buffer->view->strides[x_buffer->view->rank - 2] && x_buffer->view->strides[x_buffer->view->rank - 1] == 1; 
    bool_t y_transpose = y_buffer->view->shape[y_buffer->view->rank - 1] == y_buffer->view->strides[y_buffer->view->rank - 2] && y_buffer->view->strides[y_buffer->view->rank - 1] == 1;  

    switch (z_buffer->view->rank)
    {
    case 2:
        switch (z_buffer->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose,
                                           x_buffer->data, x_buffer->view->offset,
                                           y_buffer->data, y_buffer->view->offset,
                                           z_buffer->data, z_buffer->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose,
                                      x_buffer->data, x_buffer->view->offset,
                                      y_buffer->data, y_buffer->view->offset,
                                      z_buffer->data, z_buffer->view->offset);
            break;
        case CU_RUNTIME:
            cu_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose,
                                     x_buffer->data, x_buffer->view->offset,
                                     y_buffer->data, y_buffer->view->offset,
                                     z_buffer->data, z_buffer->view->offset);
            break;
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) z_buffer->runtime), NULL);
        }
        break;
    case 3:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            switch (z_buffer->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose,
                                               x_buffer->data, x_buffer->view->offset + i * x_buffer->view->strides[0],
                                               y_buffer->data, y_buffer->view->offset + i * y_buffer->view->strides[0],
                                               z_buffer->data, z_buffer->view->offset + i * z_buffer->view->strides[0]);
                break;
            case MKL_RUNTIME:
                mkl_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose,
                                          x_buffer->data, x_buffer->view->offset + i * x_buffer->view->strides[0],
                                          y_buffer->data, y_buffer->view->offset + i * y_buffer->view->strides[0],
                                          z_buffer->data, z_buffer->view->offset + i * z_buffer->view->strides[0]);
                break;
            case CU_RUNTIME:
                cu_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose,
                                         x_buffer->data, x_buffer->view->offset + i * x_buffer->view->strides[0],
                                         y_buffer->data, y_buffer->view->offset + i * y_buffer->view->strides[0],
                                         z_buffer->data, z_buffer->view->offset + i * z_buffer->view->strides[0]);
                break;
            default:
                return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) z_buffer->runtime), NULL);
            }
        }
        break;
    case 4:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[0]; ++j)
            {
                switch (z_buffer->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose,
                                                   x_buffer->data, x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                                   y_buffer->data, y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                                   z_buffer->data, z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                    break;
                case MKL_RUNTIME:
                    mkl_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose,
                                              x_buffer->data, x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                              y_buffer->data, y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                              z_buffer->data, z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                    break;
                case CU_RUNTIME:
                    cu_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose,
                                             x_buffer->data, x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                             y_buffer->data, y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                             z_buffer->data, z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                    break;
                default:
                    return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) z_buffer->runtime), NULL);
                }
            }
        }
        break;
    default:
        break;
    }

    return NULL;
}

typedef enum reduction_t
{
    RUNTIME_SUMMATION,
    RUNTIME_MAXIMUM
} reduction_t;

static nw_error_t *runtime_reduction(reduction_t reduction, buffer_t *x, buffer_t *result, uint64_t axis)
{
    uint64_t idim;
    uint64_t jdim;
    uint64_t kdim;

    switch (x->view->rank)
    {
    case 1:
        switch (reduction)
        {
        case RUNTIME_SUMMATION:
            switch (x->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset, result->data, result->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset, result->data, result->view->offset);
                break;
            case CU_RUNTIME:
                cu_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset, result->data, result->view->offset);
                break;
            default:
                break;
            }
            break;
        case RUNTIME_MAXIMUM:
            switch (x->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset, result->data, result->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset, result->data, result->view->offset);
                break;
            case CU_RUNTIME:
                cu_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset, result->data, result->view->offset);
                break;
            default:
                break;
            }
            break;
        default:
            break;
        }
        break;
    case 2:
        switch (axis)
        {
        case 0:
            idim = 1;
            break;
        case 1:
            idim = 0;
            break;
        default:
            return ERROR(ERROR_AXIS, string_create("invalid axis dimension."), NULL);
        }
        
        for (uint64_t i = 0; i < x->view->shape[idim]; ++i)
        {
            switch (reduction)
            {
            case RUNTIME_SUMMATION:
                switch (x->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                       result->data, result->view->offset + i * result->view->strides[idim]);
                    break;
                case MKL_RUNTIME:
                    mkl_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                  result->data, result->view->offset + i * result->view->strides[idim]);
                    break;
                case CU_RUNTIME:
                    cu_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                 result->data, result->view->offset + i * result->view->strides[idim]);
                    break;
                default:
                    break;
                }
                break;
            case RUNTIME_MAXIMUM:
                switch (x->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                     result->data, result->view->offset + i * result->view->strides[idim]);
                    break;
                case MKL_RUNTIME:
                    mkl_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                result->data, result->view->offset + i * result->view->strides[idim]);
                    break;
                case CU_RUNTIME:
                    cu_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                               result->data, result->view->offset + i * result->view->strides[idim]);
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
        }
        break;
    case 3:
        switch (axis)
        {
        case 0:
            idim = 1;
            jdim = 2;
            break;
        case 1:
            idim = 0;
            jdim = 2;
            break;
        case 2:
            idim = 0;
            jdim = 1;
            break;
        default:
            return ERROR(ERROR_AXIS, string_create("invalid axis dimension."), NULL);
        }
        
        for (uint64_t i = 0; i < x->view->shape[idim]; ++i)
        {
            for (uint64_t j = 0; j < x->view->shape[jdim]; ++j)
            {
                switch (reduction)
                {
                case RUNTIME_SUMMATION:
                    switch (x->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                           result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
                    case MKL_RUNTIME:
                        mkl_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                      result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
                    case CU_RUNTIME:
                        cu_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                     result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
                    default:
                        break;
                    }
                    break;
                case RUNTIME_MAXIMUM:
                    switch (x->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                         result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
                    case MKL_RUNTIME:
                        mkl_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                    result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
                    case CU_RUNTIME:
                        cu_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                   result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
                    default:
                        break;
                    }
                    break;
                default:
                    break;
                }
            }
        }

        break;
    case 4:
        switch (axis)
        {
        case 0:
            idim = 1;
            jdim = 2;
            kdim = 3;
            break;
        case 1:
            idim = 0;
            jdim = 2;
            kdim = 3;
            break;
        case 2:
            idim = 0;
            jdim = 1;
            kdim = 3;
            break;
        case 3:
            idim = 0;
            jdim = 1;
            kdim = 2;
            break;
        default:
            return ERROR(ERROR_AXIS, string_create("invalid axis dimension."), NULL);
        }
        
        for (uint64_t i = 0; i < x->view->shape[idim]; ++i)
        {
            for (uint64_t j = 0; j < x->view->shape[jdim]; ++j)
            {
                for (uint64_t k = 0; k < x->view->shape[kdim]; ++k)
                {
                    switch (reduction)
                    {
                    case RUNTIME_SUMMATION:
                        switch (x->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                               result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
                        case MKL_RUNTIME:
                            mkl_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                          result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
                        case CU_RUNTIME:
                            cu_summation(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                         result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_MAXIMUM:
                        switch (x->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                             result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
                        case MKL_RUNTIME:
                            mkl_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                        result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
                        case CU_RUNTIME:
                            cu_maximum(x->datatype, x->view->shape[axis], x->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                       result->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
                        default:
                            break;
                        }
                        break;
                    default:
                        break;
                    }
                }
            }
        }

        break;
    default:
        break;
    }

    return NULL;
}

nw_error_t *runtime_summation(buffer_t *x, buffer_t *result, uint64_t axis)
{
    nw_error_t *error = runtime_reduction(RUNTIME_SUMMATION, x, result, axis);
    if (error != NULL)
    {
        return ERROR(ERROR_REDUCTION, string_create("failed to apply reduction operation."), error);
    }

    return NULL;
}

nw_error_t *runtime_maximum(buffer_t *x, buffer_t *result, uint64_t axis)
{
    nw_error_t *error = runtime_reduction(RUNTIME_MAXIMUM, x, result, axis);
    if (error != NULL)
    {
        return ERROR(ERROR_REDUCTION, string_create("failed to apply reduction operation."), error);
    }

    return NULL;
}

string_t runtime_string(runtime_t runtime)
{
    switch (runtime)
    {
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
