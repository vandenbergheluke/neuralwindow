/**@file buffer.c
 * @brief
 *
 */

#include <buffer.h>
#include <view.h>
#ifndef CPU_ONLY
#include <cu_runtime.h>
#endif
#include <mkl_runtime.h>
#include <openblas_runtime.h>
#include <string.h>

nw_error_t *storage_create(storage_t **storage, runtime_t runtime, datatype_t datatype, uint64_t n, void *data)
{
    CHECK_NULL_ARGUMENT(storage, "storage");

    *storage = (storage_t *) malloc(sizeof(storage_t));
    if (storage == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate storage of size %lu bytes.", 
                     (unsigned long) sizeof(buffer_t)),
                     NULL);
    }

    if (!n)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     "storage must store more than 0 bytes of data.",
                     NULL);
    }

    (*storage)->runtime = runtime;
    (*storage)->datatype = datatype;
    (*storage)->n = n;
    (*storage)->reference_count = 0;

    nw_error_t *error = runtime_malloc(*storage);
    if (error != NULL)
    {
        free(*storage);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate buffer data for runtime %s and datatype %s.",
                     runtime_string(runtime), datatype_string(datatype)),
                     error);
    }

    if (data != NULL)
    {
        memcpy((*storage)->data, data, (*storage)->n * datatype_size(datatype));
    }

    return NULL;
}

void storage_destroy(storage_t *storage)
{
    if (storage == NULL)
    {
        return;
    }

    if (storage->reference_count < 2)
    {
        runtime_free(storage);
        free(storage);
    }
    else
    {
        --(storage->reference_count);
    }
}


nw_error_t *buffer_create(buffer_t **buffer,
                          view_t *view,
                          storage_t *storage,
                          bool_t copy)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(storage, "storage");

    nw_error_t *error = NULL;

    *buffer = (buffer_t *) malloc(sizeof(buffer_t));
    if (buffer == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate buffer of size %lu bytes.", 
                     (unsigned long) sizeof(buffer_t)),
                     NULL);
    }

    (*buffer)->view = view;

    if (copy)
    {
        error = storage_create(&(*buffer)->storage,
                               storage->runtime,
                               storage->datatype,
                               storage->n,
                               storage->data);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create storage copy."),
                         error);
        }
    }
    else
    {
        (*buffer)->storage = storage;
    }
    ++((*buffer)->storage->reference_count);

    return NULL;
}

void buffer_destroy(buffer_t *buffer)
{
    if (buffer == NULL)
    {
        return;
    }

    storage_destroy(buffer->storage);
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
#ifndef CPU_ONLY
    case CU_RUNTIME:
        error = cu_create_context();
        break;
#endif
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
#ifndef CPU_ONLY
    case CU_RUNTIME:
        cu_destroy_context();
        break;
#endif
    default:
        break;
    }
}

nw_error_t *runtime_malloc(storage_t *storage)
{
    CHECK_NULL_ARGUMENT(storage, "storage");

    if (storage->n == 0)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("cannot allocate 0 bytes."),
                     NULL);
    }

    nw_error_t *error = NULL;

    switch (storage->runtime)
    {
    case OPENBLAS_RUNTIME:
        error = openblas_memory_allocate(&storage->data,
                                         storage->n * datatype_size(storage->datatype));
        break;
    case MKL_RUNTIME:
        error = mkl_memory_allocate(&storage->data,
                                    storage->n * datatype_size(storage->datatype));
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        error = cu_memory_allocate(&storage->data,
                                   storage->n * datatype_size(storage->datatype));
        break;
#endif
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.",
                      (int) storage->runtime),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate %lu bytes for runtime %s.", 
                     (unsigned long) (storage->n *datatype_size(storage->datatype)),
                     runtime_string(storage->runtime)),
                     error);
    }
    
    return NULL;
}

void runtime_free(storage_t *storage)
{
    if (storage == NULL)
    {
        return;
    }

    switch (storage->runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_memory_free(storage->data);
        break;
    case MKL_RUNTIME:
        mkl_memory_free(storage->data);
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        cu_memory_free(storage->data);
        break;
#endif
    default:
        break;
    }
}

static nw_error_t *runtime_unary(runtime_unary_type_t runtime_unary_type, buffer_t *x, buffer_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->view, "x->view");
    CHECK_NULL_ARGUMENT(x->storage->data, "x->storage->data");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(result->view, "result->view");
    CHECK_NULL_ARGUMENT(result->storage->data, "result->storage->data");
    CHECK_NULL_ARGUMENT(x->view->shape, "x->view->shape");
    CHECK_NULL_ARGUMENT(x->view->strides, "x->view->strides");
    CHECK_NULL_ARGUMENT(result->view->shape, "result->view->shape");
    CHECK_NULL_ARGUMENT(result->view->strides, "result->view->strides");

    if (x->storage->datatype != result->storage->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("conflicting datatypes x (%s) and result (%s).",
                     datatype_string(x->storage->datatype), datatype_string(result->storage->datatype)),
                     NULL);
    }

    if (x->storage->runtime != result->storage->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT,
                     string_create("conflicting runtimes x (%s) and result (%s).",
                     runtime_string(x->storage->runtime), runtime_string(result->storage->runtime)),
                     NULL);
    }

    if (x->storage->n != result->storage->n)
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting number of elements in x buffer (%lu) and result buffer (%lu).",
                     (unsigned long) x->storage->n, (unsigned long) result->storage->n), NULL);
    }

    if (!shapes_equal(x->view->shape, x->view->rank, result->view->shape, result->view->rank))
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting shapes x and result."),
                     NULL);
    }

    switch (runtime_unary_type)
    {
    case RUNTIME_EXPONENTIAL:
        switch (x->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_exponential(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_exponential(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_exponential(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME,
                        string_create("unknown runtime %d.",
                        (int) x->storage->runtime), NULL);
        }
        break;
    case RUNTIME_LOGARITHM:
        switch (x->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_logarithm(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_logarithm(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_logarithm(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME,
                        string_create("unknown runtime %d.",
                        (int) x->storage->runtime), NULL);
        }
        break;
    case RUNTIME_SINE:
        switch (x->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_sine(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_sine(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_sine(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME,
                        string_create("unknown runtime %d.",
                        (int) x->storage->runtime), NULL);
        }
        break;
    case RUNTIME_COSINE:
        switch (x->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_cosine(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_cosine(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_cosine(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME,
                        string_create("unknown runtime %d.",
                        (int) x->storage->runtime), NULL);
        }
        break;
    case RUNTIME_SQUARE_ROOT:
        switch (x->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_square_root(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_square_root(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_square_root(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME,
                        string_create("unknown runtime %d.",
                        (int) x->storage->runtime), NULL);
        }
        break;
    case RUNTIME_RECIPROCAL:
        switch (x->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_reciprocal(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_reciprocal(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_reciprocal(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME,
                        string_create("unknown runtime %d.",
                        (int) x->storage->runtime), NULL);
        }
        break;
    case RUNTIME_COPY:
        switch (x->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_copy(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_copy(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_copy(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME,
                        string_create("unknown runtime %d.",
                        (int) x->storage->runtime), NULL);
        }
        break;
    case RUNTIME_NEGATION:
        switch (x->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_negation(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_negation(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_negation(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME,
                        string_create("unknown runtime %d.",
                        (int) x->storage->runtime), NULL);
        }
        break;
    case RUNTIME_RECTIFIED_LINEAR:
        switch (x->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_rectified_linear(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_rectified_linear(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_rectified_linear(x->storage->datatype, x->storage->n, x->storage->data, (uint64_t) 1, x->view->offset, result->storage->data, (uint64_t) 1, result->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME,
                        string_create("unknown runtime %d.",
                        (int) x->storage->runtime), NULL);
        }
        break;
    case RUNTIME_CONTIGUOUS:
        switch (x->view->rank)
        {
        case 1:
            switch (x->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_copy(x->storage->datatype, x->view->shape[0], x->storage->data, x->view->strides[0], x->view->offset, result->storage->data, result->view->strides[0], result->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_copy(x->storage->datatype, x->view->shape[0], x->storage->data, x->view->strides[0], x->view->offset, result->storage->data, result->view->strides[0], result->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_copy(x->storage->datatype, x->view->shape[0], x->storage->data, x->view->strides[0], x->view->offset, result->storage->data, result->view->strides[0], result->view->offset);
                break;
#endif
            default:
                return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->storage->datatype), NULL);
            }
            break;
        case 2:
            for (uint64_t i = 0; i < x->view->shape[0]; ++i)
            {
                switch (x->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_copy(x->storage->datatype, x->view->shape[1], x->storage->data, x->view->strides[1], x->view->offset + i * x->view->strides[0], 
                                result->storage->data, result->view->strides[1], result->view->offset + i * result->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_copy(x->storage->datatype, x->view->shape[1], x->storage->data, x->view->strides[1], x->view->offset + i * x->view->strides[0], 
                            result->storage->data, result->view->strides[1], result->view->offset + i * result->view->strides[0]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_copy(x->storage->datatype, x->view->shape[1], x->storage->data, x->view->strides[1], x->view->offset + i * x->view->strides[0], 
                            result->storage->data, result->view->strides[1], result->view->offset + i * result->view->strides[0]);
                    break;
#endif
                default:
                    return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->storage->datatype), NULL);
                }
            }
            break;
        case 3:
            for (uint64_t i = 0; i < x->view->shape[0]; ++i)
            {
                for (uint64_t j = 0; j < x->view->shape[1]; ++j)
                {
                    switch (x->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_copy(x->storage->datatype, x->view->shape[2], x->storage->data, x->view->strides[2], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1], 
                                    result->storage->data, result->view->strides[2], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_copy(x->storage->datatype, x->view->shape[2], x->storage->data, x->view->strides[2], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1], 
                                result->storage->data, result->view->strides[2], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_copy(x->storage->datatype, x->view->shape[2], x->storage->data, x->view->strides[2], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1], 
                                result->storage->data, result->view->strides[2], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1]);
                        break;
#endif
                    default:
                        return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->storage->datatype), NULL);
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
                        switch (x->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_copy(x->storage->datatype, x->view->shape[3], x->storage->data, x->view->strides[3], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1] + k * x->view->strides[2], 
                                        result->storage->data, result->view->strides[3], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1] + k * result->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_copy(x->storage->datatype, x->view->shape[3], x->storage->data, x->view->strides[3], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1] + k * x->view->strides[2], 
                                    result->storage->data, result->view->strides[3], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1] + k * result->view->strides[2]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_copy(x->storage->datatype, x->view->shape[3], x->storage->data, x->view->strides[3], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1] + k * x->view->strides[2], 
                                    result->storage->data, result->view->strides[3], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1] + k * result->view->strides[2]);
                            break;
#endif
                        default:
                            return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->storage->datatype), NULL);
                        }
                    }
                }
            }
            break;
        case 5:
            for (uint64_t i = 0; i < x->view->shape[0]; ++i)
            {
                for (uint64_t j = 0; j < x->view->shape[1]; ++j)
                {
                    for (uint64_t k = 0; k < x->view->shape[2]; ++k)
                    {
                        for (uint64_t l = 0; l < x->view->shape[3]; ++l)
                        {
                            switch (x->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_copy(x->storage->datatype, x->view->shape[4], x->storage->data, x->view->strides[4], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1] + k * x->view->strides[2] + l * x->view->strides[3], 
                                            result->storage->data, result->view->strides[4], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1] + k * result->view->strides[2] + l * result->view->strides[3]);
                                break;
                            case MKL_RUNTIME:
                                mkl_copy(x->storage->datatype, x->view->shape[4], x->storage->data, x->view->strides[4], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1] + k * x->view->strides[2] + l * x->view->strides[3], 
                                        result->storage->data, result->view->strides[4], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1] + k * result->view->strides[2] + l * result->view->strides[3]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_copy(x->storage->datatype, x->view->shape[4], x->storage->data, x->view->strides[4], x->view->offset + i * x->view->strides[0] + j * x->view->strides[1] + k * x->view->strides[2] + l * x->view->strides[3], 
                                        result->storage->data, result->view->strides[4], result->view->offset + i * result->view->strides[0] + j * result->view->strides[1] + k * result->view->strides[2] + l * result->view->strides[3]);
                                break;
#endif
                            default:
                                return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) x->storage->datatype), NULL);
                            }
                        }
                    }
                }
            }
            break;
        default:
            return ERROR(ERROR_RANK_CONFLICT, string_create("only support tensors rank between 1-4."), NULL);
        }
        break;
    default:
        break;
    }

    return NULL;
}

nw_error_t *runtime_exponential(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_EXPONENTIAL, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_logarithm(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_LOGARITHM, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_sine(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_SINE, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_cosine(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_COSINE, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_square_root(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_SQUARE_ROOT, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_reciprocal(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_RECIPROCAL, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_copy(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_COPY, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_contiguous(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_CONTIGUOUS, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_negation(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_NEGATION, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_rectified_linear(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_RECTIFIED_LINEAR, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}


static nw_error_t *runtime_binary_elementwise(runtime_binary_elementwise_type_t runtime_binary_elementwise_type, 
                                              buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    switch (z_buffer->view->rank)
    {
    case 1:
        switch (runtime_binary_elementwise_type)
        {
        case RUNTIME_ADDITION:
            switch (z_buffer->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_addition(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                  x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_addition(z_buffer->storage->datatype, z_buffer->view->shape[0],
                             x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                             y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                             z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_addition(z_buffer->storage->datatype, z_buffer->view->shape[0],
                            x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                            y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                            z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#endif
            default:
                break;
            }
            break;
        case RUNTIME_SUBTRACTION:
            switch (z_buffer->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                     x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                     y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                     z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[0],
                               x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                               y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                               z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#endif
            default:
                break;
            }
            break;
        case RUNTIME_MULTIPLICATION:
            switch (z_buffer->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                        x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                        y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                        z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                   x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                   y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                   z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                  x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#endif
            default:
                break;
            }
            break;
        case RUNTIME_DIVISION:
            switch (z_buffer->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_division(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                  x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_division(z_buffer->storage->datatype, z_buffer->view->shape[0],
                             x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                             y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                             z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_division(z_buffer->storage->datatype, z_buffer->view->shape[0],
                            x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                            y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                            z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#endif
            default:
                break;
            }
            break;
        case RUNTIME_POWER:
            switch (z_buffer->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_power(z_buffer->storage->datatype, z_buffer->view->shape[0],
                               x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                               y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                               z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_power(z_buffer->storage->datatype, z_buffer->view->shape[0],
                          x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                          y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                          z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_power(z_buffer->storage->datatype, z_buffer->view->shape[0],
                         x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                         y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                         z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#endif
            default:
                break;
            }
            break;
        case RUNTIME_COMPARE_EQUAL:
            switch (z_buffer->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                       x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                       y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                       z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                  x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                 x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                 y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                 z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#endif
            default:
                break;
            }
            break;
        case RUNTIME_COMPARE_GREATER:
            switch (z_buffer->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                       x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                       y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                       z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                  x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                  y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                  z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[0],
                                 x_buffer->storage->data, x_buffer->view->strides[0], x_buffer->view->offset,
                                 y_buffer->storage->data, y_buffer->view->strides[0], y_buffer->view->offset,
                                 z_buffer->storage->data, z_buffer->view->strides[0], z_buffer->view->offset);
                break;
#endif
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
            switch (runtime_binary_elementwise_type)
            {
            case RUNTIME_ADDITION:
                switch (z_buffer->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_addition(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                      x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_addition(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                 x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                 y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                 z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_addition(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#endif
                default:
                    break;
                }
                break;
            case RUNTIME_SUBTRACTION:
                switch (z_buffer->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                         x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                         y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                         z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                    x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                    y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                    z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                   x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                   y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                   z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#endif
                default:
                    break;
                }
                break;
            case RUNTIME_MULTIPLICATION:
                switch (z_buffer->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                            x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                            y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                            z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                       x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                       y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                       z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                      x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#endif
                default:
                    break;
                }
                break;
            case RUNTIME_DIVISION:
                switch (z_buffer->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_division(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                      x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_division(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                 x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                 y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                 z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_division(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#endif
                default:
                    break;
                }
                break;
            case RUNTIME_POWER:
                switch (z_buffer->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_power(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                   x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                   y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                   z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_power(z_buffer->storage->datatype, z_buffer->view->shape[1],
                              x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                              y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                              z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_power(z_buffer->storage->datatype, z_buffer->view->shape[1],
                             x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                             y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                             z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#endif
                default:
                    break;
                }
                break;
            case RUNTIME_COMPARE_EQUAL:
                switch (z_buffer->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                           x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                           y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                           z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                      x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                     x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                     y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                     z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#endif
                default:
                    break;
                }
                break;
            case RUNTIME_COMPARE_GREATER:
                switch (z_buffer->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                           x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                           y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                           z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
                case MKL_RUNTIME:
                    mkl_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                      x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                      y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                      z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[1],
                                     x_buffer->storage->data, x_buffer->view->strides[1], x_buffer->view->offset + i * x_buffer->view->strides[0],
                                     y_buffer->storage->data, y_buffer->view->strides[1], y_buffer->view->offset + i * y_buffer->view->strides[0],
                                     z_buffer->storage->data, z_buffer->view->strides[1], z_buffer->view->offset + i * z_buffer->view->strides[0]);
                    break;
#endif
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
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[1]; ++j)
            {
                switch (runtime_binary_elementwise_type)
                {
                case RUNTIME_ADDITION:
                    switch (z_buffer->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_addition(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                          x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_addition(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                     x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                     y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                     z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_addition(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                    x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                    y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                    z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#endif
                    default:
                        break;
                    }
                    break;
                case RUNTIME_SUBTRACTION:
                    switch (z_buffer->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                             x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                             y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                             z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                        x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                        y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                        z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                       x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                       y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                       z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#endif
                    default:
                        break;
                    }
                    break;
                case RUNTIME_MULTIPLICATION:
                    switch (z_buffer->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                                x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                                y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                                z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                           x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                           y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                           z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                          x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#endif
                    default:
                        break;
                    }
                    break;
                case RUNTIME_DIVISION:
                    switch (z_buffer->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_division(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                          x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_division(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                     x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                     y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                     z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_division(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                    x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                    y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                    z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#endif
                    default:
                        break;
                    }
                    break;
                case RUNTIME_POWER:
                    switch (z_buffer->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_power(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                       x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                       y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                       z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_power(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                  x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                  y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                  z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_power(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                 x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                 y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                 z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#endif
                    default:
                        break;
                    }
                    break;
                case RUNTIME_COMPARE_EQUAL:
                    switch (z_buffer->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                               x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                               y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                               z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                          x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                         x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                         y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                         z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#endif
                    default:
                        break;
                    }
                    break;
                case RUNTIME_COMPARE_GREATER:
                    switch (z_buffer->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                               x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                               y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                               z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
                    case MKL_RUNTIME:
                        mkl_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                          x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                          y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                          z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[2],
                                         x_buffer->storage->data, x_buffer->view->strides[2], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                         y_buffer->storage->data, y_buffer->view->strides[2], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                         z_buffer->storage->data, z_buffer->view->strides[2], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                        break;
#endif
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
                    switch (runtime_binary_elementwise_type)
                    {
                    case RUNTIME_ADDITION:
                        switch (z_buffer->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_addition(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                              x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_addition(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                         x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                         y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                         z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_addition(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                        x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                        y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                        z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#endif
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_SUBTRACTION:
                        switch (z_buffer->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                                 x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                 y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                 z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                            x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                            y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                            z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                           x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                           y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                           z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#endif
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_MULTIPLICATION:
                        switch (z_buffer->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                                    x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                    y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                    z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                               x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                               y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                               z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                              x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#endif
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_DIVISION:
                        switch (z_buffer->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_division(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                              x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_division(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                         x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                         y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                         z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_division(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                        x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                        y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                        z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#endif
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_POWER:
                        switch (z_buffer->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_power(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                           x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                           y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                           z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_power(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                      x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                      y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                      z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_power(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                     x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                     y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                     z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#endif
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_COMPARE_EQUAL:
                        switch (z_buffer->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                                   x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                   y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                   z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                              x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                             x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                             y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                             z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#endif
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_COMPARE_GREATER:
                        switch (z_buffer->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                                   x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                   y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                   z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
                        case MKL_RUNTIME:
                            mkl_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                              x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                              y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                              z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[3],
                                             x_buffer->storage->data, x_buffer->view->strides[3], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                             y_buffer->storage->data, y_buffer->view->strides[3], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                             z_buffer->storage->data, z_buffer->view->strides[3], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                            break;
#endif
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
    case 5:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[1]; ++j)
            {
                for (uint64_t k = 0; k < z_buffer->view->shape[2]; ++k)
                {
                    for (uint64_t l = 0; l < z_buffer->view->shape[3]; ++l)
                    {
                        switch (runtime_binary_elementwise_type)
                        {
                        case RUNTIME_ADDITION:
                            switch (z_buffer->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_addition(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
                            case MKL_RUNTIME:
                                mkl_addition(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                            x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                            y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                            z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_addition(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                            x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                            y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                            z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#endif
                            default:
                                break;
                            }
                            break;
                        case RUNTIME_SUBTRACTION:
                            switch (z_buffer->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                    x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                    y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                    z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
                            case MKL_RUNTIME:
                                mkl_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_subtraction(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                            x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                            y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                            z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#endif
                            default:
                                break;
                            }
                            break;
                        case RUNTIME_MULTIPLICATION:
                            switch (z_buffer->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                        x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                        y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                        z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
                            case MKL_RUNTIME:
                                mkl_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_multiplication(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#endif
                            default:
                                break;
                            }
                            break;
                        case RUNTIME_DIVISION:
                            switch (z_buffer->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_division(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
                            case MKL_RUNTIME:
                                mkl_division(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                            x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                            y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                            z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_division(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                            x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                            y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                            z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#endif
                            default:
                                break;
                            }
                            break;
                        case RUNTIME_POWER:
                            switch (z_buffer->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_power(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                            x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                            y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                            z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
                            case MKL_RUNTIME:
                                mkl_power(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                        x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                        y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                        z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_power(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                        x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                        y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                        z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#endif
                            default:
                                break;
                            }
                            break;
                        case RUNTIME_COMPARE_EQUAL:
                            switch (z_buffer->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                    x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                    y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                    z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
                            case MKL_RUNTIME:
                                mkl_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_compare_equal(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#endif
                            default:
                                break;
                            }
                            break;
                        case RUNTIME_COMPARE_GREATER:
                            switch (z_buffer->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                    x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                    y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                    z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
                            case MKL_RUNTIME:
                                mkl_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_compare_greater(z_buffer->storage->datatype, z_buffer->view->shape[4],
                                                x_buffer->storage->data, x_buffer->view->strides[4], x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2] + l * x_buffer->view->strides[3],
                                                y_buffer->storage->data, y_buffer->view->strides[4], y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2] + l * y_buffer->view->strides[3],
                                                z_buffer->storage->data, z_buffer->view->strides[4], z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2] + l * z_buffer->view->strides[3]);
                                break;
#endif
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

    if (x_buffer->storage->datatype != y_buffer->storage->datatype || x_buffer->storage->datatype != z_buffer->storage->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT, string_create("conflicting datatypes %s + %s = %s.",
                     datatype_string(x_buffer->storage->datatype), datatype_string(y_buffer->storage->datatype), datatype_string(z_buffer->storage->datatype)), NULL);
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

    if (x_buffer->storage->runtime != y_buffer->storage->runtime || x_buffer->storage->runtime != z_buffer->storage->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT, string_create("conflicting runtimes %s + %s = %s.", 
                     runtime_string(x_buffer->storage->runtime), runtime_string(y_buffer->storage->runtime), runtime_string(z_buffer->storage->runtime)), NULL);
    }
    
    uint64_t m = x_buffer->view->shape[x_buffer->view->rank - 2];
    uint64_t k = x_buffer->view->shape[x_buffer->view->rank - 1];
    uint64_t n = y_buffer->view->shape[x_buffer->view->rank - 1];
    bool_t x_transpose = x_buffer->view->shape[x_buffer->view->rank - 1] == x_buffer->view->strides[x_buffer->view->rank - 2] && x_buffer->view->strides[x_buffer->view->rank - 1] == 1; 
    bool_t y_transpose = y_buffer->view->shape[y_buffer->view->rank - 1] == y_buffer->view->strides[y_buffer->view->rank - 2] && y_buffer->view->strides[y_buffer->view->rank - 1] == 1;  

    switch (z_buffer->view->rank)
    {
    case 2:
        switch (z_buffer->storage->runtime)
        {
        case OPENBLAS_RUNTIME:
            openblas_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                           x_buffer->storage->data, x_buffer->view->offset,
                                           y_buffer->storage->data, y_buffer->view->offset,
                                           z_buffer->storage->data, z_buffer->view->offset);
            break;
        case MKL_RUNTIME:
            mkl_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                      x_buffer->storage->data, x_buffer->view->offset,
                                      y_buffer->storage->data, y_buffer->view->offset,
                                      z_buffer->storage->data, z_buffer->view->offset);
            break;
#ifndef CPU_ONLY
        case CU_RUNTIME:
            cu_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                     x_buffer->storage->data, x_buffer->view->offset,
                                     y_buffer->storage->data, y_buffer->view->offset,
                                     z_buffer->storage->data, z_buffer->view->offset);
            break;
#endif
        default:
            return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) z_buffer->storage->runtime), NULL);
        }
        break;
    case 3:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            switch (z_buffer->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                               x_buffer->storage->data, x_buffer->view->offset + i * x_buffer->view->strides[0],
                                               y_buffer->storage->data, y_buffer->view->offset + i * y_buffer->view->strides[0],
                                               z_buffer->storage->data, z_buffer->view->offset + i * z_buffer->view->strides[0]);
                break;
            case MKL_RUNTIME:
                mkl_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                          x_buffer->storage->data, x_buffer->view->offset + i * x_buffer->view->strides[0],
                                          y_buffer->storage->data, y_buffer->view->offset + i * y_buffer->view->strides[0],
                                          z_buffer->storage->data, z_buffer->view->offset + i * z_buffer->view->strides[0]);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                         x_buffer->storage->data, x_buffer->view->offset + i * x_buffer->view->strides[0],
                                         y_buffer->storage->data, y_buffer->view->offset + i * y_buffer->view->strides[0],
                                         z_buffer->storage->data, z_buffer->view->offset + i * z_buffer->view->strides[0]);
                break;
#endif
            default:
                return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) z_buffer->storage->runtime), NULL);
            }
        }
        break;
    case 4:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[1]; ++j)
            {
                switch (z_buffer->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                                   x_buffer->storage->data, x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                                   y_buffer->storage->data, y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                                   z_buffer->storage->data, z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                    break;
                case MKL_RUNTIME:
                    mkl_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                              x_buffer->storage->data, x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                              y_buffer->storage->data, y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                              z_buffer->storage->data, z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                             x_buffer->storage->data, x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1],
                                             y_buffer->storage->data, y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1],
                                             z_buffer->storage->data, z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1]);
                    break;
#endif
                default:
                    return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) z_buffer->storage->runtime), NULL);
                }
            }
        }
        break;
    case 5:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[1]; ++j)
            {
                for (uint64_t k = 0; k < z_buffer->view->shape[2]; ++k)
                {
                    switch (z_buffer->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                                    x_buffer->storage->data, x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                    y_buffer->storage->data, y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1 + k * y_buffer->view->strides[2]],
                                                    z_buffer->storage->data, z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                        break;
                    case MKL_RUNTIME:
                        mkl_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                                x_buffer->storage->data, x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                y_buffer->storage->data, y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                z_buffer->storage->data, z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_matrix_multiplication(z_buffer->storage->datatype, m, k, n, x_transpose, y_transpose,
                                                x_buffer->storage->data, x_buffer->view->offset + i * x_buffer->view->strides[0] + j * x_buffer->view->strides[1] + k * x_buffer->view->strides[2],
                                                y_buffer->storage->data, y_buffer->view->offset + i * y_buffer->view->strides[0] + j * y_buffer->view->strides[1] + k * y_buffer->view->strides[2],
                                                z_buffer->storage->data, z_buffer->view->offset + i * z_buffer->view->strides[0] + j * z_buffer->view->strides[1] + k * z_buffer->view->strides[2]);
                        break;
#endif
                    default:
                        return ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) z_buffer->storage->runtime), NULL);
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


static nw_error_t *runtime_reduction(runtime_reduction_type_t runtime_reduction_type, buffer_t *x, buffer_t *result, uint64_t axis)
{
    uint64_t idim;
    uint64_t jdim;
    uint64_t kdim;
    uint64_t ldim;

    switch (x->view->rank)
    {
    case 1:
        switch (runtime_reduction_type)
        {
        case RUNTIME_SUMMATION:
            switch (x->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset, result->storage->data, result->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset, result->storage->data, result->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset, result->storage->data, result->view->offset);
                break;
#endif
            default:
                break;
            }
            break;
        case RUNTIME_MAXIMUM:
            switch (x->storage->runtime)
            {
            case OPENBLAS_RUNTIME:
                openblas_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset, result->storage->data, result->view->offset);
                break;
            case MKL_RUNTIME:
                mkl_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset, result->storage->data, result->view->offset);
                break;
#ifndef CPU_ONLY
            case CU_RUNTIME:
                cu_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset, result->storage->data, result->view->offset);
                break;
#endif
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
            switch (runtime_reduction_type)
            {
            case RUNTIME_SUMMATION:
                switch (x->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                       result->storage->data, result->view->offset + i * result->view->strides[idim]);
                    break;
                case MKL_RUNTIME:
                    mkl_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                  result->storage->data, result->view->offset + i * result->view->strides[idim]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                 result->storage->data, result->view->offset + i * result->view->strides[idim]);
                    break;
#endif
                default:
                    break;
                }
                break;
            case RUNTIME_MAXIMUM:
                switch (x->storage->runtime)
                {
                case OPENBLAS_RUNTIME:
                    openblas_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                     result->storage->data, result->view->offset + i * result->view->strides[idim]);
                    break;
                case MKL_RUNTIME:
                    mkl_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                                result->storage->data, result->view->offset + i * result->view->strides[idim]);
                    break;
#ifndef CPU_ONLY
                case CU_RUNTIME:
                    cu_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim],
                               result->storage->data, result->view->offset + i * result->view->strides[idim]);
                    break;
#endif
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
                switch (runtime_reduction_type)
                {
                case RUNTIME_SUMMATION:
                    switch (x->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                           result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
                    case MKL_RUNTIME:
                        mkl_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                      result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                     result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
#endif
                    default:
                        break;
                    }
                    break;
                case RUNTIME_MAXIMUM:
                    switch (x->storage->runtime)
                    {
                    case OPENBLAS_RUNTIME:
                        openblas_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                         result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
                    case MKL_RUNTIME:
                        mkl_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                    result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
#ifndef CPU_ONLY
                    case CU_RUNTIME:
                        cu_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim],
                                   result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim]);
                        break;
#endif
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
                    switch (runtime_reduction_type)
                    {
                    case RUNTIME_SUMMATION:
                        switch (x->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                               result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
                        case MKL_RUNTIME:
                            mkl_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                          result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                         result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
#endif
                        default:
                            break;
                        }
                        break;
                    case RUNTIME_MAXIMUM:
                        switch (x->storage->runtime)
                        {
                        case OPENBLAS_RUNTIME:
                            openblas_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                             result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
                        case MKL_RUNTIME:
                            mkl_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                        result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
#ifndef CPU_ONLY
                        case CU_RUNTIME:
                            cu_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim],
                                       result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim]);
                            break;
#endif
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
    case 5:
        switch (axis)
        {
        case 0:
            idim = 1;
            jdim = 2;
            kdim = 3;
            ldim = 4;
            break;
        case 1:
            idim = 0;
            jdim = 2;
            kdim = 3;
            ldim = 4;
            break;
        case 2:
            idim = 0;
            jdim = 1;
            kdim = 3;
            ldim = 4;
            break;
        case 3:
            idim = 0;
            jdim = 1;
            kdim = 2;
            ldim = 4;
            break;
        case 4:
            idim = 0;
            jdim = 1;
            kdim = 2;
            ldim = 3;
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
                    for (uint64_t l = 0; l < x->view->shape[ldim]; ++l)
                    {
                        switch (runtime_reduction_type)
                        {
                        case RUNTIME_SUMMATION:
                            switch (x->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim] + l * x->view->strides[ldim],
                                                result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim] + l * result->view->strides[ldim]);
                                break;
                            case MKL_RUNTIME:
                                mkl_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim] + l * x->view->strides[ldim],
                                            result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim] + l * result->view->strides[ldim]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_summation(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim] + l * x->view->strides[ldim],
                                            result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim] + l * result->view->strides[ldim]);
                                break;
#endif
                            default:
                                break;
                            }
                            break;
                        case RUNTIME_MAXIMUM:
                            switch (x->storage->runtime)
                            {
                            case OPENBLAS_RUNTIME:
                                openblas_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim] + l * x->view->strides[ldim],
                                                result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim] + l * result->view->strides[ldim]);
                                break;
                            case MKL_RUNTIME:
                                mkl_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim] + l * x->view->strides[ldim],
                                            result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim] + l * result->view->strides[ldim]);
                                break;
#ifndef CPU_ONLY
                            case CU_RUNTIME:
                                cu_maximum(x->storage->datatype, x->view->shape[axis], x->storage->data, x->view->strides[axis], x->view->offset + i * x->view->strides[idim] + j * x->view->strides[jdim] + k * x->view->strides[kdim] + l * x->view->strides[ldim],
                                        result->storage->data, result->view->offset + i * result->view->strides[idim] + j * result->view->strides[jdim] + k * result->view->strides[kdim] + l * result->view->strides[ldim]);
                                break;
#endif
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
