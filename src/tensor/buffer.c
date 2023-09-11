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

static void runtime_unary_execute(runtime_unary_type_t runtime_unary_type,
                                  runtime_t runtime,
                                  datatype_t datatype,
                                  uint64_t n,
                                  void *x_data,
                                  uint64_t x_stride,
                                  uint64_t x_offset,
                                  void *y_data,
                                  uint64_t y_stride,
                                  uint64_t y_offset)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        switch (runtime_unary_type)
        {
        case RUNTIME_EXPONENTIAL:
            openblas_exponential(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_LOGARITHM:
            openblas_logarithm(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_SINE:
            openblas_sine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_COSINE:
            openblas_cosine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_SQUARE_ROOT:
            openblas_square_root(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_CONTIGUOUS:
            openblas_copy(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_NEGATION:
            openblas_negation(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_RECTIFIED_LINEAR:
            openblas_rectified_linear(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_SIGMOID:
            openblas_sigmoid(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_RECIPROCAL:
            openblas_reciprocal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        default:
            break;
        }
        break;
    case MKL_RUNTIME:
        switch (runtime_unary_type)
        {
        case RUNTIME_EXPONENTIAL:
            mkl_exponential(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_LOGARITHM:
            mkl_logarithm(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_SINE:
            mkl_sine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_COSINE:
            mkl_cosine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_SQUARE_ROOT:
            mkl_square_root(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_CONTIGUOUS:
            mkl_copy(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_NEGATION:
            mkl_negation(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_RECTIFIED_LINEAR:
            mkl_rectified_linear(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_SIGMOID:
            mkl_sigmoid(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_RECIPROCAL:
            mkl_reciprocal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        default:
            break;
        }
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        switch (runtime_unary_type)
        {
        case RUNTIME_EXPONENTIAL:
            cu_exponential(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_LOGARITHM:
            cu_logarithm(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_SINE:
            cu_sine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_COSINE:
            cu_cosine(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_SQUARE_ROOT:
            cu_square_root(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_CONTIGUOUS:
            cu_copy(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_NEGATION:
            cu_negation(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_RECTIFIED_LINEAR:
            cu_rectified_linear(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_SIGMOID:
            cu_sigmoid(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        case RUNTIME_RECIPROCAL:
            cu_reciprocal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            break;
        default:
            break;
        }
        break;
#endif
    default:
        break;
    }
}

static nw_error_t *runtime_unary(runtime_unary_type_t runtime_unary_type,
                                 buffer_t *x_buffer, 
                                 buffer_t *y_buffer)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(x_buffer->view, "x_buffer->view");
    CHECK_NULL_ARGUMENT(y_buffer->view, "y_buffer->view");
    CHECK_NULL_ARGUMENT(x_buffer->view->strides, "x_buffer->view->strides");
    CHECK_NULL_ARGUMENT(y_buffer->view->strides, "y_buffer->view->strides");
    CHECK_NULL_ARGUMENT(x_buffer->view->shape, "x_buffer->view->shape");
    CHECK_NULL_ARGUMENT(y_buffer->view->shape, "y_buffer->view->shape");
    CHECK_NULL_ARGUMENT(x_buffer->storage, "x_buffer->storage");
    CHECK_NULL_ARGUMENT(y_buffer->storage, "y_buffer->storage");
    CHECK_NULL_ARGUMENT(x_buffer->storage->data, "x_buffer->storage->data");
    CHECK_NULL_ARGUMENT(y_buffer->storage->data, "y_buffer->storage->data");

    datatype_t datatype = y_buffer->storage->datatype;
    runtime_t runtime = y_buffer->storage->runtime;
    uint64_t rank = y_buffer->view->rank;
    void *x_data = x_buffer->storage->data;
    void *y_data = y_buffer->storage->data;

    uint64_t n;
    uint64_t x_stride;
    uint64_t y_stride;
    uint64_t x_offset;
    uint64_t y_offset;

    switch (rank)
    {
    case 0:
        n = 1;
        x_stride = 0;
        y_stride = 0;
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        runtime_unary_execute(runtime_unary_type, 
                              runtime, datatype, n, 
                              x_data, x_stride, x_offset, 
                              y_data, y_stride, y_offset);
        break;
    case 1:
        n = y_buffer->view->shape[0];
        x_stride = x_buffer->view->strides[0];
        y_stride = y_buffer->view->strides[0];
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        runtime_unary_execute(runtime_unary_type, 
                              runtime, datatype, n, 
                              x_data, x_stride, x_offset, 
                              y_data, y_stride, y_offset);
        break;
    case 2:
        for (uint64_t i = 0; i < y_buffer->view->shape[0]; ++i)
        {
            n = y_buffer->view->shape[1];
            x_stride = x_buffer->view->strides[1];
            y_stride = y_buffer->view->strides[1];
            x_offset = x_buffer->view->offset
                       + i * x_buffer->view->strides[0];
            y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[0];
            runtime_unary_execute(runtime_unary_type, 
                                  runtime, datatype, n, 
                                  x_data, x_stride, x_offset, 
                                  y_data, y_stride, y_offset);
        }
        break;
    case 3:
        for (uint64_t i = 0; i < y_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < y_buffer->view->shape[1]; ++j)
            {
                n = y_buffer->view->shape[2];
                x_stride = x_buffer->view->strides[2];
                y_stride = y_buffer->view->strides[2];
                x_offset = x_buffer->view->offset
                           + i * x_buffer->view->strides[0]
                           + j * x_buffer->view->strides[1];
                y_offset = y_buffer->view->offset
                           + i * y_buffer->view->strides[0]
                           + j * y_buffer->view->strides[1];

                runtime_unary_execute(runtime_unary_type, 
                                      runtime, datatype, n, 
                                      x_data, x_stride, x_offset, 
                                      y_data, y_stride, y_offset);
            }
        }
        break;
    case 4:
        for (uint64_t i = 0; i < y_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < y_buffer->view->shape[1]; ++j)
            {
                for (uint64_t k = 0; k < y_buffer->view->shape[2]; ++k)
                {
                    n = y_buffer->view->shape[3];
                    x_stride = x_buffer->view->strides[3];
                    y_stride = y_buffer->view->strides[3];
                    x_offset = x_buffer->view->offset
                               + i * x_buffer->view->strides[0]
                               + j * x_buffer->view->strides[1]
                               + k * x_buffer->view->strides[2];
                    y_offset = y_buffer->view->offset
                               + i * y_buffer->view->strides[0]
                               + j * y_buffer->view->strides[1]
                               + k * y_buffer->view->strides[2];
                    runtime_unary_execute(runtime_unary_type, 
                                          runtime, datatype, n, 
                                          x_data, x_stride, x_offset, 
                                          y_data, y_stride, y_offset);
                }
            }
        }
        break;
    case 5:
        for (uint64_t i = 0; i < y_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < y_buffer->view->shape[1]; ++j)
            {
                for (uint64_t k = 0; k < y_buffer->view->shape[2]; ++k)
                {
                    for (uint64_t l = 0; l < y_buffer->view->shape[3]; ++l)
                    {
                        n = y_buffer->view->shape[4];
                        x_stride = x_buffer->view->strides[4];
                        y_stride = y_buffer->view->strides[4];
                        x_offset = x_buffer->view->offset
                                   + i * x_buffer->view->strides[0]
                                   + j * x_buffer->view->strides[1]
                                   + k * x_buffer->view->strides[2]
                                   + l * x_buffer->view->strides[3];
                        y_offset = y_buffer->view->offset
                                   + i * y_buffer->view->strides[0]
                                   + j * y_buffer->view->strides[1]
                                   + k * y_buffer->view->strides[2]
                                   + l * y_buffer->view->strides[3];
                        runtime_unary_execute(runtime_unary_type, 
                                              runtime, datatype, n, 
                                              x_data, x_stride, x_offset, 
                                              y_data, y_stride, y_offset);
                    }
                }
            }
        }
        break;
    default:
        return ERROR(ERROR_RANK_CONFLICT, string_create("unsupported rank %d", (int) rank), NULL);
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

nw_error_t *runtime_sigmoid(buffer_t *x, buffer_t *result)
{
    nw_error_t *error = runtime_unary(RUNTIME_SIGMOID, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_UNARY, 
                     string_create("failed to apply unary operation."),
                     error);
    }

    return NULL;
}


static void runtime_binary_elementwise_execute(runtime_binary_elementwise_type_t runtime_binary_elementwise_type,
                                               runtime_t runtime,
                                               datatype_t datatype,
                                               uint64_t n,
                                               void *x_data,
                                               uint64_t x_stride,
                                               uint64_t x_offset,
                                               void *y_data,
                                               uint64_t y_stride,
                                               uint64_t y_offset,
                                               void *z_data,
                                               uint64_t z_stride,
                                               uint64_t z_offset)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        switch (runtime_binary_elementwise_type)
        {
        case RUNTIME_ADDITION:
            openblas_addition(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_SUBTRACTION:
            openblas_subtraction(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_MULTIPLICATION:
            openblas_multiplication(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_DIVISION:
            openblas_division(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_POWER:  
            openblas_power(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_COMPARE_EQUAL:
            openblas_compare_equal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_COMPARE_GREATER:
            openblas_compare_greater(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        default:
            break;
        }
        break;
    case MKL_RUNTIME:
        switch (runtime_binary_elementwise_type)
        {
        case RUNTIME_ADDITION:
            mkl_addition(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_SUBTRACTION:
            mkl_subtraction(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_MULTIPLICATION:
            mkl_multiplication(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_DIVISION:
            mkl_division(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_POWER:  
            mkl_power(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_COMPARE_EQUAL:
            mkl_compare_equal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_COMPARE_GREATER:
            mkl_compare_greater(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        default:
            break;
        }
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        switch (runtime_binary_elementwise_type)
        {
        case RUNTIME_ADDITION:
            cu_addition(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_SUBTRACTION:
            cu_subtraction(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_MULTIPLICATION:
            cu_multiplication(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_DIVISION:
            cu_division(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_POWER:  
            cu_power(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_COMPARE_EQUAL:
            cu_compare_equal(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        case RUNTIME_COMPARE_GREATER:
            cu_compare_greater(datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset, z_data, z_stride, z_offset);
            break;
        default:
            break;
        }
        break;
#endif
    default:
        break;
    }
}

static nw_error_t *runtime_binary_elementwise(runtime_binary_elementwise_type_t runtime_binary_elementwise_type, 
                                              buffer_t *x_buffer,
                                              buffer_t *y_buffer,
                                              buffer_t *z_buffer)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(z_buffer, "z_buffer");
    CHECK_NULL_ARGUMENT(x_buffer->view, "x_buffer->view");
    CHECK_NULL_ARGUMENT(y_buffer->view, "y_buffer->view");
    CHECK_NULL_ARGUMENT(z_buffer->view, "z_buffer->view");
    CHECK_NULL_ARGUMENT(x_buffer->view->strides, "x_buffer->view->strides");
    CHECK_NULL_ARGUMENT(y_buffer->view->strides, "y_buffer->view->strides");
    CHECK_NULL_ARGUMENT(z_buffer->view->strides, "z_buffer->view->strides");
    CHECK_NULL_ARGUMENT(x_buffer->view->shape, "x_buffer->view->shape");
    CHECK_NULL_ARGUMENT(y_buffer->view->shape, "y_buffer->view->shape");
    CHECK_NULL_ARGUMENT(z_buffer->view->shape, "z_buffer->view->shape");
    CHECK_NULL_ARGUMENT(x_buffer->storage, "x_buffer->storage");
    CHECK_NULL_ARGUMENT(y_buffer->storage, "y_buffer->storage");
    CHECK_NULL_ARGUMENT(z_buffer->storage, "z_buffer->storage");
    CHECK_NULL_ARGUMENT(x_buffer->storage->data, "x_buffer->storage->data");
    CHECK_NULL_ARGUMENT(y_buffer->storage->data, "y_buffer->storage->data");
    CHECK_NULL_ARGUMENT(z_buffer->storage->data, "z_buffer->storage->data");

    datatype_t datatype = z_buffer->storage->datatype;
    runtime_t runtime = z_buffer->storage->runtime;
    uint64_t rank = z_buffer->view->rank;
    void *x_data = x_buffer->storage->data;
    void *y_data = y_buffer->storage->data;
    void *z_data = z_buffer->storage->data;

    uint64_t n;
    uint64_t x_stride;
    uint64_t y_stride;
    uint64_t z_stride;
    uint64_t x_offset;
    uint64_t y_offset;
    uint64_t z_offset;

    switch (rank)
    {
    case 0:
        n = 1;
        x_stride = 0;
        y_stride = 0;
        z_stride = 0;
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        z_offset = z_buffer->view->offset;
        runtime_binary_elementwise_execute(runtime_binary_elementwise_type, 
                                           runtime, datatype, n, 
                                           x_data, x_stride, x_offset, 
                                           y_data, y_stride, y_offset,
                                           z_data, z_stride, z_offset);
        break;
    case 1:
        n = z_buffer->view->shape[0];
        x_stride = x_buffer->view->strides[0];
        y_stride = y_buffer->view->strides[0];
        z_stride = z_buffer->view->strides[0];
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        z_offset = z_buffer->view->offset;
        runtime_binary_elementwise_execute(runtime_binary_elementwise_type, 
                                           runtime, datatype, n, 
                                           x_data, x_stride, x_offset, 
                                           y_data, y_stride, y_offset,
                                           z_data, z_stride, z_offset);
        break;
    case 2:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            n = z_buffer->view->shape[1];
            x_stride = x_buffer->view->strides[1];
            y_stride = y_buffer->view->strides[1];
            z_stride = z_buffer->view->strides[1];
            x_offset = x_buffer->view->offset
                       + i * x_buffer->view->strides[0];
            y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[0];
            z_offset = z_buffer->view->offset
                       + i * z_buffer->view->strides[0];
            runtime_binary_elementwise_execute(runtime_binary_elementwise_type, 
                                               runtime, datatype, n, 
                                               x_data, x_stride, x_offset, 
                                               y_data, y_stride, y_offset,
                                               z_data, z_stride, z_offset);
        }
        break;
    case 3:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[1]; ++j)
            {
                n = z_buffer->view->shape[2];
                x_stride = x_buffer->view->strides[2];
                y_stride = y_buffer->view->strides[2];
                z_stride = z_buffer->view->strides[2];
                x_offset = x_buffer->view->offset
                           + i * x_buffer->view->strides[0]
                           + j * x_buffer->view->strides[1];
                y_offset = y_buffer->view->offset
                           + i * y_buffer->view->strides[0]
                           + j * y_buffer->view->strides[1];
                z_offset = z_buffer->view->offset
                           + i * z_buffer->view->strides[0]
                           + j * z_buffer->view->strides[1];

                runtime_binary_elementwise_execute(runtime_binary_elementwise_type, 
                                                   runtime, datatype, n, 
                                                   x_data, x_stride, x_offset, 
                                                   y_data, y_stride, y_offset,
                                                   z_data, z_stride, z_offset);
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
                    n = z_buffer->view->shape[3];
                    x_stride = x_buffer->view->strides[3];
                    y_stride = y_buffer->view->strides[3];
                    z_stride = z_buffer->view->strides[3];
                    x_offset = x_buffer->view->offset
                               + i * x_buffer->view->strides[0]
                               + j * x_buffer->view->strides[1]
                               + k * x_buffer->view->strides[2];
                    y_offset = y_buffer->view->offset
                               + i * y_buffer->view->strides[0]
                               + j * y_buffer->view->strides[1]
                               + k * y_buffer->view->strides[2];
                    z_offset = z_buffer->view->offset
                               + i * z_buffer->view->strides[0]
                               + j * z_buffer->view->strides[1]
                               + k * z_buffer->view->strides[2];

                    runtime_binary_elementwise_execute(runtime_binary_elementwise_type, 
                                                       runtime, datatype, n, 
                                                       x_data, x_stride, x_offset, 
                                                       y_data, y_stride, y_offset,
                                                       z_data, z_stride, z_offset);
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
                        n = z_buffer->view->shape[4];
                        x_stride = x_buffer->view->strides[4];
                        y_stride = y_buffer->view->strides[4];
                        z_stride = z_buffer->view->strides[4];
                        x_offset = x_buffer->view->offset
                                   + i * x_buffer->view->strides[0]
                                   + j * x_buffer->view->strides[1]
                                   + k * x_buffer->view->strides[2]
                                   + l * x_buffer->view->strides[3];
                        y_offset = y_buffer->view->offset
                                   + i * y_buffer->view->strides[0]
                                   + j * y_buffer->view->strides[1]
                                   + k * y_buffer->view->strides[2]
                                   + l * y_buffer->view->strides[3];
                        z_offset = z_buffer->view->offset
                                   + i * z_buffer->view->strides[0]
                                   + j * z_buffer->view->strides[1]
                                   + k * z_buffer->view->strides[2]
                                   + l * z_buffer->view->strides[3];

                        runtime_binary_elementwise_execute(runtime_binary_elementwise_type, 
                                                           runtime, datatype, n, 
                                                           x_data, x_stride, x_offset, 
                                                           y_data, y_stride, y_offset,
                                                           z_data, z_stride, z_offset);
                    }
                }
            }
        }
        break;
    default:
        return ERROR(ERROR_RANK_CONFLICT, 
                     string_create("unsupported rank %d",
                     (int) rank),
                     NULL);
    }

    return NULL;
}

nw_error_t *runtime_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_ADDITION, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, 
                     string_create("failed to apply binary elementwise operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_subtraction(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_SUBTRACTION, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, 
                     string_create("failed to apply binary elementwise operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_MULTIPLICATION, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, 
                     string_create("failed to apply binary elementwise operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_division(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_DIVISION, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, 
                     string_create("failed to apply binary elementwise operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_power(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_POWER, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, 
                     string_create("failed to apply binary elementwise operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_compare_equal(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_COMPARE_EQUAL, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, 
                     string_create("failed to apply binary elementwise operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_compare_greater(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    nw_error_t *error = runtime_binary_elementwise(RUNTIME_COMPARE_GREATER, x_buffer, y_buffer, z_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_BINARY_ELEMENTWISE, 
                     string_create("failed to apply binary elementwise operation."),
                     error);
    }

    return NULL;
}

static void runtime_matrix_multiplication_execute(runtime_t runtime,
                                                  datatype_t datatype,
                                                  uint64_t m,
                                                  uint64_t k,
                                                  uint64_t n,
                                                  bool_t x_transpose,
                                                  bool_t y_transpose,
                                                  void *x_data,
                                                  uint64_t x_offset,
                                                  void *y_data,
                                                  uint64_t y_offset,
                                                  void *z_data,
                                                  uint64_t z_offset)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        openblas_matrix_multiplication(datatype, m, k, n, 
                                       x_transpose, y_transpose,
                                       x_data, x_offset,
                                       y_data, y_offset,
                                       z_data, z_offset);
        break;
    case MKL_RUNTIME:
        mkl_matrix_multiplication(datatype, m, k, n, 
                                  x_transpose, y_transpose,
                                  x_data, x_offset,
                                  y_data, y_offset,
                                  z_data, z_offset);
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        cu_matrix_multiplication(datatype, m, k, n, 
                                 x_transpose, y_transpose,
                                 x_data, x_offset,
                                 y_data, y_offset,
                                 z_data, z_offset);
        break;
#endif
    default:
        break;
    }
}


nw_error_t *runtime_matrix_multiplication(buffer_t *x_buffer,
                                          buffer_t *y_buffer,
                                          buffer_t *z_buffer)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(z_buffer, "z_buffer");
    CHECK_NULL_ARGUMENT(x_buffer->view, "x_buffer->view");
    CHECK_NULL_ARGUMENT(y_buffer->view, "y_buffer->view");
    CHECK_NULL_ARGUMENT(z_buffer->view, "z_buffer->view");
    CHECK_NULL_ARGUMENT(x_buffer->view->strides, "x_buffer->view->strides");
    CHECK_NULL_ARGUMENT(y_buffer->view->strides, "y_buffer->view->strides");
    CHECK_NULL_ARGUMENT(z_buffer->view->strides, "z_buffer->view->strides");
    CHECK_NULL_ARGUMENT(x_buffer->view->shape, "x_buffer->view->shape");
    CHECK_NULL_ARGUMENT(y_buffer->view->shape, "y_buffer->view->shape");
    CHECK_NULL_ARGUMENT(z_buffer->view->shape, "z_buffer->view->shape");
    CHECK_NULL_ARGUMENT(x_buffer->storage, "x_buffer->storage");
    CHECK_NULL_ARGUMENT(y_buffer->storage, "y_buffer->storage");
    CHECK_NULL_ARGUMENT(z_buffer->storage, "z_buffer->storage");
    CHECK_NULL_ARGUMENT(x_buffer->storage->data, "x_buffer->storage->data");
    CHECK_NULL_ARGUMENT(y_buffer->storage->data, "y_buffer->storage->data");
    CHECK_NULL_ARGUMENT(z_buffer->storage->data, "z_buffer->storage->data");

    datatype_t datatype = z_buffer->storage->datatype;
    runtime_t runtime = z_buffer->storage->runtime;
    uint64_t rank = z_buffer->view->rank;
    void *x_data = x_buffer->storage->data;
    void *y_data = y_buffer->storage->data;
    void *z_data = z_buffer->storage->data;
    uint64_t m = x_buffer->view->shape[x_buffer->view->rank - 2];
    uint64_t k = x_buffer->view->shape[x_buffer->view->rank - 1];
    uint64_t n = y_buffer->view->shape[y_buffer->view->rank - 1];
    bool_t x_transpose = x_buffer->view->shape[x_buffer->view->rank - 2] == 
                         x_buffer->view->strides[x_buffer->view->rank - 1] && 
                         x_buffer->view->strides[x_buffer->view->rank - 2] == 1; 
    bool_t y_transpose = y_buffer->view->shape[y_buffer->view->rank - 2] == 
                         y_buffer->view->strides[y_buffer->view->rank - 1] && 
                         y_buffer->view->strides[y_buffer->view->rank - 2] == 1;  
    uint64_t x_offset;
    uint64_t y_offset;
    uint64_t z_offset;

    switch (rank)
    {
    case 2:
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        z_offset = z_buffer->view->offset;

        runtime_matrix_multiplication_execute(runtime, datatype, m, k, n,
                                              x_transpose, y_transpose, 
                                              x_data, x_offset, 
                                              y_data, y_offset,
                                              z_data, z_offset);
        break;
    case 3:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            x_offset = x_buffer->view->offset
                       + i * x_buffer->view->strides[0];
            y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[0];
            z_offset = z_buffer->view->offset
                       + i * z_buffer->view->strides[0];

            runtime_matrix_multiplication_execute(runtime, datatype, m, k, n,
                                                  x_transpose, y_transpose, 
                                                  x_data, x_offset, 
                                                  y_data, y_offset,
                                                  z_data, z_offset);
        }
        break;
    case 4:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[1]; ++j)
            {
                x_offset = x_buffer->view->offset
                           + i * x_buffer->view->strides[0]
                           + j * x_buffer->view->strides[1];
                y_offset = y_buffer->view->offset
                           + i * y_buffer->view->strides[0]
                           + j * y_buffer->view->strides[1];
                z_offset = z_buffer->view->offset
                           + i * z_buffer->view->strides[0]
                           + j * z_buffer->view->strides[1];

                runtime_matrix_multiplication_execute(runtime, datatype, m, k, n,
                                                      x_transpose, y_transpose, 
                                                      x_data, x_offset, 
                                                      y_data, y_offset,
                                                      z_data, z_offset);
            }
        }
        break;
    case 5:
        for (uint64_t i = 0; i < z_buffer->view->shape[0]; ++i)
        {
            for (uint64_t j = 0; j < z_buffer->view->shape[1]; ++j)
            {
                for (uint64_t l = 0; l < z_buffer->view->shape[2]; ++l)
                {
                    x_offset = x_buffer->view->offset
                               + i * x_buffer->view->strides[0]
                               + j * x_buffer->view->strides[1]
                               + l * x_buffer->view->strides[2];
                    y_offset = y_buffer->view->offset
                               + i * y_buffer->view->strides[0]
                               + j * y_buffer->view->strides[1]
                               + l * y_buffer->view->strides[2];
                    z_offset = z_buffer->view->offset
                               + i * z_buffer->view->strides[0]
                               + j * z_buffer->view->strides[1]
                               + l * z_buffer->view->strides[2];

                    runtime_matrix_multiplication_execute(runtime, datatype, m, k, n,
                                                          x_transpose, y_transpose, 
                                                          x_data, x_offset, 
                                                          y_data, y_offset,
                                                          z_data, z_offset);
                }
            }
        }
        break;
    default:
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("unsupported rank %d",
                     (int) z_buffer->view->rank),
                     NULL);
    }

    return NULL;
}

static void runtime_reduction_execute(runtime_reduction_type_t runtime_reduction_type,
                                      runtime_t runtime,
                                      datatype_t datatype,
                                      uint64_t n,
                                      void *x_data,
                                      uint64_t x_stride,
                                      uint64_t x_offset,
                                      void *y_data,
                                      uint64_t y_offset)
{
    switch (runtime)
    {
    case OPENBLAS_RUNTIME:
        switch (runtime_reduction_type)
        {
        case RUNTIME_MAXIMUM:
            openblas_maximum(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        case RUNTIME_SUMMATION:
            openblas_summation(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        default:
            break;
        }
        break;
    case MKL_RUNTIME:
        switch (runtime_reduction_type)
        {
        case RUNTIME_MAXIMUM:
            mkl_maximum(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        case RUNTIME_SUMMATION:
            mkl_summation(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        default:
            break;
        }
        break;
#ifndef CPU_ONLY
    case CU_RUNTIME:
        switch (runtime_reduction_type)
        {
        case RUNTIME_MAXIMUM:
            cu_maximum(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        case RUNTIME_SUMMATION:
            cu_summation(datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
            break;
        default:
            break;
        }
        break;
#endif
    default:
        break;
    }

}

static nw_error_t *runtime_reduction(runtime_reduction_type_t runtime_reduction_type,
                                     buffer_t *x_buffer,
                                     buffer_t *y_buffer,
                                     uint64_t axis)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(x_buffer->view, "x_buffer->view");
    CHECK_NULL_ARGUMENT(y_buffer->view, "y_buffer->view");
    CHECK_NULL_ARGUMENT(x_buffer->view->strides, "x_buffer->view->strides");
    CHECK_NULL_ARGUMENT(y_buffer->view->strides, "y_buffer->view->strides");
    CHECK_NULL_ARGUMENT(x_buffer->view->shape, "x_buffer->view->shape");
    CHECK_NULL_ARGUMENT(y_buffer->view->shape, "y_buffer->view->shape");
    CHECK_NULL_ARGUMENT(x_buffer->storage, "x_buffer->storage");
    CHECK_NULL_ARGUMENT(y_buffer->storage, "y_buffer->storage");
    CHECK_NULL_ARGUMENT(x_buffer->storage->data, "x_buffer->storage->data");
    CHECK_NULL_ARGUMENT(y_buffer->storage->data, "y_buffer->storage->data");

    uint64_t idim;
    uint64_t jdim;
    uint64_t kdim;
    uint64_t ldim;

    datatype_t datatype = x_buffer->storage->datatype;
    runtime_t runtime = x_buffer->storage->runtime;
    uint64_t rank = x_buffer->view->rank;
    uint64_t n = x_buffer->view->shape[axis];
    void *x_data = x_buffer->storage->data;
    void *y_data = y_buffer->storage->data;
    uint64_t x_stride = x_buffer->view->strides[axis];
    uint64_t x_offset;
    uint64_t y_offset;

    switch (rank)
    {
    case 1:
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        runtime_reduction_execute(runtime_reduction_type, runtime, datatype, n, 
                                  x_data, x_stride, x_offset, 
                                  y_data, y_offset);
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
        
        for (uint64_t i = 0; i < x_buffer->view->shape[idim]; ++i)
        {
            x_offset = x_buffer->view->offset
                       + i * x_buffer->view->strides[idim];
            y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[idim];
            runtime_reduction_execute(runtime_reduction_type, runtime, datatype, n, 
                                    x_data, x_stride, x_offset, 
                                    y_data, y_offset);
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
        
        for (uint64_t i = 0; i < x_buffer->view->shape[idim]; ++i)
        {
            for (uint64_t j = 0; j < x_buffer->view->shape[jdim]; ++j)
            {
                x_offset = x_buffer->view->offset
                           + i * x_buffer->view->strides[idim]
                           + j * x_buffer->view->strides[jdim];
                y_offset = y_buffer->view->offset
                           + i * y_buffer->view->strides[idim]
                           + j * y_buffer->view->strides[jdim];
                runtime_reduction_execute(runtime_reduction_type, runtime, datatype, n, 
                                          x_data, x_stride, x_offset, 
                                          y_data, y_offset);
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
        
        for (uint64_t i = 0; i < x_buffer->view->shape[idim]; ++i)
        {
            for (uint64_t j = 0; j < x_buffer->view->shape[jdim]; ++j)
            {
                for (uint64_t k = 0; k < x_buffer->view->shape[kdim]; ++k)
                {
                    x_offset = x_buffer->view->offset
                               + i * x_buffer->view->strides[idim]
                               + j * x_buffer->view->strides[jdim]
                               + k * x_buffer->view->strides[kdim];
                    y_offset = y_buffer->view->offset
                               + i * y_buffer->view->strides[idim]
                               + j * y_buffer->view->strides[jdim]
                               + k * y_buffer->view->strides[kdim];
                    runtime_reduction_execute(runtime_reduction_type, runtime, datatype, n, 
                                              x_data, x_stride, x_offset, 
                                              y_data, y_offset);
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
        
        for (uint64_t i = 0; i < x_buffer->view->shape[idim]; ++i)
        {
            for (uint64_t j = 0; j < x_buffer->view->shape[jdim]; ++j)
            {
                for (uint64_t k = 0; k < x_buffer->view->shape[kdim]; ++k)
                {
                    for (uint64_t l = 0; l < x_buffer->view->shape[ldim]; ++l)
                    {
                        x_offset = x_buffer->view->offset
                                   + i * x_buffer->view->strides[idim]
                                   + j * x_buffer->view->strides[jdim]
                                   + k * x_buffer->view->strides[kdim]
                                   + l * x_buffer->view->strides[ldim];
                        y_offset = y_buffer->view->offset
                                   + i * y_buffer->view->strides[idim]
                                   + j * y_buffer->view->strides[jdim]
                                   + k * y_buffer->view->strides[kdim]
                                   + l * y_buffer->view->strides[ldim];
                        runtime_reduction_execute(runtime_reduction_type, runtime, datatype, n, 
                                                  x_data, x_stride, x_offset, 
                                                  y_data, y_offset);
                    }
                }
            }
        }
        break;
    default:
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("unsupported rank %d",
                     (int) x_buffer->view->rank),
                     NULL);
    }

    return NULL;
}

nw_error_t *runtime_summation(buffer_t *x, buffer_t *result, uint64_t axis)
{
    nw_error_t *error = runtime_reduction(RUNTIME_SUMMATION, x, result, axis);
    if (error != NULL)
    {
        return ERROR(ERROR_REDUCTION,
                     string_create("failed to apply reduction operation."),
                     error);
    }

    return NULL;
}

nw_error_t *runtime_maximum(buffer_t *x, buffer_t *result, uint64_t axis)
{
    nw_error_t *error = runtime_reduction(RUNTIME_MAXIMUM, x, result, axis);
    if (error != NULL)
    {
        return ERROR(ERROR_REDUCTION,
                     string_create("failed to apply reduction operation."),
                     error);
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
