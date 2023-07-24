#include <buffer.h>
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

    // We can do this with CUDA runtime because of unified memory.
    // TODO: There are many cases where we avoid copying the data.
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
    case OPENBLAS_RUNTIME:
    case MKL_RUNTIME:
        // As far as I am aware we don't need to initialize a context with 
        // MKL or OpenBLAS.
        error = NULL;
        break;
    case CU_RUNTIME:
        // TODO: Allow for CUDA context configuration.
        error = cu_create_context();
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.", (int) runtime),
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
    case OPENBLAS_RUNTIME:
        error = openblas_memory_allocate(&buffer->data, size);
        break;
    case MKL_RUNTIME:
        error = mkl_memory_allocate(&buffer->data, size);
        break;
    case CU_RUNTIME:
        error = cu_memory_allocate(&buffer->data, size);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.", (int) buffer->runtime),
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
}

error_t *runtime_addition(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(z_buffer, "z_buffer");

    error_t *error;

    // TODO: Do they have to be the same datatype?
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

    // TODO: If shapes are not equal, broadcasting should be attempted.
    if (!view_shape_equal(x_buffer->view, y_buffer->view) ||
        !view_shape_equal(x_buffer->view, z_buffer->view))
    {
        // TODO: print tensor shapes would be quite helpful.
        // Dumping an arbitrary length shape into a string
        // that we can properly memory manage is awkward with this
        // error handling setup.
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting tensor shapes."),
                     NULL);
    }

    // TODO: Can we add tensors that are non-contiguous without rearranging
    // memory. If we can't we should apply a contiguous operation to
    // rearrange the memory instead of just returning an error.
    if (!view_is_contiguous(x_buffer->view) ||
        !view_is_contiguous(y_buffer->view) ||
        !view_is_contiguous(z_buffer->view))
    {
        return ERROR(ERROR_CONTIGUOUS,
                     string_create("not all tensors are contiguous."),
                     NULL);
    }

    // Mixing different backends should be avoided. Pytorch also returns an
    // error when performing operations between tensors that are on the
    // host memory and gpu memory.
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
                      string_create("unknown runtime %d.", (int) z_buffer->runtime),
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

error_t *runtime_matrix_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    // TODO: A lot of null checks are redundant and are here for debugging purposes
    // as the code base is expected to change significantly with different interactions
    // between functions. Later in developement when things are stable we will look at removing these.
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(z_buffer, "z_buffer");
    CHECK_NULL_ARGUMENT(x_buffer->view, "x_buffer->view");
    CHECK_NULL_ARGUMENT(y_buffer->view, "y_buffer->view");
    CHECK_NULL_ARGUMENT(z_buffer->view, "z_buffer->view");
    CHECK_NULL_ARGUMENT(x_buffer->view->shape, "x_buffer->view->shape");
    CHECK_NULL_ARGUMENT(y_buffer->view->shape, "y_buffer->view->shape");
    CHECK_NULL_ARGUMENT(z_buffer->view->shape, "z_buffer->view->shape");
    CHECK_NULL_ARGUMENT(x_buffer->view->strides, "x_buffer->view->strides");
    CHECK_NULL_ARGUMENT(y_buffer->view->strides, "y_buffer->view->strides");
    CHECK_NULL_ARGUMENT(z_buffer->view->strides, "z_buffer->view->strides");

    error_t *error;

    // TODO: Do they have to be the same datatype?
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

    // For now lets just support matrix multiplication of 2D tensors.
    // This is all we need for the feed forward neural network milestone.
    // The story will change in future milestones.
    // I'm not sure how broadcasting works for the matmul case.
    if (x_buffer->view->rank != 2 ||
        y_buffer->view->rank != 2 ||
        z_buffer->view->rank != 2)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("conflicting ranks not dimension 2 %u + %u = %u.",
                                   x_buffer->view->rank,
                                   y_buffer->view->rank,
                                   z_buffer->view->rank),
                     NULL);
    }

    // TODO: For a 2D tensor, non-contiguous could be because of a transpose.
    // We can avoid copying the data to be contiguous by utilizing
    // the transpose operation arguments in the BLAS function calls.
    // For simplicity, lets assume the tensors must be contiguous for now.
    if (!view_is_contiguous(x_buffer->view) ||
        !view_is_contiguous(y_buffer->view) ||
        !view_is_contiguous(z_buffer->view))
    {
        return ERROR(ERROR_CONTIGUOUS,
                     string_create("not all tensors are contiguous."),
                     NULL);
    }

    // By definition of matrix multiplication these need to be the shape constraints.
    // TODO: Multiple dimension tensors and investigate broadcasting.
    if (!(x_buffer->view->shape[1] == y_buffer->view->shape[0] &&
          z_buffer->view->shape[0] == x_buffer->view->shape[0] &&
          z_buffer->view->shape[1] == y_buffer->view->shape[1]))
    {
        // TODO: print tensor shapes.
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("conflicting tensor shapes."),
                     NULL);
    }
    uint32_t m = x_buffer->view->shape[0];
    uint32_t k = x_buffer->view->shape[1];
    uint32_t n = y_buffer->view->shape[1];

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
    case OPENBLAS_RUNTIME:
        error = openblas_matrix_multiplication(z_buffer->datatype, m, k, n, x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    case MKL_RUNTIME:
        error = mkl_matrix_multiplication(z_buffer->datatype, m, k, n, x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    case CU_RUNTIME:
        error = cu_matrix_multiplication(z_buffer->datatype, m, k, n, x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME,
                      string_create("unknown runtime %d.", (int) z_buffer->runtime),
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