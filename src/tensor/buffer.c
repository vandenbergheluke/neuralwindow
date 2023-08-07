#include <buffer.h>
#include <cu_runtime.h>
#include <mkl_runtime.h>
#include <openblas_runtime.h>

error_t *buffer_create(buffer_t **buffer, runtime_t runtime, datatype_t datatype, view_t *view, void *data, size_t size, bool_t new)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(view, "view");

    *buffer = (buffer_t *) malloc(sizeof(buffer_t));
    if (buffer == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate buffer of size %zu bytes.", sizeof(buffer_t)), NULL);
    }

    (*buffer)->runtime = runtime;
    (*buffer)->datatype = datatype;
    (*buffer)->view = view;
    (*buffer)->new = new;

    if (size == 0)
    {
        (*buffer)->size = shape_size((*buffer)->view->shape, (*buffer)->view->rank) * datatype_size((*buffer)->datatype);
    }
    else
    {
        (*buffer)->size = size;
    }


    error_t *error;

    if (new)
    {
        error = runtime_malloc(*buffer);
        if (error != NULL)
        {
            free(buffer);
            return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate buffer data."), error);
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

    if (buffer->new)
    {
        runtime_free(buffer);
    }
    view_destroy(buffer->view);
    free(buffer);
}

error_t *runtime_create_context(runtime_t runtime)
{
    error_t *error;
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
        error = ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) runtime), NULL);
        break;
    }
    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create context."), error);
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
    CHECK_NULL_ARGUMENT(buffer->view, "buffer->view");

    if (buffer->size == 0)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate 0 bytes."), NULL);
    }

    error_t *error;
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
        error = ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) buffer->runtime), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes for runtime %s.", buffer->size, runtime_string(buffer->runtime)), error);
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

error_t *runtime_copy(buffer_t *x_buffer, void *y_buffer)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(data, "data");

    error_t *error;
    uint32_t number_of_elements = shape_size(buffer->view->shape, buffer->view->rank);
    size_t element_size = datatype_size(buffer->datatype);
    size_t size = number_of_elements * element_size;

    if (data != NULL)
    {
        memcpy(buffer->data, data, size);
    }
    
    return NULL;
}

error_t *runtime_binary_elementwise(runtime_binary_elementwise_type_t runtime_binary_elementwise_type, buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
    if (x_buffer->datatype != y_buffer->datatype || x_buffer->datatype != z_buffer->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT, string_create("conflicting datatypes %s + %s = %s.",
                     datatype_string(x_buffer->datatype), datatype_string(y_buffer->datatype), datatype_string(z_buffer->datatype)), NULL);
    }

    if (!shapes_equal(x_buffer->view->shape, x_buffer->view->rank, y_buffer->view->shape, y_buffer->view->rank) || 
        !shapes_equal(x_buffer->view->shape, x_buffer->view->rank, z_buffer->view->shape, z_buffer->view->rank))
    {
        return ERROR(ERROR_SHAPE_CONFLICT, string_create("conflicting tensor shapes."), NULL);
    }

    if (x_buffer->runtime != y_buffer->runtime || x_buffer->runtime != z_buffer->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT, string_create("conflicting runtimes %s + %s = %s.",
                     runtime_string(x_buffer->runtime), runtime_string(y_buffer->runtime), runtime_string(z_buffer->runtime)), NULL);
    }

    error_t *error;
    if (is_contiguous(x_buffer->view->shape, x_buffer->view->rank, x_buffer->view->strides) &&
        is_contiguous(y_buffer->view->shape, y_buffer->view->rank, y_buffer->view->strides))
    {
        switch (runtime_binary_elementwise_type)
        {
        case RUNTIME_ADDITION:
            switch (z_buffer->runtime)
            {
            case OPENBLAS_RUNTIME:
                error = openblas_addition(z_buffer->datatype, shape_size(z_buffer->view->shape, z_buffer->view->rank), x_buffer->data, y_buffer->data, z_buffer->data);
                break;
            case MKL_RUNTIME:
                error = mkl_addition(z_buffer->datatype, shape_size(z_buffer->view->shape, z_buffer->view->rank), x_buffer->data, y_buffer->data, z_buffer->data);
                break;
            case CU_RUNTIME:
                error = cu_addition(z_buffer->datatype, shape_size(z_buffer->view->shape, z_buffer->view->rank), x_buffer->data, y_buffer->data, z_buffer->data);
                break;
            default:
                error = ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) z_buffer->runtime), NULL);
                break;
            }
            
            if (error != NULL)
            {
                return ERROR(ERROR_ADDITION, string_create("addition operation failed for runtime %s.", runtime_string(z_buffer->runtime)), error);
            }
            break;
        default:
            break;
        }
    }
    else
    {
        switch (z_buffer->view->rank)
        {
        case 2:
            for (uint32_t i = 0; i < z_buffer->view->shape[0]; i++)    
            {
                
            }
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




    return NULL;
}

error_t *runtime_matrix_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t *z_buffer)
{
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
    uint32_t m;
    uint32_t k;
    uint32_t n;
    bool_t x_transpose;
    bool_t y_transpose;

    if (x_buffer->datatype != y_buffer->datatype || x_buffer->datatype != z_buffer->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT, string_create("conflicting datatypes %s + %s = %s.",
                     datatype_string(x_buffer->datatype), datatype_string(y_buffer->datatype), datatype_string(z_buffer->datatype)), NULL);
    }

    if (x_buffer->view->rank != 2 || y_buffer->view->rank != 2 || z_buffer->view->rank != 2)
    {
        return ERROR(ERROR_RANK_CONFLICT, string_create("conflicting ranks not dimension 2 %u + %u = %u.", 
                     x_buffer->view->rank, y_buffer->view->rank, z_buffer->view->rank), NULL);
    }

    if (!(x_buffer->view->shape[1] == y_buffer->view->shape[0] &&
          z_buffer->view->shape[0] == x_buffer->view->shape[0] &&
          z_buffer->view->shape[1] == y_buffer->view->shape[1]))
    {
        return ERROR(ERROR_SHAPE_CONFLICT, string_create("conflicting tensor shapes."), NULL);
    }

    if (x_buffer->runtime != y_buffer->runtime || x_buffer->runtime != z_buffer->runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT, string_create("conflicting runtimes %s + %s = %s.",
                     runtime_string(x_buffer->runtime), runtime_string(y_buffer->runtime), runtime_string(z_buffer->runtime)), NULL);
    }

    m = x_buffer->view->shape[0];
    k = x_buffer->view->shape[1];
    n = y_buffer->view->shape[1];
    x_transpose = x_buffer->view->shape[1] == x_buffer->view->strides[0] && x_buffer->view->strides[1] == 1; 
    y_transpose = y_buffer->view->shape[1] == y_buffer->view->strides[0] && y_buffer->view->strides[1] == 1;  

    switch (z_buffer->runtime)
    {
    case OPENBLAS_RUNTIME:
        error = openblas_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose, x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    case MKL_RUNTIME:
        error = mkl_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose, x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    case CU_RUNTIME:
        error = cu_matrix_multiplication(z_buffer->datatype, m, k, n, x_transpose, y_transpose, x_buffer->data, y_buffer->data, z_buffer->data);
        break;
    default:
        error = ERROR(ERROR_UNKNOWN_RUNTIME, string_create("unknown runtime %d.", (int) z_buffer->runtime), NULL);
        break;
    }
    
    if (error != NULL)
    {
        return ERROR(ERROR_ADDITION, string_create("addition operation failed for runtime %s.", runtime_string(z_buffer->runtime)), error);
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