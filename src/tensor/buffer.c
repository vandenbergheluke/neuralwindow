#include <buffer.h>
#include <view.h>
#include <string.h>
#include <sort.h>

nw_error_t *storage_create(storage_t **storage, runtime_t runtime, datatype_t datatype, int64_t n, void *data, bool_t copy)
{
    CHECK_NULL_ARGUMENT(storage, "storage");

    *storage = (storage_t *) malloc(sizeof(storage_t));
    if (!*storage)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(storage_t)), NULL);
    }

    if (!n)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, "storage must store more than 0 bytes of data.", NULL);
    }

    (*storage)->runtime = runtime;
    (*storage)->datatype = datatype;
    (*storage)->n = n;
    (*storage)->reference_count = 0;

    if (copy)
    {
        nw_error_t *error = runtime_malloc(&(*storage)->data, n, datatype, runtime);
        if (error)
        {
            free(*storage);
            return ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocate buffer data for runtime %s and datatype %s.",
                         runtime_string(runtime), datatype_string(datatype)), error);
        }

        runtime_synchronize(runtime);

        if (data)
        {
            memcpy((*storage)->data, data, (*storage)->n * datatype_size(datatype));
        }

    }
    else
    {
        (*storage)->data = data;
    }


    return NULL;
}

void storage_destroy(storage_t *storage)
{
    if (storage)
    {
        if (storage->reference_count < 2)
        {
            runtime_free(storage->data, storage->runtime);
            free(storage);
        }
        else
        {
            --(storage->reference_count);
        }
    }
}

nw_error_t *buffer_create(buffer_t **buffer, view_t *view, storage_t *storage, bool_t copy)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(view, "view");
    CHECK_NULL_ARGUMENT(storage, "storage");

    nw_error_t *error = NULL;

    *buffer = (buffer_t *) malloc(sizeof(buffer_t));
    if (!*buffer)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate buffer of size %zu bytes.", sizeof(buffer_t)), NULL);
    }

    (*buffer)->view = view;

    if (copy)
    {
        error = storage_create(&(*buffer)->storage, storage->runtime, storage->datatype, storage->n, storage->data, copy);
        if (error)
        {
            free(*buffer);
            return ERROR(ERROR_CREATE, string_create("failed to create storage copy."), error);
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
    if (buffer)
    {
        storage_destroy(buffer->storage);
        view_destroy(buffer->view);
        free(buffer);
    }
}

nw_error_t *storage_save(storage_t *storage, FILE *file)
{
    CHECK_NULL_ARGUMENT(storage, "storage");
    CHECK_NULL_ARGUMENT(file, "file");

    if (!fwrite(&storage->n, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&storage->runtime, sizeof(runtime_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&storage->datatype, sizeof(datatype_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(storage->data, datatype_size(storage->datatype), storage->n, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    return NULL;
}

nw_error_t *storage_load(storage_t **storage, FILE *file)
{
    CHECK_NULL_ARGUMENT(storage, "storage");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *storage = (storage_t *) malloc(sizeof(storage_t));
    if (!*storage)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(storage_t)), NULL);
        goto cleanup;
    }

    (*storage)->reference_count = 0;
    (*storage)->data = NULL;

    if (!fread(&(*storage)->n, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*storage)->runtime, sizeof(runtime_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*storage)->datatype, sizeof(datatype_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    error = runtime_malloc(&(*storage)->data, (*storage)->n, (*storage)->datatype, (*storage)->runtime);
    if (error)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION,
                      string_create("failed to allocate buffer data for runtime %s and datatype %s.",
                                    runtime_string((*storage)->runtime), datatype_string((*storage)->datatype)),
                      error);
        goto cleanup;
    }

    if (!fread((*storage)->data, datatype_size((*storage)->datatype), (*storage)->n, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    (*storage)->reference_count = 1;

    return error;

cleanup:

    error_print(error);
    storage_destroy(*storage);

    return error;
}

nw_error_t *buffer_save(buffer_t *buffer, FILE *file)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    error = view_save(buffer->view, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save view."), error);
    }

    error = storage_save(buffer->storage, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save view."), error);
    }

    return error;
}

nw_error_t *buffer_load(buffer_t **buffer, FILE *file)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *buffer = (buffer_t *) malloc(sizeof(buffer_t));
    if (!*buffer)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate buffer of size %zu bytes.", sizeof(buffer_t)), NULL);
        goto cleanup;
    }

    (*buffer)->view = NULL;
    (*buffer)->storage = NULL;

    error = view_load(&(*buffer)->view, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load view."), error);
        goto cleanup;
    }

    error = storage_load(&(*buffer)->storage, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load storage."), error);
        goto cleanup;
    }

    return error;

cleanup:

    buffer_destroy(*buffer);

    return error;
}

nw_error_t *buffer_unary(unary_operation_type_t unary_operation_type, buffer_t *x_buffer, buffer_t **y_buffer)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(x_buffer->view, "x_buffer->view");
    CHECK_NULL_ARGUMENT(x_buffer->view->strides, "x_buffer->view->strides");
    CHECK_NULL_ARGUMENT(x_buffer->view->shape, "x_buffer->view->shape");
    CHECK_NULL_ARGUMENT(x_buffer->storage, "x_buffer->storage");
    CHECK_NULL_ARGUMENT(x_buffer->storage->data, "x_buffer->storage->data");

    nw_error_t *error = NULL;
    bool_t overwrite = (bool_t) *y_buffer;

    if (unary_operation_type == AS_OPERATION)
    {
        view_t *view = NULL;

        error = view_copy(x_buffer->view, &view);
        if (error)
        {
            return ERROR(ERROR_COPY, string_create("failed to copy view."), error);
        }

        error = buffer_create(y_buffer, view, x_buffer->storage, false);
        if (error)
        {
            view_destroy(view);
            return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
        }

        return error;
    }

    if (!overwrite)
    {
        error = buffer_creation(EMPTY_OPERATION, y_buffer, x_buffer->view->shape, x_buffer->view->rank, NULL, 
                                0, x_buffer->storage->runtime, x_buffer->storage->datatype, NULL, 0, 0);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
            goto cleanup;
        }
    }

    datatype_t datatype = (*y_buffer)->storage->datatype;
    runtime_t runtime = (*y_buffer)->storage->runtime;
    int64_t rank = (*y_buffer)->view->rank;
    void *x_data = x_buffer->storage->data;
    void *y_data = (*y_buffer)->storage->data;

    int64_t n;
    int64_t x_stride;
    int64_t y_stride;
    int64_t x_offset;
    int64_t y_offset;

    switch (rank)
    {
    case 0:
        n = 1;
        x_stride = 0;
        y_stride = 0;
        x_offset = x_buffer->view->offset;
        y_offset = (*y_buffer)->view->offset;
        runtime_unary(unary_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
        break;
    case 1:
        n = (*y_buffer)->view->shape[0];
        x_stride = x_buffer->view->strides[0];
        y_stride = (*y_buffer)->view->strides[0];
        x_offset = x_buffer->view->offset;
        y_offset = (*y_buffer)->view->offset;
        runtime_unary(unary_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
        break;
    case 2:
        for (int64_t i = 0; i < (*y_buffer)->view->shape[0]; ++i)
        {
            n = (*y_buffer)->view->shape[1];
            x_stride = x_buffer->view->strides[1];
            y_stride = (*y_buffer)->view->strides[1];
            x_offset = x_buffer->view->offset
                       + i * x_buffer->view->strides[0];
            y_offset = (*y_buffer)->view->offset
                       + i * (*y_buffer)->view->strides[0];
            runtime_unary(unary_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
        }
        break;
    case 3:
        for (int64_t i = 0; i < (*y_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*y_buffer)->view->shape[1]; ++j)
            {
                n = (*y_buffer)->view->shape[2];
                x_stride = x_buffer->view->strides[2];
                y_stride = (*y_buffer)->view->strides[2];
                x_offset = x_buffer->view->offset
                           + i * x_buffer->view->strides[0]
                           + j * x_buffer->view->strides[1];
                y_offset = (*y_buffer)->view->offset
                           + i * (*y_buffer)->view->strides[0]
                           + j * (*y_buffer)->view->strides[1];
                runtime_unary(unary_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
            }
        }
        break;
    case 4:
        for (int64_t i = 0; i < (*y_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*y_buffer)->view->shape[1]; ++j)
            {
                for (int64_t k = 0; k < (*y_buffer)->view->shape[2]; ++k)
                {
                    n = (*y_buffer)->view->shape[3];
                    x_stride = x_buffer->view->strides[3];
                    y_stride = (*y_buffer)->view->strides[3];
                    x_offset = x_buffer->view->offset
                               + i * x_buffer->view->strides[0]
                               + j * x_buffer->view->strides[1]
                               + k * x_buffer->view->strides[2];
                    y_offset = (*y_buffer)->view->offset
                               + i * (*y_buffer)->view->strides[0]
                               + j * (*y_buffer)->view->strides[1]
                               + k * (*y_buffer)->view->strides[2];
                    runtime_unary(unary_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
                }
            }
        }
        break;
    case 5:
        for (int64_t i = 0; i < (*y_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*y_buffer)->view->shape[1]; ++j)
            {
                for (int64_t k = 0; k < (*y_buffer)->view->shape[2]; ++k)
                {
                    for (int64_t l = 0; l < (*y_buffer)->view->shape[3]; ++l)
                    {
                        n = (*y_buffer)->view->shape[4];
                        x_stride = x_buffer->view->strides[4];
                        y_stride = (*y_buffer)->view->strides[4];
                        x_offset = x_buffer->view->offset
                                   + i * x_buffer->view->strides[0]
                                   + j * x_buffer->view->strides[1]
                                   + k * x_buffer->view->strides[2]
                                   + l * x_buffer->view->strides[3];
                        y_offset = (*y_buffer)->view->offset
                                   + i * (*y_buffer)->view->strides[0]
                                   + j * (*y_buffer)->view->strides[1]
                                   + k * (*y_buffer)->view->strides[2]
                                   + l * (*y_buffer)->view->strides[3];
                        runtime_unary(unary_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_stride, y_offset);
                    }
                }
            }
        }
        break;
    default:
        error = ERROR(ERROR_RANK, string_create("unsupported rank %d", (int) rank), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    if (!overwrite)
    {
        buffer_destroy(*y_buffer);
    }

    return error;
}

static nw_error_t *buffer_matrix_multiplication(buffer_t *x_buffer, buffer_t *y_buffer, buffer_t **z_buffer)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(z_buffer, "z_buffer");
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

    nw_error_t *error = NULL;
    bool_t overwrite = (bool_t) *z_buffer;
    view_t *view = NULL;
    runtime_t runtime;
    datatype_t datatype;

    if (x_buffer->storage->datatype != y_buffer->storage->datatype)
    {
        error = ERROR(ERROR_DATATYPE, string_create("datatypes are incompatible."), NULL);
        goto cleanup;
    }
    else
    {
        datatype = x_buffer->storage->datatype;
    }

    if (x_buffer->storage->runtime != y_buffer->storage->runtime)
    {
        error = ERROR(ERROR_RUNTIME, string_create("runtimes are incompatible."), NULL);
        goto cleanup;
    }
    else
    {
        runtime = x_buffer->storage->runtime;
    }

    if (!overwrite)
    {
        error = view_matrix_multiplication(x_buffer->view, y_buffer->view, &view);
        if (error)
        {
            error = ERROR(ERROR_SHAPE, string_create("incompatible shapes for matrix multiplication."), error);
            goto cleanup;
        }

        error = buffer_creation(EMPTY_OPERATION, z_buffer, view->shape, view->rank,
                                view->strides, view->offset, runtime, datatype, NULL, 0, NULL);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
            goto cleanup;
        }
    }

    void *x_data = x_buffer->storage->data;
    void *y_data = y_buffer->storage->data;
    void *z_data = (*z_buffer)->storage->data;
    int64_t m = x_buffer->view->shape[x_buffer->view->rank - 2];
    int64_t k = x_buffer->view->shape[x_buffer->view->rank - 1];
    int64_t n = y_buffer->view->shape[y_buffer->view->rank - 1];
    bool_t x_transpose = false; 
    bool_t y_transpose = false;  
    int64_t x_offset;
    int64_t y_offset;
    int64_t z_offset;

    switch ((*z_buffer)->view->rank)
    {
    case 2:
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        z_offset = (*z_buffer)->view->offset;
        runtime_matrix_multiplication(runtime, datatype, m, k, n,
                                      x_transpose, y_transpose, 
                                      x_data, x_offset, 
                                      y_data, y_offset,
                                      z_data, z_offset);
        break;
    case 3:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            x_offset = x_buffer->view->offset
                       + i * x_buffer->view->strides[0];
            y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[0];
            z_offset = (*z_buffer)->view->offset
                       + i * (*z_buffer)->view->strides[0];
            runtime_matrix_multiplication(runtime, datatype, m, k, n,
                                          x_transpose, y_transpose, 
                                          x_data, x_offset, 
                                          y_data, y_offset,
                                          z_data, z_offset);
        }
        break;
    case 4:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*z_buffer)->view->shape[1]; ++j)
            {
                x_offset = x_buffer->view->offset
                           + i * x_buffer->view->strides[0]
                           + j * x_buffer->view->strides[1];
                y_offset = y_buffer->view->offset
                           + i * y_buffer->view->strides[0]
                           + j * y_buffer->view->strides[1];
                z_offset = (*z_buffer)->view->offset
                           + i * (*z_buffer)->view->strides[0]
                           + j * (*z_buffer)->view->strides[1];
                runtime_matrix_multiplication(runtime, datatype, m, k, n,
                                              x_transpose, y_transpose, 
                                              x_data, x_offset, 
                                              y_data, y_offset,
                                              z_data, z_offset);
            }
        }
        break;
    case 5:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*z_buffer)->view->shape[1]; ++j)
            {
                for (int64_t l = 0; l < (*z_buffer)->view->shape[2]; ++l)
                {
                    x_offset = x_buffer->view->offset
                               + i * x_buffer->view->strides[0]
                               + j * x_buffer->view->strides[1]
                               + l * x_buffer->view->strides[2];
                    y_offset = y_buffer->view->offset
                               + i * y_buffer->view->strides[0]
                               + j * y_buffer->view->strides[1]
                               + l * y_buffer->view->strides[2];
                    z_offset = (*z_buffer)->view->offset
                               + i * (*z_buffer)->view->strides[0]
                               + j * (*z_buffer)->view->strides[1]
                               + l * (*z_buffer)->view->strides[2];
                    runtime_matrix_multiplication(runtime, datatype, m, k, n,
                                                  x_transpose, y_transpose, 
                                                  x_data, x_offset, 
                                                  y_data, y_offset,
                                                  z_data, z_offset);
                }
            }
        }
        break;
    default:
        error = ERROR(ERROR_RANK, string_create("unsupported rank %d", (int) (*z_buffer)->view->rank), NULL);
        goto cleanup;
    }

    view_destroy(view);

    return error;

cleanup:

    if (!overwrite)
    {
        buffer_destroy(*z_buffer);
    }

    view_destroy(view);

    return error;
}

static nw_error_t *buffer_binary_elementwise(binary_operation_type_t binary_operation_type, buffer_t *x_buffer, buffer_t *y_buffer, buffer_t **z_buffer)
{
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(z_buffer, "z_buffer");
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

    nw_error_t *error = NULL;
    bool_t overwrite = (bool_t) *z_buffer;
    int64_t rank = MAX(x_buffer->view->rank, y_buffer->view->rank);
    int64_t shape[rank];
    runtime_t runtime;
    datatype_t datatype;

    if (x_buffer->storage->datatype != y_buffer->storage->datatype)
    {
        error = ERROR(ERROR_DATATYPE, string_create("datatypes are incompatible."), NULL);
        goto cleanup;
    }
    else
    {
        datatype = x_buffer->storage->datatype;
    }

    if (x_buffer->storage->runtime != y_buffer->storage->runtime)
    {
        error = ERROR(ERROR_RUNTIME, string_create("runtimes are incompatible."), NULL);
        goto cleanup;
    }
    else
    {
        runtime = x_buffer->storage->runtime;
    }

    if (!overwrite)
    {
        if (!view_shapes_equal(x_buffer->view, y_buffer->view))
        {
            error = ERROR(ERROR_SHAPE, string_create("incompatible tensor shapes."), NULL);
            goto cleanup;
        }
        else
        {
            memcpy(shape, x_buffer->view->shape, rank * sizeof(int64_t));
        }

        error = buffer_creation(EMPTY_OPERATION, z_buffer, shape, rank, NULL, 0, runtime, datatype, NULL, 0, NULL);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
            goto cleanup;
        }
    }

    void *x_data = x_buffer->storage->data;
    void *y_data = y_buffer->storage->data;
    void *z_data = (*z_buffer)->storage->data;

    int64_t n;
    int64_t x_stride;
    int64_t y_stride;
    int64_t z_stride;
    int64_t x_offset;
    int64_t y_offset;
    int64_t z_offset;

    switch (rank)
    {
    case 0:
        n = 1;
        x_stride = 0;
        y_stride = 0;
        z_stride = 0;
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        z_offset = (*z_buffer)->view->offset;
        runtime_binary_elementwise(binary_operation_type, 
                                   runtime, datatype, n, 
                                   x_data, x_stride, x_offset, 
                                   y_data, y_stride, y_offset,
                                   z_data, z_stride, z_offset);
        break;
    case 1:
        n = (*z_buffer)->view->shape[0];
        x_stride = x_buffer->view->strides[0];
        y_stride = y_buffer->view->strides[0];
        z_stride = (*z_buffer)->view->strides[0];
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        z_offset = (*z_buffer)->view->offset;
        runtime_binary_elementwise(binary_operation_type, 
                                   runtime, datatype, n, 
                                   x_data, x_stride, x_offset, 
                                   y_data, y_stride, y_offset,
                                   z_data, z_stride, z_offset);
        break;
    case 2:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            n = (*z_buffer)->view->shape[1];
            x_stride = x_buffer->view->strides[1];
            y_stride = y_buffer->view->strides[1];
            z_stride = (*z_buffer)->view->strides[1];
            x_offset = x_buffer->view->offset
                       + i * x_buffer->view->strides[0];
            y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[0];
            z_offset = (*z_buffer)->view->offset
                       + i * (*z_buffer)->view->strides[0];
            runtime_binary_elementwise(binary_operation_type, 
                                       runtime, datatype, n, 
                                       x_data, x_stride, x_offset, 
                                       y_data, y_stride, y_offset,
                                       z_data, z_stride, z_offset);
        }
        break;
    case 3:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*z_buffer)->view->shape[1]; ++j)
            {
                n = (*z_buffer)->view->shape[2];
                x_stride = x_buffer->view->strides[2];
                y_stride = y_buffer->view->strides[2];
                z_stride = (*z_buffer)->view->strides[2];
                x_offset = x_buffer->view->offset
                           + i * x_buffer->view->strides[0]
                           + j * x_buffer->view->strides[1];
                y_offset = y_buffer->view->offset
                           + i * y_buffer->view->strides[0]
                           + j * y_buffer->view->strides[1];
                z_offset = (*z_buffer)->view->offset
                           + i * (*z_buffer)->view->strides[0]
                           + j * (*z_buffer)->view->strides[1];

                runtime_binary_elementwise(binary_operation_type, 
                                           runtime, datatype, n, 
                                           x_data, x_stride, x_offset, 
                                           y_data, y_stride, y_offset,
                                           z_data, z_stride, z_offset);
            }
        }
        break;
    case 4:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*z_buffer)->view->shape[1]; ++j)
            {
                for (int64_t k = 0; k < (*z_buffer)->view->shape[2]; ++k)
                {
                    n = (*z_buffer)->view->shape[3];
                    x_stride = x_buffer->view->strides[3];
                    y_stride = y_buffer->view->strides[3];
                    z_stride = (*z_buffer)->view->strides[3];
                    x_offset = x_buffer->view->offset
                               + i * x_buffer->view->strides[0]
                               + j * x_buffer->view->strides[1]
                               + k * x_buffer->view->strides[2];
                    y_offset = y_buffer->view->offset
                               + i * y_buffer->view->strides[0]
                               + j * y_buffer->view->strides[1]
                               + k * y_buffer->view->strides[2];
                    z_offset = (*z_buffer)->view->offset
                               + i * (*z_buffer)->view->strides[0]
                               + j * (*z_buffer)->view->strides[1]
                               + k * (*z_buffer)->view->strides[2];

                    runtime_binary_elementwise(binary_operation_type, 
                                               runtime, datatype, n, 
                                               x_data, x_stride, x_offset, 
                                               y_data, y_stride, y_offset,
                                               z_data, z_stride, z_offset);
                }
            }
        }
        break;
    case 5:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*z_buffer)->view->shape[1]; ++j)
            {
                for (int64_t k = 0; k < (*z_buffer)->view->shape[2]; ++k)
                {
                    for (int64_t l = 0; l < (*z_buffer)->view->shape[3]; ++l)
                    {
                        n = (*z_buffer)->view->shape[4];
                        x_stride = x_buffer->view->strides[4];
                        y_stride = y_buffer->view->strides[4];
                        z_stride = (*z_buffer)->view->strides[4];
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
                        z_offset = (*z_buffer)->view->offset
                                   + i * (*z_buffer)->view->strides[0]
                                   + j * (*z_buffer)->view->strides[1]
                                   + k * (*z_buffer)->view->strides[2]
                                   + l * (*z_buffer)->view->strides[3];

                        runtime_binary_elementwise(binary_operation_type, 
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
        error = ERROR(ERROR_RANK, string_create("unsupported rank %d", (int) rank), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    if (!overwrite)
    {
        buffer_destroy(*z_buffer);
    }

    return error;
}

nw_error_t *buffer_binary(binary_operation_type_t operation_type, buffer_t *x_buffer, buffer_t *y_buffer, buffer_t **z_buffer)
{
    nw_error_t *error = NULL;

    switch (operation_type)
    {
    case ADDITION_OPERATION:
    case SUBTRACTION_OPERATION:
    case MULTIPLICATION_OPERATION:
    case DIVISION_OPERATION:
    case POWER_OPERATION:
    case COMPARE_EQUAL_OPERATION:
    case COMPARE_GREATER_OPERATION:
        error = buffer_binary_elementwise(operation_type, x_buffer, y_buffer, z_buffer);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        error = buffer_matrix_multiplication(x_buffer, y_buffer, z_buffer);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_BINARY, string_create("failed binary operation."), error);
    }

    return error;
}

nw_error_t *buffer_ternary(ternary_operation_type_t ternary_operation_type, buffer_t *w_buffer, buffer_t *x_buffer, buffer_t *y_buffer, buffer_t **z_buffer)
{
    CHECK_NULL_ARGUMENT(w_buffer, "w_buffer");
    CHECK_NULL_ARGUMENT(x_buffer, "x_buffer");
    CHECK_NULL_ARGUMENT(y_buffer, "y_buffer");
    CHECK_NULL_ARGUMENT(z_buffer, "z_buffer");
    CHECK_NULL_ARGUMENT(w_buffer->view, "w_buffer->view");
    CHECK_NULL_ARGUMENT(x_buffer->view, "x_buffer->view");
    CHECK_NULL_ARGUMENT(y_buffer->view, "y_buffer->view");
    CHECK_NULL_ARGUMENT(w_buffer->view->strides, "w_buffer->view->strides");
    CHECK_NULL_ARGUMENT(x_buffer->view->strides, "x_buffer->view->strides");
    CHECK_NULL_ARGUMENT(y_buffer->view->strides, "y_buffer->view->strides");
    CHECK_NULL_ARGUMENT(w_buffer->view->shape, "w_buffer->view->shape");
    CHECK_NULL_ARGUMENT(x_buffer->view->shape, "x_buffer->view->shape");
    CHECK_NULL_ARGUMENT(y_buffer->view->shape, "y_buffer->view->shape");
    CHECK_NULL_ARGUMENT(w_buffer->storage, "w_buffer->storage");
    CHECK_NULL_ARGUMENT(x_buffer->storage, "x_buffer->storage");
    CHECK_NULL_ARGUMENT(y_buffer->storage, "y_buffer->storage");
    CHECK_NULL_ARGUMENT(w_buffer->storage->data, "w_buffer->storage->data");
    CHECK_NULL_ARGUMENT(x_buffer->storage->data, "x_buffer->storage->data");
    CHECK_NULL_ARGUMENT(y_buffer->storage->data, "y_buffer->storage->data");

    nw_error_t *error = NULL;
    bool_t overwrite = (bool_t) *z_buffer;
    int64_t rank;

    if (x_buffer->view->rank != y_buffer->view->rank || y_buffer->view->rank != w_buffer->view->rank)
    {
        return ERROR(ERROR_RANK, string_create("ranks are incompatible."), NULL);
    }
    else
    {
        rank = w_buffer->view->rank;
    }

    int64_t shape[rank];
    runtime_t runtime;
    datatype_t datatype;

    if (x_buffer->storage->datatype != y_buffer->storage->datatype || y_buffer->storage->datatype != w_buffer->storage->datatype)
    {
        error = ERROR(ERROR_DATATYPE, string_create("datatypes are incompatible."), NULL);
        goto cleanup;
    }
    else
    {
        datatype = w_buffer->storage->datatype;
    }

    if (x_buffer->storage->runtime != y_buffer->storage->runtime || y_buffer->storage->runtime != w_buffer->storage->runtime)
    {
        error = ERROR(ERROR_RUNTIME, string_create("runtimes are incompatible."), NULL);
        goto cleanup;
    }
    else
    {
        runtime = w_buffer->storage->runtime;
    }

    if (!overwrite)
    {
        if (!view_shapes_equal(x_buffer->view, y_buffer->view) || !view_shapes_equal(y_buffer->view, w_buffer->view))
        {
            error = ERROR(ERROR_SHAPE, string_create("incompatible tensor shapes."), NULL);
            goto cleanup;
        }
        else
        {
            memcpy(shape, w_buffer->view->shape, rank * sizeof(int64_t));
        }

        error = buffer_creation(EMPTY_OPERATION, z_buffer, shape, rank, NULL, 0, runtime, datatype, NULL, 0, NULL);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
            goto cleanup;
        }
    }

    void *w_data = w_buffer->storage->data;
    void *x_data = x_buffer->storage->data;
    void *y_data = y_buffer->storage->data;
    void *z_data = (*z_buffer)->storage->data;

    int64_t n;
    int64_t w_stride;
    int64_t x_stride;
    int64_t y_stride;
    int64_t z_stride;
    int64_t w_offset;
    int64_t x_offset;
    int64_t y_offset;
    int64_t z_offset;

    switch (rank)
    {
    case 0:
        n = 1;
        w_stride = 0;
        x_stride = 0;
        y_stride = 0;
        z_stride = 0;
        w_offset = w_buffer->view->offset;
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        z_offset = (*z_buffer)->view->offset;
        runtime_ternary(ternary_operation_type, 
                        runtime, datatype, n, 
                        w_data, w_stride, w_offset, 
                        x_data, x_stride, x_offset, 
                        y_data, y_stride, y_offset,
                        z_data, z_stride, z_offset);
        break;
    case 1:
        n = (*z_buffer)->view->shape[0];
        w_stride = w_buffer->view->strides[0];
        x_stride = x_buffer->view->strides[0];
        y_stride = y_buffer->view->strides[0];
        z_stride = (*z_buffer)->view->strides[0];
        w_offset = w_buffer->view->offset;
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        z_offset = (*z_buffer)->view->offset;
        runtime_ternary(ternary_operation_type, 
                        runtime, datatype, n, 
                        w_data, w_stride, w_offset, 
                        x_data, x_stride, x_offset, 
                        y_data, y_stride, y_offset,
                        z_data, z_stride, z_offset);
        break;
    case 2:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            n = (*z_buffer)->view->shape[1];
            w_stride = w_buffer->view->strides[1];
            x_stride = x_buffer->view->strides[1];
            y_stride = y_buffer->view->strides[1];
            z_stride = (*z_buffer)->view->strides[1];
            w_offset = w_buffer->view->offset
                       + i * w_buffer->view->strides[0];
            x_offset = x_buffer->view->offset
                       + i * x_buffer->view->strides[0];
            y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[0];
            z_offset = (*z_buffer)->view->offset
                       + i * (*z_buffer)->view->strides[0];
            runtime_ternary(ternary_operation_type, 
                            runtime, datatype, n, 
                            w_data, w_stride, w_offset, 
                            x_data, x_stride, x_offset, 
                            y_data, y_stride, y_offset,
                            z_data, z_stride, z_offset);
        }
        break;
    case 3:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*z_buffer)->view->shape[1]; ++j)
            {
                n = (*z_buffer)->view->shape[2];
                w_stride = w_buffer->view->strides[2];
                x_stride = x_buffer->view->strides[2];
                y_stride = y_buffer->view->strides[2];
                z_stride = (*z_buffer)->view->strides[2];
                w_offset = w_buffer->view->offset
                           + i * w_buffer->view->strides[0]
                           + j * w_buffer->view->strides[1];
                x_offset = x_buffer->view->offset
                           + i * x_buffer->view->strides[0]
                           + j * x_buffer->view->strides[1];
                y_offset = y_buffer->view->offset
                           + i * y_buffer->view->strides[0]
                           + j * y_buffer->view->strides[1];
                z_offset = (*z_buffer)->view->offset
                           + i * (*z_buffer)->view->strides[0]
                           + j * (*z_buffer)->view->strides[1];

                runtime_ternary(ternary_operation_type, 
                                runtime, datatype, n, 
                                w_data, w_stride, w_offset, 
                                x_data, x_stride, x_offset, 
                                y_data, y_stride, y_offset,
                                z_data, z_stride, z_offset);
            }
        }
        break;
    case 4:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*z_buffer)->view->shape[1]; ++j)
            {
                for (int64_t k = 0; k < (*z_buffer)->view->shape[2]; ++k)
                {
                    n = (*z_buffer)->view->shape[3];
                    w_stride = w_buffer->view->strides[3];
                    x_stride = x_buffer->view->strides[3];
                    y_stride = y_buffer->view->strides[3];
                    z_stride = (*z_buffer)->view->strides[3];
                    w_offset = w_buffer->view->offset
                               + i * w_buffer->view->strides[0]
                               + j * w_buffer->view->strides[1]
                               + k * w_buffer->view->strides[2];
                    x_offset = x_buffer->view->offset
                               + i * x_buffer->view->strides[0]
                               + j * x_buffer->view->strides[1]
                               + k * x_buffer->view->strides[2];
                    y_offset = y_buffer->view->offset
                               + i * y_buffer->view->strides[0]
                               + j * y_buffer->view->strides[1]
                               + k * y_buffer->view->strides[2];
                    z_offset = (*z_buffer)->view->offset
                               + i * (*z_buffer)->view->strides[0]
                               + j * (*z_buffer)->view->strides[1]
                               + k * (*z_buffer)->view->strides[2];

                    runtime_ternary(ternary_operation_type, 
                                    runtime, datatype, n, 
                                    w_data, w_stride, w_offset, 
                                    x_data, x_stride, x_offset, 
                                    y_data, y_stride, y_offset,
                                    z_data, z_stride, z_offset);
                }
            }
        }
        break;
    case 5:
        for (int64_t i = 0; i < (*z_buffer)->view->shape[0]; ++i)
        {
            for (int64_t j = 0; j < (*z_buffer)->view->shape[1]; ++j)
            {
                for (int64_t k = 0; k < (*z_buffer)->view->shape[2]; ++k)
                {
                    for (int64_t l = 0; l < (*z_buffer)->view->shape[3]; ++l)
                    {
                        n = (*z_buffer)->view->shape[4];
                        w_stride = w_buffer->view->strides[4];
                        x_stride = x_buffer->view->strides[4];
                        y_stride = y_buffer->view->strides[4];
                        z_stride = (*z_buffer)->view->strides[4];
                        w_offset = w_buffer->view->offset
                                   + i * w_buffer->view->strides[0]
                                   + j * w_buffer->view->strides[1]
                                   + k * w_buffer->view->strides[2]
                                   + l * w_buffer->view->strides[3];
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
                        z_offset = (*z_buffer)->view->offset
                                   + i * (*z_buffer)->view->strides[0]
                                   + j * (*z_buffer)->view->strides[1]
                                   + k * (*z_buffer)->view->strides[2]
                                   + l * (*z_buffer)->view->strides[3];
                        runtime_ternary(ternary_operation_type, 
                                        runtime, datatype, n, 
                                        w_data, w_stride, w_offset, 
                                        x_data, x_stride, x_offset, 
                                        y_data, y_stride, y_offset,
                                        z_data, z_stride, z_offset);
                    }
                }
            }
        }
        break;
    default:
        error = ERROR(ERROR_RANK, string_create("unsupported rank %d", (int) rank), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    if (!overwrite)
    {
        buffer_destroy(*z_buffer);
    }

    return error;
}

static nw_error_t *runtime_reduction_dimension(reduction_operation_type_t reduction_operation_type, buffer_t *x_buffer, buffer_t *y_buffer, int64_t axis, bool_t keep_dimension)
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

    int64_t idim;
    int64_t jdim;
    int64_t kdim;
    int64_t ldim;

    datatype_t datatype = x_buffer->storage->datatype;
    runtime_t runtime = x_buffer->storage->runtime;
    int64_t rank = x_buffer->view->rank;
    int64_t n = x_buffer->view->shape[axis];
    void *x_data = x_buffer->storage->data;
    void *y_data = y_buffer->storage->data;
    int64_t x_stride = x_buffer->view->strides[axis];
    int64_t x_offset;
    int64_t y_offset;

    switch (rank)
    {
    case 1:
        x_offset = x_buffer->view->offset;
        y_offset = y_buffer->view->offset;
        runtime_reduction(reduction_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
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
        
        for (int64_t i = 0; i < x_buffer->view->shape[idim]; ++i)
        {
            x_offset = x_buffer->view->offset
                       + i * x_buffer->view->strides[idim];
            y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[(idim >= axis && !keep_dimension) ? idim - 1 : idim];
            runtime_reduction(reduction_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
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
        
        for (int64_t i = 0; i < x_buffer->view->shape[idim]; ++i)
        {
            for (int64_t j = 0; j < x_buffer->view->shape[jdim]; ++j)
            {
                x_offset = x_buffer->view->offset
                           + i * x_buffer->view->strides[idim]
                           + j * x_buffer->view->strides[jdim];
                y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[(idim >= axis && !keep_dimension) ? idim - 1 : idim]
                       + j * y_buffer->view->strides[(jdim >= axis && !keep_dimension) ? jdim - 1 : jdim];
                runtime_reduction(reduction_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
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
        
        for (int64_t i = 0; i < x_buffer->view->shape[idim]; ++i)
        {
            for (int64_t j = 0; j < x_buffer->view->shape[jdim]; ++j)
            {
                for (int64_t k = 0; k < x_buffer->view->shape[kdim]; ++k)
                {
                    x_offset = x_buffer->view->offset
                               + i * x_buffer->view->strides[idim]
                               + j * x_buffer->view->strides[jdim]
                               + k * x_buffer->view->strides[kdim];
                    y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[(idim >= axis && !keep_dimension) ? idim - 1 : idim]
                       + j * y_buffer->view->strides[(jdim >= axis && !keep_dimension) ? jdim - 1 : jdim]
                       + k * y_buffer->view->strides[(kdim >= axis && !keep_dimension) ? kdim - 1 : kdim];
                    runtime_reduction(reduction_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
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
        
        for (int64_t i = 0; i < x_buffer->view->shape[idim]; ++i)
        {
            for (int64_t j = 0; j < x_buffer->view->shape[jdim]; ++j)
            {
                for (int64_t k = 0; k < x_buffer->view->shape[kdim]; ++k)
                {
                    for (int64_t l = 0; l < x_buffer->view->shape[ldim]; ++l)
                    {
                        x_offset = x_buffer->view->offset
                                   + i * x_buffer->view->strides[idim]
                                   + j * x_buffer->view->strides[jdim]
                                   + k * x_buffer->view->strides[kdim]
                                   + l * x_buffer->view->strides[ldim];
                        y_offset = y_buffer->view->offset
                       + i * y_buffer->view->strides[(idim >= axis && !keep_dimension) ? idim - 1 : idim]
                       + j * y_buffer->view->strides[(jdim >= axis && !keep_dimension) ? jdim - 1 : jdim]
                       + k * y_buffer->view->strides[(kdim >= axis && !keep_dimension) ? kdim - 1 : kdim]
                       + l * y_buffer->view->strides[(ldim >= axis && !keep_dimension) ? ldim - 1 : ldim];
                        runtime_reduction(reduction_operation_type, runtime, datatype, n, x_data, x_stride, x_offset, y_data, y_offset);
                    }
                }
            }
        }
        break;
    default:
        return ERROR(ERROR_RANK, string_create("unsupported rank %d", (int) x_buffer->view->rank), NULL);
    }

    return NULL;
}

nw_error_t *buffer_reduction(reduction_operation_type_t reduction_operation_type, buffer_t *x, int64_t *axis, int64_t length, buffer_t **result, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->view, "x->view");
    CHECK_NULL_ARGUMENT(x->storage, "x->storage");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_UNIQUE(axis, length, "axis");

    nw_error_t *error = NULL;
    bool_t overwrite = (bool_t) *result;
    datatype_t datatype = x->storage->datatype;
    runtime_t runtime = x->storage->runtime;
    buffer_t *intermediate_buffer = NULL;
    view_t *reduced_view = NULL;
    int64_t sorted_axis[length];

    if (!overwrite)
    {
        error = view_reduce(x->view, &reduced_view, axis, length, keep_dimension);
        if (error)
        {
            error = ERROR(ERROR_REDUCTION, string_create("failed to reduce tensor."), error);
            goto cleanup;
        }

        error = buffer_creation(EMPTY_OPERATION, result, reduced_view->shape, reduced_view->rank, reduced_view->strides, reduced_view->offset, x->storage->runtime, x->storage->datatype, NULL, 0, NULL);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
            goto cleanup;
        }

        view_destroy(reduced_view);
        reduced_view = NULL;
    }

    error = descending_sort(axis, length, sorted_axis);
    if (error)
    {
        error = ERROR(ERROR_SORT, string_create("failed to sort axis"), error);
        goto cleanup;
    }

    for (int64_t i = 0; i < length; ++i)
    {
        error = view_reduce(x->view, &reduced_view, &sorted_axis[i], (int64_t) 1, keep_dimension);
        if (error)
        {
            error = ERROR(ERROR_REDUCTION, string_create("failed to reduce tensor."), error);
            goto cleanup;
        }

        if (i + 1 < length)
        {
            error = buffer_creation(EMPTY_OPERATION, &intermediate_buffer, reduced_view->shape, reduced_view->rank, 
                                    reduced_view->strides, reduced_view->offset, runtime, datatype, NULL, 0, NULL);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
                goto cleanup;
            }
        }
        else
        {
            intermediate_buffer = *result;
        }

        error = runtime_reduction_dimension(reduction_operation_type, x, intermediate_buffer, sorted_axis[i], keep_dimension);
        if (error)
        {
            error = ERROR(ERROR_REDUCTION, string_create("failed to reduce tensor dimension."), error);
            goto cleanup;
        }

        if (i > 0)
        {
            buffer_destroy(x);
        }

        x = intermediate_buffer;

        view_destroy(reduced_view);
        reduced_view = NULL;
    }

    return error; 

cleanup:

    if (intermediate_buffer != *result)
    {
        buffer_destroy(intermediate_buffer);
    }

    if (!overwrite)
    {
        buffer_destroy(*result);
    }

    view_destroy(reduced_view);

    return error;
}

static void runtime_padding(const buffer_t *x, buffer_t *y, int64_t *arguments, int64_t length, int64_t index, bool_t in_bounds, int64_t x_offset, int64_t y_offset)
{
    if (!x->view->rank)
    {
        return;
    }

    for (int64_t i  = 0; i < y->view->shape[index]; ++i)
    {
        int64_t offset_i = i - arguments[2 * index]; 
        bool_t in_bounds_i = in_bounds && offset_i >= 0 && offset_i < x->view->shape[index];
        int64_t x_offset_i = x_offset + offset_i * x->view->strides[index];
        int64_t y_offset_i = y_offset + i * y->view->strides[index];
        if (index == x->view->rank - 1)
        {
            switch (x->storage->datatype)
            {
            case FLOAT32:
                ((float32_t *) y->storage->data)[y_offset_i] = (in_bounds_i) ? ((float32_t *) x->storage->data)[x_offset_i] : (float32_t) 0.0;
                break;
            case FLOAT64:
                ((float64_t *) y->storage->data)[y_offset_i] = (in_bounds_i) ? ((float64_t *) x->storage->data)[x_offset_i] : (float64_t) 0.0;
                break;
            default:
                break;
            }
        }
        else
        {
            runtime_padding(x, y, arguments, length, index + 1, in_bounds_i, x_offset_i, y_offset_i);
        }
    }
}

nw_error_t *buffer_structure(structure_operation_type_t structure_operation_type, buffer_t *x, int64_t *arguments, int64_t length, buffer_t **result)
{
    nw_error_t *error = NULL;
    view_t *view = NULL;

    if (structure_operation_type == EXPAND_OPERATION)
    {
        error = view_expand(x->view, &view, arguments, length);
        if (error)
        {
            return ERROR(ERROR_EXPAND, string_create("failed to expand."), error);
        }

    }
    else if (structure_operation_type == PERMUTE_OPERATION)
    {
        error = view_permute(x->view, &view, arguments, length);
        if (error)
        {
            return ERROR(ERROR_PERMUTE, string_create("failed to permute."), error);
        }
    }
    else if (structure_operation_type == RESHAPE_OPERATION)
    {
        error = view_create(&view, x->view->offset, length, arguments, NULL);
        if (error)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
        }
    }
    else if (structure_operation_type == SLICE_OPERATION)
    {
        error = view_slice(x->view, &view, arguments, length);
        if (error)
        {
            return ERROR(ERROR_SLICE, string_create("failed to slice."), error);
        }
    }
    else if (structure_operation_type == PADDING_OPERATION)
    {
        error = view_padding(x->view, &view, arguments, length);
        if (error)
        {
            return ERROR(ERROR_SLICE, string_create("failed to slice."), error);
        }

        error = buffer_creation(EMPTY_OPERATION, result, view->shape, view->rank, view->strides, view->offset, 
                                x->storage->runtime, x->storage->datatype, NULL, 0, NULL);
        if (error)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
        }

        runtime_padding(x, *result, arguments, length, 0, true, x->view->offset, (*result)->view->offset);

        view_destroy(view);

        return error;
    }
    else if (structure_operation_type == IMAGE_TO_COLUMN_OPERATION || structure_operation_type == COLUMN_TO_IMAGE_OPERATION)
    {
        int64_t batch_size = x->view->shape[0];
        int64_t kernel_size = arguments[0];
        int64_t stride = arguments[1];
        int64_t padding = arguments[2];
        int64_t channels = arguments[3];
        int64_t height = arguments[4];
        int64_t width = arguments[5];
        int64_t padding_value = arguments[6];
        int64_t output_height = (height + 2 * padding - kernel_size) / stride + 1;
        int64_t output_width = (width + 2 * padding - kernel_size) / stride + 1;
        runtime_t runtime = x->storage->runtime;
        datatype_t datatype = x->storage->datatype;
        bool_t im2col = structure_operation_type == IMAGE_TO_COLUMN_OPERATION;
        int64_t rank = (im2col) ? 3 : 4;
        int64_t *shape = (im2col) ? 
                          (int64_t[]){batch_size, channels * kernel_size * kernel_size, output_height * output_width} :
                          (int64_t[]){batch_size, channels, height, width};
        void *value = malloc(datatype_size(datatype));
        if (!value)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
        }

        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) value = (float32_t) padding_value;
            break;
        case FLOAT64:
            *(float64_t *) value = (float64_t) padding_value;
            break;
        default:
            break;
        }

        error = buffer_creation(ZEROES_OPERATION, result, shape, rank, NULL, 0, runtime, datatype, NULL, 0, NULL);
        if (error)
        {
            free(value);
            return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
        }
        runtime_image_to_column(datatype, x->storage->data, batch_size, channels, height, width, kernel_size, 
                                output_height, output_width, stride, padding, (*result)->storage->data, !im2col, value);


        free(value);
        return error;
    }

    error = buffer_create(result, view, x->storage, false);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return error;
}

static nw_error_t *buffer_create_empty(buffer_t **buffer, const int64_t *shape, int64_t rank, const int64_t *strides,
                                       int64_t offset, runtime_t runtime, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;
    view_t *view = NULL;
    storage_t *storage = NULL;
    int64_t n = 0;

    error = view_create(&view, offset, rank, shape, strides);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create view."), error);
        goto cleanup;
    }

    error = view_physical_size(view, &n);
    if (error)
    {
        error = ERROR(ERROR_N, string_create("failed to obtain storage size."), error);
        goto cleanup;
    }

    error = storage_create(&storage, runtime, datatype, n, NULL, true);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create storage."), error);
        goto cleanup;
    }

    error = buffer_create(buffer, view, storage, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
        goto cleanup;
    }

    return error;

cleanup:

    view_destroy(view);
    storage_destroy(storage);

    return error;
}

static nw_error_t *buffer_create_nonempty(buffer_t **buffer, const int64_t *shape, int64_t rank, const int64_t *strides,
                                          int64_t offset, runtime_t runtime, datatype_t datatype, void *data, bool_t copy)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;
    view_t *view = NULL;
    storage_t *storage = NULL;
    int64_t n = 0;

    error = view_create(&view, offset, rank, shape, strides);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create view."), error);
        goto cleanup;
    }

    error = view_physical_size(view, &n);
    if (error)
    {
        error = ERROR(ERROR_N, string_create("failed to obtain storage size."), error);
        goto cleanup;
    }

    error = storage_create(&storage, runtime, datatype, n, data, copy);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create storage."), error);
        goto cleanup;
    }

    error = buffer_create(buffer, view, storage, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
        goto cleanup;
    }

    return error;

cleanup:

    view_destroy(view);
    storage_destroy(storage);

    return error;
}

static nw_error_t *buffer_create_init(buffer_t **buffer, creation_operation_type_t creation_operation_type, const int64_t *shape, int64_t rank,
                                      const int64_t *strides, int64_t offset, runtime_t runtime, datatype_t datatype, void **arguments, int64_t length)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");
    CHECK_NULL_ARGUMENT(shape, "shape");
    if (length)
    {
        CHECK_NULL_ARGUMENT(arguments, "arguments");
    }

    nw_error_t *error = NULL;
    *buffer = NULL;

    error = buffer_create_empty(buffer, shape, rank, strides, offset, runtime, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
        goto cleanup;
    }

    switch (creation_operation_type)
    {
    case EMPTY_OPERATION:
        break;
    case ZEROES_OPERATION:
        runtime_zeroes((*buffer)->storage->data, (*buffer)->storage->n, (*buffer)->storage->datatype);
        break;
    case ONES_OPERATION:
        runtime_ones((*buffer)->storage->data, (*buffer)->storage->n, (*buffer)->storage->datatype);
        break;
    case UNIFORM_OPERATION:
        if (length == 2)
        {
            runtime_uniform((*buffer)->storage->data, (*buffer)->storage->n, (*buffer)->storage->datatype, arguments[0], arguments[1]);
        }
        else
        {
            error = ERROR(ERROR_ARGUMENTS, string_create("invalid number of arguments."), NULL);
        }
        break;
    case NORMAL_OPERATION:
        if (length == 2)
        {
            runtime_normal((*buffer)->storage->data, (*buffer)->storage->n, (*buffer)->storage->datatype, arguments[0], arguments[1]);
        }
        else
        {
            error = ERROR(ERROR_ARGUMENTS, string_create("invalid number of arguments."), NULL);
        }
        break;
    case ARANGE_OPERATION:
        if (length == 3)
        {
            runtime_arange((*buffer)->storage->data, (*buffer)->storage->datatype, arguments[0], arguments[1], arguments[2]);
        }
        else
        {
            error = ERROR(ERROR_ARGUMENTS, string_create("invalid number of arguments."), NULL);
        }
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown runtime creation operation."), NULL);
        break;
    }

    if (error)
    {
        error = ERROR(ERROR_INITIALIZATION, string_create("failed to initialize buffer."), error);
        goto cleanup;
    }

    return error;

cleanup:

    buffer_destroy(*buffer);

    return error;
}

nw_error_t *buffer_creation(creation_operation_type_t creation_operation_type, buffer_t **buffer, const int64_t *shape, int64_t rank, const int64_t *strides,
                            int64_t offset, const runtime_t runtime, datatype_t datatype, void **arguments, int64_t length, void *data)
{
    CHECK_NULL_ARGUMENT(buffer, "buffer");

    nw_error_t *error = NULL;

    switch (creation_operation_type)
    {
    case EMPTY_OPERATION:
    case ZEROES_OPERATION:
    case ONES_OPERATION:
    case UNIFORM_OPERATION:
    case NORMAL_OPERATION:
    case ARANGE_OPERATION:
        error = buffer_create_init(buffer, creation_operation_type, shape, rank, strides, offset, runtime, datatype, arguments, length);
        break;
    case FROM_OPERATION:
        error = buffer_create_nonempty(buffer, shape, rank, strides, offset, runtime, datatype, data, false);
        break;
    case COPY_OPERATION:
        error = buffer_create_nonempty(buffer, shape, rank, strides, offset, runtime, datatype, data, true);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type."), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return error;
}
