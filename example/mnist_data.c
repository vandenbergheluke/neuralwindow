#include <mnist_data.h>
#include <runtime.h>
#include <tensor.h>

static uint32_t uint32_big_endian(uint8_t *buffer)
{
    uint32_t value = 0;
    value |= buffer[0] << 24;
    value |= buffer[1] << 16;
    value |= buffer[2] << 8;
    value |= buffer[3];
    return value;
}

nw_error_t *mnist_setup(void *arguments) 
{
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    nw_error_t *error = NULL;
    mnist_dataset_t *mnist_dataset = (mnist_dataset_t *) arguments;

    uint8_t buffer[4];
    size_t read;

    mnist_dataset->images_file = fopen(mnist_dataset->images_path, "rb");
    if (!mnist_dataset->images_file)
    {
        return ERROR(ERROR_FILE, string_create("failed to open %s.", mnist_dataset->images_path), NULL);
    }

    mnist_dataset->labels_file = fopen(mnist_dataset->labels_path, "rb");
    if (!mnist_dataset->images_file)
    {
        return ERROR(ERROR_FILE, string_create("failed to open %s.", mnist_dataset->labels_path), NULL);
    }

    // Magic Number
    read = fread(buffer, sizeof(buffer), 1, mnist_dataset->images_file);
    if (!read)
    {
        ERROR(ERROR_FILE, string_create("failed to read file."), NULL);
    }

    // Number of samples
    read = fread(buffer, sizeof(buffer), 1, mnist_dataset->images_file);
    if (!read)
    {
        ERROR(ERROR_FILE, string_create("failed to read file."), NULL);
    }

    // Height
    read = fread(buffer, sizeof(buffer), 1, mnist_dataset->images_file);
    if (!read)
    {
        ERROR(ERROR_FILE, string_create("failed to read file."), NULL);
    }
    mnist_dataset->height = (int64_t) uint32_big_endian(buffer);

    // Width
    read = fread(buffer, sizeof(buffer), 1, mnist_dataset->images_file);
    if (!read)
    {
        ERROR(ERROR_FILE, string_create("failed to read file."), NULL);
    }
    mnist_dataset->width = (int64_t) uint32_big_endian(buffer);

    mnist_dataset->number_of_labels = 10;
    mnist_dataset->image_offset = 16;
    mnist_dataset->label_offset = 8;

    return error;
}

nw_error_t *mnist_teardown(void *arguments) 
{
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    int status;
    mnist_dataset_t *mnist_dataset = (mnist_dataset_t *) arguments;

    status = fclose(mnist_dataset->images_file);
    if (status)
    {
        return ERROR(ERROR_FILE, string_create("failed to close file %s.", mnist_dataset->images_path), NULL);
    }

    status = fclose(mnist_dataset->labels_file);
    if (status)
    {
        return ERROR(ERROR_FILE, string_create("failed to close file %s.", mnist_dataset->labels_path), NULL);
    }
    
    return NULL;
}

nw_error_t *mnist_dataloader(int64_t index, batch_t *batch, void *arguments)
{
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(batch, "batch");

    int status;
    size_t read;
    nw_error_t *error = NULL;
    mnist_dataset_t *mnist_dataset = (mnist_dataset_t *) arguments;
    int64_t number_of_pixels = mnist_dataset->height * mnist_dataset->width;
    int64_t number_of_labels = mnist_dataset->number_of_labels;
    int64_t batch_size = batch->batch_size;
    int64_t m = batch_size * number_of_labels;
    int64_t n = batch_size * number_of_pixels;
    void *data = NULL;
    void *labels = NULL;
    datatype_t datatype = batch->datatype;
    runtime_t runtime = batch->runtime;
    uint8_t file_buffer[1];
    bool_t copy = runtime == CU_RUNTIME;
    size_t size = datatype_size(datatype);

    data = (void *) malloc(size * n);
    if (!data)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", (size_t) (size * n)), NULL);
    }

    labels = (void *) malloc(size * m);
    if (!labels)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", (size_t) (size * m)), NULL);
    }

    status = fseek(mnist_dataset->images_file, mnist_dataset->image_offset + index * number_of_pixels , SEEK_SET);
    if (status)
    {
        return ERROR(ERROR_FILE, string_create("failed to move to offset in file."), NULL);
    }

    status = fseek(mnist_dataset->labels_file, mnist_dataset->label_offset + index , SEEK_SET);
    if (status)
    {
        return ERROR(ERROR_FILE, string_create("failed to move to offset in file."), NULL);
    }

    for (int64_t i = 0; i < n; ++i)
    {
        read = fread(file_buffer, sizeof(file_buffer), 1, mnist_dataset->images_file);
        if (!read)
        {
            return ERROR(ERROR_FILE, string_create("failed to read file."), NULL);
        }

        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) data)[i] = (float32_t) *file_buffer / (float64_t) 255.0;
            if (mnist_dataset->normalize)
            {
                ((float32_t *) data)[i] -= (float32_t) 0.5;
                ((float32_t *) data)[i] /= (float32_t) 0.5;
            }
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = (float64_t) *file_buffer / (float64_t) 255.0;
            if (mnist_dataset->normalize)
            {
                ((float64_t *) data)[i] -= (float64_t) 0.5;
                ((float64_t *) data)[i] /= (float64_t) 0.5;
            }
            break;
        default:
            return ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
        }
    }

    for (int64_t i = 0; i < batch_size; ++i)
    {
        read = fread(file_buffer, sizeof(file_buffer), 1, mnist_dataset->labels_file);
        if (!read)
        {
            return ERROR(ERROR_FILE, string_create("failed to read file."), NULL);
        }

        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) labels)[i] = (float32_t) *file_buffer;
            break;
        case FLOAT64:
            ((float64_t *) labels)[i] = (float64_t) *file_buffer;
            break;
        default:
            return ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
        }
    }

    error = tensor_from_data(&batch->x, data, runtime, datatype, 4, (int64_t[]) {batch_size, 1, mnist_dataset->height, mnist_dataset->width}, copy, false, true);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_from_data(&batch->y, labels, runtime, datatype, 2, (int64_t[]) {batch_size, 1}, copy, false, true);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    if (copy)
    {
        free(data);
        free(labels);
    }

    return error;
}


