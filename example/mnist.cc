#include <iostream>
extern "C"
{
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <data.h>
#include <cost.h>
#include <layer.h>
#include <train.h>
#include <optimizer.h>
}
#include <torch/torch.h>

typedef struct mnist_dataset_t
{
    string_t images_path;
    string_t labels_path;
    FILE *images_file;
    FILE *labels_file;
    uint64_t height;
    uint64_t width;
    uint64_t number_of_labels;
    uint64_t offset;
} mnist_dataset_t;

static uint32_t uint32_big_endian(uint8_t *buffer)
{
    uint32_t value = 0;
    value |= buffer[0] << 24;
    value |= buffer[1] << 16;
    value |= buffer[2] << 8;
    value |= buffer[3];
    return value;
}

nw_error_t *mnist_metrics(dataset_type_t dataset_type, const tensor_t *y_pred, const tensor_t *y_true)
{
    CHECK_NULL_ARGUMENT(y_pred, "y_pred");
    CHECK_NULL_ARGUMENT(y_true, "y_true");

    switch (dataset_type)
    {
    case TRAIN:
        break;
    case VALID:
        break;
    case TEST:
        break;
    default:
        break;
    }

    return NULL;
}

nw_error_t *mnist_setup(void *arguments) 
{
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    nw_error_t *error = NULL;
    mnist_dataset_t *mnist_dataset = (mnist_dataset_t *) arguments;

    uint8_t buffer[4];

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
    fread(buffer, sizeof(buffer), 1, mnist_dataset->images_file);

    // Number of samples
    fread(buffer, sizeof(buffer), 1, mnist_dataset->images_file);

    // Height
    fread(buffer, sizeof(buffer), 1, mnist_dataset->images_file);
    mnist_dataset->height = (uint64_t) uint32_big_endian(buffer);

    // Width
    fread(buffer, sizeof(buffer), 1, mnist_dataset->images_file);
    mnist_dataset->width = (uint64_t) uint32_big_endian(buffer);

    mnist_dataset->number_of_labels = 10;
    mnist_dataset->offset = 16;
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

nw_error_t *mnist_dataloader(uint64_t index, batch_t *batch, void *arguments)
{
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(batch, "batch");

    int status;
    nw_error_t *error = NULL;
    mnist_dataset_t *mnist_dataset = (mnist_dataset_t *) arguments;
    uint64_t number_of_pixels = mnist_dataset->height * mnist_dataset->width;
    uint64_t number_of_labels = mnist_dataset->number_of_labels;
    uint64_t images_offset = mnist_dataset->offset;
    uint64_t batch_size = batch->batch_size;
    uint64_t m = batch_size * number_of_labels;
    uint64_t n = batch_size * number_of_pixels;
    void *data = NULL;
    void *labels = NULL;
    datatype_t datatype = batch->datatype;
    runtime_t runtime = batch->runtime;
    uint8_t file_buffer[1];

    switch (datatype)
    {
    case FLOAT32:
        data = (void *) malloc(sizeof(float32_t) * n);
        labels = (void *) malloc(sizeof(float32_t) * m);
        break;
    case FLOAT64:
        data = (void *) malloc(sizeof(float64_t) * n);
        labels = (void *) malloc(sizeof(float32_t) * m);
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
    }

    status = fseek(mnist_dataset->images_file, images_offset + index * number_of_pixels , SEEK_SET);
    if (status)
    {
        return ERROR(ERROR_FILE, string_create("failed to move to offset in file."), NULL);
    }

    for (uint64_t i = 0; i < n; ++i)
    {
        fread(file_buffer, sizeof(file_buffer), 1, mnist_dataset->images_file);

        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) data)[i] = (float32_t) *file_buffer;
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = (float64_t) *file_buffer;
            break;
        default:
            return ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
        }
    }

    for (uint64_t i = 0; i < batch_size; ++i)
    {
        fread(file_buffer, sizeof(file_buffer), 1, mnist_dataset->labels_file);

        for (uint64_t j = 0; j < number_of_labels; ++j)
        {
            switch (datatype)
            {
            case FLOAT32:
                ((float32_t *) labels)[i] = ((uint8_t) j == *file_buffer) ? (float32_t) 1.0 : (float32_t) 0.0;
                break;
            case FLOAT64:
                ((float64_t *) labels)[i] = ((uint8_t) j == *file_buffer) ? (float64_t) 1.0 : (float64_t) 0.0;
                break;
            default:
                return ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
            }
        }
    }

    error = tensor_from_data(&batch->x, data, runtime, datatype, 2, (uint64_t[]) {batch_size, number_of_pixels}, runtime == CU_RUNTIME, false);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_from_data(&batch->y, data, runtime, datatype, 2, (uint64_t[]) {batch_size, number_of_labels}, runtime == CU_RUNTIME, false);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *mnist_model_create(model_t **model, runtime_t runtime, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = NULL;

    *model = (model_t *) malloc(sizeof(model_t));
    if (!*model)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(model_t)), NULL);
    }

    (*model)->datatype = datatype;
    (*model)->runtime = runtime;


    return error;
}

void mnist_model_destroy(model_t *model)
{

}


int main(void)
{
    mnist_dataset_t mnist_dataset = (mnist_dataset_t) {
        .images_path = "../data/train-images-idx3-ubyte",
        .labels_path = "../data/train-labels-idx1-ubyte",
        .images_file = NULL,
        .labels_file = NULL,
    };

    nw_error_t *error = NULL;
    uint64_t epochs = 2;

    // error = train(&mnist_setup, 
    //               &mnist_teardown,
    //               &mnist_dataloader,
    //               &dataset,
    //               model, 
    //               &categorical_cross_entropy,
    //               optimizer,
    //               epochs,
    //               &mnist_metrics);
    // if (error)
    // {
    //     error = ERROR(ERROR_TRAIN, string_create("model training failed."), error);
    //     goto fail;
    // }
    
    return EXIT_SUCCESS;

fail:

    error_print(error);
    error_destroy(error);

    return EXIT_FAILURE;
}