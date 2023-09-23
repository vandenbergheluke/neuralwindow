#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <cost.h>
#include <layer.h>
#include <init.h>
#include <train.h>
#include <optimizer.h>
#include <metric.h>

typedef struct mnist_dataset_t
{
    string_t images_path;
    string_t labels_path;
    FILE *images_file;
    FILE *labels_file;
    uint64_t height;
    uint64_t width;
    uint64_t number_of_labels;
    uint64_t image_offset;
    uint64_t label_offset;
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

nw_error_t *mnist_metrics(dataset_type_t dataset_type, const tensor_t *y_true, const tensor_t *y_pred)
{
    CHECK_NULL_ARGUMENT(y_pred, "y_pred");
    CHECK_NULL_ARGUMENT(y_true, "y_true");
    
    nw_error_t *error = NULL;
    tensor_t *accuracy = NULL;
    tensor_t *probabilities = NULL;

    error = tensor_exponential(y_pred, &probabilities);
    if (error)
    {
        return ERROR(ERROR_EXPONENTIAL, string_create("failed to exponentiate tensor."), error);
    }

    // PRINTLN_TEMP_TENSOR("probabilities", probabilities);
    // PRINTLN_TEMP_TENSOR("y_true", y_true);


    switch (dataset_type)
    {
    case TRAIN:
        error = multiclass_accuracy(probabilities, y_true, &accuracy);
        break;
    case VALID:
        error = multiclass_accuracy(probabilities, y_true, &accuracy);
        break;
    case TEST:
        error = multiclass_accuracy(probabilities, y_true, &accuracy);
        break;
    default:
        error = ERROR(ERROR_DATASET_TYPE, string_create("unknown dataset type %d.", dataset_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
    }

    LOG_SCALAR_TENSOR("accuracy", accuracy);

    tensor_destroy(accuracy);
    tensor_destroy(probabilities);

    return error;
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
    mnist_dataset->height = (uint64_t) uint32_big_endian(buffer);

    // Width
    read = fread(buffer, sizeof(buffer), 1, mnist_dataset->images_file);
    if (!read)
    {
        ERROR(ERROR_FILE, string_create("failed to read file."), NULL);
    }
    mnist_dataset->width = (uint64_t) uint32_big_endian(buffer);

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

nw_error_t *mnist_dataloader(uint64_t index, batch_t *batch, void *arguments)
{
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(batch, "batch");

    int status;
    size_t read;
    nw_error_t *error = NULL;
    mnist_dataset_t *mnist_dataset = (mnist_dataset_t *) arguments;
    uint64_t number_of_pixels = mnist_dataset->height * mnist_dataset->width;
    uint64_t number_of_labels = mnist_dataset->number_of_labels;
    uint64_t batch_size = batch->batch_size;
    uint64_t m = batch_size * number_of_labels;
    uint64_t n = batch_size * number_of_pixels;
    void *data = NULL;
    void *labels = NULL;
    datatype_t datatype = batch->datatype;
    runtime_t runtime = batch->runtime;
    uint8_t file_buffer[1];
    bool_t copy = runtime == CU_RUNTIME;

    switch (datatype)
    {
    case FLOAT32:
        data = (void *) malloc(sizeof(float32_t) * n);
        labels = (void *) malloc(sizeof(float32_t) * m);
        break;
    case FLOAT64:
        data = (void *) malloc(sizeof(float64_t) * n);
        labels = (void *) malloc(sizeof(float64_t) * m);
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
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

    for (uint64_t i = 0; i < n; ++i)
    {
        read = fread(file_buffer, sizeof(file_buffer), 1, mnist_dataset->images_file);
        if (!read)
        {
            ERROR(ERROR_FILE, string_create("failed to read file."), NULL);
        }

        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) data)[i] = (float32_t) *file_buffer / (float64_t) 255.0;
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = (float64_t) *file_buffer / (float64_t) 255.0;
            break;
        default:
            return ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
        }
    }

    for (uint64_t i = 0; i < batch_size; ++i)
    {
        read = fread(file_buffer, sizeof(file_buffer), 1, mnist_dataset->labels_file);
        if (!read)
        {
            ERROR(ERROR_FILE, string_create("failed to read file."), NULL);
        }

        for (uint64_t j = 0; j < number_of_labels; ++j)
        {
            switch (datatype)
            {
            case FLOAT32:
                ((float32_t *) labels)[i * number_of_labels + j] = ((uint8_t) j == *file_buffer) ? (float32_t) 1.0 : (float32_t) 0.0;
                break;
            case FLOAT64:
                ((float64_t *) labels)[i * number_of_labels + j] = ((uint8_t) j == *file_buffer) ? (float64_t) 1.0 : (float64_t) 0.0;
                break;
            default:
                return ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
            }
        }
    }

    error = tensor_from_data(&batch->x, data, runtime, datatype, 2, (uint64_t[]) {batch_size, number_of_pixels}, copy, false);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_from_data(&batch->y, labels, runtime, datatype, 2, (uint64_t[]) {batch_size, number_of_labels}, copy, false);
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

    layer_t *input_layer = NULL;
    layer_t *output_layer = NULL;
    block_t *block = NULL;
    activation_t *input_activation = NULL;
    parameter_init_t *weight_init = NULL;
    parameter_init_t *bias_init = NULL;
    activation_t *output_activation = NULL;
    float32_t mean_float32, standard_deviation_float32;
    float64_t mean_float64, standard_deviation_float64;
    void *mean = NULL;
    void *standard_deviation = NULL;

    switch (datatype)
    {
    case FLOAT32:
        mean_float32 = (float32_t) 0.0;
        standard_deviation_float32 = (float32_t) 1.0;
        mean = (void *) &mean_float32;
        standard_deviation = (void *) &standard_deviation_float32;
        break;
    case FLOAT64:
        mean_float64 = (float64_t) 0.0;
        standard_deviation_float64 = (float64_t) 1.0;
        mean = (void *) &mean_float64;
        standard_deviation = (void *) &standard_deviation_float64;
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
    }

    error = normal_parameter_init(&weight_init, mean, standard_deviation);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create normal initializer."), error);
    }

    error = zeroes_parameter_init(&bias_init);
    if (error)
    {
        parameter_init_destroy(weight_init);
        return ERROR(ERROR_CREATE, string_create("failed to create zero initializer."), error);
    }

    error = rectified_linear_activation_create(&input_activation);
    if (error)
    {
        parameter_init_destroy(weight_init);
        parameter_init_destroy(bias_init);
        return ERROR(ERROR_CREATE, string_create("failed to create rectified linear activation."), error);
    }

    error = linear_layer_create(&input_layer, 784, 128, runtime, datatype, true,
                                input_activation, weight_init, bias_init);
    if (error)
    {
        parameter_init_destroy(weight_init);
        parameter_init_destroy(bias_init);
        activation_destroy(input_activation);
        return ERROR(ERROR_CREATE, string_create("failed to create linear layer."), error);
    }

    error = logsoftmax_activation_create(&output_activation, (uint64_t) 1);
    if (error)
    {
        parameter_init_destroy(weight_init);
        parameter_init_destroy(bias_init);
        layer_destroy(input_layer);
        return ERROR(ERROR_CREATE, string_create("failed to create softmax activation."), error);
    }

    error = linear_layer_create(&output_layer, 128, 10, runtime, datatype, true,
                                output_activation, weight_init, bias_init);
    if (error)
    {
        parameter_init_destroy(weight_init);
        parameter_init_destroy(bias_init);
        activation_destroy(output_activation);
        layer_destroy(input_layer);
        return ERROR(ERROR_CREATE, string_create("failed to create linear layer."), error);
    }

    uint64_t depth = 2;
    error = block_create(&block, depth, input_layer, output_layer);
    if (error)
    {
        parameter_init_destroy(weight_init);
        parameter_init_destroy(bias_init);
        layer_destroy(input_layer);
        layer_destroy(output_layer);
        return ERROR(ERROR_CREATE, string_create("failed to create block."), error);
    }

    error = model_create(model, block);
    if (error)
    {
        parameter_init_destroy(weight_init);
        parameter_init_destroy(bias_init);
        block_destroy(block);
        return ERROR(ERROR_CREATE, string_create("failed to create model."), error);
    }

    parameter_init_destroy(weight_init);
    parameter_init_destroy(bias_init);

    return error;
}

void mnist_model_destroy(model_t *model)
{
    model_destroy(model);
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
    uint64_t epochs = 1000;
    model_t *model = NULL;
    runtime_t runtime = OPENBLAS_RUNTIME;
    datatype_t datatype = FLOAT32;
    uint64_t number_of_samples = 60000;
    batch_t *batch = NULL;
    uint64_t batch_size = 128;
    bool_t shuffle = true;
    float32_t train_split = 0.7;
    float32_t valid_split = 0.2;
    float32_t test_split = 0.1;
    optimizer_t *optimizer = NULL;
    float32_t learning_rate = 0.0001;

    error = batch_create(&batch, batch_size, datatype, runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create batch."), error);
        goto cleanup;
    }

    error = mnist_model_create(&model, runtime, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create model."), error);
        goto cleanup;
    }

    error = optimizer_stochastic_gradient_descent_create(&optimizer, learning_rate, 0.0, 0.0, 0.0, 0.0);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create optimizer."), error);
        goto cleanup;
    }

    error = fit(epochs, number_of_samples, batch, shuffle, train_split, valid_split, test_split, model, optimizer,
                &mnist_dataset, &mnist_setup, &mnist_teardown, &mnist_dataloader, &negative_log_likelihood, &mnist_metrics);
    if (error)
    {
        error = ERROR(ERROR_TRAIN, string_create("failed to fit model."), error);
        goto cleanup;
    }

cleanup:

    optimizer_destroy(optimizer);
    batch_destroy(batch);
    mnist_model_destroy(model);

    if (error)
    {
        error_print(error);
        error_destroy(error);

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}