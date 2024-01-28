#include <view.h>
#include <runtime.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <cost.h>
#include <layer.h>
#include <init.h>
#include <train.h>
#include <optimizer.h>
#include <metric.h>
#include <measure.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mgl2/base_cf.h>
#include <mgl2/canvas_cf.h>
#include <mgl2/mgl_cf.h>

typedef struct mnist_dataset_t
{
    string_t images_path;
    string_t labels_path;
    FILE *images_file;
    FILE *labels_file;
    int64_t height;
    int64_t width;
    int64_t number_of_labels;
    int64_t image_offset;
    int64_t label_offset;
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

void *plt_accuracies = NULL;
void *plt_costs = NULL;
float32_t *plt_count = NULL;


nw_error_t *bounded_plot(string_t title, string_t save_path,
          string_t x_str, void* x, int x_n,
          string_t y_str, void* y, int y_n,
          float64_t y_min, float64_t y_max,
          datatype_t datatype)
{
    HMGL graph = mgl_create_graph(800,400);

    HMDT x_mgl = mgl_create_data();
    HMDT y_mgl = mgl_create_data();

    mgl_data_set_float(x_mgl, x, x_n, 1, 1);
    switch (datatype)
    {
    case FLOAT32:
        mgl_data_set_float(y_mgl, (float32_t *) y, y_n, 1, 1);
        break;
    case FLOAT64:
        mgl_data_set_double(y_mgl, (float64_t *) y, y_n, 1, 1);
        break;
    default:
        mgl_delete_graph(graph);

        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
    }

    mgl_fill_background(graph, 1, 1, 1, 1);

    //mgl_subplot(graph, 3, 3, 4, "");
    mgl_inplot(graph, 0, 1, 0, 1);
    mgl_title(graph, title, "", 5);
    mgl_set_range_dat(graph, 'x', x_mgl, 0);
    mgl_set_range_val(graph, 'y', y_min, y_max);
    mgl_axis(graph, "xy", "", "");
    // |    long dashed line
    // h    grey
    mgl_axis_grid(graph, "xy", "|h", "");
    mgl_label(graph, 'x', x_str, 0, "");
    mgl_label(graph, 'y', y_str, 0, "");
    mgl_box(graph);
    // u    blue purple
    mgl_plot_xy(graph, x_mgl, y_mgl, "2u", "");

    mgl_write_png(graph, save_path, "w");

    mgl_delete_graph(graph);
    mgl_delete_data(x_mgl);
    mgl_delete_data(y_mgl);

    return NULL;
}

nw_error_t *plot(string_t title, string_t save_path,
          string_t x_str, void* x, int x_n,
          string_t y_str, void* y, int y_n,
          datatype_t datatype)
{
    HMGL graph = mgl_create_graph(800,400);

    HMDT x_mgl = mgl_create_data();
    HMDT y_mgl = mgl_create_data();

    mgl_data_set_float(x_mgl, x, x_n, 1, 1);
    switch (datatype)
    {
    case FLOAT32:
        mgl_data_set_float(y_mgl, (float32_t *) y, y_n, 1, 1);
        break;
    case FLOAT64:
        mgl_data_set_double(y_mgl, (float64_t *) y, y_n, 1, 1);
        break;
    default:
        mgl_delete_graph(graph);

        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
    }

    mgl_fill_background(graph, 1, 1, 1, 1);

    //mgl_subplot(graph, 3, 3, 4, "");
    mgl_inplot(graph, 0, 1, 0, 1);
    mgl_title(graph, title, "", 5);
    mgl_set_range_dat(graph, 'x', x_mgl, 0);
    mgl_set_range_dat(graph, 'y', y_mgl, 0);
    mgl_axis(graph, "xy", "", "");
    // |    long dashed line
    // h    grey
    mgl_axis_grid(graph, "xy", "|h", "");
    mgl_label(graph, 'x', x_str, 0, "");
    mgl_label(graph, 'y', y_str, 0, "");
    mgl_box(graph);
    // u    blue purple
    mgl_plot_xy(graph, x_mgl, y_mgl, "2u", "");

    mgl_write_png(graph, save_path, "w");

    mgl_delete_graph(graph);
    mgl_delete_data(x_mgl);
    mgl_delete_data(y_mgl);

    return NULL;
}

nw_error_t *mnist_metrics(dataset_type_t dataset_type, 
                          const tensor_t *y_true,
                          const tensor_t *y_pred,
                          const tensor_t *cost,
                          int64_t epoch,
                          int64_t epochs,
                          int64_t iteration,
                          int64_t iterations)
{
    CHECK_NULL_ARGUMENT(y_pred, "y_pred");
    CHECK_NULL_ARGUMENT(y_true, "y_true");
    CHECK_NULL_ARGUMENT(y_true, "cost");
    static void *accuracy_data = NULL;
    static void *cost_data = NULL;
    static int64_t time = 0; 
    tensor_t *total_accuracy = NULL;
    tensor_t *total_cost = NULL;
    tensor_t *mean_accuracy = NULL;
    tensor_t *mean_cost = NULL;
    nw_error_t *error = NULL;
    tensor_t *accuracy = NULL;
    tensor_t *probabilities = NULL;
    runtime_t runtime = cost->buffer->storage->runtime;
    datatype_t datatype = cost->buffer->storage->datatype;

    error = tensor_exponential(y_pred, &probabilities);
    if (error)
    {
        error = ERROR(ERROR_EXPONENTIAL, string_create("failed to exponentiate tensor."), error);
        goto cleanup;
    }

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
        error = ERROR(ERROR_METRICS, string_create("failed to compute metrics."), error);
        goto cleanup;
    }

    if (!(iteration - 1))
    {
        size_t size = iterations * datatype_size(datatype);
        accuracy_data = (void *) malloc(size);
        if (!accuracy_data)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
            goto cleanup;
        }

        cost_data = (void *) malloc(size);
        if (!cost_data)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
            goto cleanup;
        }

        time = get_time_nanoseconds();
    }

    error = tensor_item(accuracy, accuracy_data + (iteration - 1) * datatype_size(datatype));
    if (error)
    {
        error = ERROR(ERROR_ITEM, string_create("failed to get tensor item."), NULL);
        goto cleanup;
    }

    error = tensor_item(cost, cost_data + (iteration - 1) * datatype_size(datatype));
    if (error)
    {
        error = ERROR(ERROR_ITEM, string_create("failed to get tensor item."), NULL);
        goto cleanup;
    }

    if (iteration == iterations)
    {
        int64_t shape[] = {iterations};
        int64_t rank = 1;
        
        error = tensor_from_data(&total_accuracy, accuracy_data, runtime, datatype, rank, shape, false, false, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_from_data(&total_cost, cost_data, runtime, datatype, rank, shape, false, false, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_mean(total_accuracy, &mean_accuracy, NULL, 0, false);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to get mean of tensor."), error);
            goto cleanup;
        }

        error = tensor_mean(total_cost, &mean_cost, NULL, 0, false);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to get mean of tensor."), error);
            goto cleanup;
        }

        error = tensor_item(mean_accuracy, plt_accuracies + datatype_size(datatype) * (epoch - 1));
        if (error)
        {
            error = ERROR(ERROR_ITEM, string_create("failed to get tensor item."), NULL);
            goto cleanup;
        }

        error = tensor_item(mean_cost, plt_costs + datatype_size(datatype) * (epoch - 1));
        if (error)
        {
            error = ERROR(ERROR_ITEM, string_create("failed to get tensor item."), NULL);
            goto cleanup;
        }

        plt_count[epoch - 1] = epoch;

        if (epoch == epochs) {
            switch (dataset_type)
            {
            case TRAIN:
                error = bounded_plot("MNIST Training Accuracy",
                          "img/mnist_accuracy_train.png",
                          "Epoch", plt_count, epochs,
                          "Accuracy", plt_accuracies, epochs,
                          0.0, 1.0, datatype);
                if (error)
                {
                    error = ERROR(ERROR_METRICS, string_create("failed to plot accuracy."), error);
                    goto cleanup;
                }

                error = plot("MNIST Training Cost",
                          "img/mnist_cost_train.png",
                          "Epoch", plt_count, epochs,
                          "Cost", plt_costs, epochs,
                          datatype);
                if (error)
                {
                    error = ERROR(ERROR_METRICS, string_create("failed to plot cost."), error);
                    goto cleanup;
                }
                break;
            case VALID:
                error = bounded_plot("MNIST Validation Accuracy",
                          "img/mnist_accuracy_valid.png",
                          "Epoch", plt_count, epochs,
                          "Accuracy", plt_accuracies, epochs,
                          0.0, 1.0, datatype);
                if (error)
                {
                    error = ERROR(ERROR_METRICS, string_create("failed to plot accuracy."), error);
                    goto cleanup;
                }

                error = plot("MNIST Validation Cost",
                          "img/mnist_cost_valid.png",
                          "Epoch", plt_count, epochs,
                          "Cost", plt_costs, epochs,
                          datatype);
                if (error)
                {
                    error = ERROR(ERROR_METRICS, string_create("failed to plot cost."), error);
                    goto cleanup;
                }
                break;
            case TEST:
                break;
            default:
                error = ERROR(ERROR_DATASET_TYPE, string_create("unknown dataset type %d.", dataset_type), NULL);
                goto cleanup;
            }
        }

        LOG("Dataset %s - %lu/%lu Epochs", dataset_type_string(dataset_type), epoch, epochs);
        LOG_SCALAR_TENSOR(" - Cost", mean_cost);
        LOG_SCALAR_TENSOR("- Accuracy", mean_accuracy);
        LOG("- Time: %lfs", (float64_t) (get_time_nanoseconds() - time) * (float64_t) 1e-9);
        LOG_NEWLINE;
    }

cleanup:

    tensor_destroy(accuracy);
    tensor_destroy(probabilities);
    tensor_destroy(mean_accuracy);
    tensor_destroy(mean_cost);
    tensor_destroy(total_accuracy);
    tensor_destroy(total_cost);

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
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = (float64_t) *file_buffer / (float64_t) 255.0;
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

        for (int64_t j = 0; j < number_of_labels; ++j)
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

    error = tensor_from_data(&batch->x, data, runtime, datatype, 2, (int64_t[]) {batch_size, number_of_pixels}, copy, false, true);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_from_data(&batch->y, labels, runtime, datatype, 2, (int64_t[]) {batch_size, number_of_labels}, copy, false, true);
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

nw_error_t *mnist_model_create(model_t **model, runtime_t runtime, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = NULL;

    layer_t *input_layer = NULL;
    layer_t *dropout_layer = NULL;
    layer_t *output_layer = NULL;
    block_t *block = NULL;
    activation_t *input_activation = NULL;
    parameter_init_t *weight_init = NULL;
    parameter_init_t *bias_init = NULL;
    activation_t *output_activation = NULL;
    void *mean = NULL;
    void *standard_deviation = NULL;
    void *probability = NULL;
    size_t size = datatype_size(datatype);

    mean = (void *) malloc(size);
    if (!mean)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    standard_deviation = (void *) malloc(size);
    if (!standard_deviation)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    probability = (void *) malloc(size);
    if (!probability)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) mean = (float32_t) 0.0;
        *(float32_t *) standard_deviation = (float32_t) 1.0;
        *(float32_t *) probability = (float32_t) 0.5;
        break;
    case FLOAT64:
        *(float64_t *) mean = (float64_t) 0.0;
        *(float64_t *) standard_deviation = (float64_t) 1.0;
        *(float64_t *) probability = (float64_t) 0.5;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    error = normal_parameter_init(&weight_init, mean, standard_deviation);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create normal initializer."), error);
        goto cleanup;
    }

    error = zeroes_parameter_init(&bias_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create zero initializer."), error);
        goto cleanup;
    }

    error = rectified_linear_activation_create(&input_activation);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create rectified linear activation."), error);
        goto cleanup;
    }

    error = linear_layer_create(&input_layer, 784, 128, runtime, datatype, true, input_activation, weight_init, bias_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create linear layer."), error);
        goto cleanup;
    }

    error = dropout_layer_create(&dropout_layer, probability, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create dropout layer."), error);
        goto cleanup;
    }

    error = logsoftmax_activation_create(&output_activation, (int64_t) 1);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create softmax activation."), error);
        goto cleanup;
    }

    error = linear_layer_create(&output_layer, 128, 10, runtime, datatype, true, output_activation, weight_init, bias_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create linear layer."), error);
        goto cleanup;
    }

    error = block_create(&block, 3, input_layer, dropout_layer, output_layer);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create block."), error);
        goto cleanup;
    }

    error = model_create(model, block);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create model."), error);
        goto cleanup;
    }
    
cleanup:

    free(mean);
    free(standard_deviation);
    free(probability);
    parameter_init_destroy(weight_init);
    parameter_init_destroy(bias_init);
    if (!error)
    {
        return error;
    }

    if (!input_layer)
    {
        activation_destroy(input_activation);
    }
    if (!output_layer)
    {
        activation_destroy(output_activation);
    }
    if (!block)
    {
        layer_destroy(input_layer);
        layer_destroy(output_layer);
        layer_destroy(dropout_layer);
    }
    block_destroy(block);

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
    int64_t epochs = 300;
    model_t *model = NULL;
    runtime_t runtime = OPENBLAS_RUNTIME;
    datatype_t datatype = FLOAT32;
    int64_t number_of_samples = 60000;
    batch_t *batch = NULL;
    int64_t batch_size = 128;
    bool_t shuffle = true;
    float32_t train_split = 0.7;
    float32_t valid_split = 0.2;
    float32_t test_split = 0.1;
    optimizer_t *optimizer = NULL;
    float32_t learning_rate = 0.0001;
    float32_t momentum = 0.0;
    float32_t dampening = 0.0;
    float32_t weight_decay = 0.0;
    bool_t nesterov = false;

    mkdir("img", S_IRWXU);

    plt_accuracies = malloc(epochs * datatype_size(datatype));
    plt_costs = malloc(epochs * datatype_size(datatype));
    plt_count = malloc(epochs * sizeof(float32_t));

    error = runtime_create_context(runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create context."), error);
        goto cleanup;
    }

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

    error = optimizer_stochastic_gradient_descent_create(&optimizer, datatype, (void *) &learning_rate, (void *) &momentum, (void *) &dampening, &weight_decay, nesterov);
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

    free(plt_accuracies);
    free(plt_costs);
    free(plt_count);

    runtime_destroy_context(runtime);
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
