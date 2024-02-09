#include <mnist_data.h>
#include <plots.h>
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

void *plt_accuracies = NULL;
void *plt_costs = NULL;
float32_t *plt_count = NULL;

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
    CHECK_NULL_ARGUMENT(cost, "cost");
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

    error = tensor_softmax(y_pred, &probabilities, -1);
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

nw_error_t *mnist_model_create(model_t **model, runtime_t runtime, datatype_t datatype, int64_t batch_size)
{
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = NULL;

    layer_t *reshape_layer = NULL;
    layer_t *input_layer = NULL;
    layer_t *input_activation = NULL;
    layer_t *output_layer = NULL;
    block_t *block = NULL;
    parameter_init_t *weight_init = NULL;
    parameter_init_t *bias_init = NULL;
    void *mean = NULL;
    void *standard_deviation = NULL;
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

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) mean = (float32_t) 0.0;
        *(float32_t *) standard_deviation = (float32_t) 1.0;
        break;
    case FLOAT64:
        *(float64_t *) mean = (float64_t) 0.0;
        *(float64_t *) standard_deviation = (float64_t) 1.0;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    error = normal_parameter_init(&weight_init, mean, standard_deviation, datatype);
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

    error = rectified_linear_activation_layer_create(&input_activation);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create rectified linear activation."), error);
        goto cleanup;
    }

    error = reshape_layer_create(&reshape_layer, (int64_t[]){batch_size, 784}, 2);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create reshape layer."), error);
        goto cleanup;
    }

    error = linear_layer_create(&input_layer, 784, 128, runtime, datatype, weight_init, bias_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create linear layer."), error);
        goto cleanup;
    }

    error = linear_layer_create(&output_layer, 128, 10, runtime, datatype, weight_init, bias_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create linear layer."), error);
        goto cleanup;
    }

    error = block_create(&block, 4, reshape_layer, input_layer, input_activation, output_layer);
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
    parameter_init_destroy(weight_init);
    parameter_init_destroy(bias_init);
    if (!error)
    {
        return error;
    }

    if (!block)
    {
        layer_destroy(reshape_layer);
        layer_destroy(input_layer);
        layer_destroy(input_activation);
        layer_destroy(output_layer);
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
        .normalize = false,
    };

    nw_error_t *error = mnist_setup(&mnist_dataset);
    if (error)
    {
        error = ERROR(ERROR_SETUP, string_create("failed to setup."), error);
        goto cleanup;
    }

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

    error = mnist_model_create(&model, runtime, datatype, batch_size);
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
                &mnist_dataset, &mnist_dataloader, &categorical_cross_entropy, &mnist_metrics);
    if (error)
    {
        error = ERROR(ERROR_TRAIN, string_create("failed to fit model."), error);
        goto cleanup;
    }

    error = mnist_teardown(&mnist_dataset);
    if (error)
    {
        error = ERROR(ERROR_TEARDOWN, string_create("failed to teardown."), error);
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