#include <simpsons_data.h>
#include <optimizer.h>
#include <plots.h>
#include <cost.h>
#include <metric.h>
#include <layer.h>
#include <view.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

void *plt_accuracies = NULL;
void *plt_costs = NULL;
float32_t *plt_count = NULL;

nw_error_t *transformer_metrics(dataset_type_t dataset_type, const tensor_t *y_true, const tensor_t *y_pred, const tensor_t *cost,
                                int64_t epoch, int64_t epochs, int64_t iteration, int64_t iterations)
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
                error = bounded_plot("Transformer Training Accuracy", "img/transformer_accuracy_train.png", "Epoch", plt_count, epochs, "Accuracy", plt_accuracies, epochs, 0.0, 1.0, datatype);
                if (error)
                {
                    error = ERROR(ERROR_METRICS, string_create("failed to plot accuracy."), error);
                    goto cleanup;
                }

                error = plot("Transformer Training Cost", "img/transformer_cost_train.png", "Epoch", plt_count, epochs, "Cost", plt_costs, epochs, datatype);
                if (error)
                {
                    error = ERROR(ERROR_METRICS, string_create("failed to plot cost."), error);
                    goto cleanup;
                }
                break;
            case VALID:
                error = bounded_plot("Transformer Validation Accuracy", "img/transformer_accuracy_valid.png", "Epoch", plt_count, epochs, "Accuracy", plt_accuracies, epochs,
                          0.0, 1.0, datatype);
                if (error)
                {
                    error = ERROR(ERROR_METRICS, string_create("failed to plot accuracy."), error);
                    goto cleanup;
                }

                error = plot("Transformer Validation Cost", "img/transformer_cost_valid.png", "Epoch", plt_count, epochs, "Cost", plt_costs, epochs, datatype);
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

nw_error_t *generate(runtime_t runtime, datatype_t datatype, model_t *model, string_t prompt, int64_t prompt_length, 
                     int64_t max_tokens, int64_t vocabulary_size, int64_t block_size, int64_t character_to_integer[], char integer_to_character[])
{
    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(character_to_integer, "character_to_integer");
    CHECK_NULL_ARGUMENT(integer_to_character, "integer_to_character");

    with_no_gradient(true);
    model_inference(model, true);
    void *data = NULL;
    tensor_t *x = NULL;
    tensor_t *y = NULL;
    tensor_t *probabilities = NULL;
    tensor_t *sample = NULL;
    tensor_t *indicies = NULL;
    tensor_t *one_hot = NULL;
    tensor_t *one_hot_expand = NULL;
    tensor_t *x_new = NULL;
    tensor_t *x_sliced = NULL;
    nw_error_t *error = NULL;
    bool_t copy = runtime == CU_RUNTIME;
    size_t size = datatype_size(datatype) * prompt_length * vocabulary_size;
    void *token = NULL;
    void *start = NULL;
    void *stop = NULL;
    void *step = NULL;

    data = (void *) malloc(size);
    if (!data)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    size = datatype_size(datatype);
    token = (void *) malloc(size);
    if (!token)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    start = (void *) malloc(size);
    if (!start)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    stop = (void *) malloc(size);
    if (!stop)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    step = (void *) malloc(size);
    if (!step)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *((float32_t *) start) = (float32_t) 0.0;
        *((float32_t *) stop) = (float32_t) vocabulary_size;
        *((float32_t *) step) = (float32_t) 1.0;
        break;
    case FLOAT64:
        *((float64_t *) start) = (float64_t) 0.0;
        *((float64_t *) stop) = (float64_t) vocabulary_size;
        *((float64_t *) step) = (float64_t) 1.0;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
        goto cleanup;
    }


    for (int64_t i = 0; i < prompt_length; ++i)
    {
        int64_t character_index = character_to_integer[prompt[i]];
        for (int64_t j = 0; j < vocabulary_size; ++j)
        {
            switch (datatype)
            {
            case FLOAT32:
                ((float32_t *) data)[i * vocabulary_size + j] = (character_index == j) ? (float32_t) 1.0 : (float32_t) 0.0;
                break;
            case FLOAT64:
                ((float64_t *) data)[i * vocabulary_size + j] = (character_index == j) ? (float64_t) 1.0 : (float64_t) 0.0;
                break;
            default:
                error = ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
                goto cleanup;
            }
        }
    }

    error = tensor_from_data(&x, data, runtime, datatype, 3, (int64_t[]) {1, prompt_length, vocabulary_size}, copy, false, true);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    if (copy)
    {
        free(data);
    }

    fprintf(stdout, prompt);
    for (int64_t i = 0; i < max_tokens; ++i)
    {
        error = model_forward(model, x, &y);
        if (error)
        {
            error = ERROR(ERROR_FORWARD, string_create("failed to get model prediction."), error);
            goto cleanup;
        }

        error = tensor_exponential(y, &probabilities);
        if (error)
        {
            error = ERROR(ERROR_EXPONENTIAL, string_create("failed to exponentiate tensor."), error);
            goto cleanup;
        }

        error = tensor_multinomial(probabilities, &sample, 1);
        if (error)
        {
            error = ERROR(ERROR_SAMPLE, string_create("failed to sample tensor."), error);
            goto cleanup;
        }

        error = tensor_item(sample, token);
        if (error)
        {
            error = ERROR(ERROR_ITEM, string_create("failed to extract item from tensor."), error);
            goto cleanup;
        }

        switch (datatype)
        {
        case FLOAT32:
            fprintf(stdout, integer_to_character[(int64_t) (*(float32_t *) token)]);
            break;
        case FLOAT64:
            fprintf(stdout, integer_to_character[(int64_t) (*(float64_t *) token)]);
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
            goto cleanup;
        }

        error = tensor_arange(&indicies, start, stop, step, runtime, datatype, false, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_compare_equal(indicies, sample, &one_hot);
        if (error)
        {
            error = ERROR(ERROR_COMPARE_EQUAL, string_create("failed to compare equal tensors."), error);
            goto cleanup;
        }

        error = tensor_expand(one_hot, (int64_t[]){1, 1, vocabulary_size}, 3, &one_hot_expand);
        if (error)
        {
            error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
            goto cleanup;
        }

        if (x->buffer->view->shape[1] == block_size)
        {
            error = tensor_slice(x, &x_sliced, (int64_t[]){0, 1, 1, block_size, 0, vocabulary_size}, 6);
            if (error)
            {
                error = ERROR(ERROR_SLICE, string_create("failed to slice tensor."), error);
                goto cleanup;
            }
        }
        else
        {
            x_sliced = x;
        }

        error = tensor_concatenation(x_sliced, one_hot_expand, &x_new, 1);
        if (error)
        {
            error = ERROR(ERROR_CONCATENATION, string_create("failed to concatenate tensors."), error);
            goto cleanup;
        }

        if (x != x_sliced)
        {
            tensor_destroy(x_sliced);
        }
        tensor_destroy(x);
        tensor_destroy(y);
        tensor_destroy(probabilities);
        tensor_destroy(sample);
        tensor_destroy(indicies);
        tensor_destroy(one_hot);
        tensor_destroy(one_hot_expand);

        x = x_new;
        y = NULL;
        probabilities = NULL;
        sample = NULL;
        indicies = NULL;
        one_hot = NULL;
        one_hot_expand = NULL;
        x_new = NULL;
        x_sliced = NULL;
    }

cleanup:

    if (x != x_new)
    {
        tensor_destroy(x_new);
    }
    if (x != x_sliced)
    {
        tensor_destroy(x_sliced);
    }
    tensor_destroy(x);
    tensor_destroy(y);
    tensor_destroy(probabilities);
    tensor_destroy(sample);
    tensor_destroy(indicies);
    tensor_destroy(one_hot);
    tensor_destroy(one_hot_expand);
    free(token);
    free(start);
    free(stop);
    free(step);

    model_inference(model, false);
    with_no_gradient(false);

    return error;
}

int main(void)
{
    simpsons_dataset_t simpsons_dataset = (simpsons_dataset_t) {
        .data_path = "../data/simpsons.txt",
        .data_file = NULL,
        .block_size = 1024,
    };

    nw_error_t *error = NULL;
    int64_t epochs = 10;
    model_t *model = NULL;
    runtime_t runtime = OPENBLAS_RUNTIME;
    datatype_t datatype = FLOAT32;
    int64_t number_of_samples = 7116231 / (simpsons_dataset.block_size + 1);
    batch_t *batch = NULL;
    int64_t batch_size = 12;
    bool_t shuffle = true;
    float32_t train_split = 0.8;
    float32_t valid_split = 0.1;
    float32_t test_split = 0.1;
    optimizer_t *optimizer = NULL;
    float32_t learning_rate = 0.0001;
    float32_t beta1 = 0.9;
    float32_t beta2 = 0.99;
    float32_t epsilon = 1e-8;
    float32_t weight_decay = 0.0;

    mkdir("img", S_IRWXU);
    mkdir("txt", S_IRWXU);

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

    error = transformer_model_create(&model, runtime, datatype, batch_size);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create model."), error);
        goto cleanup;
    }

    error = optimizer_adam_create(&optimizer, datatype, (void *) &learning_rate, (void *) &beta1, (void *) &beta2, (void *) &weight_decay, (void *) &epsilon);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create optimizer."), error);
        goto cleanup;
    }

    error = fit(epochs, number_of_samples, batch, shuffle, train_split, valid_split, test_split, model, optimizer,
                &simpsons_dataset, &simpsons_setup, &simpsons_teardown, &simpsons_dataloader, &negative_log_likelihood, &transformer_metrics);
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
    transformer_model_destroy(model);

    if (error)
    {
        error_print(error);
        error_destroy(error);

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}