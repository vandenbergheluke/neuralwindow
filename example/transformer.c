#include <simpsons_data.h>
#include <optimizer.h>
#include <plots.h>
#include <cost.h>
#include <metric.h>
#include <layer.h>
#include <tensor.h>
#include <buffer.h>
#include <view.h>
#include <init.h>
#include <errors.h>
#include <measure.h>
#include <random.h>
#include <string.h>

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
                error = bounded_plot("Transformer Training Accuracy",
                          "img/transformer_accuracy_train.png",
                          "Epoch", plt_count, epochs,
                          "Accuracy", plt_accuracies, epochs,
                          0.0, 1.0, datatype);
                if (error)
                {
                    error = ERROR(ERROR_METRICS, string_create("failed to plot accuracy."), error);
                    goto cleanup;
                }

                error = plot("Transformer Training Cost",
                          "img/transformer_cost_train.png",
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
                error = bounded_plot("Transformer Validation Accuracy",
                          "img/mnist_accuracy_valid.png",
                          "Epoch", plt_count, epochs,
                          "Accuracy", plt_accuracies, epochs,
                          0.0, 1.0, datatype);
                if (error)
                {
                    error = ERROR(ERROR_METRICS, string_create("failed to plot accuracy."), error);
                    goto cleanup;
                }

                error = plot("Transformer Validation Cost",
                          "img/transformer_cost_valid.png",
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

nw_error_t *generate(model_t *model, void *arguments, runtime_t runtime, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    with_no_gradient(true);
    model_inference(model, true);
    simpsons_dataset_t *simpsons_dataset = (simpsons_dataset_t *) arguments;
    void *data = NULL;
    tensor_t *x = NULL;
    tensor_t *y = NULL;
    tensor_t *probabilities = NULL;
    tensor_t *last_position_probabilities = NULL;
    tensor_t *sample = NULL;
    tensor_t *sample_expand = NULL;
    tensor_t *x_new = NULL;
    tensor_t *x_sliced = NULL;
    nw_error_t *error = NULL;
    bool_t copy = runtime == CU_RUNTIME;
    size_t size = datatype_size(datatype) * simpsons_dataset->prompt_length;
    void *token = NULL;

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

    for (int64_t i = 0; i < simpsons_dataset->prompt_length; ++i)
    {
        int64_t character_index = simpsons_dataset->character_to_integer[(int) simpsons_dataset->prompt[i]];
        switch (datatype)
        {
        case FLOAT32:
            ((float32_t *) data)[i] = (float32_t) character_index;
            break;
        case FLOAT64:
            ((float64_t *) data)[i] = (float64_t) character_index;
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
            goto cleanup;
        }
    }

    error = tensor_from_data(&x, data, runtime, datatype, 2, (int64_t[]) {1, simpsons_dataset->prompt_length}, copy, false, true);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    if (copy)
    {
        free(data);
    }

    fprintf(stdout, "Prompt: %s\nOutput: ", simpsons_dataset->prompt);
    for (int64_t i = 0; i < simpsons_dataset->max_tokens; ++i)
    {
        int64_t sequence_length = x->buffer->view->shape[1];

        error = model_forward(model, x, &y);
        if (error)
        {
            error = ERROR(ERROR_FORWARD, string_create("failed to get model prediction."), error);
            goto cleanup;
        }

        error = tensor_softmax(y, &probabilities, -1);
        if (error)
        {
            error = ERROR(ERROR_SOFTMAX, string_create("failed to softmax tensor."), error);
            goto cleanup;
        }

        error = tensor_slice(probabilities, &last_position_probabilities, (int64_t[]) {sequence_length - 1, sequence_length, 0, simpsons_dataset->vocabulary_size}, 4);
        if (error)
        {
            error = ERROR(ERROR_SLICE, string_create("failed to slice tensor."), error);
            goto cleanup;
        }

        error = tensor_multinomial_sample(last_position_probabilities, token);
        if (error)
        {
            error = ERROR(ERROR_SAMPLE, string_create("failed to sample tensor."), error);
            goto cleanup;
        }

        switch (datatype)
        {
        case FLOAT32:
            fprintf(stdout, "%c", simpsons_dataset->integer_to_character[(int64_t) (*(float32_t *) token)]);
            break;
        case FLOAT64:
            fprintf(stdout, "%c", simpsons_dataset->integer_to_character[(int64_t) (*(float64_t *) token)]);
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unsupported datatype."), NULL);
            goto cleanup;
        }

        error = tensor_constant(token, datatype, runtime, false, false, &sample);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_expand(sample, (int64_t[]){1, 1}, 2, &sample_expand);
        if (error)
        {
            error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
            goto cleanup;
        }

        if (sequence_length == simpsons_dataset->block_size)
        {
            error = tensor_slice(x, &x_sliced, (int64_t[]){0, 1, 1, simpsons_dataset->block_size}, 4);
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

        error = tensor_concatenation(x_sliced, sample_expand, &x_new, 1);
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
        tensor_destroy(last_position_probabilities);
        tensor_destroy(probabilities);
        tensor_destroy(sample);
        tensor_destroy(sample_expand);

        x = x_new;
        y = NULL;
        probabilities = NULL;
        last_position_probabilities = NULL;
        sample = NULL;
        sample_expand = NULL;
        x_new = NULL;
        x_sliced = NULL;
    }
    fprintf(stdout, "\n");

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
    // tensor_destroy(last_position_probabilities);
    tensor_destroy(sample);
    tensor_destroy(sample_expand);
    free(token);

    model_inference(model, false);
    with_no_gradient(false);

    return error;
}

nw_error_t *transformer_block_create(layer_t **layer, int64_t embedding_size, int64_t number_of_heads, void *dropout_probability,
                                     void *epsilon, parameter_init_t *weight_init, parameter_init_t *bias_init, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(dropout_probability, "dropout_probability");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");
    CHECK_NULL_ARGUMENT(weight_init, "weight_init");

    nw_error_t *error = NULL;
    layer_t *layer_norm_1 = NULL, *layer_norm_2 = NULL;
    layer_t *causal_multihead_self_attention = NULL;
    layer_t *linear_1 = NULL, *linear_2 = NULL;
    layer_t *gelu = NULL;
    layer_t *dropout = NULL;
    block_t *residual_block_1 = NULL;
    block_t *residual_block_2 = NULL;
    layer_t *residual_block_layer_1 = NULL;
    layer_t *residual_block_layer_2 = NULL;
    block_t *transformer_block = NULL;

    error = layer_normalization_layer_create(&layer_norm_1, (int64_t[]){embedding_size}, 1, epsilon, true, false, datatype, runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create layer normalization layer."), error);
        goto cleanup;
    }

    error = layer_normalization_layer_create(&layer_norm_2, (int64_t[]){embedding_size}, 1, epsilon, true, false, datatype, runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create layer normalization layer."), error);
        goto cleanup;
    }

    error = causal_multihead_self_attention_layer_create(&causal_multihead_self_attention, number_of_heads, embedding_size, dropout_probability, 
                                                         datatype, runtime, weight_init, bias_init, weight_init, bias_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create causal multihead attention layer."), error);
        goto cleanup;
    }

    error = linear_layer_create(&linear_1, embedding_size, 4 * embedding_size, runtime, datatype, weight_init, bias_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create linear layer."), error);
        goto cleanup;
    }

    error = linear_layer_create(&linear_2, 4 * embedding_size, embedding_size, runtime, datatype, weight_init, bias_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create linear layer."), error);
        goto cleanup;
    }

    error = dropout_layer_create(&dropout, dropout_probability, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create dropout layer."), error);
        goto cleanup;
    }

    error = gelu_activation_layer_create(&gelu);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create gelu activation."), error);
        goto cleanup;
    }

    error = block_create(&residual_block_1, 2, layer_norm_1, causal_multihead_self_attention);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create residual block."), error);
        goto cleanup;
    }

    error = block_create(&residual_block_2, 5, layer_norm_2, linear_1, gelu, linear_2, dropout);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create residual block."), error);
        goto cleanup;
    }

    error = residual_block_layer_create(&residual_block_layer_1, residual_block_1);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create residual block layer."), error);
        goto cleanup;
    }

    error = residual_block_layer_create(&residual_block_layer_2, residual_block_2);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create residual block layer."), error);
        goto cleanup;
    }

    error = block_create(&transformer_block, 2, residual_block_layer_1, residual_block_layer_2);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create transformer block."), error);
        goto cleanup;
    }

    error = block_layer_create(layer, transformer_block);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create transformer block layer."), error);
        goto cleanup;
    }

    return error;

cleanup:

    if (!residual_block_1)
    {
        layer_destroy(layer_norm_1);
        layer_destroy(causal_multihead_self_attention);
    }

    if (!residual_block_2)
    {
        layer_destroy(layer_norm_2);
        layer_destroy(linear_1);
        layer_destroy(gelu);
        layer_destroy(linear_2);
        layer_destroy(dropout);
    }

    if (!residual_block_layer_1)
    {
        block_destroy(residual_block_1);
    }

    if (!residual_block_layer_2)
    {
        block_destroy(residual_block_2);
    }

    if (!transformer_block)
    {
        layer_destroy(residual_block_layer_1);
        layer_destroy(residual_block_layer_2);
    }

    if (!*layer)
    {
        block_destroy(transformer_block);
    }

    return error;
}

nw_error_t *transformer_model_create(model_t **model, runtime_t runtime, datatype_t datatype, int64_t number_of_layers, int64_t vocabulary_size, 
                                     int64_t block_size, int64_t embedding_size, int64_t number_of_heads, void *dropout_probability, void *mean, void *standard_deviation, void *epsilon)
{
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = NULL;
    parameter_init_t *weight_init = NULL;
    parameter_init_t *bias_init = NULL;
    layer_t *transformer_embedding = NULL;
    layer_t *dropout = NULL;
    layer_t *layer_normalization = NULL;
    layer_t *linear = NULL;
    layer_t *transformer_blocks[number_of_layers];
    block_t *decoder_block = NULL;
    layer_t *decoder = NULL;
    layer_t *reshape = NULL;
    block_t *block = NULL;

    for (int64_t i = 0; i < number_of_layers; ++i)
    {
        transformer_blocks[i] = NULL;
    }
    
    error = normal_parameter_init(&weight_init, mean, standard_deviation, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create parameter initailizer."), error);
        goto cleanup;
    }

    error = transformer_embedding_layer_create(&transformer_embedding, vocabulary_size, embedding_size, block_size, datatype, runtime, weight_init, weight_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create transformer embedding layer."), error);
        goto cleanup;
    }

    error = dropout_layer_create(&dropout, dropout_probability, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create dropout layer."), error);
        goto cleanup;
    }

    error = layer_normalization_layer_create(&layer_normalization, (int64_t[]){embedding_size}, 1, epsilon, true, false, datatype, runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create layer normalization layer."), error);
        goto cleanup;
    }

    error = linear_layer_create(&linear, embedding_size, vocabulary_size, runtime, datatype, weight_init, bias_init);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create linear layer."), error);
        goto cleanup;
    }

    for (int64_t i = 0; i < number_of_layers; ++i)
    {
        error = transformer_block_create(&transformer_blocks[i], embedding_size, number_of_heads, dropout_probability, epsilon, weight_init, bias_init, datatype, runtime);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create transformer block."), error);
            goto cleanup;
        }
    } 

    error = block_create_from_array(&decoder_block, number_of_layers, transformer_blocks);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create decoder block."), error);
        goto cleanup;
    }

    error = block_layer_create(&decoder, decoder_block);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create decoder layer."), error);
        goto cleanup;
    }

    error = reshape_layer_create(&reshape, (int64_t[]){-1, vocabulary_size}, 2);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create reshape layer."), error);
        goto cleanup;
    }

    error = block_create(&block, 6, transformer_embedding, dropout, decoder, layer_normalization, linear, reshape);
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

    parameter_init_destroy(weight_init);
    parameter_init_destroy(bias_init);
    if (!error)
    {
        return error;
    }

    if (!block)
    {
        layer_destroy(transformer_embedding);
        layer_destroy(dropout);
        layer_destroy(layer_normalization);
        layer_destroy(linear);
        if (!decoder_block)
        {
            for (int64_t i = 0; i < number_of_layers; ++i)
            {
                layer_destroy(transformer_blocks[i]);
            }
        } 
        else if (!decoder)
        {
            block_destroy(decoder_block);
        }
    }
    block_destroy(block);

    return error;
}

void transformer_model_destroy(model_t *model)
{
    model_destroy(model);
}

int main(int argc, char **argv)
{
    simpsons_dataset_t simpsons_dataset = (simpsons_dataset_t) {
        .data_path = "../data/simpsons.txt",
        .data_file = NULL,
        .block_size = 256,
        .prompt = (argc == 3) ? argv[2] : "\n",
        .prompt_length = 1,
        .max_tokens = 1000,
    };

    nw_error_t *error = simpsons_setup(&simpsons_dataset);
    if (error)
    {
        error = ERROR(ERROR_SETUP, string_create("failed to setup."), error);
        goto cleanup;
    }

    int64_t epochs = 10;
    model_t *model = NULL;
    runtime_t runtime = MKL_RUNTIME;
    datatype_t datatype = FLOAT32;
    int64_t number_of_samples = simpsons_dataset.number_of_characters / (simpsons_dataset.block_size + 1);
    batch_t *batch = NULL;
    int64_t batch_size = 32;
    bool_t shuffle = true;
    float32_t train_split = 0.8;
    float32_t valid_split = 0.1;
    float32_t test_split = 0.1;
    optimizer_t *optimizer = NULL;
    float32_t learning_rate = 0.001;
    float32_t beta1 = 0.9;
    float32_t beta2 = 0.99;
    float32_t epsilon = 1e-5;
    float32_t weight_decay = 0.0;
    float32_t probability = 0.2;
    int64_t number_of_layers = 6;
    int64_t number_of_heads = 6;
    int64_t embedding_size = 384;
    float32_t gradient_threshold = 1.0;
    float32_t mean = 0.0;
    float32_t standard_deviation = 0.02;
    char_t* demo_var = getenv("DEMO");

    mkdir("img", S_IRWXU);
    mkdir("models", S_IRWXU);

    error = runtime_create_context(runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create context."), error);
        goto cleanup;
    }

    if (demo_var && strcmp(demo_var, "1") == 0)
    {
        if (argc < 2)
        {
            error = ERROR(ERROR_ARGUMENTS, string_create("invalid number of arguments."), error);
            goto cleanup;
        }

        error = model_load(&model, argv[1]);
        if (error)
        {
            error = ERROR(ERROR_LOAD, string_create("failed to load model."), error);
            goto cleanup;
        }
        
        error = generate(model, &simpsons_dataset, runtime, datatype);
        if (error)
        {
            error = ERROR(ERROR_GENERATE, string_create("failed to generate text."), error);
            goto cleanup;
        }
    }
    else
    {
        set_seed(1234);

        plt_accuracies = malloc(epochs * datatype_size(datatype));
        plt_costs = malloc(epochs * datatype_size(datatype));
        plt_count = malloc(epochs * sizeof(float32_t));

        error = batch_create(&batch, batch_size, datatype, runtime);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create batch."), error);
            goto cleanup;
        }

        error = transformer_model_create(&model, runtime, datatype, number_of_layers, simpsons_dataset.vocabulary_size, simpsons_dataset.block_size, 
                                        embedding_size, number_of_heads, (void *) &probability, (void *) &mean, (void *) &standard_deviation, (void *) &epsilon);
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

        error = fit(epochs, number_of_samples, batch, shuffle, train_split, valid_split, test_split, model, optimizer, &simpsons_dataset,
                    &simpsons_dataloader, &categorical_cross_entropy, &transformer_metrics, &generate, &gradient_threshold, true);
        if (error)
        {
            error = ERROR(ERROR_TRAIN, string_create("failed to fit model."), error);
            goto cleanup;
        }

        error = model_save(model, "models/transformer.bin");
        if (error)
        {
            error = ERROR(ERROR_SAVE, string_create("failed to save model."), error);
            goto cleanup;
        }

    }

    error = simpsons_teardown(&simpsons_dataset);
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
    transformer_model_destroy(model);

    if (error)
    {
        error_print(error);
        error_destroy(error);

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}