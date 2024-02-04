#include <mnist_data.h>
#include <plots.h>
#include <layer.h>
#include <optimizer.h>
#include <cost.h>
#include <init.h>
#include <function.h>
#include <random.h>

// nw_error_t *dcgan_metrics(dataset_type_t dataset_type, 
//                           const tensor_t *x,
//                           const tensor_t *cost,
//                           int64_t epoch,
//                           int64_t epochs,
//                           int64_t iteration,
//                           int64_t iterations)
// {
//     CHECK_NULL_ARGUMENT(y_pred, "y_pred");
//     CHECK_NULL_ARGUMENT(y_true, "y_true");
//     CHECK_NULL_ARGUMENT(cost, "cost");
//     static void *cost_data = NULL;
//     static int64_t time = 0; 
//     tensor_t *total_cost = NULL;
//     tensor_t *mean_cost = NULL;
//     nw_error_t *error = NULL;
//     runtime_t runtime = cost->buffer->storage->runtime;
//     datatype_t datatype = cost->buffer->storage->datatype;

//     if (!(iteration - 1))
//     {
//         size_t size = iterations * datatype_size(datatype);

//         cost_data = (void *) malloc(size);
//         if (!cost_data)
//         {
//             error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
//             goto cleanup;
//         }

//         time = get_time_nanoseconds();
//     }

//     error = tensor_item(cost, cost_data + (iteration - 1) * datatype_size(datatype));
//     if (error)
//     {
//         error = ERROR(ERROR_ITEM, string_create("failed to get tensor item."), NULL);
//         goto cleanup;
//     }

//     if (iteration == iterations)
//     {
//         int64_t shape[] = {iterations};
//         int64_t rank = 1;
        
//         error = tensor_from_data(&total_accuracy, accuracy_data, runtime, datatype, rank, shape, false, false, false);
//         if (error)
//         {
//             error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
//             goto cleanup;
//         }

//         error = tensor_from_data(&total_cost, cost_data, runtime, datatype, rank, shape, false, false, false);
//         if (error)
//         {
//             error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
//             goto cleanup;
//         }

//         error = tensor_mean(total_accuracy, &mean_accuracy, NULL, 0, false);
//         if (error)
//         {
//             error = ERROR(ERROR_DIVISION, string_create("failed to get mean of tensor."), error);
//             goto cleanup;
//         }

//         error = tensor_mean(total_cost, &mean_cost, NULL, 0, false);
//         if (error)
//         {
//             error = ERROR(ERROR_DIVISION, string_create("failed to get mean of tensor."), error);
//             goto cleanup;
//         }

//         error = tensor_item(mean_accuracy, plt_accuracies + datatype_size(datatype) * (epoch - 1));
//         if (error)
//         {
//             error = ERROR(ERROR_ITEM, string_create("failed to get tensor item."), NULL);
//             goto cleanup;
//         }

//         error = tensor_item(mean_cost, plt_costs + datatype_size(datatype) * (epoch - 1));
//         if (error)
//         {
//             error = ERROR(ERROR_ITEM, string_create("failed to get tensor item."), NULL);
//             goto cleanup;
//         }

//         plt_count[epoch - 1] = epoch;

//         if (epoch == epochs) {
//             switch (dataset_type)
//             {
//             case TRAIN:
//                 error = bounded_plot("MNIST Training Accuracy",
//                           "img/mnist_accuracy_train.png",
//                           "Epoch", plt_count, epochs,
//                           "Accuracy", plt_accuracies, epochs,
//                           0.0, 1.0, datatype);
//                 if (error)
//                 {
//                     error = ERROR(ERROR_METRICS, string_create("failed to plot accuracy."), error);
//                     goto cleanup;
//                 }

//                 error = plot("MNIST Training Cost",
//                           "img/mnist_cost_train.png",
//                           "Epoch", plt_count, epochs,
//                           "Cost", plt_costs, epochs,
//                           datatype);
//                 if (error)
//                 {
//                     error = ERROR(ERROR_METRICS, string_create("failed to plot cost."), error);
//                     goto cleanup;
//                 }
//                 break;
//             case VALID:
//                 error = bounded_plot("MNIST Validation Accuracy",
//                           "img/mnist_accuracy_valid.png",
//                           "Epoch", plt_count, epochs,
//                           "Accuracy", plt_accuracies, epochs,
//                           0.0, 1.0, datatype);
//                 if (error)
//                 {
//                     error = ERROR(ERROR_METRICS, string_create("failed to plot accuracy."), error);
//                     goto cleanup;
//                 }

//                 error = plot("MNIST Validation Cost",
//                           "img/mnist_cost_valid.png",
//                           "Epoch", plt_count, epochs,
//                           "Cost", plt_costs, epochs,
//                           datatype);
//                 if (error)
//                 {
//                     error = ERROR(ERROR_METRICS, string_create("failed to plot cost."), error);
//                     goto cleanup;
//                 }
//                 break;
//             case TEST:
//                 break;
//             default:
//                 error = ERROR(ERROR_DATASET_TYPE, string_create("unknown dataset type %d.", dataset_type), NULL);
//                 goto cleanup;
//             }
//         }

//         LOG("Dataset %s - %lu/%lu Epochs", dataset_type_string(dataset_type), epoch, epochs);
//         LOG_SCALAR_TENSOR(" - Cost", mean_cost);
//         LOG_SCALAR_TENSOR("- Accuracy", mean_accuracy);
//         LOG("- Time: %lfs", (float64_t) (get_time_nanoseconds() - time) * (float64_t) 1e-9);
//         LOG_NEWLINE;
//     }

// cleanup:

//     tensor_destroy(accuracy);
//     tensor_destroy(probabilities);
//     tensor_destroy(mean_accuracy);
//     tensor_destroy(mean_cost);
//     tensor_destroy(total_accuracy);
//     tensor_destroy(total_cost);

//     return error;
// }

nw_error_t *dcgan_model_create(model_t **generator, model_t **discriminator, runtime_t runtime, datatype_t datatype, int64_t noise_dimension, int64_t batch_size)
{
    CHECK_NULL_ARGUMENT(generator, "generator");
    CHECK_NULL_ARGUMENT(discriminator, "discriminator");

    nw_error_t *error = NULL;
    layer_t *convolution_transpose_2d_1 = NULL, *convolution_transpose_2d_2 = NULL, *convolution_transpose_2d_3 = NULL, *convolution_transpose_2d_4 = NULL;
    layer_t *relu_activation_1 = NULL, *relu_activation_2 = NULL, *relu_activation_3 = NULL;
    layer_t *convolution_2d_1 = NULL, *convolution_2d_2 = NULL, *convolution_2d_3 = NULL, *convolution_2d_4 = NULL;
    layer_t *batch_normalization_2d_1 = NULL, *batch_normalization_2d_2 = NULL, *batch_normalization_2d_3 = NULL, 
            *batch_normalization_2d_4 = NULL, *batch_normalization_2d_5 = NULL;
    layer_t *leaky_relu_activation_1 = NULL, *leaky_relu_activation_2 = NULL, *leaky_relu_activation_3 = NULL;
    layer_t *tanh_activation = NULL;
    layer_t *sigmoid_activation = NULL;
    layer_t *reshape = NULL;
    block_t *generator_block = NULL, *discriminator_block = NULL;
    void *momentum = NULL, *epsilon = NULL, *c = NULL, *gain = NULL, *mean = NULL, *std = NULL;
    parameter_init_t *generator_weight_init = NULL, *discriminator_weight_init = NULL;
    size_t size = datatype_size(datatype);

    momentum = (void *) malloc(size);
    if (!momentum)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    epsilon = (void *) malloc(size);
    if (!epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    c = (void *) malloc(size);
    if (!c)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    gain = (void *) malloc(size);
    if (!gain)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    mean = (void *) malloc(size);
    if (!mean)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    std = (void *) malloc(size);
    if (!std)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) momentum = (float32_t) 0.1;
        *(float32_t *) epsilon = (float32_t) 1e-5;
        *(float32_t *) c = (float32_t) 0.2;
        *(float32_t *) mean = (float32_t) 0.0;
        *(float32_t *) std = (float32_t) 0.02;
        break;
    case FLOAT64:
        *(float64_t *) momentum = (float64_t) 0.1;
        *(float64_t *) epsilon = (float64_t) 1e-5;
        *(float64_t *) c = (float64_t) 0.2;
        *(float64_t *) mean = (float64_t) 0.0;
        *(float64_t *) std = (float64_t) 0.02;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("failed to calculate gain."), error);
        goto cleanup;
        
    }

    error = calculate_gain(ACTIVATION_RECTIFIED_LINEAR, datatype, gain, NULL);
    if (error)
    {
        error = ERROR(ERROR_GAIN, string_create("failed to calculate gain."), error);
        goto cleanup;
    }

    error = calculate_gain(ACTIVATION_LEAKY_RECTIFIED_LINEAR, datatype, gain, c);
    if (error)
    {
        error = ERROR(ERROR_GAIN, string_create("failed to calculate gain."), error);
        goto cleanup;
    }

    // error = kaiming_normal_parameter_init(&generator_weight_init, gain, false, datatype);
    // if (error)
    // {
    //     error = ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    //     goto cleanup;
    // }

    // error = kaiming_normal_parameter_init(&discriminator_weight_init, gain, false, datatype);
    // if (error)
    // {
    //     error = ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
    //     goto cleanup;
    // }

    error = normal_parameter_init(&generator_weight_init, mean, std, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
        goto cleanup;
    }

    error = normal_parameter_init(&discriminator_weight_init, mean, std, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
        goto cleanup;
    }

    error = convolution_transpose_2d_layer_create(&convolution_transpose_2d_1, 4, 0, 1, noise_dimension, 256, runtime, datatype, generator_weight_init, NULL);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create convolution transpose 2d layer."), error);
        goto cleanup;
    }

    error = convolution_transpose_2d_layer_create(&convolution_transpose_2d_2, 3, 1, 2, 256, 128, runtime, datatype, generator_weight_init, NULL);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create convolution transpose 2d layer."), error);
        goto cleanup;
    }

    error = convolution_transpose_2d_layer_create(&convolution_transpose_2d_3, 4, 1, 2, 128, 64, runtime, datatype, generator_weight_init, NULL);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create convolution transpose 2d layer."), error);
        goto cleanup;
    }

    error = convolution_transpose_2d_layer_create(&convolution_transpose_2d_4, 4, 1, 2, 64, 1, runtime, datatype, generator_weight_init, NULL);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create convolution transpose 2d layer."), error);
        goto cleanup;
    }

    error = convolution_2d_layer_create(&convolution_2d_1, 4, 1, 2, 1, 64, runtime, datatype, discriminator_weight_init, NULL);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create convolution 2d layer."), error);
        goto cleanup;
    }

    error = convolution_2d_layer_create(&convolution_2d_2, 4, 1, 2, 64, 128, runtime, datatype, discriminator_weight_init, NULL);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create convolution 2d layer."), error);
        goto cleanup;
    }

    error = convolution_2d_layer_create(&convolution_2d_3, 4, 1, 2, 128, 256, runtime, datatype, discriminator_weight_init, NULL);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create convolution 2d layer."), error);
        goto cleanup;
    }

    error = convolution_2d_layer_create(&convolution_2d_4, 3, 0, 1, 256, 1, runtime, datatype, discriminator_weight_init, NULL);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create convolution 2d layer."), error);
        goto cleanup;
    }

    error = batch_normalization_2d_layer_create(&batch_normalization_2d_1, 256, momentum, epsilon, true, true, datatype, runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create batch normalization 2d layer."), error);
        goto cleanup;
    }

    error = batch_normalization_2d_layer_create(&batch_normalization_2d_2, 128, momentum, epsilon, true, true, datatype, runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create batch normalization 2d layer."), error);
        goto cleanup;
    }

    error = batch_normalization_2d_layer_create(&batch_normalization_2d_3, 64, momentum, epsilon, true, true, datatype, runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create batch normalization 2d layer."), error);
        goto cleanup;
    }

    error = batch_normalization_2d_layer_create(&batch_normalization_2d_4, 128, momentum, epsilon, true, true, datatype, runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create batch normalization 2d layer."), error);
        goto cleanup;
    }

    error = batch_normalization_2d_layer_create(&batch_normalization_2d_5, 256, momentum, epsilon, true, true, datatype, runtime);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create batch normalization 2d layer."), error);
        goto cleanup;
    }

    error = rectified_linear_activation_layer_create(&relu_activation_1);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create relu activation layer."), error);
        goto cleanup;
    }

    error = rectified_linear_activation_layer_create(&relu_activation_2);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create relu activation layer."), error);
        goto cleanup;
    }

    error = rectified_linear_activation_layer_create(&relu_activation_3);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create relu activation layer."), error);
        goto cleanup;
    }

    error = leaky_rectified_linear_activation_layer_create(&leaky_relu_activation_1, c, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create leaky relu activation layer."), error);
        goto cleanup;
    }

    error = leaky_rectified_linear_activation_layer_create(&leaky_relu_activation_2, c, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create leaky relu activation layer."), error);
        goto cleanup;
    }

    error = leaky_rectified_linear_activation_layer_create(&leaky_relu_activation_3, c, datatype);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create leaky relu activation layer."), error);
        goto cleanup;
    }

    error = tanh_activation_layer_create(&tanh_activation);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tanh activation layer."), error);
        goto cleanup;
    }

    error = sigmoid_activation_layer_create(&sigmoid_activation);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create sigmoid activation layer."), error);
        goto cleanup;
    }

    error = reshape_layer_create(&reshape, (int64_t[]){batch_size}, 1);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create reshape layer."), error);
        goto cleanup;
    }

    error = block_create(&generator_block, 11, 
                        convolution_transpose_2d_1, batch_normalization_2d_1, relu_activation_1,
                        convolution_transpose_2d_2, batch_normalization_2d_2, relu_activation_2,
                        convolution_transpose_2d_3, batch_normalization_2d_3, relu_activation_3,
                        convolution_transpose_2d_4, tanh_activation);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create block."), error);
        goto cleanup;
    }

    error = block_create(&discriminator_block, 11, 
                        convolution_2d_1, leaky_relu_activation_1,
                        convolution_2d_2, batch_normalization_2d_4, leaky_relu_activation_2,
                        convolution_2d_3, batch_normalization_2d_5, leaky_relu_activation_3,
                        convolution_2d_4, sigmoid_activation, reshape);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create block."), error);
        goto cleanup;
    }

    error = model_create(generator, generator_block);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create model."), error);
        goto cleanup;
    }

    error = model_create(discriminator, discriminator_block);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create model."), error);
        goto cleanup;
    }

cleanup:

    free(c);
    free(gain);
    free(momentum);
    free(epsilon);
    free(mean);
    free(std);
    parameter_init_destroy(generator_weight_init);
    parameter_init_destroy(discriminator_weight_init);
    if (!error)
    {
        return error;
    }

    if (!generator_block)
    {
        layer_destroy(convolution_transpose_2d_1);
        layer_destroy(convolution_transpose_2d_2);
        layer_destroy(convolution_transpose_2d_3);
        layer_destroy(convolution_transpose_2d_4);
        layer_destroy(batch_normalization_2d_1);
        layer_destroy(batch_normalization_2d_2);
        layer_destroy(batch_normalization_2d_3);
        layer_destroy(relu_activation_1);
        layer_destroy(relu_activation_2);
        layer_destroy(relu_activation_3);
        layer_destroy(tanh_activation);
    }
    block_destroy(generator_block);

    if (!discriminator_block)
    {
        layer_destroy(convolution_2d_1);
        layer_destroy(convolution_2d_2);
        layer_destroy(convolution_2d_3);
        layer_destroy(convolution_2d_4);
        layer_destroy(batch_normalization_2d_4);
        layer_destroy(batch_normalization_2d_5);
        layer_destroy(leaky_relu_activation_1);
        layer_destroy(leaky_relu_activation_2);
        layer_destroy(leaky_relu_activation_3);
        layer_destroy(sigmoid_activation);
        layer_destroy(reshape);
    }
    block_destroy(discriminator_block);

    return error;
}

void dcgan_model_destroy(model_t *generator, model_t *discriminator)
{
    model_destroy(generator);
    model_destroy(discriminator);
}

nw_error_t *dcgan_fit(int64_t epochs, int64_t number_of_samples, batch_t *batch, bool_t shuffle, int64_t noise_dimension,
                      model_t *generator, model_t *discriminator, optimizer_t *generator_optimizer, 
                      optimizer_t *discriminator_optimizer, void * arguments,
                      nw_error_t *(*setup)(void *), nw_error_t *(*teardown)(void *), nw_error_t *(*dataloader)(int64_t, batch_t *, void *))
{
    nw_error_t *error = NULL;
    int64_t iterations = number_of_samples / batch->batch_size;
    int64_t indicies[iterations];
    void *mean = NULL, *standard_deviation = NULL, *lower_bound = NULL, *upper_bound = NULL;
    size_t size = datatype_size(batch->datatype);

    mean = (void *) malloc(size);
    if (!mean)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    standard_deviation = (void *) malloc(size);
    if (!standard_deviation)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    lower_bound = (void *) malloc(size);
    if (!lower_bound)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    upper_bound = (void *) malloc(size);
    if (!upper_bound)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    switch (batch->datatype)
    {
    case FLOAT32:
        *(float32_t *) mean = (float32_t) 0.0;
        *(float32_t *) standard_deviation = (float32_t) 1.0;
        *(float32_t *) lower_bound = (float32_t) 0.8;
        *(float32_t *) upper_bound = (float32_t) 1.0;
        break;
    case FLOAT64:
        *(float64_t *) mean = (float64_t) 0.0;
        *(float64_t *) standard_deviation = (float64_t) 1.0;
        *(float64_t *) lower_bound = (float64_t) 0.8;
        *(float64_t *) upper_bound = (float64_t) 1.0;
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) batch->datatype), NULL);
    }

    error = (*setup)(arguments);
    if (error)
    {
        return ERROR(ERROR_SETUP, string_create("failed to setup."), error);
    }

    for (int64_t i = 0; i < iterations; ++i)
    {
        indicies[i] = i;
    }

    if (shuffle)
    {
        shuffle_array(indicies, iterations);
    }

    for (int64_t i = 0; i < epochs; ++i)
    {
        LOG("%ld/%ld Epochs", i, epochs);
        LOG_NEWLINE;
        for (int64_t j = 0; j < iterations; ++j)
        {
            LOG("%ld/%ld Batches", j, iterations);
            LOG_NEWLINE;
            error = (*dataloader)(indicies[j] * batch->batch_size, batch, arguments);
            if (error)
            {
                return ERROR(ERROR_LOAD, string_create("failed to load batch."), error);
            }

            // Train discriminator with real images.
            error = zero_gradient_model(discriminator);
            if (error)
            {
                return ERROR(ERROR_ZERO_GRADIENT, string_create("failed zero gradient."), error);
            }

            tensor_t *real_images = batch->x;
            tensor_t *real_labels = NULL;
            tensor_t *real_output = NULL;
            tensor_t *discriminator_cost_real = NULL;

            error = tensor_create_uniform(&real_labels, (int64_t[]){batch->batch_size}, 1, batch->runtime, batch->datatype, false, false, lower_bound, upper_bound);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            }

            error = model_forward(discriminator, real_images, &real_output);
            if (error)
            {
                return ERROR(ERROR_FORWARD, string_create("failed to apply model."), error);
            }

            error = binary_cross_entropy(real_labels, real_output, &discriminator_cost_real);
            if (error)
            {
                return ERROR(ERROR_CRITERION, string_create("failed to compute cost."), error);
            }

            LOG_SCALAR_TENSOR("Discriminator Cost Real", discriminator_cost_real);
            LOG_NEWLINE;

            error = tensor_backward(discriminator_cost_real, NULL);
            if (error)
            {
                return ERROR(ERROR_BACKWARD, string_create("failed to apply backpropogation."), error);
            }

            // Train discriminator with real images.
            tensor_t *noise = NULL;
            tensor_t *fake_images = NULL;
            tensor_t *fake_labels = NULL;
            tensor_t *fake_output = NULL;
            tensor_t *fake_images_detached = NULL;
            tensor_t *discriminator_cost_fake = NULL;

            error = tensor_create_normal(&noise, (int64_t[]){batch->batch_size, noise_dimension, 1, 1}, 
                                         4, batch->runtime, batch->datatype, false, true, mean, standard_deviation);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            }

            error = model_forward(generator, noise, &fake_images);
            if (error)
            {
                return ERROR(ERROR_FORWARD, string_create("failed to apply model."), error);
            }

            error = save_tensor_grayscale_to_png_file(fake_images, "image.png");
            if (error)
            {
                return ERROR(ERROR_FILE, string_create("failed to write png file."), error);
            }

            error = tensor_create_zeroes(&fake_labels, (int64_t[]){batch->batch_size}, 1, batch->runtime, batch->datatype, false, false);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            }

            error = tensor_as_tensor(fake_images, &fake_images_detached);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            }

            error = model_forward(discriminator, fake_images_detached, &fake_output);
            if (error)
            {
                return ERROR(ERROR_FORWARD, string_create("failed to apply model."), error);
            }

            error = binary_cross_entropy(fake_labels, fake_output, &discriminator_cost_fake);
            if (error)
            {
                return ERROR(ERROR_CRITERION, string_create("failed to compute cost."), error);
            }

            LOG_SCALAR_TENSOR("Discriminator Cost Fake", discriminator_cost_fake);
            LOG_NEWLINE;

            error = tensor_backward(discriminator_cost_fake, NULL);
            if (error)
            {
                return ERROR(ERROR_BACKWARD, string_create("failed to apply backpropogation."), error);
            }

            error = update_model(discriminator_optimizer, discriminator);
            if (error)
            {
                return ERROR(ERROR_STEP, string_create("failed to apply optimizer step."), error);
            }

            // Train generator
            fake_output = NULL;
            fake_labels = NULL;
            tensor_t *generator_loss = NULL;

            error = zero_gradient_model(generator);
            if (error)
            {
                return ERROR(ERROR_ZERO_GRADIENT, string_create("failed zero gradient."), error);
            }

            error = tensor_create_ones(&fake_labels, (int64_t[]){batch->batch_size}, 1, batch->runtime, batch->datatype, false, false);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            }

            error = model_forward(discriminator, fake_images, &fake_output);
            if (error)
            {
                return ERROR(ERROR_FORWARD, string_create("failed to apply model."), error);
            }

            error = binary_cross_entropy(fake_labels, fake_output, &generator_loss);
            if (error)
            {
                return ERROR(ERROR_CRITERION, string_create("failed to compute cost."), error);
            }

            LOG_SCALAR_TENSOR("Generator Loss", generator_loss);
            LOG_NEWLINE;

            error = tensor_backward(generator_loss, NULL);
            if (error)
            {
                return ERROR(ERROR_BACKWARD, string_create("failed to apply backpropogation."), error);
            }

            error = update_model(generator_optimizer, generator);
            if (error)
            {
                return ERROR(ERROR_STEP, string_create("failed to apply optimizer step."), error);
            }

            tensor_destroy(batch->x);
            tensor_destroy(batch->y);
            tensor_destroy(noise);
            batch->x = NULL;
            batch->y = NULL;
        }
    }

    free(mean);
    free(standard_deviation);
    free(lower_bound);
    free(upper_bound);

    error = (*teardown)(arguments);
    if (error)
    {
        return ERROR(ERROR_TEARDOWN, string_create("failed to teardown."), error);
    }
     
    return error;
}

int main(void)
{
    set_seed(1234);
    mnist_dataset_t mnist_dataset = (mnist_dataset_t) {
        .images_path = "../data/train-images-idx3-ubyte",
        .labels_path = "../data/train-labels-idx1-ubyte",
        .images_file = NULL,
        .labels_file = NULL,
        .normalize = true,
    };

    nw_error_t *error = NULL;
    int64_t epochs = 100;
    model_t *generator = NULL, *discriminator = NULL;
    runtime_t runtime = MKL_RUNTIME;
    datatype_t datatype = FLOAT32;
    int64_t number_of_samples = 60000;
    batch_t *batch = NULL;
    int64_t batch_size = 64;
    int64_t noise_dimension = 100;
    bool_t shuffle = true;
    optimizer_t *generator_optimizer = NULL, *discriminator_optimizer = NULL;
    float32_t learning_rate = 0.0002;
    float32_t beta1 = 0.5;
    float32_t beta2 = 0.999;
    float32_t epsilon = 1e-8;
    float32_t weight_decay = 0;

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

    error = dcgan_model_create(&generator, &discriminator, runtime, datatype, noise_dimension, batch_size);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create model."), error);
        goto cleanup;
    }

    error = optimizer_adam_create(&generator_optimizer, datatype, (void *) &learning_rate, (void *) &beta1, (void *) &beta2, (void *) &weight_decay, (void *) &epsilon);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create optimizer."), error);
        goto cleanup;
    }

    error = optimizer_adam_create(&discriminator_optimizer, datatype, (void *) &learning_rate, (void *) &beta1, (void *) &beta2, (void *) &weight_decay, (void *) &epsilon);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create optimizer."), error);
        goto cleanup;
    }

    error = dcgan_fit(epochs, number_of_samples, batch, shuffle, noise_dimension, generator, discriminator, generator_optimizer, discriminator_optimizer,
                &mnist_dataset, &mnist_setup, &mnist_teardown, &mnist_dataloader);
    if (error)
    {
        error = ERROR(ERROR_TRAIN, string_create("failed to fit model."), error);
        goto cleanup;
    }

cleanup:

    runtime_destroy_context(runtime);
    optimizer_destroy(generator_optimizer);
    optimizer_destroy(discriminator_optimizer);
    batch_destroy(batch);
    dcgan_model_destroy(generator, discriminator);

    if (error)
    {
        error_print(error);
        error_destroy(error);

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}