#include <layer.h>
#include <init.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <math.h>
#include <string.h>

extern bool_t no_gradient;

nw_error_t *model_create(model_t **model, block_t *block)
{
    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(block, "block");

    *model = (model_t *) malloc(sizeof(model_t));
    if (!*model)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(model_t)), NULL);
    }

    (*model)->block = block;

    return NULL;
}

void model_destroy(model_t *model) 
{
    if (model)
    {
        block_destroy(model->block);
        free(model);
    }
}

nw_error_t *block_create(block_t **block, int64_t depth, ...)
{
    CHECK_NULL_ARGUMENT(block, "block");
    
    va_list valist;
    va_start(valist, depth);

    *block = (block_t *) malloc(sizeof(block_t));
    if (!*block)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(block_t)), NULL);
    }

    (*block)->depth = depth;
    (*block)->layers = (layer_t **) malloc(depth * sizeof(layer_t *));
    if (!(*block)->layers)
    {
        free(*block);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(depth * sizeof(layer_t *))), NULL);
    }

    for (int64_t i = 0; i < depth; ++i)
    {
        (*block)->layers[i] = (layer_t *) va_arg(valist, layer_t *);
    }

    va_end(valist);

    return NULL;
}

void block_destroy(block_t *block)
{
    if (block)
    {
        if (block->layers)
        {
            for (int64_t i = 0; i < block->depth; ++i)
            {
                layer_destroy(block->layers[i]);
            }
            free(block->layers);
        }
        free(block);
    }
}

nw_error_t *layer_create(layer_t **layer, transform_t *transform, transform_type_t transform_type)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(transform, "transform");

    *layer = (layer_t *) malloc(sizeof(layer_t));
    if (!*layer)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_t)), NULL);
    }

    (*layer)->transform = transform;
    (*layer)->transform_type = transform_type;

    return NULL;
}

void layer_destroy(layer_t *layer)
{
    if (layer)
    {
        transform_destroy(layer->transform, layer->transform_type);
        free(layer);
    }
}

nw_error_t *transform_create(transform_t **transform, transform_type_t transform_type, void *type_transform)
{
    CHECK_NULL_ARGUMENT(transform, "transform");
    CHECK_NULL_ARGUMENT(type_transform, "type_transform");

    *transform = (transform_t *) malloc(sizeof(transform_t));
    if (!*transform)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(transform_t)), NULL);
    }

    switch (transform_type)
    {
    case LINEAR:
        (*transform)->linear = (linear_t *) type_transform;
        break;
    case CONVOLUTION_2D:
    case CONVOLUTION_TRANSPOSE_2D:
        (*transform)->convolution_2d = (convolution_2d_t *) type_transform;
        break;
    case DROPOUT:
        (*transform)->dropout = (dropout_t *) type_transform;
        break;
    case BATCH_NORMALIZATION_2D:
        (*transform)->batch_normalization_2d = (batch_normalization_2d_t *) type_transform;
        break;
    case LAYER_NORMALIZATION:
        (*transform)->layer_normalization = (layer_normalization_t *) type_transform;
        break;
    case RESHAPE:
        (*transform)->reshape = (reshape_t *) type_transform;
        break;
    case ACTIVATION:
        (*transform)->activation = (activation_t *) type_transform;
        break;
    case BLOCK:
        (*transform)->block = (block_t *) type_transform;
        break;
    default:
        free(*transform);
        return ERROR(ERROR_LAYER_TYPE, string_create("unknown transform type %d.", (int) transform_type), NULL);
    }

    return NULL;
}

void transform_destroy(transform_t *transform, transform_type_t transform_type)
{
    if (transform)
    {
        switch (transform_type)
        {
        case LINEAR:
            linear_destroy(transform->linear);
            break;
        case CONVOLUTION_2D:
        case CONVOLUTION_TRANSPOSE_2D:
            convolution_2d_destroy(transform->convolution_2d);
            break;
        case DROPOUT:
            dropout_destroy(transform->dropout);
            break;
        case BATCH_NORMALIZATION_2D:
            batch_normalization_2d_destroy(transform->batch_normalization_2d);
            break;
        case LAYER_NORMALIZATION:
            layer_normalization_destroy(transform->layer_normalization);
            break;
        case RESHAPE:
            reshape_destroy(transform->reshape);
            break;
            break;
        case ACTIVATION:
            activation_destroy(transform->activation);
            break;
        case BLOCK:
            block_destroy(transform->block);
            break;
        default:
            break;
        }
        free(transform);
    }
}

string_t transform_type_string(transform_type_t transform_type)
{
    switch (transform_type)
    {
    case LINEAR:
        return "LINEAR";
    case CONVOLUTION_2D:
        return "CONVOLUTION_2D";
    case CONVOLUTION_TRANSPOSE_2D:
        return "CONVOLUTION_TRANSPOSE_2D";
    case DROPOUT:
        return "DROPOUT";
    case BATCH_NORMALIZATION_2D:
        return "BATCH_NORMALIZATION_2D";
    case LAYER_NORMALIZATION:
        return "LAYER_NORMALIZATION";
    case RESHAPE:
        return "RESHAPE";
    case ACTIVATION:
        return "ACTIVATION";
    case BLOCK:
        return "BLOCK";
    default:
        return "TRANSFORM_TYPE";
    }
}

nw_error_t *linear_create(linear_t **linear, tensor_t *weights, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(weights, "weights");

    *linear = (linear_t *) malloc(sizeof(linear_t));
    if (!*linear)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(linear_t)), NULL);
    }

    (*linear)->weights = weights;
    (*linear)->bias = bias;

    return NULL;
}

void linear_destroy(linear_t *linear)
{
    if (linear)
    {
        tensor_destroy(linear->weights);
        tensor_destroy(linear->bias);
        free(linear);
    }
}

nw_error_t *convolution_2d_create(convolution_2d_t **convolution_2d, int64_t padding, int64_t stride, tensor_t *kernel, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(kernel, "kernel");

    *convolution_2d = (convolution_2d_t *) malloc(sizeof(convolution_2d_t));
    if (!*convolution_2d)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(convolution_2d_t)), NULL);
    }

    (*convolution_2d)->padding = padding;
    (*convolution_2d)->stride = stride;
    (*convolution_2d)->kernel = kernel;
    (*convolution_2d)->bias = bias;

    return NULL;
}

void convolution_2d_destroy(convolution_2d_t *convolution_2d)
{
    if (convolution_2d)
    {
        tensor_destroy(convolution_2d->kernel);
        tensor_destroy(convolution_2d->bias);
        free(convolution_2d);
    }
}

nw_error_t *dropout_create(dropout_t **dropout, void *probability, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(dropout, "dropout");
    CHECK_NULL_ARGUMENT(probability, "probability");

    *dropout = (dropout_t *) malloc(sizeof(dropout_t));
    if (!*dropout)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(dropout_t)), NULL);
    }

    (*dropout)->probability = (void *) malloc(datatype_size(datatype));
    if (!(*dropout)->probability)
    {
        free(*dropout);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
    }

    memcpy((*dropout)->probability, probability, datatype_size(datatype));
    (*dropout)->inference = false;
    (*dropout)->datatype = datatype;

    return NULL;
}

void dropout_destroy(dropout_t *dropout)
{
    if (dropout)
    {
        free(dropout->probability);
        free(dropout);
    }
}

nw_error_t *batch_normalization_2d_create(batch_normalization_2d_t **batch_normalization_2d, int64_t number_of_features,
                                          void *momentum, void *epsilon, bool_t track_running_stats,
                                          bool_t affine, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(batch_normalization_2d, "batch_normalization_2d");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");
    if (track_running_stats)
    {
        CHECK_NULL_ARGUMENT(momentum, "momentum");
    }

    *batch_normalization_2d = (batch_normalization_2d_t *) malloc(sizeof(batch_normalization_2d_t));
    if (!*batch_normalization_2d)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(batch_normalization_2d_t)), NULL);
    }

    nw_error_t *error = NULL;

    (*batch_normalization_2d)->momentum = NULL;
    (*batch_normalization_2d)->epsilon = NULL;
    (*batch_normalization_2d)->bias = NULL;
    (*batch_normalization_2d)->weights = NULL;
    (*batch_normalization_2d)->running_mean = NULL;
    (*batch_normalization_2d)->running_variance = NULL;
    (*batch_normalization_2d)->track_running_stats = track_running_stats;
    (*batch_normalization_2d)->inference = false;

    (*batch_normalization_2d)->momentum = (void *) malloc(datatype_size(datatype));
    if (!(*batch_normalization_2d)->momentum)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
        goto cleanup;
    }
    memcpy((*batch_normalization_2d)->momentum, momentum, datatype_size(datatype));

    (*batch_normalization_2d)->epsilon = (void *) malloc(datatype_size(datatype));
    if (!(*batch_normalization_2d)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
        goto cleanup;
    }
    memcpy((*batch_normalization_2d)->epsilon, epsilon, datatype_size(datatype));

    if (affine)
    {
        error = tensor_create_ones(&(*batch_normalization_2d)->weights, (int64_t[]){number_of_features}, 1, runtime, datatype, true, true);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_create_zeroes(&(*batch_normalization_2d)->bias, (int64_t[]){number_of_features}, 1, runtime, datatype, true, true);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    error = tensor_create_ones(&(*batch_normalization_2d)->running_variance, (int64_t[]){number_of_features}, 1, runtime, datatype, false, true);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_create_zeroes(&(*batch_normalization_2d)->running_mean, (int64_t[]){number_of_features}, 1, runtime, datatype, false, true);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }
    
    return error;

cleanup:

    batch_normalization_2d_destroy(*batch_normalization_2d);

    return error;
}

void batch_normalization_2d_destroy(batch_normalization_2d_t *batch_normalization_2d)
{
    if (batch_normalization_2d)
    {
        tensor_destroy(batch_normalization_2d->weights);
        tensor_destroy(batch_normalization_2d->bias);
        tensor_destroy(batch_normalization_2d->running_mean);
        tensor_destroy(batch_normalization_2d->running_variance);
        free(batch_normalization_2d->epsilon);
        free(batch_normalization_2d->momentum);
        free(batch_normalization_2d);
    }
}

nw_error_t *layer_normalization_create(layer_normalization_t **layer_normalization, const int64_t *normalized_shape, int64_t length,
                                        void *epsilon, bool_t elementwise_affine, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(layer_normalization, "layer_normalization");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");
    CHECK_NULL_ARGUMENT(normalized_shape, "normalized_shape");

    *layer_normalization = (layer_normalization_t *) malloc(sizeof(layer_normalization_t));
    if (!*layer_normalization)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_normalization_t)), NULL);
    }

    nw_error_t *error = NULL;

    (*layer_normalization)->epsilon = NULL;
    (*layer_normalization)->bias = NULL;
    (*layer_normalization)->weights = NULL;
    (*layer_normalization)->length = length;
    (*layer_normalization)->normalized_shape = NULL;
    
    size_t size = datatype_size(datatype);
    (*layer_normalization)->epsilon = (void *) malloc(size);
    if (!(*layer_normalization)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*layer_normalization)->epsilon, epsilon, size);

    size = length * sizeof(int64_t);
    (*layer_normalization)->normalized_shape = (int64_t *) malloc(size);
    if (!(*layer_normalization)->normalized_shape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*layer_normalization)->normalized_shape, normalized_shape, size);

    if (elementwise_affine)
    {
        error = tensor_create_ones(&(*layer_normalization)->weights, normalized_shape, length, runtime, datatype, true, true);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_create_zeroes(&(*layer_normalization)->bias, normalized_shape, length, runtime, datatype, true, true);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    return error;

cleanup:

    layer_normalization_destroy(*layer_normalization);

    return error;
}

void layer_normalization_destroy(layer_normalization_t *layer_normalization)
{
    if (layer_normalization)
    {
        tensor_destroy(layer_normalization->weights);
        tensor_destroy(layer_normalization->bias);
        free(layer_normalization->epsilon);
        free(layer_normalization->normalized_shape);
        free(layer_normalization);
    }
}

nw_error_t *reshape_create(reshape_t **reshape, int64_t *shape, int64_t length)
{
    CHECK_NULL_ARGUMENT(reshape, "reshape");
    CHECK_NULL_ARGUMENT(shape, "shape");

    *reshape = (reshape_t *) malloc(sizeof(reshape_t));
    if (!*reshape)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(reshape_t)), NULL);
    }

    (*reshape)->shape = NULL;
    (*reshape)->length = length;

    (*reshape)->shape = (int64_t *) malloc(length * sizeof(int64_t));
    if (!(*reshape)->shape)
    {
        free(*reshape);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", length * sizeof(int64_t)), NULL);
    }

    for (int64_t i = 0; i < length; ++i)
    {
        (*reshape)->shape[i] = shape[i];
    }

    return NULL;
}

void reshape_destroy(reshape_t *reshape)
{
    if (reshape)
    {
        free(reshape->shape);
        free(reshape);
    }
}

nw_error_t *linear_layer_create(layer_t **layer, int64_t in_features, int64_t out_features, runtime_t runtime, datatype_t datatype,
                                parameter_init_t *weight_init, parameter_init_t *bias_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(weight_init, "weight_init");

    nw_error_t *error = NULL;
    tensor_t *weights = NULL;
    tensor_t *bias = NULL;
    linear_t *linear = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = LINEAR;
    int64_t *weight_shape = (int64_t[]) {in_features, out_features};
    int64_t *bias_shape = (int64_t[]) {out_features};
    int64_t weight_rank = 2;
    int64_t bias_rank = 1;

    error = initialize(&weights, weight_init, weight_shape, weight_rank, runtime, datatype, true);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
    }

    if (bias_init) 
    {
        error = initialize(&bias, bias_init, bias_shape, bias_rank, runtime, datatype, true);
        if (error)
        {
            tensor_destroy(weights);
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
        }
    }

    error = linear_create(&linear, weights, bias);
    if (error)
    {
        tensor_destroy(weights);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) linear);
    if (error)
    {
        linear_destroy(linear);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *linear_layer_create_from_parameters(layer_t **layer, tensor_t *weights, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    linear_t *linear = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = LINEAR;

    error = linear_create(&linear, weights, bias);
    if (error)
    {
        tensor_destroy(weights);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) linear);
    if (error)
    {
        linear_destroy(linear);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *convolution_transpose_2d_layer_create_from_parameters(layer_t **layer, int64_t padding, int64_t stride, tensor_t *kernel, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    convolution_2d_t *convolution_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CONVOLUTION_TRANSPOSE_2D;

    error = convolution_2d_create(&convolution_2d, padding, stride, kernel, bias);
    if (error)
    {
        tensor_destroy(kernel);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) convolution_2d);
    if (error)
    {
        convolution_2d_destroy(convolution_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}


nw_error_t *convolution_transpose_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride,
                                                  int64_t in_channels, int64_t out_channels, runtime_t runtime, datatype_t datatype,
                                                  parameter_init_t *kernel_init, parameter_init_t *bias_init)
{
    nw_error_t *error = NULL;

    error = convolution_2d_layer_create(layer, kernel_size, padding, stride, out_channels, in_channels, runtime, datatype, kernel_init, bias_init);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create convolution_2d layer."), error);
    }

    (*layer)->transform_type = CONVOLUTION_TRANSPOSE_2D;

    return error;
}

nw_error_t *convolution_2d_layer_create_from_parameters(layer_t **layer, int64_t padding, int64_t stride, tensor_t *kernel, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    convolution_2d_t *convolution_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CONVOLUTION_2D;

    error = convolution_2d_create(&convolution_2d, padding, stride, kernel, bias);
    if (error)
    {
        tensor_destroy(kernel);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) convolution_2d);
    if (error)
    {
        convolution_2d_destroy(convolution_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *convolution_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride,
                                        int64_t in_channels, int64_t out_channels, runtime_t runtime, datatype_t datatype,
                                        parameter_init_t *kernel_init, parameter_init_t *bias_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(kernel_init, "kernel_init");

    nw_error_t *error = NULL;
    tensor_t *kernel = NULL;
    tensor_t *bias = NULL;
    convolution_2d_t *convolution_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CONVOLUTION_2D;
    int64_t *kernel_shape = (int64_t[]) {out_channels, in_channels, kernel_size, kernel_size};
    int64_t *bias_shape = (int64_t[]) {out_channels};
    int64_t weight_rank = 4;
    int64_t bias_rank = 1;

    error = initialize(&kernel, kernel_init, kernel_shape, weight_rank, runtime, datatype, true);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize kernel."), error);
    }
    
    if (bias_init)
    {
        error = initialize(&bias, bias_init, bias_shape, bias_rank, runtime, datatype, true);
        if (error)
        {
            tensor_destroy(kernel);
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
        }
    }

    error = convolution_2d_create(&convolution_2d, padding, stride, kernel, bias);
    if (error)
    {
        tensor_destroy(kernel);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create convolution_2d."), error);
    }

    error = transform_create(&transform, transform_type, (void *) convolution_2d);
    if (error)
    {
        convolution_2d_destroy(convolution_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *dropout_layer_create(layer_t **layer, void *probability, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(probability, "probability");

    nw_error_t *error = NULL;
    dropout_t *dropout = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = DROPOUT;

    error = dropout_create(&dropout, probability, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create dropout layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) dropout);
    if (error)
    {
        dropout_destroy(dropout);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *batch_normalization_2d_layer_create(layer_t **layer, int64_t number_of_features,
                                                void *momentum, void *epsilon, bool_t track_running_stats,
                                                bool_t affine, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");

    nw_error_t *error = NULL;
    batch_normalization_2d_t *batch_normalization_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = BATCH_NORMALIZATION_2D;

    error = batch_normalization_2d_create(&batch_normalization_2d, number_of_features, momentum, epsilon, track_running_stats, affine, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create batch normalization layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) batch_normalization_2d);
    if (error)
    {
        batch_normalization_2d_destroy(batch_normalization_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *layer_normalization_layer_create(layer_t **layer, const int64_t *normalized_shape, int64_t length,
                                        void *epsilon, bool_t elementwise_affine, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(normalized_shape, "normalized_shape");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");

    nw_error_t *error = NULL;
    layer_normalization_t *layer_normalization = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = LAYER_NORMALIZATION;

    error = layer_normalization_create(&layer_normalization, normalized_shape, length, epsilon, elementwise_affine, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create batch normalization layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) layer_normalization);
    if (error)
    {
        layer_normalization_destroy(layer_normalization);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *reshape_layer_create(layer_t **layer, int64_t *shape, int64_t length)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    reshape_t *reshape = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = RESHAPE;

    error = reshape_create(&reshape, shape, length);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create reshape layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) reshape);
    if (error)
    {
        reshape_destroy(reshape);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *rectified_linear_activation_layer_create(layer_t **layer)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = rectified_linear_activation_create(&activation);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *sigmoid_activation_layer_create(layer_t **layer)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = sigmoid_activation_create(&activation);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *softmax_activation_layer_create(layer_t **layer, int64_t axis)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = softmax_activation_create(&activation, axis);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *logsoftmax_activation_layer_create(layer_t **layer, int64_t axis)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = logsoftmax_activation_create(&activation, axis);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *leaky_rectified_linear_activation_layer_create(layer_t **layer, void *c, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(c, "c");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = leaky_rectified_linear_activation_create(&activation, c, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *tanh_activation_layer_create(layer_t **layer)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = tanh_activation_create(&activation);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *gelu_activation_layer_create(layer_t **layer)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = gelu_activation_create(&activation);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *model_forward(model_t *model, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_MODEL("model", model);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = block_forward(model->block, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed forward pass"), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *block_forward(block_t *block, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_BLOCK("block", block);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(block->layers, "block->layers");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *feature_map = NULL;

    for (int64_t i = 0; i < block->depth; ++i)
    {
        layer_t *layer = block->layers[i];
        if (!layer)
        {
            return ERROR(ERROR_NULL, string_create("layer is null."), NULL);
        }

        transform_type_t transform_type = layer->transform_type;
        transform_t *transform = layer->transform;
        if (!transform)
        {
            return ERROR(ERROR_NULL, string_create("transform is null."), NULL);
        }

        switch (transform_type)
        {
        case LINEAR:
            error = linear_forward(transform->linear, x, &feature_map);
            break;
        case CONVOLUTION_2D:
            error = convolution_2d_forward(transform->convolution_2d, x, &feature_map);
            break;
        case CONVOLUTION_TRANSPOSE_2D:
            error = convolution_transpose_2d_forward(transform->convolution_2d, x, &feature_map);
            break;
        case DROPOUT:
            error = dropout_forward(transform->dropout, x, &feature_map);
            break;
        case BATCH_NORMALIZATION_2D:
            error = batch_normalization_2d_forward(transform->batch_normalization_2d, x, &feature_map);
            break;
        case RESHAPE:
            error = reshape_forward(transform->reshape, x, &feature_map);
            break;
        case ACTIVATION:
            error = activation_forward(transform->activation, x, &feature_map);
            break;
        case BLOCK:
            error = block_forward(transform->block, x, &feature_map);
            break;
        default:
            error = ERROR(ERROR_LAYER_TYPE, string_create("unknown transform type %d.", (int) transform_type), NULL);
            break;
        }

        if (error)
        {
            return ERROR(ERROR_FORWARD, string_create("failed forward pass."), error);
        }

        if (i > 0 && (!feature_map->requires_gradient || no_gradient) && x != feature_map)
        {
            tensor_destroy(x);
        }

        x = feature_map;
        feature_map = NULL;
    }

    *y = x;

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *linear_forward(linear_t *linear, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_LINEAR("linear", linear);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_linear(x, linear->weights, linear->bias, y);
    if (error)
    {
        return ERROR(ERROR_LINEAR, string_create("failed to matrix multiply tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *convolution_2d_forward(convolution_2d_t *convolution_2d, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_convolution_2d(x, convolution_2d->kernel, convolution_2d->bias, y, convolution_2d->stride, convolution_2d->padding);
    if (error)
    {
        return ERROR(ERROR_CONVOLUTION, string_create("failed to apply convolution_2d."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *convolution_transpose_2d_forward(convolution_2d_t *convolution_2d, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_convolution_transpose_2d(x, convolution_2d->kernel, convolution_2d->bias, y, convolution_2d->stride, convolution_2d->padding);
    if (error)
    {
        return ERROR(ERROR_CONVOLUTION, string_create("failed to apply convolution_2d transpose."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *dropout_forward(dropout_t *dropout, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_DROPOUT("dropout", dropout);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(dropout, "dropout");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_dropout(x, y, dropout->probability, dropout->inference);
    if (error)
    {
        return ERROR(ERROR_DROPOUT, string_create("failed to apply dropout."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *batch_normalization_2d_forward(batch_normalization_2d_t *batch_normalization_2d, tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(batch_normalization_2d, "batch_normalization_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    if (batch_normalization_2d->track_running_stats)
    {
        error = tensor_batch_normalization_2d(x, batch_normalization_2d->weights, batch_normalization_2d->bias, batch_normalization_2d->running_mean, 
                                              batch_normalization_2d->running_variance, y, batch_normalization_2d->inference, batch_normalization_2d->momentum, 
                                              batch_normalization_2d->epsilon);
    }
    else
    {
        error = tensor_batch_normalization_2d(x, batch_normalization_2d->weights, batch_normalization_2d->bias, NULL, NULL, y,
                                              batch_normalization_2d->inference, batch_normalization_2d->momentum, batch_normalization_2d->epsilon);
    }

    if (error)
    {
        return ERROR(ERROR_BATCH_NORMALIZATION, string_create("failed to apply batch normalization 2d."), error);
    }

    return error;
}

nw_error_t *layer_normalization_forward(layer_normalization_t *layer_normalization, tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(layer_normalization, "layer_normalization");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_layer_normalization(x, layer_normalization->weights, layer_normalization->bias, y, layer_normalization->normalized_shape,
                                       layer_normalization->length, layer_normalization->epsilon);

    if (error)
    {
        return ERROR(ERROR_BATCH_NORMALIZATION, string_create("failed to apply layer normalization."), error);
    }

    return error;
}

nw_error_t *reshape_forward(reshape_t *reshape, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(reshape, "reshape");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_reshape(x, y, reshape->shape, reshape->length);
    if (error)
    {
        return ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *model_inference(model_t *model, bool_t inference)
{
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = block_inference(model->block, inference);
    if (error)
    {
        return ERROR(ERROR_SET, string_create("failed to set inference flag."), error);
    }

    return error;
}

nw_error_t *block_inference(block_t *block, bool_t inference)
{
    CHECK_NULL_ARGUMENT(block, "block");

    nw_error_t *error = NULL;

    for (int64_t i = 0; i < block->depth; ++i)
    {
        switch (block->layers[i]->transform_type)
        {
        case DROPOUT:
            block->layers[i]->transform->dropout->inference = inference;
            break;
        case BATCH_NORMALIZATION_2D:
            block->layers[i]->transform->batch_normalization_2d->inference = inference;
            break;
        case BLOCK:
            error = block_inference(block->layers[i]->transform->block, inference);
            if (error)
            {
                return ERROR(ERROR_SET, string_create("failed to set inference flag."), error);
            }
            break;
        default:
            break;
        }
    }

    return error;
}