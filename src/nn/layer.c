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

nw_error_t *convolution_2d_create(convolution_2d_t **convolution_2d, int64_t kernel_size, int64_t padding, int64_t stride,
                                  int64_t in_channels, int64_t out_channels, tensor_t *kernel, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(kernel, "kernel");

    *convolution_2d = (convolution_2d_t *) malloc(sizeof(linear_t));
    if (!*convolution_2d)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(convolution_2d_t)), NULL);
    }

    (*convolution_2d)->kernel_size = kernel_size;
    (*convolution_2d)->padding = padding;
    (*convolution_2d)->stride = stride;
    (*convolution_2d)->in_channels = in_channels;
    (*convolution_2d)->out_channels = out_channels;
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
    (*dropout)->datatype = datatype;
    (*dropout)->inference = false;

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

nw_error_t *convolution_2d_transpose_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride,
                                                  int64_t in_channels, int64_t out_channels, runtime_t runtime, datatype_t datatype,
                                                  parameter_init_t *kernel_init, parameter_init_t *bias_init)
{
    nw_error_t *error = NULL;

    error = convolution_2d_layer_create(layer, kernel_size, padding, stride, in_channels, out_channels, runtime, datatype, kernel_init, bias_init);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create convolution_2d layer."), error);
    }

    (*layer)->transform_type = CONVOLUTION_TRANSPOSE_2D;

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

    error = convolution_2d_create(&convolution_2d, kernel_size, padding, stride, in_channels, out_channels, kernel, bias);
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
            error = convolution_2d_transpose_forward(transform->convolution_2d, x, &feature_map);
            break;
        case DROPOUT:
            error = dropout_forward(transform->dropout, x, &feature_map);
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

nw_error_t *convolution_2d_transpose_forward(convolution_2d_t *convolution_2d, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_convolution_2d_transpose(x, convolution_2d->kernel, convolution_2d->bias, y, convolution_2d->stride, convolution_2d->padding);
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
    tensor_t *probability = NULL;
    tensor_t *scale = NULL;
    tensor_t *mask= NULL;
    tensor_t *rand_tensor = NULL;
    tensor_t *x_i = NULL;
    void *min = NULL;
    void *max = NULL;
    void *scalar = NULL;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;

    if (dropout->inference)
    {
        *y = x;
        return NULL;
    }

    min = (void *) malloc(datatype_size(datatype));
    if (!min) 
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
        goto cleanup;
    }

    max = (void *) malloc(datatype_size(datatype));
    if (!max) 
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
        goto cleanup;
    }

    scalar = (void *) malloc(datatype_size(datatype));
    if (!scalar) 
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) min = (float32_t) 0.0;
        *(float32_t *) max = (float32_t) 1.0;
        *(float32_t *) scalar = (float32_t) 1.0 / ((float32_t) 1.0 - *(float32_t *) dropout->probability);
        break;
    case FLOAT64:
        *(float64_t *) min = (float64_t) 0.0;
        *(float64_t *) max = (float64_t) 1.0;
        *(float64_t *) scalar = (float64_t) 1.0 / ((float64_t) 1.0 - *(float64_t *) dropout->probability);
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype."), NULL);
        goto cleanup;
    }

    error = tensor_constant(dropout->probability, datatype, runtime, false, false, &probability);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_constant(scalar, datatype, runtime, false, false, &scale);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_create_uniform(&rand_tensor, x->buffer->view->shape, x->buffer->view->rank, runtime, datatype, false, false, min, max);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_compare_greater(rand_tensor, probability, &mask);
    if (error)
    {
        error = ERROR(ERROR_COMPARE_GREATER, string_create("failed to compare greater tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x, mask, &x_i);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x_i, scale, y);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    free(scalar);
    free(min);
    free(max);
    
    tensor_destroy(probability);
    tensor_destroy(rand_tensor);
    tensor_destroy(scale);

    if (!x->requires_gradient || no_gradient)
    {
        tensor_destroy(mask);
        tensor_destroy(x_i);
    }

    return NULL;
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