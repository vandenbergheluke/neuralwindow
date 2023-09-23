#include <init.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <layer.h>
#include <math.h>
#include <string.h>

extern bool_t no_gradient;

nw_error_t *linear_layer_create(layer_t **layer, 
                                uint64_t in_features,
                                uint64_t out_features,
                                runtime_t runtime,
                                datatype_t datatype,
                                bool_t requires_gradient,
                                activation_t *activation,
                                parameter_init_t *weight_init,
                                parameter_init_t *bias_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(activation, "activation");
    CHECK_NULL_ARGUMENT(weight_init, "weight_init");
    CHECK_NULL_ARGUMENT(bias_init, "bias_init");

    nw_error_t *error = NULL;
    tensor_t *weights = NULL;
    tensor_t *bias = NULL;
    linear_t *linear = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = LINEAR;
    uint64_t *weight_shape = (uint64_t[]) {in_features, out_features};
    uint64_t *bias_shape = (uint64_t[]) {out_features};
    uint64_t weight_rank = 2;
    uint64_t bias_rank = 1;

    error = initialize(&weights, weight_init, weight_shape, weight_rank, runtime, datatype, requires_gradient);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
    }
    
    error = initialize(&bias, bias_init, bias_shape, bias_rank, runtime, datatype, requires_gradient);
    if (error)
    {
        tensor_destroy(weights);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
    }

    error = linear_create(&linear, weights, bias, activation);
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

static nw_error_t *activation_forward(activation_t *activation, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_ACTIVATION("activation", activation);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(activation, "activation");
    CHECK_NULL_ARGUMENT(activation->activation_function, "activation->activation_function");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    activation_function_t *activation_function = activation->activation_function;
    activation_function_type_t activation_function_type = activation->activation_function_type;

    switch (activation_function_type)
    {
    case ACTIVATION_RECTIFIED_LINEAR:
        error = tensor_rectified_linear(x, y);
        break;
    case ACTIVATION_SIGMOID:
        error = tensor_sigmoid(x, y);
        break;
    case ACTIVATION_SOFTMAX:
        if (!activation_function->softmax)
        {
            error = ERROR(ERROR_NULL, string_create("activation function is null."), NULL);
        }
        else
        {
            error = tensor_softmax(x, y, activation_function->softmax->axis);
        }
        break;
    case ACTIVATION_LOGSOFTMAX:
        if (!activation_function->softmax)
        {
            error = ERROR(ERROR_NULL, string_create("activation function is null."), NULL);
        }
        else
        {
            error = tensor_logsoftmax(x, y, activation_function->softmax->axis);
        }
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown activation function %d.", (int) activation_function_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply activation function."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

static nw_error_t *linear_forward(linear_t *linear, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_LINEAR("linear", linear);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *weights = linear->weights;
    tensor_t *bias = linear->bias;
    activation_t *activation = linear->activation;

    error = tensor_matrix_multiplication(x, weights, &x_i);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(x_i, bias, &x_j);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = activation_forward(activation, x_j, y);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to apply activation function."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!(x_i->requires_gradient || bias->requires_gradient) || no_gradient)
    {
        tensor_destroy(x_j);
    }

    if (!(x->requires_gradient || weights->requires_gradient) || no_gradient)
    {
        tensor_destroy(x_i);
    }

    return error;
}

static nw_error_t *block_forward(block_t *block, tensor_t *x, tensor_t **y)
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

    for (uint64_t i = 0; i < block->depth; ++i)
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
        case BLOCK:
            error = block_forward(transform->block, x, &feature_map);
            break;
        default:
            error = ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown transform type %d.", (int) transform_type), NULL);
            break;
        }

        if (error)
        {
            return ERROR(ERROR_FORWARD, string_create("failed forward pass."), error);
        }

        if (i > 0 && (!feature_map->requires_gradient || no_gradient))
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

static nw_error_t *linear_requires_gradient(linear_t *linear, bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(linear->weights, "linear->weights");
    CHECK_NULL_ARGUMENT(linear->bias, "linear->bias");

    linear->weights->requires_gradient = requires_gradient;
    linear->bias->requires_gradient = requires_gradient;

    return NULL;
}

static nw_error_t *block_requires_gradient(block_t *block, bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(block->layers, "block->layers");

    nw_error_t *error = NULL;

    for (uint64_t i = 0; i < block->depth; ++i)
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
            error = linear_requires_gradient(transform->linear, requires_gradient);
            break;
        case BLOCK:
            error = block_requires_gradient(transform->block, requires_gradient);
            break;
        default:
            error = ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown transform type %d.", (int) transform_type), NULL);
            break;
        }

        if (error)
        {
            return ERROR(ERROR_REQUIRES_GRADIENT, string_create("failed to modify requires gradient flag."), error);
        }
    }

    return error;
}

nw_error_t *model_requires_gradient(model_t *model, bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = NULL;

    error = block_requires_gradient(model->block, requires_gradient);
    if (error)
    {
        return ERROR(ERROR_REQUIRES_GRADIENT, string_create("failed to modify requires gradient flag."), error);
    }

    return error;
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
    case DROPOUT:
        (*transform)->dropout = (dropout_t *) type_transform;
        break;
    case BLOCK:
        (*transform)->block = (block_t *) type_transform;
        break;
    default:
        free(*transform);
        return ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown transform type %d.", (int) transform_type), NULL);
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
        case DROPOUT:
            dropout_destroy(transform->dropout);
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
    case DROPOUT:
        return "DROPOUT";
    case BLOCK:
        return "BLOCK";
    default:
        return "UNKNOWN_TRANSFORM_TYPE";
    }
}

string_t activation_function_type_string(activation_function_type_t activation_function_type)
{
    switch (activation_function_type)
    {
    case ACTIVATION_RECTIFIED_LINEAR:
        return "ACTIVATION_RECTIFIED_LINEAR";
    case ACTIVATION_SIGMOID:
        return "ACTIVATION_SIGMOID";
    case ACTIVATION_SOFTMAX:
        return "ACTIVATION_SOFTMAX";
    case ACTIVATION_LOGSOFTMAX:
        return "ACTIVATION_LOGSOFTMAX";
    default:
        return "UNKNOWN_ACTIVATION_FUNCTION_TYPE";
    }
}

nw_error_t *linear_create(linear_t **linear, tensor_t *weights, tensor_t *bias, activation_t *activation)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(weights, "weights");
    CHECK_NULL_ARGUMENT(bias, "bias");

    *linear = (linear_t *) malloc(sizeof(linear_t));
    if (!*linear)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(linear_t)), NULL);
    }

    (*linear)->weights = weights;
    (*linear)->bias = bias;
    (*linear)->activation = activation;

    return NULL;
}

void linear_destroy(linear_t *linear)
{
    if (linear)
    {
        tensor_destroy(linear->weights);
        tensor_destroy(linear->bias);
        activation_destroy(linear->activation);
        free(linear);
    }
}

nw_error_t *dropout_create(dropout_t **dropout, float32_t probability)
{
    CHECK_NULL_ARGUMENT(dropout, "dropout");

    *dropout = (dropout_t *) malloc(sizeof(dropout_t));
    if (!*dropout)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(dropout_t)), NULL);
    }

    if (probability < 0 || probability > 1)
    {
        free(*dropout);
        return ERROR(ERROR_DROPOUT, string_create("Dropout probability has to be between 0 and 1, but got %f", probability), NULL);
    }
    (*dropout)->probability = probability;

    return NULL;
}

void dropout_destroy(dropout_t *dropout)
{
    if (dropout)
    {
        free(dropout);
    }
}

nw_error_t *block_create(block_t **block, uint64_t depth, ...)
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

    for (uint64_t i = 0; i < depth; ++i)
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
            for (uint64_t i = 0; i < block->depth; ++i)
            {
                layer_destroy(block->layers[i]);
            }
            free(block->layers);
        }
        free(block);
    }
}

nw_error_t *softmax_create(softmax_t **softmax, uint64_t axis)
{
    CHECK_NULL_ARGUMENT(softmax, "softmax");

    *softmax = (softmax_t *) malloc(sizeof(softmax_t));
    if (!*softmax)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(softmax_t)), NULL);
    }

    (*softmax)->axis = axis;

    return NULL;
}

void softmax_destroy(softmax_t *softmax)
{
    if (softmax)
    {
        free(softmax);
    }
}

nw_error_t *activation_function_create(activation_function_t **activation_function,
                                       activation_function_type_t activation_function_type,
                                       void *type_activation_function)
{
    CHECK_NULL_ARGUMENT(activation_function, "activation_function");

    *activation_function = (activation_function_t *) malloc(sizeof(activation_function_t));
    if (!*activation_function)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(activation_function_t)), NULL);
    }

    switch (activation_function_type)
    {
    case ACTIVATION_RECTIFIED_LINEAR:
    case ACTIVATION_SIGMOID:
        return NULL;
    case ACTIVATION_SOFTMAX:
    case ACTIVATION_LOGSOFTMAX:
        (*activation_function)->softmax = (softmax_t *) type_activation_function;
        break;
    default:
        free(*activation_function);
        return ERROR(ERROR_ACTIVATION_TYPE, string_create("unknown activation type %d.", (int) activation_function_type), NULL);
    }

    return NULL;
}

void activation_function_destroy(activation_function_t *activation_function, activation_function_type_t activation_function_type)
{
    if (activation_function)
    {
        switch (activation_function_type)
        {
        case ACTIVATION_SOFTMAX:
        case ACTIVATION_LOGSOFTMAX:
            softmax_destroy(activation_function->softmax);
            break;
        default:
            break;
        }
        free(activation_function);
    }
}

nw_error_t *activation_create(activation_t **activation,
                              activation_function_t *activation_function,
                              activation_function_type_t activation_function_type)
{
    CHECK_NULL_ARGUMENT(activation, "activation");
    CHECK_NULL_ARGUMENT(activation_function, "activation_function");

    *activation = (activation_t *) malloc(sizeof(activation_t));
    if (!*activation)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(activation_t)), NULL);
    }

    (*activation)->activation_function = activation_function;
    (*activation)->activation_function_type = activation_function_type;

    return NULL;
}

void activation_destroy(activation_t *activation)
{
    if (activation)
    {
        activation_function_destroy(activation->activation_function, activation->activation_function_type);
        free(activation);
    }
}

nw_error_t *rectified_linear_activation_create(activation_t **activation)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;
    activation_function_t *activation_function = NULL;
    activation_function_type_t activation_function_type = ACTIVATION_RECTIFIED_LINEAR;

    error = activation_function_create(&activation_function, activation_function_type, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation function."), error);
    }

    error = activation_create(activation, activation_function, activation_function_type);
    if (error)
    {
        activation_function_destroy(activation_function, activation_function_type);
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    return error;
}

nw_error_t *sigmoid_activation_create(activation_t **activation)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;
    activation_function_t *activation_function = NULL;
    activation_function_type_t activation_function_type = ACTIVATION_SIGMOID;

    error = activation_function_create(&activation_function, activation_function_type, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation function."), error);
    }

    error = activation_create(activation, activation_function, activation_function_type);
    if (error)
    {
        activation_function_destroy(activation_function, activation_function_type);
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    return error;
}

static nw_error_t *softmax_activation_type_create(activation_t **activation, uint64_t axis, activation_function_type_t activation_function_type)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;
    softmax_t *softmax = NULL;
    activation_function_t *activation_function = NULL;

    error = softmax_create(&softmax, axis);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create softmax."), error);
    }

    error = activation_function_create(&activation_function, activation_function_type, softmax);
    if (error)
    {
        softmax_destroy(softmax);
        return ERROR(ERROR_CREATE, string_create("failed to create activation function."), error);
    }

    error = activation_create(activation, activation_function, activation_function_type);
    if (error)
    {
        activation_function_destroy(activation_function, activation_function_type);
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    return error;
}

nw_error_t *softmax_activation_create(activation_t **activation, uint64_t axis)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;

    error = softmax_activation_type_create(activation, axis, ACTIVATION_SOFTMAX);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create softmax activation."), error);
    }

    return error;
}

nw_error_t *logsoftmax_activation_create(activation_t **activation, uint64_t axis)
{
    CHECK_NULL_ARGUMENT(activation, "activation");

    nw_error_t *error = NULL;

    error = softmax_activation_type_create(activation, axis, ACTIVATION_LOGSOFTMAX);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create softmax activation."), error);
    }

    return error;
}

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
