#include <init.h>
#include <tensor.h>
#include <layer.h>
#include <math.h>
#include <string.h>

// nw_error_t *linear_layer_create(layer_t **layer, 
//                                 uint64_t in_features,
//                                 uint64_t out_features,
//                                 runtime_t runtime,
//                                 datatype_t datatype,
//                                 activation_t activation,
//                                 initialization_type_t weight_initialization,
//                                 initialization_type_t bias_initialization)
// {
//     CHECK_NULL_ARGUMENT(layer, "layer");

//     nw_error_t *error = NULL;
//     tensor_t *weights = NULL;
//     tensor_t *bias = NULL;
//     uint64_t *shape = (uint64_t[]) {in_features, out_features};
//     calculate_gain(activation, )

//     switch (weight_initialization)
//     {
//     case ZEROES:
//         break;
//     case ONES:
//         break;
//     case UNIFORM:
//         break;
//     case NORMAL:
//         break;
//     case KAIMING_UNIFORM:
//         if (datatype == FLOAT32)
//         {
//             float32_t 
//         }
//         break;
//     case KAIMING_NORMAL:
//         break;
//     case GLOROT_UNIFORM:
//         break;
//     case GLOROT_NORMAL:
//         break;
//     default:
//         break;
//     }

//     return NULL;

// }

static nw_error_t *activation_forward(activation_t *activation, tensor_t *x, tensor_t **y)
{
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
            error = tensor_softmax(x, y, activation_function->softmax->axis, activation_function->softmax->length);
        }
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown activation function %d.", (int) activation), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply activation function."), error);
    }

    return error;
}

static nw_error_t *linear_forward(linear_t *linear, tensor_t *x, tensor_t **y)
{
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

cleanup:

    if (!(x_i->requires_gradient || bias->requires_gradient))
    {
        tensor_destroy(x_j);
    }

    if (!(x->requires_gradient || weights->requires_gradient))
    {
        tensor_destroy(x_i);
    }

    return error;
}


nw_error_t *block_forward(block_t *block, tensor_t *x, tensor_t **y)
{
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

        transformation_type_t transformation_type = layer->transformation_type;
        transformation_t *transformation = layer->transformation;
        if (!transformation)
        {
            return ERROR(ERROR_NULL, string_create("transformation is null."), NULL);
        }

        switch (transformation_type)
        {
        case LINEAR:
            error = linear_forward(transformation->linear, x, &feature_map);
            break;
        case BLOCK:
            error = block_forward(transformation->block, x, &feature_map);
            break;
        default:
            error = ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown transformation type %d.", (int) transformation_type), NULL);
            break;
        }

        if (error)
        {
            return ERROR(ERROR_FORWARD, string_create("failed forward pass."), error);
        }

        if (i > 0 && !feature_map->requires_gradient)
        {
            tensor_destroy(x);
        }

        x = feature_map;
    }

    *y = feature_map;

    return error;
}

nw_error_t *model_forward(model_t *model, tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = block_forward(model->block, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed forward pass"), error);
    }

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

        transformation_type_t transformation_type = layer->transformation_type;
        transformation_t *transformation = layer->transformation;
        if (!transformation)
        {
            return ERROR(ERROR_NULL, string_create("transformation is null."), NULL);
        }

        switch (transformation_type)
        {
        case LINEAR:
            error = linear_requires_gradient(transformation->linear, requires_gradient);
            break;
        case BLOCK:
            error = block_requires_gradient(transformation->block, requires_gradient);
            break;
        default:
            error = ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown transformation type %d.", (int) transformation_type), NULL);
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

nw_error_t *layer_create(layer_t **layer, transformation_t *transformation, transformation_type_t transformation_type)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(transformation, "transformation");

    *layer = (layer_t *) malloc(sizeof(layer_t));
    if (!*layer)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_t)), NULL);
    }

    (*layer)->transformation = transformation;
    (*layer)->transformation_type = transformation_type;

    return NULL;
}

void layer_destroy(layer_t *layer)
{
    if (layer)
    {
        transformation_destroy(layer->transformation, layer->transformation_type);
        free(layer);
    }
}

nw_error_t *transformation_create(transformation_t **transformation, transformation_type_t transformation_type, void *type_transformation)
{
    CHECK_NULL_ARGUMENT(transformation, "transformation");
    CHECK_NULL_ARGUMENT(type_transformation, "type_transformation");

    *transformation = (transformation_t *) malloc(sizeof(transformation_t));
    if (!*transformation)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(transformation_t)), NULL);
    }

    switch (transformation_type)
    {
    case LINEAR:
        (*transformation)->linear = (linear_t *) type_transformation;
        break;
    case DROPOUT:
        (*transformation)->dropout = (dropout_t *) type_transformation;
        break;
    case BLOCK:
        (*transformation)->block = (block_t *) type_transformation;
        break;
    default:
        free(*transformation);
        return ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown transformation type %d.", (int) transformation_type), NULL);
    }

    return NULL;
}

void transformation_destroy(transformation_t *transformation, transformation_type_t transformation_type)
{
    if (transformation)
    {
        switch (transformation_type)
        {
        case LINEAR:
            linear_destroy(transformation->linear);
            break;
        case DROPOUT:
            dropout_destroy(transformation->dropout);
            break;
        case BLOCK:
            block_destroy(transformation->block);
            break;
        default:
            break;
        }
        free(transformation);
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

nw_error_t *block_create(block_t **block, layer_t **layers, uint64_t depth)
{
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(layers, "layers");

    *block = (block_t *) malloc(sizeof(block_t));
    if (!*block)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(block_t)), NULL);
    }

    (*block)->depth = depth;
    (*block)->layers = (layer_t **) malloc(depth * sizeof(layer_t *));
    if (!(*block)->layers)
    {
        free(block);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(depth * sizeof(layer_t *))), NULL);
    }

    for (uint64_t i = 0; i < depth; ++i)
    {
        (*block)->layers[i] = layers[i];
    }

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

nw_error_t *softmax_create(softmax_t **softmax, uint64_t *axis, uint64_t length)
{
    CHECK_NULL_ARGUMENT(softmax, "softmax");

    *softmax = (softmax_t *) malloc(sizeof(softmax_t));
    if (!*softmax)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(softmax_t)), NULL);
    }

    (*softmax)->length = length;
    (*softmax)->axis = (uint64_t *) malloc(length * sizeof(uint64_t));
    if (!(*softmax)->axis)
    {
        free(*softmax);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", (size_t)(length * sizeof(uint64_t))), NULL);
    }

    if (axis)
    {
        memcpy((*softmax)->axis, axis, (size_t)(length * sizeof(uint64_t)));
    }

    return NULL;
}

void softmax_destroy(softmax_t *softmax)
{
    if (softmax)
    {
        free(softmax->axis);
        free(softmax);
    }
}

nw_error_t *activation_function(activation_function_t **activation_function,
                                activation_function_type_t activation_function_type,
                                void *type_activation_function)
{
    CHECK_NULL_ARGUMENT(activation_function, "activation_function");
    CHECK_NULL_ARGUMENT(type_activation_function, "type_activation_function");

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