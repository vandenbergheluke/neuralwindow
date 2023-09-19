#include <init.h>
#include <tensor.h>
#include <layer.h>
#include <math.h>

nw_error_t *linear_layer_create(layer_t **layer, 
                                uint64_t in_features,
                                uint64_t out_features,
                                runtime_t runtime,
                                datatype_t datatype,
                                activation_t activation,
                                initialization_type_t weight_initialization,
                                initialization_type_t bias_initialization)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    tensor_t *weights = NULL;
    tensor_t *bias = NULL;
    uint64_t *shape = (uint64_t[]) {in_features, out_features};
    calculate_gain(activation, )

    switch (weight_initialization)
    {
    case ZEROES:
        break;
    case ONES:
        break;
    case UNIFORM:
        break;
    case NORMAL:
        break;
    case KAIMING_UNIFORM:
        if (datatype == FLOAT32)
        {
            float32_t 
        }
        break;
    case KAIMING_NORMAL:
        break;
    case GLOROT_UNIFORM:
        break;
    case GLOROT_NORMAL:
        break;
    default:
        break;
    }

    return NULL;

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

nw_error_t *linear_create(linear_t **linear, tensor_t *weights, tensor_t *bias, activation_t activation)
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