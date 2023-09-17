/**@file layer.c
 * @brief
 *
 */

#include <layer.h>
#include <tensor.h>

nw_error_t *parameters_create(parameters_t **parameters, tensor_t *weights, tensor_t *mask, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(parameters, "parameters");
    CHECK_NULL_ARGUMENT(weights, "weights");
    CHECK_NULL_ARGUMENT(mask, "mask");
    CHECK_NULL_ARGUMENT(bias, "bias");

    *parameters = (parameters_t *) malloc(sizeof(parameters_t));
    if (!*parameters)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(parameters_t)), NULL);
    }

    (*parameters)->weights = weights;
    (*parameters)->mask = mask;
    (*parameters)->bias = bias;

    return NULL;
}

void parameters_destroy(parameters_t *parameters)
{
    if (parameters)
    {
        tensor_destroy(parameters->weights);
        tensor_destroy(parameters->mask);
        tensor_destroy(parameters->bias);
        free(parameters);
    }
}

nw_error_t *linear_create(linear_t **linear, uint64_t input_features, uint64_t output_features, parameters_t *parameters)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(parameters, "parameters");

    *linear = (linear_t *) malloc(sizeof(linear_t));
    if (!*linear)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(linear_t)), NULL);
    }

    (*linear)->input_features = input_features;
    (*linear)->output_features = output_features;
    (*linear)->parameters = parameters;

    return NULL;
}

void linear_destroy(linear_t *linear)
{
    if (linear)
    {
        parameters_destroy(linear->parameters);
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

nw_error_t *layer_create(layer_t **layer, layer_type_t layer_type, void *type_layer)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(type_layer, "type_layer");

    *layer = (layer_t *) malloc(sizeof(layer_t));
    if (!*layer)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_t)), NULL);
    }

    switch (layer_type)
    {
    case LINEAR:
        (*layer)->linear = (linear_t *) type_layer;
        break;
    case DROPOUT:
        (*layer)->dropout = (dropout_t *) type_layer;
        break;
    case MODULE:
        (*layer)->module = (module_t *) type_layer;
        break;
    default:
        free(*layer);
        return ERROR(ERROR_UKNOWN_LAYER_TYPE, string_create("unknown layer type %d.", (int) layer_type), NULL);
    }

    return NULL;
}

void layer_destroy(layer_t *layer, layer_type_t layer_type)
{
    if (layer)
    {
        switch (layer_type)
        {
        case LINEAR:
            linear_destroy(layer->linear);
            break;
        case DROPOUT:
            dropout_destroy(layer->dropout);
            break;
        case MODULE:
            module_destroy(layer->module);
            break;
        default:
            break;
        }
        free(layer);
    }
}

nw_error_t *unit_create(unit_t **unit, layer_type_t layer_type, layer_t *layer)
{
    CHECK_NULL_ARGUMENT(unit, "unit");
    CHECK_NULL_ARGUMENT(layer, "layer");

    *unit = (unit_t *) malloc(sizeof(unit_t));
    if (!*unit)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(unit_t)), NULL);
    }

    (*unit)->layer = layer;
    (*unit)->layer_type = layer_type;

    return NULL;
}

void unit_destroy(unit_t *unit)
{
    if (unit)
    {
        layer_destroy(unit->layer, unit->layer_type);
        free(unit);
    }
}

nw_error_t *module_create(module_t **module, unit_t **units, uint64_t depth)
{
    CHECK_NULL_ARGUMENT(module, "module");
    CHECK_NULL_ARGUMENT(units, "units");

    *module = (module_t *) malloc(sizeof(module_t));
    if (!*module)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(module_t)), NULL);
    }

    (*module)->depth = depth;
    (*module)->units = (unit_t **)malloc(depth * sizeof(unit_t *));
    if (!(*module)->units)
    {
        free(module);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(depth * sizeof(unit_t *))), NULL);
    }

    for (uint64_t i = 0; i < depth; ++i)
    {
        (*module)->units[i] = NULL;
    }

    return NULL;
}

void module_destroy(module_t *module)
{
    if (module)
    {
        if (module->units)
        {
            for (uint64_t i = 0; i < module->depth; ++i)
            {
                unit_destroy(module->units[i]);
            }
            free(module->units);
        }
        free(module);
    }
}