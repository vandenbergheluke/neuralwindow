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

nw_error_t *block_create_from_array(block_t **block, int64_t depth, layer_t **layers)
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
        free(*block);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(depth * sizeof(layer_t *))), NULL);
    }

    for (int64_t i = 0; i < depth; ++i)
    {
        (*block)->layers[i] = (layer_t *) layers[i];
    }

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
    case MAX_POOLING_2D:
    case AVERAGE_POOLING_2D:
        (*transform)->pooling_2d = (pooling_2d_t *) type_transform;
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
    case EMBEDDING:
        (*transform)->embedding = (embedding_t *) type_transform;
        break;
    case TRANSFORMER_EMBEDDING:
        (*transform)->transformer_embedding = (transformer_embedding_t *) type_transform;
        break;
    case CAUSAL_MULTIHEAD_SELF_ATTENTION:
        (*transform)->causal_multihead_self_attention = (causal_multihead_self_attention_t *) type_transform;
        break;
    case ACTIVATION:
        (*transform)->activation = (activation_t *) type_transform;
        break;
    case BLOCK:
    case RESIDUAL_BLOCK:
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
        case MAX_POOLING_2D:
        case AVERAGE_POOLING_2D:
            pooling_2d_destroy(transform->pooling_2d);
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
        case EMBEDDING:
            embedding_destroy(transform->embedding);
            break;
        case TRANSFORMER_EMBEDDING:
            transformer_embedding_destroy(transform->transformer_embedding);
            break;
        case CAUSAL_MULTIHEAD_SELF_ATTENTION:
            causal_multihead_self_attention_destroy(transform->causal_multihead_self_attention);
            break;
        case ACTIVATION:
            activation_destroy(transform->activation);
            break;
        case BLOCK:
        case RESIDUAL_BLOCK:
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
    case MAX_POOLING_2D:
        return "MAX_POOLING_2D";
    case AVERAGE_POOLING_2D:
        return "AVERAGE_POOLING_2D";
    case DROPOUT:
        return "DROPOUT";
    case BATCH_NORMALIZATION_2D:
        return "BATCH_NORMALIZATION_2D";
    case LAYER_NORMALIZATION:
        return "LAYER_NORMALIZATION";
    case RESHAPE:
        return "RESHAPE";
   case EMBEDDING:
        return "EMBEDDING";
   case TRANSFORMER_EMBEDDING:
        return "TRANSFORMER_EMBEDDING";
   case CAUSAL_MULTIHEAD_SELF_ATTENTION:
        return "CAUSAL_MULTIHEAD_SELF_ATTENTION";
    case ACTIVATION:
        return "ACTIVATION";
    case BLOCK:
        return "BLOCK";
    case RESIDUAL_BLOCK:
        return "RESIDUAL_BLOCK";
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

nw_error_t *pooling_2d_create(pooling_2d_t **pooling_2d, int64_t padding, int64_t stride, int64_t kernel)
{
    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");

    *pooling_2d = (pooling_2d_t *) malloc(sizeof(pooling_2d_t));
    if (!*pooling_2d)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(pooling_2d_t)), NULL);
    }

    (*pooling_2d)->padding = padding;
    (*pooling_2d)->stride = stride;
    (*pooling_2d)->kernel = kernel;

    return NULL;
}

void pooling_2d_destroy(pooling_2d_t *pooling_2d)
{
    if (pooling_2d)
    {
        free(pooling_2d);
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
    (*batch_normalization_2d)->datatype = datatype;

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
                                        void *epsilon, bool_t bias_affine, bool_t weights_affine, datatype_t datatype, runtime_t runtime)
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
    (*layer_normalization)->datatype = datatype;
    
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

    if (weights_affine)
    {
        error = tensor_create_ones(&(*layer_normalization)->weights, normalized_shape, length, runtime, datatype, true, true);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    if (bias_affine)
    {
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

nw_error_t *embedding_create(embedding_t **embedding, int64_t vocabulary_size, int64_t embedding_size, tensor_t *vocabulary_counter, tensor_t *weights)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(vocabulary_counter, "vocabulary_counter");
    CHECK_NULL_ARGUMENT(weights, "weights");

    *embedding = (embedding_t *) malloc(sizeof(embedding_t));
    if (!*embedding)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(embedding_t)), NULL);
    }

    (*embedding)->vocabulary_size = vocabulary_size;
    (*embedding)->embedding_size = embedding_size;
    (*embedding)->vocabulary_counter = vocabulary_counter;
    (*embedding)->weights = weights;

    return NULL;
}

void embedding_destroy(embedding_t *embedding)
{
    tensor_destroy(embedding->vocabulary_counter);
    tensor_destroy(embedding->weights);
    free(embedding);
}

nw_error_t *transformer_embedding_create(transformer_embedding_t **transformer_embedding, embedding_t *token_embedding, embedding_t *position_embedding)
{
    CHECK_NULL_ARGUMENT(transformer_embedding, "transformer_embedding");
    CHECK_NULL_ARGUMENT(token_embedding, "token_embedding");
    CHECK_NULL_ARGUMENT(position_embedding, "position_embedding");

    *transformer_embedding = (transformer_embedding_t *) malloc(sizeof(transformer_embedding_t));
    if (!*transformer_embedding)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(transformer_embedding_t)), NULL);
    }

    (*transformer_embedding)->token_embedding = token_embedding;
    (*transformer_embedding)->position_embedding = position_embedding;

    return NULL;
}

void transformer_embedding_destroy(transformer_embedding_t *transformer_embedding)
{
    embedding_destroy(transformer_embedding->token_embedding);
    embedding_destroy(transformer_embedding->position_embedding);
    free(transformer_embedding);
}

nw_error_t *causal_multihead_self_attention_create(causal_multihead_self_attention_t **causal_multihead_self_attention, tensor_t *input_weights, tensor_t *input_bias, tensor_t *output_weights,
                                                   tensor_t *output_bias, int64_t number_of_heads, int64_t embedding_size, void *dropout_probability, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(causal_multihead_self_attention, "causal_multihead_self_attention");
    CHECK_NULL_ARGUMENT(input_weights, "input_weights");
    CHECK_NULL_ARGUMENT(output_weights, "output_weights");

    *causal_multihead_self_attention = (causal_multihead_self_attention_t *) malloc(sizeof(causal_multihead_self_attention_t));
    if (!*causal_multihead_self_attention)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(causal_multihead_self_attention_t)), NULL);
    }

    (*causal_multihead_self_attention)->dropout_probability = (void *) malloc(datatype_size(datatype));
    if (!(*causal_multihead_self_attention)->dropout_probability)
    {
        free(*causal_multihead_self_attention);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
    }

    memcpy((*causal_multihead_self_attention)->dropout_probability, dropout_probability, datatype_size(datatype));
    (*causal_multihead_self_attention)->inference = false;
    (*causal_multihead_self_attention)->number_of_heads = number_of_heads;
    (*causal_multihead_self_attention)->embedding_size = embedding_size;
    (*causal_multihead_self_attention)->input_weights = input_weights;
    (*causal_multihead_self_attention)->input_bias = input_bias;
    (*causal_multihead_self_attention)->output_weights = output_weights;
    (*causal_multihead_self_attention)->output_bias = output_bias;
    (*causal_multihead_self_attention)->datatype = datatype;

    return NULL;
}

void causal_multihead_self_attention_destroy(causal_multihead_self_attention_t *causal_multihead_self_attention)
{
    if (causal_multihead_self_attention)
    {
        tensor_destroy(causal_multihead_self_attention->input_weights);
        tensor_destroy(causal_multihead_self_attention->input_bias);
        tensor_destroy(causal_multihead_self_attention->output_weights);
        tensor_destroy(causal_multihead_self_attention->output_bias);
        free(causal_multihead_self_attention->dropout_probability);
        free(causal_multihead_self_attention);
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

nw_error_t *max_pooling_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    pooling_2d_t *pooling_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = MAX_POOLING_2D;

    error = pooling_2d_create(&pooling_2d, padding, stride, kernel_size);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create pooling_2d layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) pooling_2d);
    if (error)
    {
        pooling_2d_destroy(pooling_2d);
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

nw_error_t *average_pooling_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    pooling_2d_t *pooling_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = AVERAGE_POOLING_2D;

    error = pooling_2d_create(&pooling_2d, padding, stride, kernel_size);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create pooling_2d layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) pooling_2d);
    if (error)
    {
        pooling_2d_destroy(pooling_2d);
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
                                             void *epsilon, bool_t weights, bool_t bias, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(normalized_shape, "normalized_shape");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");

    nw_error_t *error = NULL;
    layer_normalization_t *layer_normalization = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = LAYER_NORMALIZATION;

    error = layer_normalization_create(&layer_normalization, normalized_shape, length, epsilon, weights, bias, datatype, runtime);
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

static nw_error_t *vocabulary_count_create(tensor_t **vocabulary_count, int64_t vocabulary_size, datatype_t datatype, runtime_t runtime)
{
    nw_error_t *error = NULL;
    void *start = NULL;
    void *stop = NULL;
    void *step = NULL;
    size_t size = datatype_size(datatype);

    start = (void *) malloc(size);
    if (!start)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    stop = (void *) malloc(size);
    if (!stop)
    {
        free(start);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    step = (void *) malloc(size);
    if (!step)
    {
        free(start);
        free(stop);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) start = (float32_t) 0.0;
        *(float32_t *) stop = (float32_t) vocabulary_size;
        *(float32_t *) step = (float32_t) 1.0;
        break;
    case FLOAT64:
        *(float64_t *) start = (float64_t) 0.0;
        *(float64_t *) stop = (float64_t) vocabulary_size;
        *(float64_t *) step = (float64_t) 1.0;
        break;
    default:
        free(start);
        free(stop);
        free(step);
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype."), NULL);
    }

    error = tensor_arange(vocabulary_count, start, stop, step, runtime, datatype, false, true);
    if (error)
    {
        free(start);
        free(stop);
        free(step);
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    free(start);
    free(stop);
    free(step);

    return error;
}

static nw_error_t *embedding_initialize(embedding_t **embedding, int64_t vocabulary_size, int64_t embedding_size, datatype_t datatype, runtime_t runtime, parameter_init_t *embedding_init)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(embedding_init, "embedding_init");

    nw_error_t *error = NULL;
    tensor_t *weights = NULL;
    int64_t *embedding_shape = (int64_t[]) {vocabulary_size, embedding_size};
    int64_t embedding_rank = 2;
    tensor_t *vocabulary_count = NULL;

    error = vocabulary_count_create(&vocabulary_count, vocabulary_size, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = initialize(&weights, embedding_init, embedding_shape, embedding_rank, runtime, datatype, true);
    if (error)
    {
        tensor_destroy(vocabulary_count);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize embedding."), error);
    }

    error = embedding_create(embedding, vocabulary_size, embedding_size, vocabulary_count, weights);
    if (error)
    {
        tensor_destroy(vocabulary_count);
        tensor_destroy(weights);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }
    
    return error;
}

nw_error_t *embedding_layer_create(layer_t **layer, int64_t vocabulary_size, int64_t embedding_size, datatype_t datatype, runtime_t runtime, parameter_init_t *embedding_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    embedding_t *embedding = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = EMBEDDING;

    error = embedding_initialize(&embedding, vocabulary_size, embedding_size, datatype, runtime, embedding_init);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = transform_create(&transform, transform_type, (void *) embedding);
    if (error)
    {
        embedding_destroy(embedding);
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

static nw_error_t *embedding_create_from_parameters(embedding_t **embedding, tensor_t *weights)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(weights, "weights");

    nw_error_t *error = NULL;
    tensor_t *vocabulary_count = NULL;
    int64_t vocabulary_size = weights->buffer->view->shape[0];
    int64_t embedding_size = weights->buffer->view->shape[1];
    datatype_t datatype = weights->buffer->storage->datatype;
    runtime_t runtime = weights->buffer->storage->runtime;

    error = vocabulary_count_create(&vocabulary_count, vocabulary_size, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = embedding_create(embedding, vocabulary_size, embedding_size, vocabulary_count, weights);
    if (error)
    {
        tensor_destroy(vocabulary_count);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    return error;
}

nw_error_t *embedding_layer_create_from_parameters(layer_t **layer, tensor_t *weights)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(weights, "weights");

    nw_error_t *error = NULL;
    embedding_t *embedding = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = EMBEDDING;

    error = embedding_create_from_parameters(&embedding,  weights);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) embedding);
    if (error)
    {
        embedding_destroy(embedding);
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

nw_error_t *transformer_embedding_layer_create(layer_t **layer, int64_t vocabulary_size, int64_t embedding_size, int64_t block_size, datatype_t datatype, runtime_t runtime,
                                               parameter_init_t *token_embedding_init, parameter_init_t *position_embedding_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(token_embedding_init, "token_embedding_init");
    CHECK_NULL_ARGUMENT(position_embedding_init, "position_embedding_init");

    nw_error_t *error = NULL;
    embedding_t *token_embedding = NULL;
    embedding_t *position_embedding = NULL;
    transform_t *transform = NULL;
    transformer_embedding_t *transformer_embedding = NULL;
    transform_type_t transform_type = TRANSFORMER_EMBEDDING;

    error = embedding_initialize(&token_embedding, vocabulary_size, embedding_size, datatype, runtime, token_embedding_init);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = embedding_initialize(&position_embedding, block_size, embedding_size, datatype, runtime, position_embedding_init);
    if (error)
    {
        embedding_destroy(token_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = transformer_embedding_create(&transformer_embedding, token_embedding, position_embedding);
    if (error)
    {
        embedding_destroy(token_embedding);
        embedding_destroy(position_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create transformer embedding."), error);
    }

    error = transform_create(&transform, transform_type, (void *) transformer_embedding);
    if (error)
    {
        transformer_embedding_destroy(transformer_embedding);
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

nw_error_t *transformer_embedding_layer_create_from_parameters(layer_t **layer, tensor_t *token_weights, tensor_t *position_weights)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(token_weights, "token_weights");
    CHECK_NULL_ARGUMENT(position_weights, "position_weights");

    nw_error_t *error = NULL;
    embedding_t *token_embedding = NULL;
    embedding_t *position_embedding = NULL;
    transform_t *transform = NULL;
    transformer_embedding_t *transformer_embedding = NULL;
    transform_type_t transform_type = TRANSFORMER_EMBEDDING;

    error = embedding_create_from_parameters(&token_embedding, token_weights);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = embedding_create_from_parameters(&position_embedding, position_weights);
    if (error)
    {
        embedding_destroy(token_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = transformer_embedding_create(&transformer_embedding, token_embedding, position_embedding);
    if (error)
    {
        embedding_destroy(token_embedding);
        embedding_destroy(position_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create transformer embedding."), error);
    }

    error = transform_create(&transform, transform_type, (void *) transformer_embedding);
    if (error)
    {
        transformer_embedding_destroy(transformer_embedding);
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

nw_error_t *causal_multihead_self_attention_layer_create(layer_t **layer, int64_t number_of_heads, int64_t embedding_size, void *dropout_probability, datatype_t datatype, runtime_t runtime,
                                                         parameter_init_t *input_weight_init, parameter_init_t *input_bias_init, parameter_init_t *output_weight_init, parameter_init_t *output_bias_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(dropout_probability, "dropout_probability");
    CHECK_NULL_ARGUMENT(input_weight_init, "input_weight_init");
    CHECK_NULL_ARGUMENT(output_weight_init, "output_weight_init");

    nw_error_t *error = NULL;
    tensor_t *input_weights = NULL;
    tensor_t *input_bias = NULL;
    tensor_t *output_weights = NULL;
    tensor_t *output_bias = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CAUSAL_MULTIHEAD_SELF_ATTENTION;
    causal_multihead_self_attention_t *causal_multihead_self_attention = NULL;

    int64_t *input_weight_shape = (int64_t[]) {embedding_size, 3 * embedding_size};
    int64_t *input_bias_shape = (int64_t[]) {3 * embedding_size};
    int64_t *output_weight_shape = (int64_t[]) {embedding_size, embedding_size};
    int64_t *output_bias_shape = (int64_t[]) {embedding_size};
    int64_t weight_rank = 2;
    int64_t bias_rank = 1;

    error = initialize(&input_weights, input_weight_init, input_weight_shape, weight_rank, runtime, datatype, true);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
    }

    if (input_bias_init) 
    {
        error = initialize(&input_bias, input_bias_init, input_bias_shape, bias_rank, runtime, datatype, true);
        if (error)
        {
            tensor_destroy(input_weights);
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
        }
    }

    error = initialize(&output_weights, output_weight_init, output_weight_shape, weight_rank, runtime, datatype, true);
    if (error)
    {
        tensor_destroy(input_weights);
        tensor_destroy(input_bias);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
    }

    if (output_bias_init) 
    {
        error = initialize(&output_bias, output_bias_init, output_bias_shape, bias_rank, runtime, datatype, true);
        if (error)
        {
            tensor_destroy(input_weights);
            tensor_destroy(input_bias);
            tensor_destroy(output_weights);
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
        }
    }

    error = causal_multihead_self_attention_create(&causal_multihead_self_attention, input_weights, input_bias, output_weights, 
                                                   output_bias, number_of_heads, embedding_size, dropout_probability, datatype);
    if (error)
    {
        tensor_destroy(input_weights);
        tensor_destroy(input_bias);
        tensor_destroy(output_weights);
        tensor_destroy(output_bias);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
    }

    error = transform_create(&transform, transform_type, (void *) causal_multihead_self_attention);
    if (error)
    {
        causal_multihead_self_attention_destroy(causal_multihead_self_attention);
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

nw_error_t *causal_multihead_self_attention_layer_create_from_parameters(layer_t **layer, int64_t number_of_heads, int64_t embedding_size, void *dropout_probability, datatype_t datatype,
                                                                         tensor_t *input_weights, tensor_t *input_bias, tensor_t *output_weights, tensor_t *output_bias)
{
    nw_error_t *error = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CAUSAL_MULTIHEAD_SELF_ATTENTION;
    causal_multihead_self_attention_t *causal_multihead_self_attention = NULL;

    error = causal_multihead_self_attention_create(&causal_multihead_self_attention, input_weights, input_bias, output_weights, 
                                                   output_bias, number_of_heads, embedding_size, dropout_probability, datatype);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
    }

    error = transform_create(&transform, transform_type, (void *) causal_multihead_self_attention);
    if (error)
    {
        causal_multihead_self_attention_destroy(causal_multihead_self_attention);
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

nw_error_t *residual_block_layer_create(layer_t **layer, block_t *block)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = RESIDUAL_BLOCK;

    error = transform_create(&transform, transform_type, (void *) block);
    if (error)
    {
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

nw_error_t *block_layer_create(layer_t **layer, block_t *block)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = BLOCK;

    error = transform_create(&transform, transform_type, (void *) block);
    if (error)
    {
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
        case MAX_POOLING_2D:
            error = max_pooling_2d_forward(transform->pooling_2d, x, &feature_map);
            break;
        case AVERAGE_POOLING_2D:
            error = average_pooling_2d_forward(transform->pooling_2d, x, &feature_map);
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
        case LAYER_NORMALIZATION:
            error = layer_normalization_forward(transform->layer_normalization, x, &feature_map);
            break;
        case EMBEDDING:
            error = embedding_forward(transform->embedding, x, &feature_map);
            break;
        case TRANSFORMER_EMBEDDING:
            error = transformer_embedding_forward(transform->transformer_embedding, x, &feature_map);
            break;
        case CAUSAL_MULTIHEAD_SELF_ATTENTION:
            error = causal_multihead_self_attention_forward(transform->causal_multihead_self_attention, x, &feature_map);
            break;
        case ACTIVATION:
            error = activation_forward(transform->activation, x, &feature_map);
            break;
        case RESIDUAL_BLOCK:
            error = residual_block_forward(transform->block, x, &feature_map);
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

nw_error_t *max_pooling_2d_forward(pooling_2d_t *pooling_2d, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_max_pool_2d(x, y, pooling_2d->kernel, pooling_2d->stride, pooling_2d->padding);
    if (error)
    {
        return ERROR(ERROR_POOLING, string_create("failed to apply pooling."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *average_pooling_2d_forward(pooling_2d_t *pooling_2d, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_average_pool_2d(x, y, pooling_2d->kernel, pooling_2d->stride, pooling_2d->padding);
    if (error)
    {
        return ERROR(ERROR_POOLING, string_create("failed to apply pooling."), error);
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

nw_error_t *embedding_forward(embedding_t *embedding, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_embedding(x, embedding->weights, embedding->vocabulary_counter, y);
    if (error)
    {
        return ERROR(ERROR_EMBEDDING, string_create("failed to embed tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *transformer_embedding_forward(transformer_embedding_t *transformer_embedding, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(transformer_embedding, "transformer_embedding");
    CHECK_NULL_ARGUMENT(x, "x");

    nw_error_t *error = NULL;
    tensor_t *token_embedding = NULL;
    tensor_t *position_embedding = NULL;
    tensor_t *positions = NULL;
    tensor_t *positions_expand = NULL;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;
    int64_t block_size = x->buffer->view->shape[1];
    void *start = NULL;
    void *stop = NULL;
    void *step = NULL;
    size_t size = datatype_size(datatype);

    start = (void *) malloc(size);
    if (!start)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes", size), NULL);
        goto cleanup;
    }

    stop = (void *) malloc(size);
    if (!stop)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes", size), NULL);
        goto cleanup;
    }

    step = (void *) malloc(size);
    if (!step)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) start = (float32_t) 0.0;
        *(float32_t *) stop = (float32_t) block_size;
        *(float32_t *) step = 1.0;
        break;
    case FLOAT64:
        *(float64_t *) start = (float64_t) 0.0;
        *(float64_t *) stop = (float64_t) block_size;
        *(float64_t *) step = 1.0;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype."), NULL);
        goto cleanup;
    }

    error = tensor_arange(&positions, start, stop, step, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_expand(positions, (int64_t[]){1, block_size}, 2, &positions_expand);
    if (error)
    {
        error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
        goto cleanup;
    }

    error = embedding_forward(transformer_embedding->token_embedding, x, &token_embedding);
    if (error)
    {
        error = ERROR(ERROR_EMBEDDING, string_create("failed to embed tensor."), error);
        goto cleanup;
    }

    error = embedding_forward(transformer_embedding->position_embedding, positions_expand, &position_embedding);
    if (error)
    {
        error = ERROR(ERROR_EMBEDDING, string_create("failed to embed tensor."), error);
        goto cleanup;
    }

    error = tensor_addition(token_embedding, position_embedding, y);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    free(start);
    free(stop);
    free(step);
    tensor_destroy(positions);
    tensor_destroy(positions_expand);

    if (!(x->requires_gradient || transformer_embedding->token_embedding->weights->requires_gradient || transformer_embedding->position_embedding->weights->requires_gradient) || no_gradient)
    {
        tensor_destroy(token_embedding);
        tensor_destroy(position_embedding);
    }

    return error;
}

nw_error_t *causal_multihead_self_attention_forward(causal_multihead_self_attention_t *causal_multihead_self_attention, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(causal_multihead_self_attention, "causal_multihead_self_attention");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_causal_multihead_self_attention(x, causal_multihead_self_attention->input_weights, causal_multihead_self_attention->input_bias,
                                                   causal_multihead_self_attention->output_weights, causal_multihead_self_attention->output_bias,
                                                   causal_multihead_self_attention->number_of_heads, causal_multihead_self_attention->dropout_probability,
                                                   causal_multihead_self_attention->inference, y);
    if (error)
    {
        return ERROR(ERROR_ATTENTION, string_create("failed to apply causal multihead self attention."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *residual_block_forward(block_t *block, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *z = NULL;

    error = block_forward(block, x, &z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply forward operation."), error);
    }

    error = tensor_addition(x, z, y);
    if (error)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
    }

    if (!(x->requires_gradient || z->requires_gradient) || no_gradient)
    {
        tensor_destroy(z);
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
        case CAUSAL_MULTIHEAD_SELF_ATTENTION:
            block->layers[i]->transform->causal_multihead_self_attention->inference = inference;
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

nw_error_t *model_save(model_t *model, string_t path)
{
    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(path, "path");

    FILE *file = fopen(path, "wb");
    if (!file)
    {
        return ERROR(ERROR_FILE, string_create("failed to open file."), NULL);
    }

    nw_error_t *error = block_save(model->block, file);
    if (error)
    {
        fclose(file);
        return ERROR(ERROR_SAVE, string_create("failed to save block."), error);
    }    

    fclose(file);

    return error;
}

nw_error_t *block_save(block_t *block, FILE *file)
{
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&block->depth, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    for (int64_t i = 0; i < block->depth; ++i)
    {
        error = layer_save(block->layers[i], file);
        if (error)
        {
            return ERROR(ERROR_SAVE, string_create("failed to save layer."), error);
        }
    }

    return error;
}

nw_error_t *layer_save(layer_t *layer, FILE *file)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&layer->transform_type, sizeof(transform_type_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    switch (layer->transform_type)
    {
    case LINEAR:
        error = linear_save(layer->transform->linear, file);
        break;
    case CONVOLUTION_2D:
    case CONVOLUTION_TRANSPOSE_2D:
        error = convolution_2d_save(layer->transform->convolution_2d, file);
        break;
    case MAX_POOLING_2D:
    case AVERAGE_POOLING_2D:
        error = pooling_2d_save(layer->transform->pooling_2d, file);
        break;
    case DROPOUT:
        error = dropout_save(layer->transform->dropout, file);
        break;
    case BATCH_NORMALIZATION_2D:
        error = batch_normalization_2d_save(layer->transform->batch_normalization_2d, file);
        break;
    case RESHAPE:
        error = reshape_save(layer->transform->reshape, file);
        break;
    case LAYER_NORMALIZATION:
        error = layer_normalization_save(layer->transform->layer_normalization, file);
        break;
    case EMBEDDING:
        error = embedding_save(layer->transform->embedding, file);
        break;
    case TRANSFORMER_EMBEDDING:
        error = transformer_embedding_save(layer->transform->transformer_embedding, file);
        break;
    case CAUSAL_MULTIHEAD_SELF_ATTENTION:
        error = causal_multihead_self_attention_save(layer->transform->causal_multihead_self_attention, file);
        break;
    case ACTIVATION:
        error = activation_save(layer->transform->activation, file);
        break;
    case RESIDUAL_BLOCK:
    case BLOCK:
        error = block_save(layer->transform->block, file);
        break;
    default:
        error = ERROR(ERROR_LAYER_TYPE, string_create("unknown transform type %d.", (int) layer->transform_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save transform."), error);
    }
    
    return error;
}

nw_error_t *linear_save(linear_t *linear, FILE *file)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    error = tensor_save(linear->weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(linear->bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    return error;
}

nw_error_t *convolution_2d_save(convolution_2d_t *convolution_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    error = tensor_save(convolution_2d->kernel, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(convolution_2d->bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    if (!fwrite(&convolution_2d->padding, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&convolution_2d->stride, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    return error;
}

nw_error_t *pooling_2d_save(pooling_2d_t *pooling_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&pooling_2d->kernel, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&pooling_2d->padding, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&pooling_2d->stride, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    return error;
}

nw_error_t *dropout_save(dropout_t *dropout, FILE *file)
{
    CHECK_NULL_ARGUMENT(dropout, "dropout");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&dropout->datatype, sizeof(datatype_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(dropout->probability, datatype_size(dropout->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&dropout->inference, sizeof(bool_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    return error;
}

nw_error_t *batch_normalization_2d_save(batch_normalization_2d_t *batch_normalization_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(batch_normalization_2d, "batch_normalization_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&batch_normalization_2d->datatype, sizeof(datatype_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&batch_normalization_2d->track_running_stats, sizeof(bool_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&batch_normalization_2d->inference, sizeof(bool_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(batch_normalization_2d->momentum, datatype_size(batch_normalization_2d->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(batch_normalization_2d->epsilon, datatype_size(batch_normalization_2d->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    error = tensor_save(batch_normalization_2d->weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(batch_normalization_2d->bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(batch_normalization_2d->running_mean, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(batch_normalization_2d->running_variance, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }
    
    return error;
}

nw_error_t *layer_normalization_save(layer_normalization_t *layer_normalization, FILE *file)
{
    CHECK_NULL_ARGUMENT(layer_normalization, "layer_normalization");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&layer_normalization->datatype, sizeof(datatype_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(layer_normalization->epsilon, datatype_size(layer_normalization->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&layer_normalization->length, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(layer_normalization->normalized_shape, sizeof(int64_t), layer_normalization->length, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    error = tensor_save(layer_normalization->weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(layer_normalization->bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    return error;
}

nw_error_t *reshape_save(reshape_t *reshape, FILE *file)
{
    CHECK_NULL_ARGUMENT(reshape, "reshape");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&reshape->length, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(reshape->shape, sizeof(int64_t), reshape->length, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    return error;
}

nw_error_t *embedding_save(embedding_t *embedding, FILE *file)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&embedding->vocabulary_size, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&embedding->embedding_size, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    error = tensor_save(embedding->vocabulary_counter, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(embedding->weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    return error;
}

nw_error_t *transformer_embedding_save(transformer_embedding_t *transformer_embedding, FILE *file)
{
    CHECK_NULL_ARGUMENT(transformer_embedding, "transformer_embedding");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    error = embedding_save(transformer_embedding->token_embedding, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save embedding."), error);
    }

    error = embedding_save(transformer_embedding->position_embedding, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save embedding."), error);
    }

    return error;
}

nw_error_t *causal_multihead_self_attention_save(causal_multihead_self_attention_t *causal_multihead_self_attention, FILE *file)
{
    CHECK_NULL_ARGUMENT(causal_multihead_self_attention, "causal_multihead_self_attention");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&causal_multihead_self_attention->datatype, sizeof(datatype_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(causal_multihead_self_attention->dropout_probability, datatype_size(causal_multihead_self_attention->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&causal_multihead_self_attention->embedding_size, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&causal_multihead_self_attention->number_of_heads, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&causal_multihead_self_attention->inference, sizeof(bool_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    error = tensor_save(causal_multihead_self_attention->input_weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(causal_multihead_self_attention->input_bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(causal_multihead_self_attention->output_weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(causal_multihead_self_attention->output_bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    return error;
}

nw_error_t *model_load(model_t **model, string_t path)
{
    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(path, "path");

    FILE *file = fopen(path, "rb");
    if (!file)
    {
        return ERROR(ERROR_FILE, string_create("failed to open file."), NULL);
    }
    
    *model = (model_t *) malloc(sizeof(model_t));
    if (!*model)
    {
        fclose(file);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(model_t)), NULL);
    }
    (*model)->block = NULL;


    nw_error_t *error = block_load(&(*model)->block, file);
    if (error)
    {
        fclose(file);
        model_destroy(*model);
        return ERROR(ERROR_LOAD, string_create("failed to load block."), error);
    }    

    fclose(file);

    return error;
}

nw_error_t *block_load(block_t **block, FILE *file)
{
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *block = (block_t *) malloc(sizeof(block_t));
    if (!*block)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(block_t)), NULL);
        goto cleanup;
    }
    (*block)->layers = NULL;
    (*block)->depth = 0;

    if (!fread(&(*block)->depth, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read to file."), NULL);
        goto cleanup;
    }

    (*block)->layers = (layer_t **) malloc((*block)->depth * sizeof(layer_t *));
    if (!(*block)->layers)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", (*block)->depth * sizeof(layer_t *)), NULL);
        goto cleanup;
    }

    for (int64_t i = 0; i < (*block)->depth; ++i)
    {
        error = layer_load(&(*block)->layers[i], file);
        if (error)
        {
            error = ERROR(ERROR_LOAD, string_create("failed to load layer."), error);
            goto cleanup;
        }
    }

    return error;

cleanup:

    block_destroy(*block);

    return error;
}

nw_error_t *layer_load(layer_t **layer, FILE *file)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *layer = (layer_t *) malloc(sizeof(layer_t));
    if (!*layer)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_t)), NULL);
        goto cleanup;
    }
    (*layer)->transform = NULL;

    if (!fread(&(*layer)->transform_type, sizeof(transform_type_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    (*layer)->transform = (transform_t *) malloc(sizeof(transform_t));
    if (!(*layer)->transform)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(transform_t)), NULL);
        goto cleanup;
    }

    switch ((*layer)->transform_type)
    {
    case LINEAR:
        error = linear_load(&(*layer)->transform->linear, file);
        break;
    case CONVOLUTION_2D:
    case CONVOLUTION_TRANSPOSE_2D:
        error = convolution_2d_load(&(*layer)->transform->convolution_2d, file);
        break;
    case MAX_POOLING_2D:
    case AVERAGE_POOLING_2D:
        error = pooling_2d_load(&(*layer)->transform->pooling_2d, file);
        break;
    case DROPOUT:
        error = dropout_load(&(*layer)->transform->dropout, file);
        break;
    case BATCH_NORMALIZATION_2D:
        error = batch_normalization_2d_load(&(*layer)->transform->batch_normalization_2d, file);
        break;
    case RESHAPE:
        error = reshape_load(&(*layer)->transform->reshape, file);
        break;
    case LAYER_NORMALIZATION:
        error = layer_normalization_load(&(*layer)->transform->layer_normalization, file);
        break;
    case EMBEDDING:
        error = embedding_load(&(*layer)->transform->embedding, file);
        break;
    case TRANSFORMER_EMBEDDING:
        error = transformer_embedding_load(&(*layer)->transform->transformer_embedding, file);
        break;
    case CAUSAL_MULTIHEAD_SELF_ATTENTION:
        error = causal_multihead_self_attention_load(&(*layer)->transform->causal_multihead_self_attention, file);
        break;
    case ACTIVATION:
        error = activation_load(&(*layer)->transform->activation, file);
        break;
    case RESIDUAL_BLOCK:
    case BLOCK:
        error = block_load(&(*layer)->transform->block, file);
        break;
    default:
        error = ERROR(ERROR_LAYER_TYPE, string_create("unknown transform type %d.", (int) (*layer)->transform_type), NULL);
        break;
    }

    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed load transform."), error);
        goto cleanup;
    }

    return error;

cleanup:

    layer_destroy(*layer);

    return error;
}

nw_error_t *linear_load(linear_t **linear, FILE *file)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;
    
    *linear = (linear_t *) malloc(sizeof(linear_t));
    if (!*linear)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(linear_t)), NULL);
        goto cleanup;
    }

    (*linear)->weights = NULL;
    (*linear)->bias = NULL;
    
    error = tensor_load(&(*linear)->weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*linear)->bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    return error;

cleanup:

    linear_destroy(*linear);

    return error;
}

nw_error_t *convolution_2d_load(convolution_2d_t **convolution_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *convolution_2d = (convolution_2d_t *) malloc(sizeof(convolution_2d_t));
    if (!*convolution_2d)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(convolution_2d_t)), NULL);
        goto cleanup;
    }

    (*convolution_2d)->kernel = NULL;
    (*convolution_2d)->bias = NULL;

    error = tensor_load(&(*convolution_2d)->kernel, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*convolution_2d)->bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    if (!fread(&(*convolution_2d)->padding, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*convolution_2d)->stride, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    convolution_2d_destroy(*convolution_2d);

    return error;
}

nw_error_t *pooling_2d_load(pooling_2d_t **pooling_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *pooling_2d = (pooling_2d_t *) malloc(sizeof(pooling_2d_t));
    if (!*pooling_2d)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(pooling_2d_t)), NULL);
        goto cleanup;
    }

    if (!fread(&(*pooling_2d)->kernel, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*pooling_2d)->padding, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*pooling_2d)->stride, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    pooling_2d_destroy(*pooling_2d);

    return error;
}

nw_error_t *dropout_load(dropout_t **dropout, FILE *file)
{
    CHECK_NULL_ARGUMENT(dropout, "dropout");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *dropout = (dropout_t *) malloc(sizeof(dropout_t));
    if (!*dropout)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(dropout_t)), NULL);
        goto cleanup;
    }
    (*dropout)->probability = NULL;

    if (!fread(&(*dropout)->datatype, sizeof(datatype_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    (*dropout)->probability = (void *) malloc(datatype_size((*dropout)->datatype));
    if (!(*dropout)->probability)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size((*dropout)->datatype)), NULL);
        goto cleanup;
    }

    if (!fread((*dropout)->probability, datatype_size((*dropout)->datatype), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*dropout)->inference, sizeof(bool_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    dropout_destroy(*dropout);

    return error;
}

nw_error_t *batch_normalization_2d_load(batch_normalization_2d_t **batch_normalization_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(batch_normalization_2d, "batch_normalization_2d");
    CHECK_NULL_ARGUMENT(file, "file");

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

    if (!fread(&(*batch_normalization_2d)->datatype, sizeof(datatype_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*batch_normalization_2d)->track_running_stats, sizeof(bool_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*batch_normalization_2d)->inference, sizeof(bool_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    size_t size = datatype_size((*batch_normalization_2d)->datatype);

    (*batch_normalization_2d)->momentum = (void *) malloc(size);
    if (!(*batch_normalization_2d)->momentum)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    (*batch_normalization_2d)->epsilon = (void *) malloc(size);
    if (!(*batch_normalization_2d)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    if (!fread((*batch_normalization_2d)->momentum, size, 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread((*batch_normalization_2d)->epsilon, size, 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    error = tensor_load(&(*batch_normalization_2d)->weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*batch_normalization_2d)->bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*batch_normalization_2d)->running_mean, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*batch_normalization_2d)->running_variance, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }
    
    return error;

cleanup:

    batch_normalization_2d_destroy(*batch_normalization_2d);

    return error;
}

nw_error_t *layer_normalization_load(layer_normalization_t **layer_normalization, FILE *file)
{
    CHECK_NULL_ARGUMENT(layer_normalization, "layer_normalization");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *layer_normalization = (layer_normalization_t *) malloc(sizeof(layer_normalization_t));
    if (!*layer_normalization)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_normalization_t)), NULL);
        goto cleanup;
    }

    (*layer_normalization)->epsilon = NULL;
    (*layer_normalization)->bias = NULL;
    (*layer_normalization)->weights = NULL;
    (*layer_normalization)->normalized_shape = NULL;

    if (!fread(&(*layer_normalization)->datatype, sizeof(datatype_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    size_t size = datatype_size((*layer_normalization)->datatype);
    
    (*layer_normalization)->epsilon = (void *) malloc(size);
    if (!(*layer_normalization)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    if (!fread((*layer_normalization)->epsilon, size, 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*layer_normalization)->length, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    size = (*layer_normalization)->length * sizeof(int64_t);

    (*layer_normalization)->normalized_shape = (int64_t *) malloc(size);
    if (!(*layer_normalization)->normalized_shape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    if (!fread((*layer_normalization)->normalized_shape, sizeof(int64_t), (*layer_normalization)->length, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    error = tensor_load(&(*layer_normalization)->weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*layer_normalization)->bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    return error;

cleanup:

    layer_normalization_destroy(*layer_normalization);

    return error;
}

nw_error_t *reshape_load(reshape_t **reshape, FILE *file)
{
    CHECK_NULL_ARGUMENT(reshape, "reshape");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *reshape = (reshape_t *) malloc(sizeof(reshape_t));
    if (!*reshape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(reshape_t)), NULL);
        goto cleanup;
    }

    (*reshape)->shape = NULL;

    if (!fread(&(*reshape)->length, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    (*reshape)->shape = (int64_t *) malloc((*reshape)->length * sizeof(int64_t));
    if (!(*reshape)->shape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", (*reshape)->length * sizeof(int64_t)), NULL);
        goto cleanup;
    }

    if (!fread((*reshape)->shape, sizeof(int64_t), (*reshape)->length, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    reshape_destroy(*reshape);

    return error;
}

nw_error_t *embedding_load(embedding_t **embedding, FILE *file)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *embedding = (embedding_t *) malloc(sizeof(embedding_t));
    if (!*embedding)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(embedding_t)), NULL);
        goto cleanup;
    }

    (*embedding)->vocabulary_counter = NULL;
    (*embedding)->weights = NULL;

    if (!fread(&(*embedding)->vocabulary_size, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*embedding)->embedding_size, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    error = tensor_load(&(*embedding)->vocabulary_counter, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*embedding)->weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    return error;

cleanup:

    embedding_destroy(*embedding);

    return error;
}

nw_error_t *transformer_embedding_load(transformer_embedding_t **transformer_embedding, FILE *file)
{
    CHECK_NULL_ARGUMENT(transformer_embedding, "transformer_embedding");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *transformer_embedding = (transformer_embedding_t *) malloc(sizeof(transformer_embedding_t));
    if (!*transformer_embedding)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(transformer_embedding_t)), NULL);
    }

    (*transformer_embedding)->token_embedding = NULL;
    (*transformer_embedding)->position_embedding = NULL;

    error = embedding_load(&(*transformer_embedding)->token_embedding, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load embedding."), error);
        goto cleanup;
    }

    error = embedding_load(&(*transformer_embedding)->position_embedding, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load embedding."), error);
        goto cleanup;
    }

    return error;

cleanup:

    transformer_embedding_destroy(*transformer_embedding);

    return error;
}

nw_error_t *causal_multihead_self_attention_load(causal_multihead_self_attention_t **causal_multihead_self_attention, FILE *file)
{
    CHECK_NULL_ARGUMENT(causal_multihead_self_attention, "causal_multihead_self_attention");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *causal_multihead_self_attention = (causal_multihead_self_attention_t *) malloc(sizeof(causal_multihead_self_attention_t));
    if (!*causal_multihead_self_attention)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(causal_multihead_self_attention_t)), NULL);
        goto cleanup;
    }

    (*causal_multihead_self_attention)->input_weights = NULL;
    (*causal_multihead_self_attention)->input_bias = NULL;
    (*causal_multihead_self_attention)->output_weights = NULL;
    (*causal_multihead_self_attention)->output_bias = NULL;
    (*causal_multihead_self_attention)->dropout_probability = NULL;

    if (!fread(&(*causal_multihead_self_attention)->datatype, sizeof(datatype_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    (*causal_multihead_self_attention)->dropout_probability = (void *) malloc(datatype_size((*causal_multihead_self_attention)->datatype));
    if (!(*causal_multihead_self_attention)->dropout_probability)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size((*causal_multihead_self_attention)->datatype)), NULL);
        goto cleanup;
    }

    if (!fread((*causal_multihead_self_attention)->dropout_probability, datatype_size((*causal_multihead_self_attention)->datatype), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*causal_multihead_self_attention)->embedding_size, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*causal_multihead_self_attention)->number_of_heads, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*causal_multihead_self_attention)->inference, sizeof(bool_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    error = tensor_load(&(*causal_multihead_self_attention)->input_weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*causal_multihead_self_attention)->input_bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*causal_multihead_self_attention)->output_weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*causal_multihead_self_attention)->output_bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    return error;

cleanup:

    causal_multihead_self_attention_destroy(*causal_multihead_self_attention);

    return error;
}

nw_error_t *model_parameter_count(model_t *model, int64_t *count)
{
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = NULL;

    error = block_parameter_count(model->block, count);
    if (error)
    {
        return ERROR(ERROR_N, string_create("failed count parameters."), error);
    }

    return error;
}

nw_error_t *block_parameter_count(block_t *block, int64_t *count)
{
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(block->layers, "block->layers");

    nw_error_t *error = NULL;

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
            error = linear_parameter_count(transform->linear, count);
            break;
        case CONVOLUTION_2D:
        case CONVOLUTION_TRANSPOSE_2D:
            error = convolution_2d_parameter_count(transform->convolution_2d, count);
            break;
        case BATCH_NORMALIZATION_2D:
            error = batch_normalization_2d_parameter_count(transform->batch_normalization_2d, count);
            break;
        case LAYER_NORMALIZATION:
            error = layer_normalization_parameter_count(transform->layer_normalization, count);
            break;
        case EMBEDDING:
            error = embedding_parameter_count(transform->embedding, count);
            break;
        case TRANSFORMER_EMBEDDING:
            error = transformer_embedding_parameter_count(transform->transformer_embedding, count);
            break;
        case CAUSAL_MULTIHEAD_SELF_ATTENTION:
            error = causal_multihead_self_attention_parameter_count(transform->causal_multihead_self_attention, count);
            break;
        case ACTIVATION:
        case RESHAPE:
        case MAX_POOLING_2D:
        case AVERAGE_POOLING_2D:
        case DROPOUT:
            break;
        case RESIDUAL_BLOCK:
        case BLOCK:
            error = block_parameter_count(transform->block, count);
            break;
        default:
            error = ERROR(ERROR_LAYER_TYPE, string_create("unknown transform type %d.", (int) transform_type), NULL);
            break;
        }

        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    return error;
}

nw_error_t *linear_parameter_count(linear_t *linear, int64_t *count)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(count, "count");

    nw_error_t *error = NULL;
    int64_t size = 0;

    if (linear->weights) 
    {
        error = view_physical_size(linear->weights->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    if (linear->bias) 
    {
        error = view_physical_size(linear->bias->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    return error;
}

nw_error_t *convolution_2d_parameter_count(convolution_2d_t *convolution_2d, int64_t *count)
{
    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(count, "count");

    nw_error_t *error = NULL;
    int64_t size = 0;

    if (convolution_2d->kernel) 
    {
        error = view_physical_size(convolution_2d->kernel->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    if (convolution_2d->bias) 
    {
        error = view_physical_size(convolution_2d->bias->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    return error;
}

nw_error_t *batch_normalization_2d_parameter_count(batch_normalization_2d_t *batch_normalization_2d, int64_t *count)
{
    CHECK_NULL_ARGUMENT(batch_normalization_2d, "batch_normalization_2d");
    CHECK_NULL_ARGUMENT(count, "count");

    nw_error_t *error = NULL;
    int64_t size = 0;

    if (batch_normalization_2d->weights) 
    {
        error = view_physical_size(batch_normalization_2d->weights->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    if (batch_normalization_2d->bias) 
    {
        error = view_physical_size(batch_normalization_2d->bias->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    return error;
}

nw_error_t *layer_normalization_parameter_count(layer_normalization_t *layer_normalization, int64_t *count)
{
    CHECK_NULL_ARGUMENT(layer_normalization, "layer_normalization");
    CHECK_NULL_ARGUMENT(count, "count");

    nw_error_t *error = NULL;
    int64_t size = 0;

    if (layer_normalization->weights) 
    {
        error = view_physical_size(layer_normalization->weights->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    if (layer_normalization->bias) 
    {
        error = view_physical_size(layer_normalization->bias->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    return error;
}

nw_error_t *embedding_parameter_count(embedding_t *embedding, int64_t *count)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(count, "count");

    nw_error_t *error = NULL;
    int64_t size = 0;

    if (embedding->weights) 
    {
        error = view_physical_size(embedding->weights->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    return error;
}

nw_error_t *transformer_embedding_parameter_count(transformer_embedding_t *transformer_embedding, int64_t *count)
{
    CHECK_NULL_ARGUMENT(transformer_embedding, "transformer_embedding");
    CHECK_NULL_ARGUMENT(count, "count");

    nw_error_t *error = NULL;
    int64_t size = 0;

    if (transformer_embedding->position_embedding) 
    {
        error = embedding_parameter_count(transformer_embedding->position_embedding, count);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    if (transformer_embedding->token_embedding) 
    {
        error = embedding_parameter_count(transformer_embedding->token_embedding, count);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    return error;
}

nw_error_t *causal_multihead_self_attention_parameter_count(causal_multihead_self_attention_t *causal_multihead_self_attention, int64_t *count)
{
    CHECK_NULL_ARGUMENT(causal_multihead_self_attention, "causal_multihead_self_attention");
    CHECK_NULL_ARGUMENT(count, "count");

    nw_error_t *error = NULL;
    int64_t size = 0;

    if (causal_multihead_self_attention->input_weights) 
    {
        error = view_physical_size(causal_multihead_self_attention->input_weights->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    if (causal_multihead_self_attention->input_bias) 
    {
        error = view_physical_size(causal_multihead_self_attention->input_bias->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    if (causal_multihead_self_attention->output_weights) 
    {
        error = view_physical_size(causal_multihead_self_attention->output_weights->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    if (causal_multihead_self_attention->output_bias) 
    {
        error = view_physical_size(causal_multihead_self_attention->output_bias->buffer->view, &size);
        if (error)
        {
            return ERROR(ERROR_N, string_create("failed to count parameters."), error);
        }
    }

    *count += size;

    return error;
}
