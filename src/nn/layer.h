#ifndef LAYER_H
#define LAYER_H

#include <datatype.h>
#include <errors.h>
#include <buffer.h>
#include <activation.h>

// Forward declarations
typedef struct tensor_t tensor_t;
typedef struct block_t block_t;
typedef struct parameter_init_t parameter_init_t;

typedef struct linear_t
{
    tensor_t *weights;
    tensor_t *bias;
} linear_t;

typedef struct convolution_2d_t
{
    int64_t padding;
    int64_t stride;
    tensor_t *kernel;
    tensor_t *bias;
} convolution_2d_t;

typedef struct pooling_2d_t
{
    int64_t padding;
    int64_t stride;
    int64_t kernel;
} pooling_2d_t;

typedef struct dropout_t
{
    void *probability;
    datatype_t datatype;
    bool_t inference;
} dropout_t;

typedef struct batch_normalization_2d_t
{
    void *momentum;
    void *epsilon;
    bool_t track_running_stats;
    tensor_t *weights;
    tensor_t *bias;
    tensor_t *running_mean;
    tensor_t *running_variance;
    bool_t inference;
    datatype_t datatype;
} batch_normalization_2d_t;

typedef struct layer_normalization_t
{
    void *epsilon;
    int64_t *normalized_shape;
    int64_t length;
    tensor_t *weights;
    tensor_t *bias;
    datatype_t datatype;
} layer_normalization_t;

typedef struct reshape_t
{
    int64_t *shape;
    int64_t length;
} reshape_t;

typedef struct embedding_t
{
    int64_t vocabulary_size;
    int64_t embedding_size;
    tensor_t *vocabulary_counter;
    tensor_t *weights;
} embedding_t;

typedef struct transformer_embedding_t
{
    embedding_t *token_embedding;    
    embedding_t *position_embedding;    
} transformer_embedding_t;

typedef struct causal_multihead_self_attention_t
{
    tensor_t *input_weights;
    tensor_t *input_bias;
    tensor_t *output_weights;
    tensor_t *output_bias;
    int64_t number_of_heads;
    int64_t embedding_size;
    void *dropout_probability;
    bool_t inference;
    datatype_t datatype;
} causal_multihead_self_attention_t;

typedef union transform_t
{
    linear_t *linear;
    convolution_2d_t *convolution_2d;
    pooling_2d_t *pooling_2d;
    dropout_t *dropout;
    batch_normalization_2d_t *batch_normalization_2d;
    layer_normalization_t *layer_normalization;
    reshape_t *reshape;
    embedding_t *embedding;
    transformer_embedding_t *transformer_embedding;
    causal_multihead_self_attention_t *causal_multihead_self_attention;
    activation_t *activation;
    block_t *block;
} transform_t;

typedef enum transform_type_t
{
    LINEAR,
    CONVOLUTION_2D,
    CONVOLUTION_TRANSPOSE_2D,
    MAX_POOLING_2D,
    AVERAGE_POOLING_2D,
    DROPOUT,
    BATCH_NORMALIZATION_2D,
    LAYER_NORMALIZATION,
    RESHAPE,
    EMBEDDING,
    TRANSFORMER_EMBEDDING,
    CAUSAL_MULTIHEAD_SELF_ATTENTION,
    ACTIVATION,
    RESIDUAL_BLOCK,
    BLOCK
} transform_type_t;

typedef struct layer_t
{
    transform_t *transform;
    transform_type_t transform_type;
} layer_t;

typedef struct block_t
{
    layer_t **layers;
    int64_t depth;
} block_t;

typedef struct model_t
{
    block_t *block;
} model_t;

// Model Creation
nw_error_t *model_create(model_t **model, block_t *block);
void model_destroy(model_t *model);

nw_error_t *block_create(block_t **block, int64_t depth, ...);
nw_error_t *block_create_from_array(block_t **block, int64_t depth, layer_t **layers);
void block_destroy(block_t *block);

nw_error_t *layer_create(layer_t **layer, transform_t *transform, transform_type_t transform_type);
void layer_destroy(layer_t *layer);

nw_error_t *transform_create(transform_t **transform, transform_type_t transform_type, void *type_transform);
void transform_destroy(transform_t *transform, transform_type_t transform_type);
string_t transform_type_string(transform_type_t transform_type);

nw_error_t *linear_create(linear_t **linear, tensor_t *weights, tensor_t *bias);
void linear_destroy(linear_t *linear);

nw_error_t *convolution_2d_create(convolution_2d_t **convolution_2d, int64_t padding, int64_t stride, tensor_t *kernel, tensor_t *bias);
void convolution_2d_destroy(convolution_2d_t *convolution_2d);

nw_error_t *pooling_2d_create(pooling_2d_t **pooling_2d, int64_t padding, int64_t stride, int64_t kernel);
void pooling_2d_destroy(pooling_2d_t *pooling_2d);

nw_error_t *dropout_create(dropout_t **dropout, void *probability, datatype_t datatype);
void dropout_destroy(dropout_t *dropout);

nw_error_t *batch_normalization_2d_create(batch_normalization_2d_t **batch_normalization_2d, int64_t number_of_features,
                                          void *momentum, void *epsilon, bool_t track_running_stats,
                                          bool_t affine, datatype_t datatype, runtime_t runtime);
void batch_normalization_2d_destroy(batch_normalization_2d_t *batch_normalization_2d);

nw_error_t *layer_normalization_create(layer_normalization_t **layer_normalization, const int64_t *normalized_shape, int64_t length,
                                        void *epsilon, bool_t weights, bool_t bias, datatype_t datatype, runtime_t runtime);
void layer_normalization_destroy(layer_normalization_t *layer_normalization);

nw_error_t *reshape_create(reshape_t **reshape, int64_t *shape, int64_t length);
void reshape_destroy(reshape_t *reshape);

nw_error_t *embedding_create(embedding_t **embedding, int64_t vocabulary_size, int64_t embedding_size, tensor_t *vocabulary_counter, tensor_t *weights);
void embedding_destroy(embedding_t *embedding);

nw_error_t *transformer_embedding_create(transformer_embedding_t **transformer_embedding, embedding_t *token_embedding, embedding_t *position_embedding);
void transformer_embedding_destroy(transformer_embedding_t *transformer_embedding);

nw_error_t *causal_multihead_self_attention_create(causal_multihead_self_attention_t **causal_multihead_self_attention, tensor_t *input_weights, tensor_t *input_bias, tensor_t *output_weights,
                                                   tensor_t *output_bias, int64_t number_of_heads, int64_t embedding_size, void *dropout_probability, datatype_t datatype);
void causal_multihead_self_attention_destroy(causal_multihead_self_attention_t *causal_multihead_self_attention);

nw_error_t *linear_layer_create(layer_t **layer, int64_t in_features, int64_t out_features, runtime_t runtime, datatype_t datatype,
                                parameter_init_t *weight_init, parameter_init_t *bias_init);
nw_error_t *linear_layer_create_from_parameters(layer_t **layer, tensor_t *weights, tensor_t *bias);
nw_error_t *convolution_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride, int64_t in_channels, int64_t out_channels, 
                                        runtime_t runtime, datatype_t datatype, parameter_init_t *kernel_init, parameter_init_t *bias_init);
nw_error_t *convolution_2d_layer_create_from_parameters(layer_t **layer, int64_t padding, int64_t stride, tensor_t *kernel, tensor_t *bias);
nw_error_t *max_pooling_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride);
nw_error_t *average_pooling_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride);
nw_error_t *convolution_transpose_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride, int64_t in_channels, int64_t out_channels,
                                                  runtime_t runtime, datatype_t datatype, parameter_init_t *kernel_init, parameter_init_t *bias_init);
nw_error_t *convolution_transpose_2d_layer_create_from_parameters(layer_t **layer, int64_t padding, int64_t stride, tensor_t *kernel, tensor_t *bias);
nw_error_t *dropout_layer_create(layer_t **layer, void *probability, datatype_t datatype);
nw_error_t *batch_normalization_2d_layer_create(layer_t **layer, int64_t number_of_features,
                                                void *momentum, void *epsilon, bool_t track_running_stats,
                                                bool_t affine, datatype_t datatype, runtime_t runtime);
nw_error_t *layer_normalization_layer_create(layer_t **layer, const int64_t *normalized_shape, int64_t length,
                                        void *epsilon, bool_t weights, bool_t bias, datatype_t datatype, runtime_t runtime);
nw_error_t *reshape_layer_create(layer_t **layer, int64_t *shape, int64_t length);
nw_error_t *embedding_layer_create(layer_t **layer, int64_t vocabulary_size, int64_t embedding_size, datatype_t datatype, runtime_t runtime, parameter_init_t *embedding_init);
nw_error_t *embedding_layer_create_from_parameters(layer_t **layer, tensor_t *weights);
nw_error_t *transformer_embedding_layer_create(layer_t **layer, int64_t vocabulary_size, int64_t embedding_size, int64_t block_size, datatype_t datatype, runtime_t runtime,
                                               parameter_init_t *token_embedding_init, parameter_init_t *position_embedding_init);
nw_error_t *transformer_embedding_layer_create_from_parameters(layer_t **layer, tensor_t *token_weights, tensor_t *position_weights);
nw_error_t *causal_multihead_self_attention_layer_create(layer_t **layer, int64_t number_of_heads, int64_t embedding_size, void *dropout_probability, datatype_t datatype, runtime_t runtime,
                                                         parameter_init_t *input_weight_init, parameter_init_t *input_bias_init, parameter_init_t *output_weight_init, parameter_init_t *output_bias_init);
nw_error_t *causal_multihead_self_attention_layer_create_from_parameters(layer_t **layer, int64_t number_of_heads, int64_t embedding_size, void *dropout_probability, datatype_t datatype,
                                                                         tensor_t *input_weights, tensor_t *input_bias, tensor_t *output_weights, tensor_t *output_bias);
nw_error_t *residual_block_layer_create(layer_t **layer, block_t *block);
nw_error_t *block_layer_create(layer_t **layer, block_t *block);

nw_error_t *rectified_linear_activation_layer_create(layer_t **layer);
nw_error_t *sigmoid_activation_layer_create(layer_t **layer);
nw_error_t *softmax_activation_layer_create(layer_t **layer, int64_t axis);
nw_error_t *logsoftmax_activation_layer_create(layer_t **layer, int64_t axis);
nw_error_t *leaky_rectified_linear_activation_layer_create(layer_t **layer, void *c, datatype_t datatype);
nw_error_t *tanh_activation_layer_create(layer_t **layer);
nw_error_t *gelu_activation_layer_create(layer_t **layer);

// Model Forward
nw_error_t *model_forward(model_t *model, tensor_t *x, tensor_t **y);
nw_error_t *block_forward(block_t *block, tensor_t *x, tensor_t **y);
nw_error_t *residual_block_forward(block_t *block, tensor_t *x, tensor_t **y);
nw_error_t *linear_forward(linear_t *linear, tensor_t *x, tensor_t **y);
nw_error_t *convolution_2d_forward(convolution_2d_t *convolution_2d, tensor_t *x, tensor_t **y);
nw_error_t *convolution_transpose_2d_forward(convolution_2d_t *convolution_2d, tensor_t *x, tensor_t **y);
nw_error_t *max_pooling_2d_forward(pooling_2d_t *pooling_2d, tensor_t *x, tensor_t **y);
nw_error_t *average_pooling_2d_forward(pooling_2d_t *pooling_2d, tensor_t *x, tensor_t **y);
nw_error_t *dropout_forward(dropout_t *dropout, tensor_t *x, tensor_t **y);
nw_error_t *batch_normalization_2d_forward(batch_normalization_2d_t *batch_normalization_2d, tensor_t *x, tensor_t **y);
nw_error_t *layer_normalization_forward(layer_normalization_t *layer_normalization, tensor_t *x, tensor_t **y);
nw_error_t *reshape_forward(reshape_t *reshape, tensor_t *x, tensor_t **y);
nw_error_t *embedding_forward(embedding_t *embedding, tensor_t *x, tensor_t **y);
nw_error_t *transformer_embedding_forward(transformer_embedding_t *transformer_embedding, tensor_t *x, tensor_t **y);
nw_error_t *causal_multihead_self_attention_forward(causal_multihead_self_attention_t *causal_multihead_self_attention, tensor_t *x, tensor_t **y);

// Inference set
nw_error_t *model_inference(model_t *model, bool_t inference);
nw_error_t *block_inference(block_t *block, bool_t inference);

// Save Model
nw_error_t *model_save(model_t *model, string_t path);
nw_error_t *block_save(block_t *block, FILE *file);
nw_error_t *layer_save(layer_t *layer, FILE *file);
nw_error_t *linear_save(linear_t *linear, FILE *file);
nw_error_t *convolution_2d_save(convolution_2d_t *convolution_2d, FILE *file);
nw_error_t *pooling_2d_save(pooling_2d_t *pooling_2d, FILE *file);
nw_error_t *dropout_save(dropout_t *dropout, FILE *file);
nw_error_t *batch_normalization_2d_save(batch_normalization_2d_t *batch_normalization_2d, FILE *file);
nw_error_t *layer_normalization_save(layer_normalization_t *layer_normalization, FILE *file);
nw_error_t *reshape_save(reshape_t *reshape, FILE *file);
nw_error_t *embedding_save(embedding_t *embedding, FILE *file);
nw_error_t *transformer_embedding_save(transformer_embedding_t *transformer_embedding, FILE *file);
nw_error_t *causal_multihead_self_attention_save(causal_multihead_self_attention_t *causal_multihead_self_attention, FILE *file);
nw_error_t *model_load(model_t **model, string_t path);
nw_error_t *block_load(block_t **block, FILE *file);
nw_error_t *layer_load(layer_t **layer, FILE *file);
nw_error_t *linear_load(linear_t **linear, FILE *file);
nw_error_t *convolution_2d_load(convolution_2d_t **convolution_2d, FILE *file);
nw_error_t *pooling_2d_load(pooling_2d_t **pooling_2d, FILE *file);
nw_error_t *dropout_load(dropout_t **dropout, FILE *file);
nw_error_t *batch_normalization_2d_load(batch_normalization_2d_t **batch_normalization_2d, FILE *file);
nw_error_t *layer_normalization_load(layer_normalization_t **layer_normalization, FILE *file);
nw_error_t *reshape_load(reshape_t **reshape, FILE *file);
nw_error_t *embedding_load(embedding_t **embedding, FILE *file);
nw_error_t *transformer_embedding_load(transformer_embedding_t **transformer_embedding, FILE *file);
nw_error_t *causal_multihead_self_attention_load(causal_multihead_self_attention_t **causal_multihead_self_attention, FILE *file);

#endif
