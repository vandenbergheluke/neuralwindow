#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <errors.h>
#include <datatype.h>
#include <layer.h>
#include <queue.h>
#include <map.h>

typedef struct model_t model_t;

typedef struct stochastic_gradient_descent_t
{
    datatype_t datatype;
    void *learning_rate;
    void *momentum;
    void *dampening;
    void *weight_decay;
    bool_t nesterov;
    map_t *momentum_buffer;
} stochastic_gradient_descent_t;

typedef struct rms_prop_t
{
    datatype_t datatype;
    void *learning_rate;
    void *momentum;
    void *alpha;
    void *weight_decay;
    void *epsilon; 
    bool_t centered;
    map_t *square_average;
    map_t *average_gradient;
    map_t *momentum_buffer;
} rms_prop_t;

typedef struct adam_t
{
    datatype_t datatype;
    void *learning_rate;
    void *beta_1;
    void *beta_2;
    void *weight_decay;
    void *epsilon; 
    map_t *first_moment;
    map_t *second_moment;
    map_t *iteration;
} adam_t;

typedef enum algorithm_type_t
{
    STOCASTIC_GRADIENT_DESCENT,
    RMS_PROP,
    ADAM,
} algorithm_type_t;

typedef union algorithm_t
{
    stochastic_gradient_descent_t *stochastic_gradient_descent;    
    rms_prop_t *rms_prop;
    adam_t *adam;
} algorithm_t;

typedef struct optimizer_t
{
    algorithm_t *algorithm;
    algorithm_type_t algorithm_type;
} optimizer_t;
 
// Optimizer
nw_error_t *optimizer_create(optimizer_t **optimizer, algorithm_t *algorithm, algorithm_type_t algorithm_type);
void optimizer_destroy(optimizer_t *optimizer);

 // Algorithm
nw_error_t *algorithm_create(algorithm_t **algorithm, algorithm_type_t algorithm_type, void *type_algorithm);
void algorithm_destroy(algorithm_t *algorithm, algorithm_type_t algorithm_type);
string_t algorithm_type_string(algorithm_type_t algorithm_type);

// Optimizer Specializations
nw_error_t *optimizer_stochastic_gradient_descent_create(optimizer_t **optimizer, datatype_t datatype, void *learning_rate,
                                                         void *momentum, void *dampening, void *weight_decay, bool_t nesterov);
nw_error_t *optimizer_rms_prop_create(optimizer_t **optimizer, datatype_t datatype, void *learning_rate, void *momentum,
                                      void *alpha, void *weight_decay, void *epsilon, bool_t centered);
nw_error_t *optimizer_adam_create(optimizer_t **optimizer, datatype_t datatype, void *learning_rate,
                                  void *beta_1, void *beta_2, void *weight_decay, void *epsilon);

// SGD
nw_error_t *stochastic_gradient_descent_create(stochastic_gradient_descent_t **stochastic_gradient_descent, datatype_t datatype,
                                               void *learning_rate, void *momentum, void *dampening, void *weight_decay, bool_t nesterov);
void stochastic_gradient_descent_destroy(stochastic_gradient_descent_t *stochastic_gradient_descent);

// RMSProp
nw_error_t *rms_prop_create(rms_prop_t **rms_prop, datatype_t datatype, void *learning_rate, void *momentum,
                            void *alpha, void *weight_decay, void *epsilon, bool_t centered);
void rms_prop_destroy(rms_prop_t *rms_prop);

// ADAM
nw_error_t *adam_create(adam_t **adam, datatype_t datatype, void *learning_rate, void *beta_1, void *beta_2, void *weight_decay, void *epsilon);
void adam_destroy(adam_t *adam);

// Update
nw_error_t *update_model(optimizer_t *optimizer, model_t *model);
nw_error_t *update_block(optimizer_t *optimizer, block_t *block);
nw_error_t *update_linear(optimizer_t *optimizer, linear_t *linear);
nw_error_t *update_convolution_2d(optimizer_t *optimizer, convolution_2d_t *convolution_2d);
nw_error_t *update_batch_normalization_2d(optimizer_t *optimizer, batch_normalization_2d_t *batch_normalization_2d);
nw_error_t *update_layer_normalization(optimizer_t *optimizer, layer_normalization_t *layer_normalization);
nw_error_t *update_embedding(optimizer_t *optimizer, embedding_t *embedding);
nw_error_t *update_transformer_embedding(optimizer_t *optimizer, transformer_embedding_t *transformer_embedding);
nw_error_t *update_causal_multihead_self_attention(optimizer_t *optimizer, causal_multihead_self_attention_t *causal_multihead_self_attention);
nw_error_t *update_parameters(optimizer_t *optimizer, tensor_t *parameters);

// Update Specializations
nw_error_t *stochastic_gradient_descent(stochastic_gradient_descent_t *optimizer, tensor_t *parameters);
nw_error_t *rms_prop(rms_prop_t *optimizer, tensor_t *parameters);
nw_error_t *adam(adam_t *optimizer, tensor_t *parameters);

// Clip Gradient
nw_error_t *clip_gradient_norm_model(model_t *model, void *threshold);
nw_error_t *clip_gradient_norm_block(block_t *block, void *threshold);
nw_error_t *clip_gradient_norm_linear(linear_t *linear, void *threshold);
nw_error_t *clip_gradient_norm_convolution_2d(convolution_2d_t *convolution_2d, void *threshold);
nw_error_t *clip_gradient_norm_batch_normalization_2d(batch_normalization_2d_t *batch_normalization_2d, void *threshold);
nw_error_t *clip_gradient_norm_layer_normalization(layer_normalization_t *layer_normalization, void *threshold);
nw_error_t *clip_gradient_norm_embedding(embedding_t *embedding, void *threshold);
nw_error_t *clip_gradient_norm_transformer_embedding(transformer_embedding_t *transformer_embedding, void *threshold);
nw_error_t *clip_gradient_norm_causal_multihead_self_attention(causal_multihead_self_attention_t *causal_multihead_self_attention, void *threshold);
nw_error_t *clip_gradient_norm_parameters(tensor_t *parameters, void *threshold);

// Zero Gradient
nw_error_t *zero_gradient_model(model_t *model);
nw_error_t *zero_gradient_block(block_t *block);
nw_error_t *zero_gradient_linear(linear_t *linear);
nw_error_t *zero_gradient_convolution_2d(convolution_2d_t *convolution_2d);
nw_error_t *zero_gradient_batch_normalization_2d(batch_normalization_2d_t *batch_normalization_2d);
nw_error_t *zero_gradient_layer_normalization(layer_normalization_t *layer_normalization);
nw_error_t *zero_gradient_embedding(embedding_t *embedding);
nw_error_t *zero_gradient_transformer_embedding(transformer_embedding_t *transformer_embedding);
nw_error_t *zero_gradient_causal_multihead_self_attention(causal_multihead_self_attention_t *causal_multihead_self_attention);
void zero_gradient_parameters(tensor_t *parameters);
#endif