#ifndef LAYER_H
#define LAYER_H

#include <datatype.h>
#include <errors.h>
#include <buffer.h>

// Forward declarations
typedef struct tensor_t tensor_t;
typedef struct block_t block_t;
typedef struct parameter_init_t parameter_init_t;

typedef struct softmax_t
{
    int64_t axis;
} softmax_t;

typedef union activation_function_t
{
    softmax_t *softmax;
} activation_function_t;

typedef enum activation_function_type_t
{
    ACTIVATION_RECTIFIED_LINEAR,
    ACTIVATION_SIGMOID,
    ACTIVATION_SOFTMAX,
    ACTIVATION_LOGSOFTMAX,
} activation_function_type_t;

typedef struct activation_t
{
    activation_function_t *activation_function;
    activation_function_type_t activation_function_type;
} activation_t;

typedef struct linear_t
{
    tensor_t *weights;
    tensor_t *bias;
    activation_t *activation;
} linear_t;

typedef struct convolution_t
{
    int64_t kernel_size;
    int64_t padding;
    int64_t stride;
    int64_t in_channels;
    int64_t out_channels;
    tensor_t *kernel;
    tensor_t *bias;
    activation_t *activation;
} convolution_t;

typedef struct dropout_t
{
    float32_t probability;
} dropout_t;

typedef union transform_t
{
    linear_t *linear;
    convolution_t *convolution;
    dropout_t *dropout;
    block_t *block;
} transform_t;

typedef enum transform_type_t
{
    LINEAR,
    CONVOLUTION,
    DROPOUT,
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

nw_error_t *model_forward(model_t *model, tensor_t *x, tensor_t **y);
nw_error_t *model_requires_gradient(model_t *model, bool_t requires_gradient);
nw_error_t *layer_create(layer_t **layer, transform_t *transform, transform_type_t transform_type);
void layer_destroy(layer_t *layer);
nw_error_t *transform_create(transform_t **transform, transform_type_t transform_type, void *type_transform);
void transform_destroy(transform_t *transform, transform_type_t transform_type);
string_t transform_type_string(transform_type_t transform_type);
string_t activation_function_type_string(activation_function_type_t activation_function_type);
nw_error_t *linear_create(linear_t **linear, tensor_t *weights, tensor_t *bias, activation_t *activation);
void linear_destroy(linear_t *linear);
nw_error_t *dropout_create(dropout_t **dropout, float32_t probability);
void dropout_destroy(dropout_t *dropout);
nw_error_t *convolution_create(convolution_t **convolution, int64_t kernel_size, int64_t padding, int64_t stride,
                               int64_t in_channels, int64_t out_channels, tensor_t *kernel, tensor_t *bias, activation_t *activation);
void convolution_destroy(convolution_t *convolution);
nw_error_t *block_create(block_t **block, int64_t depth, ...);
void block_destroy(block_t *block);
nw_error_t *softmax_create(softmax_t **softmax, int64_t axis);
void softmax_destroy(softmax_t *softmax);
nw_error_t *activation_function_create(activation_function_t **activation_function,
                                       activation_function_type_t activation_function_type,
                                       void *type_activation_function);
void activation_function_destroy(activation_function_t *activation_function, activation_function_type_t activation_function_type);
nw_error_t *activation_create(activation_t **activation,
                              activation_function_t *activation_function,
                              activation_function_type_t activation_function_type);
void activation_destroy(activation_t *activation);
nw_error_t *model_create(model_t **model, block_t *block);
void model_destroy(model_t *model);
nw_error_t *rectified_linear_activation_create(activation_t **activation);
nw_error_t *sigmoid_activation_create(activation_t **activation);
nw_error_t *softmax_activation_create(activation_t **activation, int64_t axis);
nw_error_t *logsoftmax_activation_create(activation_t **activation, int64_t axis);
nw_error_t *linear_layer_create(layer_t **layer, 
                                int64_t in_features,
                                int64_t out_features,
                                runtime_t runtime,
                                datatype_t datatype,
                                bool_t requires_gradient,
                                activation_t *activation,
                                parameter_init_t *weight_init,
                                parameter_init_t *bias_init);
nw_error_t *convolution_layer_create(layer_t **layer,
                                     int64_t kernel_size, int64_t padding, int64_t stride,
                                     int64_t in_channels, int64_t out_channels,
                                     runtime_t runtime, datatype_t datatype,
                                     bool_t requires_gradient, 
                                     activation_t *activation,
                                     parameter_init_t *kernel_init,
                                     parameter_init_t *bias_init);

#endif
