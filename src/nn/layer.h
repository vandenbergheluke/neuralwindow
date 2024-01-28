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
    int64_t kernel_size;
    int64_t padding;
    int64_t stride;
    int64_t in_channels;
    int64_t out_channels;
    tensor_t *kernel;
    tensor_t *bias;
} convolution_2d_t;

typedef struct dropout_t
{
    void *probability;
    datatype_t datatype;
    bool_t inference;
} dropout_t;

typedef union transform_t
{
    linear_t *linear;
    convolution_2d_t *convolution_2d;
    dropout_t *dropout;
    activation_t *activation;
    block_t *block;
} transform_t;

typedef enum transform_type_t
{
    LINEAR,
    CONVOLUTION_2D,
    CONVOLUTION_TRANSPOSE_2D,
    DROPOUT,
    ACTIVATION,
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
void block_destroy(block_t *block);

nw_error_t *layer_create(layer_t **layer, transform_t *transform, transform_type_t transform_type);
void layer_destroy(layer_t *layer);

nw_error_t *transform_create(transform_t **transform, transform_type_t transform_type, void *type_transform);
void transform_destroy(transform_t *transform, transform_type_t transform_type);
string_t transform_type_string(transform_type_t transform_type);

nw_error_t *linear_create(linear_t **linear, tensor_t *weights, tensor_t *bias);
void linear_destroy(linear_t *linear);

nw_error_t *convolution_2d_create(convolution_2d_t **convolution_2d, int64_t kernel_size, int64_t padding, int64_t stride,
                                  int64_t in_channels, int64_t out_channels, tensor_t *kernel, tensor_t *bias);
void convolution_2d_destroy(convolution_2d_t *convolution_2d);

nw_error_t *dropout_create(dropout_t **dropout, void *probability, datatype_t datatype);
void dropout_destroy(dropout_t *dropout);

nw_error_t *linear_layer_create(layer_t **layer, int64_t in_features, int64_t out_features, runtime_t runtime, datatype_t datatype,
                                parameter_init_t *weight_init, parameter_init_t *bias_init);
nw_error_t *linear_layer_create_from_parameters(layer_t **layer, tensor_t *weights, tensor_t *bias);
nw_error_t *convolution_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride, int64_t in_channels, int64_t out_channels, runtime_t runtime, 
                                        datatype_t datatype, parameter_init_t *kernel_init, parameter_init_t *bias_init);
nw_error_t *convolution_2d_transpose_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride, int64_t in_channels, int64_t out_channels,
                                                  runtime_t runtime, datatype_t datatype, parameter_init_t *kernel_init, parameter_init_t *bias_init);
nw_error_t *dropout_layer_create(layer_t **layer, void *probability, datatype_t datatype);
nw_error_t *rectified_linear_activation_layer_create(layer_t **layer);
nw_error_t *sigmoid_activation_layer_create(layer_t **layer);
nw_error_t *softmax_activation_layer_create(layer_t **layer, int64_t axis);
nw_error_t *logsoftmax_activation_layer_create(layer_t **layer, int64_t axis);
nw_error_t *leaky_rectified_linear_activation_layer_create(layer_t **layer, void *c, datatype_t datatype);

// Model Forward
nw_error_t *model_forward(model_t *model, tensor_t *x, tensor_t **y);
nw_error_t *block_forward(block_t *block, tensor_t *x, tensor_t **y);
nw_error_t *linear_forward(linear_t *linear, tensor_t *x, tensor_t **y);
nw_error_t *convolution_2d_forward(convolution_2d_t *convolution_2d, tensor_t *x, tensor_t **y);
nw_error_t *convolution_2d_transpose_forward(convolution_2d_t *convolution_2d, tensor_t *x, tensor_t **y);
nw_error_t *dropout_forward(dropout_t *dropout, tensor_t *x, tensor_t **y);

// Inference set
nw_error_t *model_inference(model_t *model, bool_t inference);
nw_error_t *block_inference(block_t *block, bool_t inference);

#endif
