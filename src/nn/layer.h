#ifndef LAYER_H
#define LAYER_H

#include <datatype.h>
#include <tensor.h>

typedef struct parameters_t;
typedef struct linear_t;
typedef struct dropout_t;
typedef enum padding_mode_t;
typedef struct convolution_t;
typedef struct convolution_transpose_t;
typedef struct batch_normalization_t;
typedef struct layer_normalization_t;
typedef struct maximum_pool_t;
typedef struct average_pool_t;
typedef struct padding_t;
typedef union instance_t;
typedef enum instance_type_t;
typedef struct unit_t;
typedef struct module_t;

typedef struct parameters_t
{
    tensor_t *weights;
    tensor_t *bias;
} parameters_t;

typedef struct linear_t
{
    uint32_t input_features;
    uint32_t output_features;
    bool_t bias;
    parameters_t *parameters;
} linear_t;

typedef struct dropout_t
{
    float32_t probability;
    uint32_t dimension;
} dropout_t;

typedef enum padding_mode_t
{
    Zeros,
    Reflect,
    Replicate,
    Circular
} padding_mode_t;

typedef struct convolution_t
{
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t *kernel_size;
    uint32_t *stride;
    uint32_t *padding;
    padding_mode_t padding_mode;
    uint32_t *dilation;
    uint32_t groups;
    bool_t bias;
    uint32_t dimension;
    parameters_t parameters;
} convolution_t;

typedef struct convolution_transpose_t
{
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t *kernel_size;
    uint32_t *stride;
    uint32_t *padding;
    uint32_t *output_padding;
    uint32_t *dilation;
    uint32_t groups;
    bool_t bias;
    uint32_t dimension;
    parameters_t parameters;
} convolution_transpose_t;

typedef struct batch_normalization_t
{
    uint32_t features;
    float32_t epsilon;
    float32_t momentum;
    bool_t affine;
    bool_t track_running_statistics;
    uint32_t dimension;
    parameters_t parameters;
} batch_normalization_t;

typedef struct layer_normalization_t
{
    uint32_t features;
    float32_t epsilon;
    float32_t momentum;
    bool_t affine;
    bool_t track_running_statistics;
    uint32_t dimension;
    parameters_t parameters;
} batch_normalization_t;

typedef struct maximum_pool_t
{
    uint32_t *kernel_size;
    uint32_t *stride;
    uint32_t *padding;
    uint32_t *dilation;
    bool_t ceil_mode;
    uint32_t dimension;
} maximum_pool_t;

typedef struct average_pool_t
{
    uint32_t *kernel_size;
    uint32_t *stride;
    uint32_t *padding;
    bool_t ceil_mode;
    uint32_t dimension;
} average_pool_t;

typedef struct padding_t
{
    uint32_t *padding;
    padding_mode_t padding_mode;
} padding_t;

typedef union layer_t
{
    linear_t *linear;
    dropout_t *dropout;
    convolution_t *convolution;
    convolution_transpose_t *convolution_transpose;
    batch_normalization_t *batch_normalization;
    maximum_pool_t *maximum_pool;
    average_pool_t *average_pool;
    padding_t *padding;
    struct module_t *module;
} layer_t;

typedef enum layer_type_t
{
    LINEAR,
    DROPOUT,
    CONVOLUTION,
    CONVOLUTION_TRANSPOSE,
    BATCH_NORMALIZATION,
    MAXIMUM_POOL,
    AVERAGE_POOL,
    PADDING,
    MODULE
} layer_type_t;

typedef struct unit_t
{
    layer_t *layer;
    layer_type_t layer_type;
} unit_t;

typedef struct module_t
{
    unit_t **units;
    uint32_t depth;
} module_t;

#endif