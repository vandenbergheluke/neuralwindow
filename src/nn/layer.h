#ifndef LAYER_H
#define LAYER_H

#include <datatype.h>
#include <errors.h>
#include <buffer.h>

// Forward declarations
typedef struct tensor_t tensor_t;
typedef struct block_t block_t;

typedef struct softmax_t
{
    uint64_t *axis;
    uint64_t length;
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

typedef struct dropout_t
{
    float32_t probability;
} dropout_t;

typedef union transformation_t
{
    linear_t *linear;
    dropout_t *dropout;
    block_t *block;
} transformation_t;

typedef enum transformation_type_t
{
    LINEAR,
    DROPOUT,
    BLOCK
} transformation_type_t;

typedef struct layer_t
{
    transformation_t *transformation;
    transformation_type_t transformation_type;
} layer_t;

typedef struct block_t
{
    layer_t **layers;
    uint64_t depth;
} block_t;

typedef struct model_t
{
    runtime_t runtime;
    datatype_t datatype;
    block_t *block;
} model_t;

nw_error_t *model_forward(model_t *model, tensor_t *x, tensor_t **y);
nw_error_t *model_requires_gradient(model_t *model, bool_t requires_gradient);




#endif
