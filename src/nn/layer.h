#ifndef LAYER_H
#define LAYER_H

#include <datatype.h>
#include <errors.h>
#include <buffer.h>

// Forward declarations
typedef struct tensor_t tensor_t;
typedef struct block_t block_t;

// typedef union activation_function_t
// {
// } activation_function_t;

typedef enum activation_t
{
    ACTIVATION_RECTIFIED_LINEAR,
    ACTIVATION_SIGMOID,
    ACTIVATION_SOFTMAX,
} activation_t;

// typedef struct activation_t
// {
//     activation_function_t activation_function;
//     activation_function_type_t activation_function_type;
// } activation_t;

typedef struct linear_t
{
    tensor_t *weights;
    tensor_t *bias;
    activation_t activation;
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

nw_error_t *layer_create(layer_t **layer, transformation_t *transformation, transformation_type_t transformation_type);
void layer_destroy(layer_t *layer);
nw_error_t *transformation_create(transformation_t **transformation, transformation_type_t transformation_type, void *type_transformation);
void transformation_destroy(transformation_t *transformation, transformation_type_t transformation_type);
nw_error_t *linear_create(linear_t **linear, tensor_t *weights, tensor_t *bias, activation_t activation);
void linear_destroy(linear_t *linear);
nw_error_t *dropout_create(dropout_t **dropout, float32_t probability);
void dropout_destroy(dropout_t *dropout);
nw_error_t *block_create(block_t **block, layer_t **layers, uint64_t depth);
void block_destroy(block_t *block);



#endif
