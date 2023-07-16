#ifndef LAYER_H
#define LAYER_H

#include <datatype.h>
#include <tensor.h>

struct parameters_t;
struct linear_t;
struct dropout_t;
union layer_t;
enum layer_type_t;
struct unit_t;
struct module_t;

typedef struct parameters_t
{
    tensor_t *weights;
    tensor_t *mask;
    tensor_t *bias;
} parameters_t;

typedef struct linear_t
{
    uint32_t input_features;
    uint32_t output_features;
    parameters_t *parameters;
} linear_t;

typedef struct dropout_t
{
    float32_t probability;
} dropout_t;

typedef union layer_t
{
    linear_t *linear;
    dropout_t *dropout;
    struct module_t *module;
} layer_t;

typedef enum layer_type_t
{
    LINEAR,
    DROPOUT,
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