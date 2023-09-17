/**@file layer.h
 * @brief
 *
 */

#ifndef LAYER_H
#define LAYER_H

#include <datatype.h>
#include <errors.h>

// Forward declarations
typedef struct tensor_t tensor_t;

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
    uint64_t input_features;
    uint64_t output_features;
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
    uint64_t depth;
} module_t;

nw_error_t *parameters_create(parameters_t **parameters, tensor_t *weights, tensor_t *mask, tensor_t *bias);
void parameters_destroy(parameters_t *parameters);
nw_error_t *linear_create(linear_t **linear, uint64_t input_features, uint64_t output_features, parameters_t *parameters);
void linear_destroy(linear_t *linear);
nw_error_t *dropout_create(dropout_t **dropout, float32_t probability);
void dropout_destroy(dropout_t *dropout);
nw_error_t *layer_create(layer_t **layer, layer_type_t layer_type, void *type_layer);
void layer_destroy(layer_t *layer, layer_type_t layer_type);
nw_error_t *unit_create(unit_t **unit, layer_type_t layer_type, layer_t *layer);
void layer_destroy(layer_t *layer, layer_type_t layer_type);
nw_error_t *module_create(module_t **module, unit_t **units, uint64_t depth);
void module_destroy(module_t *module);

#endif
