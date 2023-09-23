#ifndef INIT_H
#define INIT_H

#include <errors.h>

typedef struct tensor_t tensor_t;
typedef enum runtime_t runtime_t;
typedef enum activation_function_type_t activation_function_type_t;

typedef struct uniform_init_t
{
    void *lower_bound;
    void *upper_bound;
} uniform_init_t;

typedef struct normal_init_t
{
    void *mean;
    void *standard_deviation;
} normal_init_t;

typedef struct kaiming_init_t
{
    void *fan;
    void *gain;
} kaiming_init_t;

typedef struct glorot_init_t
{
    void *gain;
    void *fan_in;
    void *fan_out;
} glorot_init_t;

typedef enum initializaton_type_t
{
    ZEROES,
    ONES,
    UNIFORM,
    NORMAL,
    KAIMING_UNIFORM,
    KAIMING_NORMAL,
    GLOROT_UNIFORM,
    GLOROT_NORMAL,
} init_type_t;

typedef union init_t
{
    uniform_init_t *uniform_init;
    normal_init_t *normal_init;
    kaiming_init_t *kaiming_init;
    glorot_init_t *glorot_init;
} init_t;

typedef struct parameter_init_t
{
    init_type_t init_type;
    init_t *init;
} parameter_init_t;

nw_error_t *parameter_init_create(parameter_init_t **parameter_init, init_t *init, init_type_t init_type);
void parameter_init_destroy(parameter_init_t *parameter_init);
nw_error_t *init_create(init_t **init, init_type_t init_type, void *type_init);
void init_destroy(init_t *init, init_type_t init_type);
nw_error_t *uniform_init_create(uniform_init_t **uniform_init, void *lower_bound, void *upper_bound);
void uniform_init_destroy(uniform_init_t *uniform_init);
nw_error_t *normal_init_create(normal_init_t **normal_init, void *mean, void *standard_deviation);
void normal_init_destroy(normal_init_t *normal_init);
nw_error_t *kaiming_init_create(kaiming_init_t **kaiming_init, void *fan, void *gain);
void kaiming_init_destroy(kaiming_init_t *kaiming_init);
nw_error_t *glorot_init_create(glorot_init_t **glorot_init, void *fan_in, void *fan_out, void *gain);
void glorot_init_destroy(glorot_init_t *glorot_init);
nw_error_t *zeroes_parameter_init(parameter_init_t **parameter_init);
nw_error_t *ones_parameter_init(parameter_init_t **parameter_init);
nw_error_t *uniform_parameter_init(parameter_init_t **parameter_init, void *lower_bound, void *upper_bound);
nw_error_t *normal_parameter_init(parameter_init_t **parameter_init, void *mean, void *standard_deviation);
nw_error_t *kaiming_uniform_parameter_init(parameter_init_t **parameter_init, void *fan, void *gain);
nw_error_t *kaiming_normal_parameter_init(parameter_init_t **parameter_init, void *fan, void *gain);
nw_error_t *glorot_uniform_parameter_init(parameter_init_t **parameter_init, void *fan_in, void *fan_out, void *gain);
nw_error_t *glorot_normal_parameter_init(parameter_init_t **parameter_init, void *fan_in, void *fan_out, void *gain);
nw_error_t *initialize(tensor_t **parameters,
                       parameter_init_t *parameter_init,
                       const uint64_t *shape,
                       uint64_t rank,
                       runtime_t runtime,
                       datatype_t datatype,
                       bool_t requires_gradient);
nw_error_t *calculate_gain(activation_function_type_t activation_function_type, datatype_t datatype, void *gain);

#endif