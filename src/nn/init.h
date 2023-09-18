#ifndef INIT_H
#define INIT_H

#include <errors.h>

typedef struct uniform_initialization_t
{
    void *lower_bound;
    void *upper_bound;
} uniform_initialization_t;

typedef struct normal_initialization_t
{
    void *mean;
    void *standard_deviation;
} normal_initialization_t;

typedef struct kaiming_initialization_t
{
    void *fan;
    void *gain;
} kaiming_initialization_t;

typedef struct glorot_initialization_t
{
    void *gain;
    void *fan_in;
    void *fan_out;
} glorot_initialization_t;

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
} initialization_type_t;

typedef union initialization_t
{
    uniform_initialization_t *uniform_initialization;
    normal_initialization_t *normal_initialization;
    kaiming_initialization_t *kaiming_initialization;
    glorot_initialization_t *glorot_initialization;
} initialization_t;

typedef struct parameter_initialization_t
{
    initialization_type_t initialization_type;
    initialization_t *initialization;
} parameter_initialization_t;

nw_error_t *parameter_initialization_create(parameter_initialization_t **parameter_initialization, initialization_t *initialization, initialization_type_t initialization_type);
void parameter_initialization_destroy(parameter_initialization_t *parameter_initialization);
nw_error_t *initialization_create(initialization_t **initialization, initialization_type_t initialization_type, void *type_initialization);
void initialization_destroy(initialization_t *initialization, initialization_type_t initialization_type);
nw_error_t *uniform_initialization_create(uniform_initialization_t **uniform_initialization, void *lower_bound, void *upper_bound);
void uniform_initialization_destroy(uniform_initialization_t *uniform_initialization);
nw_error_t *normal_initialization_create(normal_initialization_t **normal_initialization, void *mean, void *standard_deviation);
void normal_initialization_destroy(normal_initialization_t *normal_initialization);
nw_error_t *kaiming_initialization_create(kaiming_initialization_t **kaiming_initialization, void *fan, void *gain);
void kaiming_initialization_destroy(kaiming_initialization_t *kaiming_initialization);
nw_error_t *glorot_initialization_create(glorot_initialization_t **glorot_initialization, void *fan_in, void *fan_out, void *gain);
void glorot_initialization_destroy(glorot_initialization_t *glorot_initialization);




#endif