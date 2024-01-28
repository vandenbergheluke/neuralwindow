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
nw_error_t *optimizer_step(optimizer_t *optimizer, model_t *model);
nw_error_t *update(algorithm_t *algorithm, algorithm_type_t algorithm_type, block_t *block);

// Update Specializations
nw_error_t *stochastic_gradient_descent(stochastic_gradient_descent_t *optimizer, tensor_t *parameters);
nw_error_t *rms_prop(rms_prop_t *optimizer, tensor_t *parameters);
nw_error_t *adam(adam_t *optimizer, tensor_t *parameters);

#endif