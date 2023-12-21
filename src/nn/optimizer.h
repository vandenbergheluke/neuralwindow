#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <errors.h>
#include <datatype.h>
#include <layer.h>
#include <queue.h>

typedef struct model_t model_t;

typedef struct stochastic_gradient_descent_t
{
    datatype_t datatype;
    void *learning_rate;
    void *momentum;
    void *dampening;
    void *weight_decay;
    bool_t nesterov;
    uint64_t momentum_buffer_size;
    tensor_t **momentum_buffer;
    queue_t *momentum_queue;
} stochastic_gradient_descent_t;

typedef enum algorithm_type_t
{
    STOCASTIC_GRADIENT_DESCENT,
} algorithm_type_t;

typedef union algorithm_t
{
    stochastic_gradient_descent_t *stochastic_gradient_descent;    
} algorithm_t;

typedef struct optimizer_t
{
    algorithm_t *algorithm;
    algorithm_type_t algorithm_type;
} optimizer_t;
 

nw_error_t *optimizer_stochastic_gradient_descent_create(optimizer_t **optimizer,
                                                         block_t *params,
                                                         datatype_t datatype,
                                                         void *learning_rate,
                                                         void *momentum,
                                                         void *dampening,
                                                         void *weight_decay,
                                                         bool_t nesterov);
nw_error_t *optimizer_step(optimizer_t *optimizer, model_t *model);
nw_error_t *optimizer_create(optimizer_t **optimizer, algorithm_t *algorithm, algorithm_type_t algorithm_type);
string_t algorithm_type_string(algorithm_type_t algorithm_type);
void optimizer_destroy(optimizer_t *optimizer);
nw_error_t *stochastic_gradient_descent_create(stochastic_gradient_descent_t **stochastic_gradient_descent,
                                               block_t *params,
                                               datatype_t datatype,
                                               void *learning_rate,
                                               void *momentum,
                                               void *dampening,
                                               void *weight_decay,
                                               bool_t nesterov);
void stochastic_gradient_descent_destroy(stochastic_gradient_descent_t *stochastic_gradient_descent);
nw_error_t *algorithm_create(algorithm_t **algorithm, algorithm_type_t algorithm_type, void *type_algorithm);
void algorithm_destroy(algorithm_t *algorithm, algorithm_type_t algorithm_type);
nw_error_t *update(algorithm_t *algorithm, algorithm_type_t algorithm_type, block_t *block);
nw_error_t *stochastic_gradient_descent(stochastic_gradient_descent_t *optimizer, tensor_t *parameters, uint64_t index);
nw_error_t *initialize_momentum_buffer(block_t *param, tensor_t **buffer, uint64_t index);
nw_error_t *update_helper(algorithm_t *algorithm, algorithm_type_t algorithm_type, block_t *block, uint64_t index);
#endif