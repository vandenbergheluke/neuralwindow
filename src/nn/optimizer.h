#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <errors.h>


typedef struct stochastic_gradient_descent_t
{
    float32_t learning_rate;
    float32_t momentum;
    float32_t dampening;
    float32_t weight_decay;
    bool_t nesterov;
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
                                                         float32_t learning_rate,
                                                         float32_t momentum,
                                                         float32_t dampening,
                                                         float32_t weight_decay,
                                                         bool_t nesterov);

#endif