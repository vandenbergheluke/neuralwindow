#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <errors.h>


typedef struct stochastic_gradient_descent_t
{
    
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
 


#endif