#ifndef COST_H
#define COST_H

#include <errors.h>

// Forward declarations
typedef struct tensor_t tensor_t;

nw_error_t *categorical_cross_entropy(const tensor_t *y_true, const tensor_t *y_prediction, tensor_t **cost);

#endif