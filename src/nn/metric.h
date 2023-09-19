#ifndef METRIC_H
#define METRIC_H

#include <errors.h>

typedef struct tensor_t tensor_t;

nw_error_t *binary_accuracy(const tensor_t *y_pred, const tensor_t *y_true, const tensor_t *threshold, tensor_t **accuracy);
nw_error_t *multiclass_accuracy(const tensor_t *y_pred, const tensor_t *y_true, tensor_t **accuracy);

#endif