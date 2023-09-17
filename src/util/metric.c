#include <tensor.h>
#include <metric.h>

nw_error_t *binary_accuracy(const tensor_t *y_pred, const tensor_t *y_true, const tensor_t *threshold, tensor_t **accuracy)
{
    nw_error_t *error = NULL;
    tensor_t *y_i = NULL;
    tensor_t *y_j = NULL;

    error = tensor_compare_greater(y_pred, threshold, &y_i);

    error = tensor_compare_equal(y_i, y_true, &y_j);

    error = tensor_mean(y_j, accuracy, NULL, 0, false);

    return error;
}

nw_error_t *multiclass_accuracy(const tensor_t *y_pred, const tensor_t *y_true, const tensor_t *threshold, tensor_t **accuracy)
{
    
}
