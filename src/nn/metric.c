#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <metric.h>

extern bool_t no_gradient;

nw_error_t *binary_accuracy(const tensor_t *y_pred, const tensor_t *y_true, const tensor_t *threshold, tensor_t **accuracy)
{
    CHECK_NULL_ARGUMENT(y_pred, "y_pred");
    CHECK_NULL_ARGUMENT(y_true, "y_true");
    CHECK_NULL_ARGUMENT(threshold, "threshold");
    CHECK_NULL_ARGUMENT(accuracy, "accuracy");

    nw_error_t *error = NULL;
    tensor_t *y_i = NULL;
    tensor_t *y_j = NULL;

    error = tensor_compare_greater(y_pred, threshold, &y_i);
    if (error)
    {
        error = ERROR(ERROR_COMPARE_GREATER, string_create("failed to compare greater than tensors."), error);
        goto cleanup;
    }

    error = tensor_compare_equal(y_i, y_true, &y_j);
    if (error)
    {
        error = ERROR(ERROR_COMPARE_EQUAL, string_create("failed to compare equal tensors."), error);
        goto cleanup;
    }

    error = tensor_mean(y_j, accuracy, NULL, 0, false);
    if (error)
    {
        error = ERROR(ERROR_MEAN, string_create("failed to get mean of tensor."), error);
        goto cleanup;
    }

cleanup:

    if (!y_pred->requires_gradient || no_gradient)
    {
        tensor_destroy(y_i);
        tensor_destroy(y_j);
    }

    return error;
}

nw_error_t *multiclass_accuracy(const tensor_t *y_pred, const tensor_t *y_true, tensor_t **accuracy)
{
    CHECK_NULL_ARGUMENT(y_pred, "y_pred");
    CHECK_NULL_ARGUMENT(y_true, "y_true");
    CHECK_NULL_ARGUMENT(accuracy, "accuracy");

    nw_error_t *error = NULL;
    tensor_t *y_i = NULL;
    tensor_t *y_j = NULL;
    tensor_t *y_k = NULL;
    uint64_t rank = y_pred->buffer->view->rank;

    error = tensor_argument_maximum(y_pred, &y_i, rank - 1, false);
    if (error)
    {
        error = ERROR(ERROR_MAXIMUM, string_create("failed to get maximum of tensor."), error);
        goto cleanup;
    }

    error = tensor_argument_maximum(y_true, &y_k, rank - 1, false);
    if (error)
    {
        error = ERROR(ERROR_MAXIMUM, string_create("failed to get maximum of tensor."), error);
        goto cleanup;
    }

    error = tensor_compare_equal(y_i, y_k, &y_j);
    if (error)
    {
        error = ERROR(ERROR_COMPARE_EQUAL, string_create("failed to compare equal tensors."), error);
        goto cleanup;
    }

    error = tensor_mean(y_j, accuracy, NULL, 0, false);
    if (error)
    {
        error = ERROR(ERROR_MEAN, string_create("failed to get mean of tensor."), error);
        goto cleanup;
    }

cleanup:

    if (!y_pred->requires_gradient || no_gradient)
    {
        tensor_destroy(y_i);
        tensor_destroy(y_j);
        tensor_destroy(y_k);
    }

    return error;
}
