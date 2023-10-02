#include <tensor.h>
#include <buffer.h>
#include <function.h>
#include <view.h>
#include <cost.h>

extern bool_t no_gradient;

nw_error_t *categorical_cross_entropy(const tensor_t *y_true, const tensor_t *y_prediction, tensor_t **cost)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("y_true", y_true);
    PRINTLN_DEBUG_TENSOR("y_prediction", y_prediction);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(y_true, "y_true");
    CHECK_NULL_ARGUMENT(y_true->buffer, "y_true->buffer");
    CHECK_NULL_ARGUMENT(y_true->buffer->view, "y_true->buffer->view");
    CHECK_NULL_ARGUMENT(y_prediction, "y_prediction");
    CHECK_NULL_ARGUMENT(y_prediction->buffer, "y_prediction->buffer");
    CHECK_NULL_ARGUMENT(y_prediction->buffer->view, "y_prediction->buffer->view");
    CHECK_NULL_ARGUMENT(cost, "cost");

    if (!tensor_shapes_equal(y_prediction, y_true))    
    {
        return ERROR(ERROR_SHAPE_CONFLICT, string_create("tensor shapes not equal."), NULL);
    }

    nw_error_t *error = NULL;
    int64_t rank = y_true->buffer->view->rank;
    tensor_t *cost_i = NULL;
    tensor_t *cost_j = NULL;
    tensor_t *cost_k = NULL;
    tensor_t *cost_l = NULL;

    error = tensor_logarithm(y_prediction, &cost_i);
    if (error)
    {
        error = ERROR(ERROR_LOGARITHM, string_create("failed to log tensor."), error);
        goto cleanup;
    }

    error = tensor_multiplication(cost_i, y_true, &cost_j);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_summation(cost_j, &cost_k, (int64_t[]) {rank - 1}, 1, false);
    if (error)
    {
        error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        goto cleanup;
    }

    error = tensor_negation(cost_k, &cost_l);
    if (error)
    {
        error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
        goto cleanup;
    }

    error = tensor_mean(cost_l, cost, NULL, 0, false);
    if (error)
    {
        error = ERROR(ERROR_MEAN, string_create("failed to get mean of tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("cost", *cost);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!y_prediction->requires_gradient || no_gradient)
    {
        tensor_destroy(cost_i);
        tensor_destroy(cost_j);
        tensor_destroy(cost_k);
        tensor_destroy(cost_l);
    }

    return error;
}

nw_error_t *negative_log_likelihood(const tensor_t *y_true, const tensor_t *y_prediction, tensor_t **cost)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("y_true", y_true);
    PRINTLN_DEBUG_TENSOR("y_prediction", y_prediction);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(y_true, "y_true");
    CHECK_NULL_ARGUMENT(y_true->buffer, "y_true->buffer");
    CHECK_NULL_ARGUMENT(y_true->buffer->view, "y_true->buffer->view");
    CHECK_NULL_ARGUMENT(y_prediction, "y_prediction");
    CHECK_NULL_ARGUMENT(y_prediction->buffer, "y_prediction->buffer");
    CHECK_NULL_ARGUMENT(y_prediction->buffer->view, "y_prediction->buffer->view");
    CHECK_NULL_ARGUMENT(cost, "cost");

    if (!tensor_shapes_equal(y_prediction, y_true))    
    {
        return ERROR(ERROR_SHAPE_CONFLICT, string_create("tensor shapes not equal."), NULL);
    }

    nw_error_t *error = NULL;
    int64_t rank = y_true->buffer->view->rank;
    tensor_t *cost_j = NULL;
    tensor_t *cost_k = NULL;
    tensor_t *cost_l = NULL;

    error = tensor_multiplication(y_prediction, y_true, &cost_j);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_summation(cost_j, &cost_k, (int64_t[]) {rank - 1}, 1, false);
    if (error)
    {
        error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        goto cleanup;
    }

    error = tensor_negation(cost_k, &cost_l);
    if (error)
    {
        error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
        goto cleanup;
    }

    error = tensor_mean(cost_l, cost, NULL, 0, false);
    if (error)
    {
        error = ERROR(ERROR_MEAN, string_create("failed to get mean of tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("cost", *cost);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!y_prediction->requires_gradient || no_gradient)
    {
        tensor_destroy(cost_j);
        tensor_destroy(cost_k);
        tensor_destroy(cost_l);
    }

    return error;
}

