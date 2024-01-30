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
        return ERROR(ERROR_SHAPE, string_create("tensor shapes not equal."), NULL);
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
        return ERROR(ERROR_SHAPE, string_create("tensor shapes not equal."), NULL);
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

nw_error_t *binary_cross_entropy(const tensor_t *y_true, const tensor_t *y_prediction, tensor_t **cost)
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

    nw_error_t *error = NULL;
    datatype_t datatype = y_prediction->buffer->storage->datatype;
    runtime_t runtime = y_prediction->buffer->storage->runtime;
    tensor_t *one_constant = NULL;
    tensor_t *cost_i = NULL;
    tensor_t *cost_j = NULL;
    tensor_t *cost_k = NULL;
    tensor_t *cost_l = NULL;
    tensor_t *cost_m = NULL;
    tensor_t *cost_n = NULL;
    tensor_t *cost_o = NULL;
    tensor_t *cost_p = NULL;

    error = tensor_create_ones(&one_constant, (int64_t[]){}, 0, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_logarithm(y_prediction, &cost_i);
    if (error)
    {
        error = ERROR(ERROR_LOGARITHM, string_create("failed to log tensor."), error);
        goto cleanup;
    }

    error = tensor_multiplication(y_true, cost_i, &cost_j);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_negation(cost_j, &cost_k);
    if (error)
    {
        error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
        goto cleanup;
    }

    error = tensor_subtraction(one_constant, y_true, &cost_l);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    error = tensor_subtraction(one_constant, y_prediction, &cost_m);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    error = tensor_logarithm(cost_m, &cost_n);
    if (error)
    {
        error = ERROR(ERROR_LOGARITHM, string_create("failed to log tensor."), error);
        goto cleanup;
    }

    error = tensor_multiplication(cost_l, cost_n, &cost_o);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_subtraction(cost_k, cost_o, &cost_p);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    error = tensor_mean(cost_p, cost, NULL, 0, false);
    if (error)
    {
        error = ERROR(ERROR_MEAN, string_create("failed to average tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("cost", *cost);
    PRINT_DEBUG_NEWLINE;

cleanup:

    tensor_destroy(one_constant);
    if (!y_prediction->requires_gradient || no_gradient)
    {
        tensor_destroy(cost_i);
        tensor_destroy(cost_j);
        tensor_destroy(cost_k);
        tensor_destroy(cost_l);
        tensor_destroy(cost_m);
        tensor_destroy(cost_n);
        tensor_destroy(cost_o);
        tensor_destroy(cost_p);
    }

    return error;
}

nw_error_t *binary_cross_entropy_logits(const tensor_t *y_true, const tensor_t *y_prediction, tensor_t **cost)
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

    nw_error_t *error = NULL;
    datatype_t datatype = y_prediction->buffer->storage->datatype;
    runtime_t runtime = y_prediction->buffer->storage->runtime;
    tensor_t *zero_constant = NULL;
    tensor_t *one_constant = NULL;
    tensor_t *cost_i = NULL;
    tensor_t *cost_j = NULL;
    tensor_t *cost_k = NULL;
    tensor_t *cost_l = NULL;
    tensor_t *cost_m = NULL;
    tensor_t *cost_n = NULL;
    tensor_t *cost_o = NULL;
    tensor_t *cost_p = NULL;
    tensor_t *cost_q = NULL;
    
    error = tensor_create_zeroes(&zero_constant, (int64_t[]){}, 0, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_create_ones(&one_constant, (int64_t[]){}, 0, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_max(y_prediction, zero_constant, &cost_i);
    if (error)
    {
        error = ERROR(ERROR_MAX, string_create("failed to max tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(y_true, y_prediction, &cost_j);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_subtraction(cost_i, cost_j, &cost_k);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensor."), error);
        goto cleanup;
    }

    error = tensor_absolute(y_prediction, &cost_l);
    if (error)
    {
        error = ERROR(ERROR_ABSOLUTE, string_create("failed to get absolute of tensor."), error);
        goto cleanup;
    }

    error = tensor_negation(cost_l, &cost_m);
    if (error)
    {
        error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
        goto cleanup;
    }

    error = tensor_exponential(cost_m, &cost_n);
    if (error)
    {
        error = ERROR(ERROR_EXPONENTIAL, string_create("failed to exponentiate tensor."), error);
        goto cleanup;
    }

    error = tensor_addition(one_constant, cost_n, &cost_o);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_logarithm(cost_o, &cost_p);
    if (error)
    {
        error = ERROR(ERROR_LOGARITHM, string_create("failed to log tensor."), error);
        goto cleanup;
    }

    error = tensor_addition(cost_k, cost_p, &cost_q);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }


    error = tensor_mean(cost_q, cost, NULL, 0, false);
    if (error)
    {
        error = ERROR(ERROR_MEAN, string_create("failed to average tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("cost", *cost);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!y_prediction->requires_gradient || no_gradient)
    {
        tensor_destroy(zero_constant);
        tensor_destroy(one_constant);
        tensor_destroy(cost_i);
        tensor_destroy(cost_j);
        tensor_destroy(cost_k);
        tensor_destroy(cost_l);
        tensor_destroy(cost_m);
        tensor_destroy(cost_n);
        tensor_destroy(cost_o);
        tensor_destroy(cost_p);
        tensor_destroy(cost_q);
    }

    return error;

}