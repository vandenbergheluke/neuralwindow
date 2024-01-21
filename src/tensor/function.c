/**
 * @file function.c
 * @brief Mid-level Operations and Automatic Differentiation Engine
 */

#include <function.h>
#include <tensor.h>
#include <view.h>
#include <buffer.h>
#include <string.h>
#include <sort.h>

extern bool_t no_gradient;

/**
 * @brief Execute exponential operation forward.
 * @param x The input operand.
 * @param result The output of the exponential operation.
 * @return Error if `x` or `result` is NULL.
 *         Error if exponential operation failed.
 *         NULL if exponential operation was successfully applied.
 */
static nw_error_t *exponential_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(EXPONENTIAL_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_EXPONENTIAL, string_create("failed to run exponential operation."), error);
    }

    return error;
}

/**
 * @brief Execute exponential operation backward.
 * @param x Input operand.
 * @param result Result of applying the exponential operation to the input operand.
 * @param gradient The incoming gradient with respect to the result.
 * @return Error if `x`, `result`, `gradient` is NULL.
 *         NULL if gradient with respect to `x` is succesfully computed.
 */
static nw_error_t *exponential_operation_backward(tensor_t *x, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;

    if (x->requires_gradient)
    {
        error = tensor_multiplication(gradient, result, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);

    return error;
}


/**
 * @brief Execute logarithm operation forward.
 * @param x The input operand.
 * @param result The output of the logarithm operation.
 * @return Error if `x` or `result` is NULL.
 *         Error if logarithm operation failed.
 *         NULL if logarithm operation was successfully applied.
 */
static nw_error_t *logarithm_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(LOGARITHM_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_LOGARITHM, string_create("failed to run logarithm operation."), error);
    }

    return error;
}

/**
 * @brief Execute logarithm operation backward.
 * @param x The input operand.
 * @param gradient The incoming gradient with respect to the result.
 * @return Error if `x` or `gradient` is NULL.
 *         NULL if the gradient with respect to `x` was successfully computed.
 */
static nw_error_t *logarithm_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;

    if (x->requires_gradient)
    {
        error = tensor_division(gradient, x, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors"), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);

    return error;
}

/**
 * @brief Execute sine operation forward.
 * @param x The input operand.
 * @param result The output of the sine operation.
 * @return Error if `x` or `result` is NULL.
 *         Error if sine operation failed.
 *         NULL if sine operation was successfully applied.
 */
static nw_error_t *sine_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(SINE_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_SINE, string_create("failed to run sine operation."), error);
    }

    return error;
}

/**
 * @brief Execute sine operation backward.
 * @param x The input operand.
 * @param gradient The incoming gradient with respect to the result.
 * @return Error if `x` or `gradient` is NULL.
 *         NULL if the gradient with respect to `x` was successfully computed.
 */
static nw_error_t *sine_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;

    if (x->requires_gradient)
    {
        error = tensor_cosine(x, &x_gradient_i);
        if (error)
        {
            error = ERROR(ERROR_COSINE, string_create("failed to cosine tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_i, gradient, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors"), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);

    return error;
}

static nw_error_t *cosine_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(COSINE_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_COSINE, string_create("failed to run cosine operation."), error);
    }

    return error;
}

/**
 * @brief Execute cosine operation backward.
 * @param x The input operand.
 * @param gradient The incoming gradient with respect to the result.
 * @return Error if `x` or `gradient` is NULL.
 *         NULL if the gradient with respect to `x` was successfully computed.
 */
static nw_error_t *cosine_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;

    if (x->requires_gradient)
    {
        error = tensor_sine(x, &x_gradient_j);
        if (error)
        {
            error = ERROR(ERROR_SINE, string_create("failed to sine tensor."), error);
            goto cleanup;
        }

        error = tensor_negation(x_gradient_j, &x_gradient_i);
        if (error)
        {
            error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_i, gradient, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors"), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);
    tensor_destroy(x_gradient_j);

    return error;
}

static nw_error_t *square_root_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(SQUARE_ROOT_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_SQUARE_ROOT, string_create("failed to run square root operation."), error);
    }
    
    return error;
}

/**
 * @brief Execute square root operation backward.
 * @param x The input operand.
 * @param result The result of the square root operation applied to the input operand.
 * @param gradient The incoming gradient with respect to the result.
 * @return Error if `x` or `gradient` is NULL.
 *         NULL if the gradient with respect to `x` was successfully computed.
 */
static nw_error_t *square_root_operation_backward(tensor_t *x, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    void *value = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    size_t size = datatype_size(datatype);

    if (x->requires_gradient)
    {
        value = (void *) malloc(size);
        if (!value)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
            goto cleanup;
        }

        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) value = (float32_t) 2.0;
            break;
        case FLOAT64:
            *(float64_t *) value = (float64_t) 2.0;
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
            goto cleanup;
        }

        error = tensor_constant(value, datatype, runtime, false, false, &x_gradient_j);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(result, x_gradient_j, &x_gradient_i);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_division(gradient, x_gradient_i, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors"), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    free(value);
    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);
    tensor_destroy(x_gradient_j);

    return error;
}

static nw_error_t *reciprocal_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(RECIPROCAL_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_RECIPROCAL, string_create("failed to run reciprocal operation."), error);
    }
    
    return error;
}

/**
 * @brief Execute square root operation backward.
 * @param x The input operand.
 * @param result The result of the reciprocal operation applied to the input operand.
 * @param gradient The incoming gradient with respect to the result.
 * @return Error if `x` or `gradient` is NULL.
 *         NULL if the gradient with respect to `x` was successfully computed.
 */
static nw_error_t *reciprocal_operation_backward(tensor_t *x, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;

    if (x->requires_gradient)
    {
        error = tensor_multiplication(result, result, &x_gradient_i);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_negation(x_gradient_i, &x_gradient_j);
        if (error)
        {
            error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_j, gradient, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);
    tensor_destroy(x_gradient_j);

    return error;
}

static nw_error_t *contiguous_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(CONTIGUOUS_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_CONTIGUOUS, string_create("failed to run contiguous operation."), error);
    }
    
    return error;
}

static nw_error_t *contiguous_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    return error;
}

static nw_error_t *negation_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(NEGATION_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_NEGATION, string_create("failed to run negation operation."), error);
    }
    
    return error;
}

static nw_error_t *negation_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;

    if (x->requires_gradient)
    {
        error = tensor_negation(gradient, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor"), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);

    return error;
}

static nw_error_t *rectified_linear_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(RECTIFIED_LINEAR_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_RECTIFIED_LINEAR, string_create("failed to run rectified linear operation."), error);
    }
    
    return error;
}

static nw_error_t *rectified_linear_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    void *value = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;
    tensor_t *x_gradient_k = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    int64_t *shape = x->buffer->view->shape;
    int64_t rank = x->buffer->view->rank;
    size_t size = datatype_size(datatype);

    if (x->requires_gradient)
    {
        value = (void *) malloc(size);
        if (!value)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
            goto cleanup;
        }

        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) value = (float32_t) 0.0;
            break;
        case FLOAT64:
            *(float64_t *) value = (float64_t) 0.0;
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
            goto cleanup;
        }

        error = tensor_constant(value, datatype, runtime, false, false, &x_gradient_i);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_expand(x_gradient_i, shape, rank, &x_gradient_k);
        if (error)
        {
            error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
            goto cleanup;
        }

        error = tensor_compare_greater(x, x_gradient_k, &x_gradient_j);
        if (error)
        {
            error = ERROR(ERROR_COMPARE_GREATER, string_create("failed to run compare greater operation."), error);
            goto cleanup;
        }
        
        error = tensor_multiplication(x_gradient_j, gradient, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    free(value);
    tensor_destroy(x_gradient);
    if (x_gradient_i != x_gradient_k)
    {
        tensor_destroy(x_gradient_i);
    }
    tensor_destroy(x_gradient_j);
    tensor_destroy(x_gradient_k);

    return error;
}

static nw_error_t *sigmoid_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(SIGMOID_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_SIGMOID, string_create("failed to run sigmoid operation."), error);
    }
    
    return error;
}

static nw_error_t *sigmoid_operation_backward(tensor_t *x, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    void *value = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;
    tensor_t *x_gradient_k = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    size_t size = datatype_size(datatype);

    if (x->requires_gradient)
    {
        value = (void *) malloc(size);
        if (!value)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
            goto cleanup;
        }

        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) value = (float32_t) 1.0;
            break;
        case FLOAT64:
            *(float64_t *) value = (float64_t) 1.0;
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
            goto cleanup;
        }

        error = tensor_constant(value, datatype, runtime, false, false, &x_gradient_i);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_subtraction(x_gradient_i, result, &x_gradient_k);
        if (error)
        {
            error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(result, x_gradient_k, &x_gradient_j);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_j, gradient, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    free(value);
    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);
    tensor_destroy(x_gradient_j);
    tensor_destroy(x_gradient_k);

    return error;
}

static nw_error_t *as_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_unary(AS_OPERATION, x->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return error;
}

static nw_error_t *unary_operation_forward(unary_operation_t *unary_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    switch (unary_operation->operation_type)
    {
    case EXPONENTIAL_OPERATION:
        error = exponential_operation_forward(unary_operation->x, result);
        break;
    case LOGARITHM_OPERATION:
        error = logarithm_operation_forward(unary_operation->x, result);
        break;
    case SINE_OPERATION:
        error = sine_operation_forward(unary_operation->x, result);
        break;
    case COSINE_OPERATION:
        error = cosine_operation_forward(unary_operation->x, result);
        break;
    case SQUARE_ROOT_OPERATION:
        error = square_root_operation_forward(unary_operation->x, result);
        break;
    case RECIPROCAL_OPERATION:
        error = reciprocal_operation_forward(unary_operation->x, result);
        break;
    case CONTIGUOUS_OPERATION:
        error = contiguous_operation_forward(unary_operation->x, result);
        break;
    case NEGATION_OPERATION:
        error = negation_operation_forward(unary_operation->x, result);
        break;
    case RECTIFIED_LINEAR_OPERATION:
        error = rectified_linear_operation_forward(unary_operation->x, result);
        break;
    case SIGMOID_OPERATION:
        error = sigmoid_operation_forward(unary_operation->x, result);
        break;
    case AS_OPERATION:
        error = as_operation_forward(unary_operation->x, result);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) unary_operation->operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply forward unary operation."), error);
    }
    
    result->requires_gradient = unary_operation->x->requires_gradient;

    return error;
}

static nw_error_t *unary_operation_backward(unary_operation_t *unary_operation, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    switch (unary_operation->operation_type)
    {
    case EXPONENTIAL_OPERATION:
        error = exponential_operation_backward(unary_operation->x, result, gradient);
        break;
    case LOGARITHM_OPERATION:
        error = logarithm_operation_backward(unary_operation->x, gradient);
        break;
    case SINE_OPERATION:
        error = sine_operation_backward(unary_operation->x, gradient);
        break;
    case COSINE_OPERATION:
        error = cosine_operation_backward(unary_operation->x, gradient);
        break;
    case SQUARE_ROOT_OPERATION:
        error = square_root_operation_backward(unary_operation->x, result, gradient);
        break;
    case RECIPROCAL_OPERATION:
        error = reciprocal_operation_backward(unary_operation->x, result, gradient);
        break;
    case CONTIGUOUS_OPERATION:
        error = contiguous_operation_backward(unary_operation->x, gradient);
        break;
    case NEGATION_OPERATION:
        error = negation_operation_backward(unary_operation->x, gradient);
        break;
    case RECTIFIED_LINEAR_OPERATION:
        error = rectified_linear_operation_backward(unary_operation->x, gradient);
        break;
    case SIGMOID_OPERATION:
        error = sigmoid_operation_backward(unary_operation->x, result, gradient);
        break;
    case AS_OPERATION:
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) unary_operation->operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_BACKWARD, string_create("failed to apply backward unary operation."), error);
    }
    
    return error;
}

static nw_error_t *addition_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_binary(ADDITION_OPERATION, x->buffer, y->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to run addition operation."), error);
    }

    return error;
}

static nw_error_t *addition_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to accumulate gradient"), error);
        }
    }

    if (y->requires_gradient)
    {
        error = tensor_accumulate_gradient(y, gradient);
        if (error)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to accumulate gradient"), error);
        }
    }

    return error;
}

static nw_error_t *subtraction_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_binary(SUBTRACTION_OPERATION, x->buffer, y->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_SUBTRACTION, string_create("failed to run subtraction operation."), error);
    }

    return error;
}

static nw_error_t *subtraction_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *y_gradient = NULL;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

    if (y->requires_gradient)
    {
        error = tensor_negation(gradient, &y_gradient);
        if (error)
        {
            error = ERROR(ERROR_NEGATION, string_create("faile to negate tensor."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }

    }

cleanup:

    tensor_destroy(y_gradient);

    return error;
}

static nw_error_t *multiplication_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_binary(MULTIPLICATION_OPERATION, x->buffer, y->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_MULTIPLICATION, string_create("failed to run multiplication operation."), error);
    }

    return error;
}

static nw_error_t *multiplication_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *y_gradient = NULL;

    if (x->requires_gradient)
    {
        error = tensor_multiplication(y, gradient, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

    if (y->requires_gradient)
    {
        error = tensor_multiplication(x, gradient, &y_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors"), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);
    tensor_destroy(y_gradient);

    return error;
}

static nw_error_t *division_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_binary(DIVISION_OPERATION, x->buffer, y->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_DIVISION, string_create("failed to run division operation."), error);
    }

    return error;
}

static nw_error_t *division_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *y_gradient = NULL;
    tensor_t *y_gradient_i = NULL;
    tensor_t *y_gradient_j = NULL;
    tensor_t *y_gradient_k = NULL;
    tensor_t *y_gradient_l = NULL;

    if (x->requires_gradient)
    {
        error = tensor_division(gradient, y, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

    if (y->requires_gradient)
    {
        error = tensor_multiplication(y, y, &y_gradient_k);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_reciprocal(y_gradient_k, &y_gradient_l);
        if (error)
        {
            error = ERROR(ERROR_RECIPROCAL, string_create("failed to get reciprocal of tensor."), error);
            goto cleanup;
        }

        error = tensor_negation(y_gradient_l, &y_gradient_i);
        if (error)
        {
            error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(y_gradient_i, x, &y_gradient_j);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }
        
        error = tensor_multiplication(y_gradient_j, gradient, &y_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);
    tensor_destroy(y_gradient);
    tensor_destroy(y_gradient_i);
    tensor_destroy(y_gradient_j);
    tensor_destroy(y_gradient_k);
    tensor_destroy(y_gradient_l);

    return error;
}

static nw_error_t *power_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_binary(POWER_OPERATION, x->buffer, y->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_POWER, string_create("failed to run power operation."), error);
    }

    return error;
}

static nw_error_t *power_operation_backward(tensor_t *x, tensor_t *y, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;
    tensor_t *y_gradient = NULL;
    tensor_t *y_gradient_i = NULL;
    tensor_t *y_gradient_j = NULL;

    if (x->requires_gradient)
    {
        error = tensor_division(result, x, &x_gradient_i);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_i, y, &x_gradient_j);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_j, gradient, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

    if (y->requires_gradient)
    {
        error = tensor_logarithm(x, &y_gradient_i);
        if (error)
        {
            error = ERROR(ERROR_LOGARITHM, string_create("failed to log tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(y_gradient_i, result, &y_gradient_j);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }
        
        error = tensor_multiplication(y_gradient_j, gradient, &y_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);
    tensor_destroy(x_gradient_j);
    tensor_destroy(y_gradient);
    tensor_destroy(y_gradient_i);
    tensor_destroy(y_gradient_j);

    return error;
}

static nw_error_t *matrix_multiplication_operation_forward(tensor_t *x, tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_binary(MATRIX_MULTIPLICATION_OPERATION, x->buffer, y->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to run matrix multiplication operation."), error);
    }

    return error;
}

static nw_error_t *matrix_multiplication_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(y->buffer, "y->buffer");
    CHECK_NULL_ARGUMENT(y->buffer->view, "y->buffer->view");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *y_gradient = NULL;
    tensor_t *y_gradient_i = NULL;

    if (x->requires_gradient)
    {
        int64_t rank = y->buffer->view->rank;

        error = tensor_transpose(y, &x_gradient_i, rank - 2, rank - 1);
        if (error)
        {
            error = ERROR(ERROR_TRANSPOSE, string_create("failed to transpose tensor"), error);
            goto cleanup;
        }

        error = tensor_matrix_multiplication(gradient, x_gradient_i, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

    if (y->requires_gradient)
    {
        int64_t rank = x->buffer->view->rank;

        error = tensor_transpose(x, &y_gradient_i, rank - 2, rank - 1);
        if (error)
        {
            error = ERROR(ERROR_TRANSPOSE, string_create("failed to transpose tensor"), error);
            goto cleanup;
        }

        error = tensor_matrix_multiplication(y_gradient_i, gradient, &y_gradient);
        if (error)
        {
            error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);
    tensor_destroy(y_gradient);
    tensor_destroy(y_gradient_i);

    return error;
}

static nw_error_t *compare_equal_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_binary(COMPARE_EQUAL_OPERATION, x->buffer, y->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_COMPARE_EQUAL, string_create("failed to run compare equal operation."), error);
    }

    return error;
}

static nw_error_t *compare_greater_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_binary(COMPARE_GREATER_OPERATION, x->buffer, y->buffer, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_COMPARE_GREATER, string_create("failed to run compare greater operation."), error);
    }

    return error;
}

static nw_error_t *binary_operation_forward(binary_operation_t *binary_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    switch (binary_operation->operation_type)
    {
    case ADDITION_OPERATION:
        error = addition_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case SUBTRACTION_OPERATION:
        error = subtraction_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case MULTIPLICATION_OPERATION:
        error = multiplication_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case DIVISION_OPERATION:
        error = division_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case POWER_OPERATION:
        error = power_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        error = matrix_multiplication_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case COMPARE_EQUAL_OPERATION:
        error = compare_equal_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case COMPARE_GREATER_OPERATION:
        error = compare_greater_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unsupported binary operation type %d.", (int) binary_operation->operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute binary operation forward pass."), error);
    }

    result->requires_gradient = binary_operation->x->requires_gradient || binary_operation->y->requires_gradient;

    return error;
}

static nw_error_t *binary_operation_backward(binary_operation_t *binary_operation, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    switch (binary_operation->operation_type)
    {
    case ADDITION_OPERATION:
        error = addition_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case SUBTRACTION_OPERATION:
        error = subtraction_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case MULTIPLICATION_OPERATION:
        error = multiplication_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case DIVISION_OPERATION:
        error = division_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case POWER_OPERATION:
        error = power_operation_backward(binary_operation->x, binary_operation->y, result, gradient);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        error = matrix_multiplication_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case COMPARE_EQUAL_OPERATION:
    case COMPARE_GREATER_OPERATION:
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unsupported operation type %d.", (int) binary_operation->operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute binary operation backward pass."), error);
    }
    
    return error;
}

static nw_error_t *summation_operation_forward(tensor_t *x, int64_t *axis, int64_t length, tensor_t *result, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_reduction(SUMMATION_OPERATION, x->buffer, axis, length, &result->buffer, keep_dimension);
    if (error)
    {
        return ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
    }

    return NULL; 
}

static nw_error_t *summation_operation_backward(tensor_t *x, int64_t *axis, int64_t length, tensor_t *gradient, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    view_t *view = NULL;

    if (x->requires_gradient)
    {
        if (!keep_dimension)
        {
            error = view_recover_dimensions(gradient->buffer->view, &view, axis, length);
            if (error)
            {
                error = ERROR(ERROR_REDUCTION, string_create("failed to recover reduce dimensions."), error);
                goto cleanup;
            }

            error = tensor_reshape(gradient, &x_gradient_i, view->shape, view->rank);
            if (error)
            {
                error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
                goto cleanup;
            }
        }
        else
        {
            x_gradient_i = gradient;
        }

        error = tensor_expand(x_gradient_i, x->buffer->view->shape, x->buffer->view->rank, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_EXPAND, string_create("failed to expand gradient."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    if (x_gradient_i != x_gradient)
    {
        tensor_destroy(x_gradient);
    }

    if (gradient != x_gradient_i)
    {
        tensor_destroy(x_gradient_i);
    }

    view_destroy(view);

    return error; 
}

static nw_error_t *maximum_operation_forward(tensor_t *x, int64_t *axis, int64_t length, tensor_t *result, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_reduction(MAXIMUM_OPERATION, x->buffer, axis, length, &result->buffer, keep_dimension);
    if (error)
    {
        return ERROR(ERROR_MAXIMUM, string_create("failed to get maximum of tensor."), error);
    }

    return error; 

}

static nw_error_t *maximum_operation_backward(tensor_t *x, int64_t *axis, int64_t length, tensor_t *result, tensor_t *gradient, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_k = NULL;
    tensor_t *x_gradient_l = NULL;
    tensor_t *x_gradient_m = NULL;
    tensor_t *x_gradient_n = NULL;
    tensor_t *x_gradient_o = NULL;
    tensor_t *x_gradient_p = NULL;
    view_t *result_view = NULL;
    view_t *gradient_view = NULL;

    if (x->requires_gradient)
    {
        if (!keep_dimension)
        {

            error = view_recover_dimensions(result->buffer->view, &result_view,  axis, length);
            if (error)
            {
                error = ERROR(ERROR_REDUCTION, string_create("failed to recover from reduce dimensions."), error);
                goto cleanup;
            }

            error = view_recover_dimensions(gradient->buffer->view, &gradient_view, axis, length);
            if (error)
            {
                error = ERROR(ERROR_REDUCTION, string_create("failed to recover from reduce dimensions."), error);
                goto cleanup;
            }

            error = tensor_reshape(result, &x_gradient_k, result_view->shape, result_view->rank);
            if (error)
            {
                error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
                goto cleanup;
            }
            
            error = tensor_reshape(gradient, &x_gradient_l, gradient_view->shape, gradient_view->rank);
            if (error)
            {
                error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
                goto cleanup;
            }
        }
        else
        {
            x_gradient_k = result;
            x_gradient_l = gradient;
        }

        error = tensor_expand(x_gradient_k, x->buffer->view->shape, x->buffer->view->rank, &x_gradient_m);
        if (error)
        {
            error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
            goto cleanup;
        }

        error = tensor_compare_equal(x_gradient_m, x, &x_gradient_n);
        if (error)
        {
            error = ERROR(ERROR_COMPARE_EQUAL, string_create("failed to compare equal tensors."), error);
            goto cleanup;
        }

        error = tensor_summation(x_gradient_n, &x_gradient_o, axis, length, true);
        if (error)
        {
            error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
            goto cleanup;
        }

        error = tensor_division(x_gradient_n, x_gradient_o, &x_gradient_p);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to divide tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_p, x_gradient_l, &x_gradient);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensor."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error) 
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    if (x_gradient_k != x_gradient_m)
    {
        tensor_destroy(x_gradient_m);    
    }
    if (result != x_gradient_k)
    {
        tensor_destroy(x_gradient_k);    
    }
    if (x_gradient_l != gradient)
    {
        tensor_destroy(x_gradient_l);    
    }
    if (x_gradient_n != x_gradient_o)
    {
        tensor_destroy(x_gradient_n);    
    }
    tensor_destroy(x_gradient_o);    
    tensor_destroy(x_gradient_p);    
    tensor_destroy(x_gradient);    
    view_destroy(result_view);
    view_destroy(gradient_view);

    return error; 
}

static nw_error_t *reduction_operation_forward(reduction_operation_t *reduction_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");

    nw_error_t *error = NULL;

    switch (reduction_operation->operation_type)
    {
    case SUMMATION_OPERATION:
        error = summation_operation_forward(reduction_operation->x, reduction_operation->axis, reduction_operation->length, result, reduction_operation->keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        error = maximum_operation_forward(reduction_operation->x, reduction_operation->axis, reduction_operation->length, result, reduction_operation->keep_dimension);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) reduction_operation->operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute reduction operation forward pass."), error);
    }

    result->requires_gradient = reduction_operation->x->requires_gradient;

    return error;
}

static nw_error_t *reduction_operation_backward(reduction_operation_t *reduction_operation, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    switch (reduction_operation->operation_type)
    {
    case SUMMATION_OPERATION:
        error = summation_operation_backward(reduction_operation->x, reduction_operation->axis, reduction_operation->length, gradient, reduction_operation->keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        error = maximum_operation_backward(reduction_operation->x, reduction_operation->axis, reduction_operation->length, result, gradient, reduction_operation->keep_dimension);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) reduction_operation->operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute reduction operation backward."), error);
    }

    return error;
}

static nw_error_t *expand_operation_forward(tensor_t *x, int64_t *shape, int64_t length, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_structure(EXPAND_OPERATION, x->buffer, shape, length, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_EXPAND, string_create("failed to expand buffer."), error);
    }

    return error;
}

static nw_error_t *expand_operation_backward(tensor_t *x, int64_t *shape, int64_t length, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    int64_t length_keep_dimension = 0;
    int64_t length_remove_dimension = 0;
    int64_t *axis_keep_dimension = NULL;
    int64_t *axis_remove_dimension = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;

    if (x->requires_gradient)
    {
        error = view_reduce_axis(x->buffer->view, shape, length, &axis_keep_dimension, &length_keep_dimension, &axis_remove_dimension, &length_remove_dimension);
        if (error)
        {
            error = ERROR(ERROR_REDUCTION, string_create("failed to get reduction axes."), error);
            goto cleanup;
        }
        
        if (length_keep_dimension)
        {
            error = tensor_summation(gradient, &x_gradient_i, axis_keep_dimension, length_keep_dimension, true);
            if (error)
            {
                error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
                goto cleanup;
            }
        }
        else
        {
            x_gradient_i = gradient;
        }

        if (length_remove_dimension)
        {
            error = tensor_summation(x_gradient_i, &x_gradient, axis_remove_dimension, length_remove_dimension, false);
            if (error)
            {
                error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
                goto cleanup;
            }
        }
        else
        {
            x_gradient = x_gradient_i;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    if (x_gradient_i != x_gradient)
    {
        tensor_destroy(x_gradient);
    }

    if (x_gradient_i != gradient)
    {
        tensor_destroy(x_gradient_i);
    }

    free(axis_keep_dimension);
    free(axis_remove_dimension);

    return error;
}

static nw_error_t *permute_operation_forward(tensor_t *x, int64_t *axis, int64_t length, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_structure(PERMUTE_OPERATION, x->buffer, axis, length, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_PERMUTE, string_create("failed to permute buffer."), error);
    }

    return error;
}

static nw_error_t *permute_operation_backward(tensor_t *x, int64_t *axis, int64_t length, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    int64_t sorted_axis[length];

    if (x->requires_gradient)
    {
        error = argument_sort(axis, length, sorted_axis);
        if (error)
        {
            error = ERROR(ERROR_SORT, string_create("failed to sort array."), error);
            goto cleanup;
        }

        error = tensor_permute(gradient, &x_gradient, sorted_axis, length);
        if (error)
        {
            error = ERROR(ERROR_PERMUTE, string_create("failed to permute tensor."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    if (gradient != x_gradient)
    {
        tensor_destroy(x_gradient);
    }

    return error;
}

static nw_error_t *reshape_operation_forward(tensor_t *x, int64_t *shape, int64_t rank, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_structure(RESHAPE_OPERATION, x->buffer, shape, rank, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_RESHAPE, string_create("failed to reshape buffer."), error);
    }

    return error;
}

static nw_error_t *reshape_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;

    if (x->requires_gradient)
    {
        error = tensor_reshape(gradient, &x_gradient, x->buffer->view->shape, x->buffer->view->rank);        
        if (error)
        {
            error = ERROR(ERROR_RESHAPE, string_create("failed to reshape gradient."), error);
            goto cleanup;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    if (gradient != x_gradient)
    {
        tensor_destroy(x_gradient);
    }

    return error;
}

static nw_error_t *image_to_column_operation_forward(tensor_t *x, int64_t *arguments, int64_t length, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_structure(IMAGE_TO_COLUMN_OPERATION, x->buffer, arguments, length, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_IMAGE_TO_COLUMN, string_create("failed to apply image to column."), error);
    }

    return error;
}

static nw_error_t *image_to_column_operation_backward(tensor_t *x, int64_t *arguments, int64_t length, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(gradient, "gradient");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    if (length != 6)
    {
        return ERROR(ERROR_ARGUMENTS, string_create("invalid number of arguments."), NULL);
    }

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;

    if (x->requires_gradient)
    {
       error = tensor_column_to_image(gradient, &x_gradient, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
       if (error)
       {
           error = ERROR(ERROR_COLUMN_TO_IMAGE, string_create("failed to apply column to image."), error);
           goto cleanup;
       }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    if (gradient != x_gradient)
    {
        tensor_destroy(x_gradient);
    }

    return error;
}

static nw_error_t *column_to_image_operation_forward(tensor_t *x, int64_t *arguments, int64_t length, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_structure(COLUMN_TO_IMAGE_OPERATION, x->buffer, arguments, length, &result->buffer);
    if (error)
    {
        return ERROR(ERROR_COLUMN_TO_IMAGE, string_create("failed to apply column to image."), error);
    }

    return error;
}

static nw_error_t *column_to_image_operation_backward(tensor_t *x, int64_t *arguments, int64_t length, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(gradient, "gradient");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    if (length != 6)
    {
        return ERROR(ERROR_ARGUMENTS, string_create("invalid number of arguments."), NULL);
    }

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;

    if (x->requires_gradient)
    {
       error = tensor_image_to_column(gradient, &x_gradient, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
       if (error)
       {
           error = ERROR(ERROR_IMAGE_TO_COLUMN, string_create("failed to apply image to column."), error);
           goto cleanup;
       }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add gradient."), error);
            goto cleanup;
        }
    }

cleanup:

    if (gradient != x_gradient)
    {
        tensor_destroy(x_gradient);
    }

    return error;
}

/**
 * @brief Apply structure operation forward.
 * @param structure_operation Structure operation to execute.
 * @return Error if `structure_operation` is NULL.
 *         Error if `structure_operation` failed to run.
 *         Error if operation type is unknown.
 *         NULL if `structure_operation` successfully executed.
 */
static nw_error_t *structure_operation_forward(structure_operation_t *structure_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");

    nw_error_t *error = NULL;

    switch (structure_operation->operation_type)
    {
    case EXPAND_OPERATION:
        error = expand_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length, result);
        break;
    case PERMUTE_OPERATION:
        error = permute_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length, result);
        break;
    case RESHAPE_OPERATION:
        error = reshape_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length, result);
        break;
    case IMAGE_TO_COLUMN_OPERATION:
        error = image_to_column_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length, result);
        break;
    case COLUMN_TO_IMAGE_OPERATION:
        error = column_to_image_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length, result);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) structure_operation->operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute structure operation forward pass."), error);
    }

    result->requires_gradient = structure_operation->x->requires_gradient;

    return error;
}

/**
 * @brief Apply structure operation backward.
 * @param structure_operation Structure operation being differeniated.
 * @param gradient Incoming gradient of the result of the structure operation.
 * @return Error if `structure_operation` or `gradient` is NULL.
 *         Error if the gradient of the structure operation with respect to the operand failed to compute.
 *         Error if operation type is unknown.
 *         NULL if `structure_operation` successfully executed.
 *         NULL if the gradient of the structure operation with respect to the operand was successfully computed.
 */
static nw_error_t *structure_operation_backward(structure_operation_t *structure_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    switch (structure_operation->operation_type)
    {
    case EXPAND_OPERATION:
        error = expand_operation_backward(structure_operation->x, structure_operation->arguments, structure_operation->length, gradient);
        break;
    case PERMUTE_OPERATION:
        error = permute_operation_backward(structure_operation->x, structure_operation->arguments, structure_operation->length, gradient);
        break;
    case RESHAPE_OPERATION:
        error = reshape_operation_backward(structure_operation->x, gradient);
        break;
    case IMAGE_TO_COLUMN_OPERATION:
        error = image_to_column_operation_backward(structure_operation->x, structure_operation->arguments, structure_operation->length, gradient);
        break;
    case COLUMN_TO_IMAGE_OPERATION:
        error = column_to_image_operation_backward(structure_operation->x, structure_operation->arguments, structure_operation->length, gradient);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) structure_operation->operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute structure operation backward pass."), error);
    }

    return error;
}

static nw_error_t *empty_operation_forward(const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_creation(EMPTY_OPERATION, &result->buffer, shape, rank, NULL, 0, runtime, datatype, NULL, 0, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty buffer."), error);
    }

    return error;
}

static nw_error_t *zeroes_operation_forward(const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_creation(ZEROES_OPERATION, &result->buffer, shape, rank, NULL, 0, runtime, datatype, NULL, 0, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create zeroes buffer."), error);
    }

    return error;
}

static nw_error_t *ones_operation_forward(const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_creation(ONES_OPERATION, &result->buffer, shape, rank, NULL, 0, runtime, datatype, NULL, 0, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create ones buffer."), error);
    }

    return error;
}

static nw_error_t *uniform_operation_forward(const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, tensor_t *result, void **arguments, int64_t length)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_creation(UNIFORM_OPERATION, &result->buffer, shape, rank, NULL, 0, runtime, datatype, arguments, length, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create uniform buffer."), error);
    }

    return error;
}

static nw_error_t *normal_operation_forward(const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, tensor_t *result, void **arguments, int64_t length)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_creation(NORMAL_OPERATION, &result->buffer, shape, rank, NULL, 0, runtime, datatype, arguments, length, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create normal buffer."), error);
    }

    return error;
}

static nw_error_t *arange_operation_forward(const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, tensor_t *result, void **arguments, int64_t length)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_creation(ARANGE_OPERATION, &result->buffer, shape, rank, NULL, 0, runtime, datatype, arguments, length, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create arange buffer."), error);
    }

    return error;
}

static nw_error_t *from_operation_forward(const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, tensor_t *result, void *data)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_creation(FROM_OPERATION, &result->buffer, shape, rank, NULL, 0, runtime, datatype, NULL, 0, data);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create arange buffer."), error);
    }

    return error;
}

static nw_error_t *copy_operation_forward(const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, tensor_t *result, void *data)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = buffer_creation(COPY_OPERATION, &result->buffer, shape, rank, NULL, 0, runtime, datatype, NULL, 0, data);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return error;
}

static nw_error_t *creation_operation_forward(creation_operation_t *creation_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(creation_operation, "creation_operation");

    nw_error_t *error = NULL;

    switch (creation_operation->operation_type)
    {
    case EMPTY_OPERATION:
        error = empty_operation_forward(creation_operation->shape, creation_operation->rank, creation_operation->runtime, creation_operation->datatype, result);
        break;
    case ZEROES_OPERATION:
        error = zeroes_operation_forward(creation_operation->shape, creation_operation->rank, creation_operation->runtime, creation_operation->datatype, result);
        break;
    case ONES_OPERATION:
        error = ones_operation_forward(creation_operation->shape, creation_operation->rank, creation_operation->runtime, creation_operation->datatype, result);
        break;
    case UNIFORM_OPERATION:
        error = uniform_operation_forward(creation_operation->shape, creation_operation->rank, creation_operation->runtime, creation_operation->datatype, result, creation_operation->arguments, creation_operation->length);
        break;
    case NORMAL_OPERATION:
        error = normal_operation_forward(creation_operation->shape, creation_operation->rank, creation_operation->runtime, creation_operation->datatype, result, creation_operation->arguments, creation_operation->length);
        break;
    case ARANGE_OPERATION:
        error = arange_operation_forward(creation_operation->shape, creation_operation->rank, creation_operation->runtime, creation_operation->datatype, result, creation_operation->arguments, creation_operation->length);
        break;
    case FROM_OPERATION:
        error = from_operation_forward(creation_operation->shape, creation_operation->rank, creation_operation->runtime, creation_operation->datatype, result, creation_operation->data);
        break;
    case COPY_OPERATION:
        error = copy_operation_forward(creation_operation->shape, creation_operation->rank, creation_operation->runtime, creation_operation->datatype, result, creation_operation->data);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) creation_operation->operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute creation operation forward pass."), error);
    }

    result->requires_gradient = creation_operation->requires_gradient;
    result->persist = creation_operation->persist;

    return error;
}

/**
 * @brief Destroy a unary operation.
 * @param unary_operation The unary operation created with `unary_operation_create` to free.
 *                        Argument can be NULL.
 */
static void unary_operation_destroy(unary_operation_t *unary_operation)
{
    if (unary_operation)
    {
        free(unary_operation);
    }
}

/**
 * @brief The unary operation constructor. 
 * @param unary_operation The address of the pointer to the unary operation to instantiate.
 * @param unary_operation_type The type of unary operation to create.
 * @param x The input operand of the unary operation.
 * @param result The resultant output of the unary operation.
 * @return Error if `unary_operation`, `x`, or `result` is NULL.
 *          
 */
static nw_error_t *unary_operation_create(unary_operation_t **unary_operation, unary_operation_type_t unary_operation_type, const tensor_t *x)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(x, "x");

    *unary_operation = (unary_operation_t *) malloc(sizeof(unary_operation_t));
    if (!*unary_operation)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(unary_operation_t)), NULL);
    }

    (*unary_operation)->operation_type = unary_operation_type;
    (*unary_operation)->x = (tensor_t *) x; 

    return NULL;
}

static void binary_operation_destroy(binary_operation_t *binary_operation)
{
    if (binary_operation)
    {
        free(binary_operation);
    }
}

static nw_error_t *binary_operation_create(binary_operation_t **binary_operation, binary_operation_type_t binary_operation_type, const tensor_t *x, const tensor_t *y)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    *binary_operation = (binary_operation_t *) malloc(sizeof(binary_operation_t));
    if (!*binary_operation)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(binary_operation_t)), NULL);
    }

    (*binary_operation)->operation_type = binary_operation_type;
    (*binary_operation)->x = (tensor_t *) x; 
    (*binary_operation)->y = (tensor_t *) y;

    return NULL;
}

static void reduction_operation_destroy(reduction_operation_t *reduction_operation)
{
    if (reduction_operation)
    {
        free(reduction_operation->axis);
        free(reduction_operation);
    }
}

static nw_error_t *reduction_operation_create(reduction_operation_t **reduction_operation, 
                                              reduction_operation_type_t reduction_operation_type,
                                              const tensor_t *x,
                                              const int64_t *axis,
                                              int64_t length,
                                              bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");

    nw_error_t *error = NULL;
    size_t size = length * sizeof(int64_t);

    *reduction_operation = (reduction_operation_t *) malloc(sizeof(reduction_operation_t));
    if (!*reduction_operation)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(reduction_operation_t)), NULL);
        goto cleanup;
    }

    (*reduction_operation)->axis = (int64_t *) malloc(size);
    if (!(*reduction_operation)->axis)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate size %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*reduction_operation)->axis, axis, size);

    (*reduction_operation)->operation_type = reduction_operation_type;
    (*reduction_operation)->x = (tensor_t *) x; 
    (*reduction_operation)->length = length;
    (*reduction_operation)->keep_dimension = keep_dimension;

    return error;

cleanup:

    reduction_operation_destroy(*reduction_operation);

    return error;
}

static void structure_operation_destroy(structure_operation_t *structure_operation)
{
    if (structure_operation)
    {
        free(structure_operation->arguments);
        free(structure_operation);
    }
}

static nw_error_t *structure_operation_create(structure_operation_t **structure_operation,
                                              structure_operation_type_t structure_operation_type,
                                              const tensor_t *x,
                                              const int64_t *arguments,
                                              int64_t length)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(x, "x");

    nw_error_t *error = NULL;
    size_t size = length * sizeof(int64_t);

    *structure_operation = (structure_operation_t *) malloc(sizeof(structure_operation_t));
    if (!*structure_operation)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(structure_operation_t)), NULL);
        goto cleanup;
    }

    (*structure_operation)->arguments = (int64_t *) malloc(size);
    if (!(*structure_operation)->arguments)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*structure_operation)->arguments, arguments, size);

    (*structure_operation)->operation_type = structure_operation_type;
    (*structure_operation)->x = (tensor_t *) x; 
    (*structure_operation)->length = length;

    return error;

cleanup:

    structure_operation_destroy(*structure_operation);

    return error;
}

static void creation_operation_destroy(creation_operation_t *creation_operation)
{
    if (creation_operation)
    {
        if (creation_operation->arguments)
        {
            for (int64_t i = 0; i < creation_operation->length; ++i)
            {
                free(creation_operation->arguments[i]);
            }
        }
        free(creation_operation->arguments);
        free(creation_operation->shape);
        free(creation_operation);
    }
}

static nw_error_t *creation_operation_create(creation_operation_t **creation_operation, creation_operation_type_t creation_operation_type,
                                             const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient,
                                             bool_t persist, const void **arguments, int64_t length, void *data)
{
    CHECK_NULL_ARGUMENT(creation_operation, "creation_operation");
    CHECK_NULL_ARGUMENT(shape, "shape");
    if (length)
    {
        CHECK_NULL_ARGUMENT(arguments, "arguments");
    }

    nw_error_t *error = NULL;
    size_t size;

    *creation_operation = (creation_operation_t *) malloc(sizeof(creation_operation_t));
    if (!*creation_operation)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(creation_operation_t)), NULL);
        goto cleanup;
    }

    (*creation_operation)->operation_type = creation_operation_type;
    (*creation_operation)->length = 0;
    (*creation_operation)->rank = rank;
    (*creation_operation)->persist = persist;
    (*creation_operation)->runtime = runtime;
    (*creation_operation)->datatype = datatype;
    (*creation_operation)->requires_gradient = requires_gradient;
    (*creation_operation)->shape = NULL;
    (*creation_operation)->arguments = NULL;
    (*creation_operation)->data = data;

    // Shape
    size = rank * sizeof(int64_t);
    (*creation_operation)->shape = (int64_t *) malloc(size);
    if (!(*creation_operation)->shape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*creation_operation)->shape, shape, size);

    // Arguments
    size = length * sizeof(void *);
    (*creation_operation)->arguments = (void **) malloc(size);
    if (!(*creation_operation)->arguments)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    
    for (int64_t i = 0; i < length; ++i)
    {
        (*creation_operation)->arguments[i] = NULL;
    }

    (*creation_operation)->length = length;
    size = datatype_size(datatype);
    for (int64_t i = 0; i < length; ++i)
    {
        (*creation_operation)->arguments[i] = (void *) malloc(size);
        if (!(*creation_operation)->arguments[i])
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
            goto cleanup;
        }
        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) (*creation_operation)->arguments[i] = *(float32_t *) arguments[i];
            break;
        case FLOAT64:
            *(float64_t *) (*creation_operation)->arguments[i] = *(float64_t *) arguments[i];
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
            goto cleanup;
        }
    }

    return error;

cleanup:

    creation_operation_destroy(*creation_operation);

    return error;
}

/**
 * @brief Destroy an operation of a given type.
 * @param operation The operation created with `operation_create` to free.
 *                  Argument can be NULL. 
 * @param operation_type The type of operation being destroyed.
 */
static void operation_destroy(operation_t *operation, operation_type_t operation_type, bool_t destroy_type_operation)
{
    if (operation)
    {
        if (destroy_type_operation)
        {
            switch (operation_type)
            {
            case UNARY_OPERATION:
                unary_operation_destroy(operation->unary_operation);
                break;
            case BINARY_OPERATION:
                binary_operation_destroy(operation->binary_operation);
                break;
            case REDUCTION_OPERATION:
                reduction_operation_destroy(operation->reduction_operation);
                break;
            case STRUCTURE_OPERATION:
                structure_operation_destroy(operation->structure_operation);
                break;
            case CREATION_OPERATION:
                creation_operation_destroy(operation->creation_operation);
                break;
            default:
                break;
            }
        }
        free(operation);
    }
}

/**
 * @brief The operation constructor.
 * @param operation The address to the pointer of the operation being instantiated.
 * @param operation_type The type of operation being created.
 * @param type_operation The operation of type `operation_type` to be assigned to the generic `operation`.
 * @return Error if `operation` or `type_operation` are NULL.
 *         Error if failed to allocate memory for `operation`.
 *         Error if `operation_type` is not a known operation type.
 *         NULL if operation was successfully created.
 */
static nw_error_t *operation_create(operation_t **operation, operation_type_t operation_type, void *type_operation)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(type_operation, "type_operation");

    nw_error_t *error = NULL;

    *operation = (operation_t *) malloc(sizeof(operation_t));
    if (!*operation)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(operation_t)), NULL);
        goto cleanup;
    }

    switch (operation_type)
    {
    case UNARY_OPERATION:
        (*operation)->unary_operation = (unary_operation_t *) type_operation;
        break;
    case BINARY_OPERATION:
        (*operation)->binary_operation = (binary_operation_t *) type_operation;
        break;
    case REDUCTION_OPERATION:
        (*operation)->reduction_operation = (reduction_operation_t *) type_operation;
        break;
    case STRUCTURE_OPERATION:
        (*operation)->structure_operation = (structure_operation_t *) type_operation;
        break;
    case CREATION_OPERATION:
        (*operation)->creation_operation = (creation_operation_t *) type_operation;
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        goto cleanup;
    }
    
    return error;

cleanup:

    operation_destroy(*operation, operation_type, false);

    return error;
}

/**
 * @brief The function constructor.
 * @param function The address of the pointer to the function being instantiated.
 * @param operation The operation the function applies.
 * @param operation_type The type of operation the function applies. 
 * @return Error in `function` or `operation` is NULL.
 *         Error if failed to allocate memory for `function`.
 *         NULL if function is created successfully.
 */
nw_error_t *function_create(function_t **function, operation_t *operation, operation_type_t operation_type)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(operation, "operation");

    *function = (function_t *) malloc(sizeof(function_t));
    if (!*function)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(function_t)), NULL);
    }

    (*function)->operation = operation;
    (*function)->operation_type = operation_type;
    
    return NULL;
}

/**
 * @brief The function destroyer.
 * @param function Free a function created with `function_create`. 
 *                 Argument can be NULL.
 */
void function_destroy(function_t *function, bool_t destroy_operation)
{
    if (function)
    {
        if (destroy_operation)
        {
            operation_destroy(function->operation, function->operation_type, true);
        }
        free(function);
    }
}


/**
 * @brief Execute an operation of a given type.
 * @param operation The operation to execute.
 * @param operation_type The type of `operation` being executed.
 * @return Error if `operation` is NULL.
 *         Error if `operation_type` is unknown.
 *         Error if `operation` failed to execute.
 *         NULL if `operation` ran successfully.
 */
static nw_error_t *operation_forward(operation_t *operation, operation_type_t operation_type, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    switch (operation_type)
    {
    case UNARY_OPERATION:
        error = unary_operation_forward(operation->unary_operation, result);
        break;
    case BINARY_OPERATION:
        error = binary_operation_forward(operation->binary_operation, result);
        break;
    case REDUCTION_OPERATION:
        error = reduction_operation_forward(operation->reduction_operation, result);
        break;
    case STRUCTURE_OPERATION:
        error = structure_operation_forward(operation->structure_operation, result);
        break;
    case CREATION_OPERATION:
        error = creation_operation_forward(operation->creation_operation, result);
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to operation forward pass."), error);
    }

    return error;
}

/**
 * @brief Compute gradient of operands for a given operation.
 * @param operation The operation being differeniated.
 * @param operation_type The type of operation being differentiated.
 * @param gradient The incoming gradient with resepct to the result of the operation.
 * @return Error if `operation` or `gradient` is NULL. 
 *         Error if `operation_type` is unknown.
 *  `      Error if gradient of operands failed to compute.
 *         NULL if the gradients with resepct to the operands were computed successfully.
 */
static nw_error_t *operation_backward(operation_t *operation, operation_type_t operation_type, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    switch (operation_type)
    {
    case UNARY_OPERATION:
        error = unary_operation_backward(operation->unary_operation, result, gradient);
        break;
    case BINARY_OPERATION:
        error = binary_operation_backward(operation->binary_operation, result, gradient);
        break;
    case REDUCTION_OPERATION:
        error = reduction_operation_backward(operation->reduction_operation, result, gradient);
        break;
    case STRUCTURE_OPERATION:
        error = structure_operation_backward(operation->structure_operation, gradient);
        break;
    case CREATION_OPERATION:
        break;
    default:
        error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_BACKWARD, string_create("failed operation backward pass."), error);
    }

    return error;
}

/**
 * @brief Execute forward pass of a function. The function is stored in the result's context.
 * @param function The function to execute.
 * @return Error if `function`, `function->operation`, `function->operation-><type>_operation`, 
 *         `function->operation-><type>_operation->result` is NULL.
 *         Error if operation type of `function` is unknown. 
 *         Error if operation failed to execute.
 *         NULL if function successfully executed.
 */
static nw_error_t *function_forward(function_t *function, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = operation_forward(function->operation, function->operation_type, result);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed operation forward pass."), error);
    }

    return error;
}

/**
 * @brief Compute the resultant gradient of the operands of the function.
 * @param function The function being differentiated.
 * @param gradient The incoming gradient with respect to the result of the function.
 * @return Error if `function` or `gradient` is NULL.
 *         Error if the gradients with respect to the operands failed to compute.
 *         NULL, if the gradients with respect to the operands were successfully computed.
 */
static nw_error_t *function_backward(function_t *function, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(gradient, "gradient");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = operation_backward(function->operation, function->operation_type, result, gradient);
    if (error)
    {
        return ERROR(ERROR_BACKWARD, string_create("failed to execute operation backward pass."), error);
    }

    return error;
}

static nw_error_t *apply_function(operation_t *operation, operation_type_t operation_type, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    function_t *function = NULL;
    bool_t overwrite = (bool_t) *result;

    if (!overwrite)
    {
        error = tensor_create_null(result);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    error = function_create(&function, operation, operation_type);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create function."), error);
        goto cleanup;
    }

    error = function_forward(function, *result);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to function forward pass."), error);
        goto cleanup;
    }

    if ((*result)->requires_gradient && !no_gradient)
    {
        if ((*result)->context)
        {
            function_destroy((*result)->context, true);
        }
        (*result)->context = function;
    }
    else
    {
        function_destroy(function, true);
    }

    return error;

cleanup:

    if (!overwrite)
    {
        tensor_destroy(*result);
    }

    function_destroy(function, false);

    return error;
}

/**
 * @brief Execute the operation of a generic function.
 * @param operation_type The type of operation being applied.
 * @param type_operation The generic operation being applied.
 * @return Error if `type_operation` is NULL.
 *         Error if function failed to execute.
 *         NULL if function executed successfully.
 */
static nw_error_t *apply_operation(operation_type_t operation_type, void *type_operation, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(type_operation, "type_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    operation_t *operation = NULL;

    error = operation_create(&operation, operation_type, type_operation);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create operation."), error);
        goto cleanup;
    }

    error = apply_function(operation, operation_type, result);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to apply function."), error);
        goto cleanup;
    }

    return error;

cleanup:

    operation_destroy(operation, operation_type, false);

    return error;
}

/**
 * @brief Execute the unary operation of a function.
 * @param unary_operation_type The type of unary operation being applied.
 * @param x The input tensor of the unary function.
 * @param result The output tensor of the unary function.
 * @return Error if `x` or `result` is NULL.
 *         Error if unary operation failed to execute.
 *         NULL if unary operation executed successfully.
 */
nw_error_t *apply_operation_unary(unary_operation_type_t unary_operation_type, const tensor_t *x, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    unary_operation_t *unary_operation = NULL;

    error = unary_operation_create(&unary_operation, unary_operation_type, x);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create unary operation."), error);
        goto cleanup;
    }

    error = apply_operation(UNARY_OPERATION, (void *) unary_operation, result);
    if (error)
    {
        error =  ERROR(ERROR_FORWARD, string_create("failed to apply unary function."), error);
        goto cleanup;
    }
    
    return error;

cleanup:

    unary_operation_destroy(unary_operation);

    return error;
}

/**
 * @brief Execute the binary operation of a function.
 * @param binary_operation_type The type of binary operation being applied.
 * @param x The first operand of the binary function.
 * @param y The second operand of the binary function.
 * @param result The output tensor of the binary function.
 * @return Error if `x`, `y`, or `result` is NULL.
 *         Error if binary function failed to execute.
 *         NULL if binary function executed successfully.
 */
nw_error_t *apply_operation_binary(binary_operation_type_t binary_operation_type, const tensor_t *x, const tensor_t *y, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    binary_operation_t *binary_operation = NULL;
    tensor_t *x_broadcasted = NULL;
    tensor_t *y_broadcasted = NULL;

    if (binary_operation_type == MATRIX_MULTIPLICATION_OPERATION)
    {
        error = tensor_broadcast_matrix_multiplication(x, y, &x_broadcasted, &y_broadcasted);
    }
    else
    {
        error = tensor_broadcast(x, y, &x_broadcasted, &y_broadcasted);
    }

    if (error)
    {
        error = ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensors."), error);
        goto cleanup;
    } 

    error = binary_operation_create(&binary_operation, binary_operation_type, x_broadcasted, y_broadcasted);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create binary operation."), error);
        goto cleanup;
    }

    error = apply_operation(BINARY_OPERATION, (void *) binary_operation, result);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to apply binary function."), error);
        goto cleanup;
    }

    if (!(x->requires_gradient || y->requires_gradient) || no_gradient)
    {
        if (x != x_broadcasted)
        {
            tensor_destroy(x_broadcasted);    
        }

        if (y != y_broadcasted)
        {
            tensor_destroy(y_broadcasted);    
        }
    }

    return error;

cleanup:

    tensor_destroy(x_broadcasted);    
    tensor_destroy(y_broadcasted);    
    binary_operation_destroy(binary_operation);

    return error;
}

/**
 * @brief Execute the reduction operation of a function.
 * @param reduction_operation_type The type of reduction operation being applied.
 * @param x The input tensor of the reduction function.
 * @param axis An array containing the indicies of the dimensions of the input tensor to reduce.
 * @param length The number of indicies in `axis`.
 * @param keep_dimension True to keep dimension of input tensor after it is reduced.
 *                       False to remove input tensor dimension after it is reduced.
 * @param result The output tensor of the reduction function.
 * @return Error if `x`, `axis`, or `result` is NULL.
 *         Error if reduction operation failed to execute.
 *         NULL if reduction operation executed successfully.
 */
nw_error_t *apply_operation_reduction(reduction_operation_type_t reduction_operation_type, const tensor_t *x, const int64_t *axis, int64_t length, bool_t keep_dimension, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    reduction_operation_t *reduction_operation = NULL;
    int64_t reduce_length = length ? length : x->buffer->view->rank;
    int64_t reduce_axis[reduce_length];
    view_t *reduced_view = NULL;

    if (x->buffer->view->rank < reduce_length)
    {
        error = ERROR(ERROR_RANK, string_create("reduce axis length greater than rank of tensor."), NULL);
        goto cleanup;
    }

    for (int64_t i = 0; i < reduce_length; ++i)
    {
        reduce_axis[i] = (!axis || !length) ? i : dimension_to_index(axis[i], x->buffer->view->rank);
    }

    CHECK_UNIQUE(reduce_axis, reduce_length, "reduce_axis");

    error = view_reduce(x->buffer->view, &reduced_view, reduce_axis, reduce_length, keep_dimension);
    if (error)
    {
        error = ERROR(ERROR_REDUCTION, string_create("failed to reduce tensor."), error);
        goto cleanup;
    }

    if (view_shapes_equal(reduced_view, x->buffer->view))
    {
        *result = (tensor_t *) x;
    }
    else
    {
        error = reduction_operation_create(&reduction_operation, reduction_operation_type, x, reduce_axis, reduce_length, keep_dimension);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create reduction operation."), error);
            goto cleanup;
        }

        error = apply_operation(REDUCTION_OPERATION, (void *) reduction_operation, result);
        if (error)
        {
            error = ERROR(ERROR_FORWARD, string_create("failed to apply reduction function."), error);
            goto cleanup;
        }
    }

    view_destroy(reduced_view);

    return error;

cleanup:

    reduction_operation_destroy(reduction_operation);
    view_destroy(reduced_view);

    return error;
}

/**
 * @brief Execute the structure operation of a function.
 * @param structure_operation_type The type of structure operation being applied.
 * @param x The input tensor of the structure function.
 * @param arguments An array containing the arguments of the structure operation.
 * @param length The number of elements in `arguments`.
 * @param result The output tensor of the structure function.
 * @return Error if `x`, `arguments`, or `result` is NULL.
 *         Error if structure operation failed to execute.
 *         NULL if structure operation executed successfully.
 */
nw_error_t *apply_operation_structure(structure_operation_type_t structure_operation_type, const tensor_t *x, const int64_t *arguments, int64_t length, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    structure_operation_t *structure_operation = NULL;

    error = structure_operation_create(&structure_operation, structure_operation_type, x, arguments, length);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create structure operation."), error);
        goto cleanup;
    }

    error = apply_operation(STRUCTURE_OPERATION, (void *) structure_operation, result);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to apply structure function."), error);
        goto cleanup;
    }
    
    return error;

cleanup:

    structure_operation_destroy(structure_operation);

    return error;
}

nw_error_t *apply_operation_creation(creation_operation_type_t creation_operation_type, const int64_t *shape, int64_t rank,
                                    runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist,
                                    const void **arguments, int64_t length, void *data, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    creation_operation_t *creation_operation = NULL;

    error = creation_operation_create(&creation_operation, creation_operation_type, shape, rank, runtime,
                                      datatype, requires_gradient, persist, arguments, length, data);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create creation."), error);
        goto cleanup;
    }

    error = apply_operation(CREATION_OPERATION, (void *) creation_operation, result);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to apply creation function."), error);
        goto cleanup;
    }
    
    return error;

cleanup:

    creation_operation_destroy(creation_operation);

    return error;
}

nw_error_t *apply_backward(tensor_t *result)
{
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = function_backward(result->context, result, result->gradient);
    if (error)
    {
        return ERROR(ERROR_BACKWARD, string_create("failed to apply backward."), error);
    }

    return error;
}