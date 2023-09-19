/**
 * @file function.c
 * @brief Mid-level Operations and Automatic Differentiation Engine
 */

#include <function.h>
#include <tensor.h>
#include <view.h>
#include <buffer.h>
#include <string.h>

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
void function_destroy(function_t *function)
{
    if (function)
    {
        operation_destroy(function->operation, function->operation_type);
        free(function);
    }
}

/**
 * @brief Execute the operation of a generic function.
 * @param operation_type The type of operation being applied.
 * @param type_operation The generic operation being applied.
 * @return Error if `type_operation` is NULL.
 *         Error if function failed to execute.
 *         NULL if function executed successfully.
 */
static nw_error_t *apply_function(operation_type_t operation_type, void *type_operation, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(type_operation, "type_operation");

    nw_error_t *error = NULL;
    operation_t *operation = NULL;
    function_t *function = NULL;

    error = operation_create(&operation, operation_type, type_operation);
    if (error)
    {
        error = ERROR(ERROR_CREATE,
                      string_create("failed to create operation of type %s.",
                      operation_type_string(operation_type)), error);
        goto cleanup;
    }

    error = function_create(&function, operation, operation_type);
    if (error)
    {
        error = ERROR(ERROR_CREATE,
                      string_create("failed to create function operation of type %s.",
                      operation_type_string(operation_type)), error);
        goto cleanup;
    }

    error = function_forward(function, result);
    if (error)
    {
        error = ERROR(ERROR_FORWARD,
                      string_create("failed to execute function forward pass of type %s.",
                      operation_type_string(operation_type)), error);
        goto cleanup;
    }

    return error;

cleanup:

    free(operation);
    free(function);

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
nw_error_t *apply_function_unary(unary_operation_type_t unary_operation_type, const tensor_t *x, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    unary_operation_t *unary_operation = NULL;

    error = unary_operation_create(&unary_operation, unary_operation_type, x);
    if (error)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create unary operation of type %s.",
                     unary_operation_type_string(unary_operation_type)), error);
    }

    error = apply_function(UNARY_OPERATION, (void *) unary_operation, result);
    if (error)
    {
        unary_operation_destroy(unary_operation);
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply unary function of type %s.",
                     unary_operation_type_string(unary_operation_type)), error);
    }
    
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
nw_error_t *apply_function_binary(binary_operation_type_t binary_operation_type, const tensor_t *x, const tensor_t *y, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    binary_operation_t *binary_operation = NULL;
    tensor_t *x_broadcasted = NULL;
    tensor_t *y_broadcasted = NULL;

    if (binary_operation_type != MATRIX_MULTIPLICATION_OPERATION)
    {
        error = tensor_broadcast(x, y, &x_broadcasted, &y_broadcasted);
        if (error)
        {
            error = ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        x_broadcasted = (tensor_t *) x;
        y_broadcasted = (tensor_t *) y;
    }

    error = binary_operation_create(&binary_operation, binary_operation_type, x_broadcasted, y_broadcasted);
    if (error)
    {
        error = ERROR(ERROR_CREATE,
                      string_create("failed to create binary operation of type %s.",
                      binary_operation_type_string(binary_operation_type)), error);
        goto cleanup;
    }

    error = apply_function(BINARY_OPERATION, (void *) binary_operation, result);
    if (error)
    {
        binary_operation_destroy(binary_operation);
        error = ERROR(ERROR_FORWARD,
                      string_create("failed to apply binary function of type %s.",
                      binary_operation_type_string(binary_operation_type)), error);
        goto cleanup;
    }

cleanup:

    if (!(x_broadcasted->requires_gradient || y_broadcasted->requires_gradient))
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
nw_error_t *apply_function_reduction(reduction_operation_type_t reduction_operation_type,
                                     const tensor_t *x,
                                     const uint64_t *axis,
                                     uint64_t length,
                                     bool_t keep_dimension,
                                     tensor_t **result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(result, "result");


    nw_error_t *error = NULL;
    reduction_operation_t *reduction_operation = NULL;
    uint64_t reduce_length = length ? length : x->buffer->view->rank;
    uint64_t reduce_axis[reduce_length];

    if (x->buffer->view->rank < reduce_length)
    {
        return ERROR(ERROR_RANK_CONFLICT, string_create("reduce axis length greater than rank of tensor."), NULL);
    }

    for (uint64_t i = 0; i < reduce_length; ++i)
    {
        reduce_axis[i] = (!axis || !length) ? i : axis[i];
    }

    CHECK_UNIQUE(reduce_axis, reduce_length, "reduce_axis");

    error = reduction_operation_create(&reduction_operation, reduction_operation_type,
                                        x, reduce_axis, reduce_length, keep_dimension);
    if (error)
    {
        return ERROR(ERROR_CREATE,
                    string_create("failed to create reduction operation of type %s.",
                    reduction_operation_type_string(reduction_operation_type)), error);
    }

    error = apply_function(REDUCTION_OPERATION, (void *) reduction_operation, result);
    if (error)
    {
        reduction_operation_destroy(reduction_operation);
        return ERROR(ERROR_FORWARD,
                        string_create("failed to apply reduction function of type %s.",
                        reduction_operation_type_string(reduction_operation_type)), error);
    }

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
nw_error_t *apply_function_structure(structure_operation_type_t structure_operation_type,
                                     const tensor_t *x,
                                     const uint64_t *arguments,
                                     uint64_t length,
                                     tensor_t **result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    structure_operation_t *structure_operation = NULL;

    error = structure_operation_create(&structure_operation, structure_operation_type, x, arguments, length);
    if (error)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create structure operation of type %s.",
                     structure_operation_type_string(structure_operation_type)), error);
    }

    error = apply_function(STRUCTURE_OPERATION, (void *) structure_operation, result);
    if (error)
    {
        structure_operation_destroy(structure_operation);
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply structure function of type %s.",
                     structure_operation_type_string(structure_operation_type)), error);
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
nw_error_t *function_forward(function_t *function, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(function->operation, "function->operation");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    operation_type_t operation_type = function->operation_type;
    operation_t *operation= function->operation;

    error = operation_forward(operation, operation_type, result);
    if (error)
    {
        return ERROR(ERROR_FORWARD, 
                     string_create("failed to execute operation forward pass of type %s.",
                     operation_type_string(function->operation_type)), error);
    }

    if (*result)
    {
        if ((*result)->requires_gradient)
        {
            (*result)->context = function;
        }
        else
        {
            function_destroy(function);
        }
    }
    else
    {
        return ERROR(ERROR_NULL, string_create("result is null."), NULL);
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
nw_error_t *function_backward(function_t *function, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    error = operation_backward(function->operation, function->operation_type, gradient);
    if (error)
    {
        return ERROR(ERROR_BACKWARD,
                     string_create("failed to execute operation backward pass of type %s.",
                     operation_type_string(function->operation_type)), error);
    }

    return error;
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
nw_error_t *operation_create(operation_t **operation, operation_type_t operation_type, void *type_operation)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(type_operation, "type_operation");

    *operation = (operation_t *) malloc(sizeof(operation_t));
    if (!*operation)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(operation_t)), NULL);
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
    default:
        free(*operation);
        return ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
    }

    return NULL;
}

/**
 * @brief Destroy an operation of a given type.
 * @param operation The operation created with `operation_create` to free.
 *                  Argument can be NULL. 
 * @param operation_type The type of operation being destroyed.
 */
void operation_destroy(operation_t *operation, operation_type_t operation_type)
{
    if (operation)
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
        default:
            break;
        }
        free(operation);
    }
}

/**
 * @brief Get string representation of `operation_type`.
 * @param operation_type Operation type to display as string.
 * @return A string literal representing the `operation_type`.
 */
string_t operation_type_string(operation_type_t operation_type)
{
    switch (operation_type)
    {
    case UNARY_OPERATION:
        return "UNARY_OPERATION";
    case BINARY_OPERATION:
        return "BINARY_OPERATION";
    case REDUCTION_OPERATION:
        return "REDUCTION_OPERATION";
    case STRUCTURE_OPERATION:
        return "STRUCTURE_OPERATION";
    default:
        return "UNKNOWN_OPERATION";
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
nw_error_t *operation_forward(operation_t *operation, operation_type_t operation_type, tensor_t **result)
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
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute operation forward pass of type %s.", 
                     operation_type_string(operation_type)), error);
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
nw_error_t *operation_backward(operation_t *operation, operation_type_t operation_type, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    switch (operation_type)
    {
    case UNARY_OPERATION:
        error = unary_operation_backward(operation->unary_operation, gradient);
        break;
    case BINARY_OPERATION:
        error = binary_operation_backward(operation->binary_operation, gradient);
        break;
    case REDUCTION_OPERATION:
        error = reduction_operation_backward(operation->reduction_operation, gradient);
        break;
    case STRUCTURE_OPERATION:
        error = structure_operation_backward(operation->structure_operation, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_BACKWARD,
                     string_create("failed to execute operation backward pass of %s.",
                     operation_type_string(operation_type)), error);
    }

    return error;
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
nw_error_t *unary_operation_create(unary_operation_t **unary_operation,
                                   unary_operation_type_t unary_operation_type,
                                   const tensor_t *x)
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
    (*unary_operation)->result = NULL;

    return NULL;
}

/**
 * @brief Destroy a unary operation.
 * @param unary_operation The unary operation created with `unary_operation_create` to free.
 *                        Argument can be NULL.
 */
void unary_operation_destroy(unary_operation_t *unary_operation)
{
    if (unary_operation)
    {
        free(unary_operation);
    }
}

/**
 * @brief Get string representation of `unary_operation_type`.
 * @param operation_type Operation type to display as string.
 * @return A string literal representing the `unary_operation_type`.
 */
string_t unary_operation_type_string(unary_operation_type_t unary_operation_type)
{
    switch (unary_operation_type)
    {
    case EXPONENTIAL_OPERATION:
        return "EXPONENTIAL_OPERATION";
    case LOGARITHM_OPERATION:
        return "LOGARITHM_OPERATION";
    case SINE_OPERATION:
        return "SINE_OPERATION";
    case COSINE_OPERATION:
        return "COSINE_OPERATION";
    case SQUARE_ROOT_OPERATION:
        return "SQUARE_ROOT_OPERATION";
    case RECIPROCAL_OPERATION:
        return "RECIPROCAL_OPERATION";
    case CONTIGUOUS_OPERATION:
        return "CONTIGUOUS_OPERATION";
    case NEGATION_OPERATION:
        return "NEGATION_OPERATION";
    case RECTIFIED_LINEAR_OPERATION:
        return "RECTIFIED_LINEAR_OPERATION";
    case SIGMOID_OPERATION:
        return "SIGMOID_OPERATION";
    default:
        return "UKNOWN_OPERATION";
    }
}

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

    error = runtime_exponential(x->buffer, result->buffer);
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
    tensor_t *x_gradient_i = NULL;

    if (x->requires_gradient)
    {
        error = tensor_as_tensor(result, &x_gradient_i, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(gradient, x_gradient_i, &x_gradient);
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

    error = runtime_logarithm(x->buffer, result->buffer);
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
    tensor_t *x_gradient_i = NULL;

    if (x->requires_gradient)
    {
        error = tensor_as_tensor(x, &x_gradient_i, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
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

    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);

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

    error = runtime_sine(x->buffer, result->buffer);
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
    tensor_t *x_gradient_j = NULL;

    if (x->requires_gradient)
    {
        error = tensor_as_tensor(x, &x_gradient_j, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_cosine(x_gradient_j, &x_gradient_i);
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
    tensor_destroy(x_gradient_j);

    return error;
}

static nw_error_t *cosine_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_cosine(x->buffer, result->buffer);
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
    tensor_t *x_gradient_k = NULL;

    if (x->requires_gradient)
    {
        error = tensor_as_tensor(x, &x_gradient_k, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor"), error);
            goto cleanup;
        }

        error = tensor_sine(x_gradient_k, &x_gradient_j);
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
    tensor_destroy(x_gradient_k);

    return error;
}

static nw_error_t *square_root_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_square_root(x->buffer, result->buffer);
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
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;
    tensor_t *x_gradient_k = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;

    if (x->requires_gradient)
    {
        switch (datatype)
        {
        case FLOAT32:
            error = tensor_constant_float32(2.0, &x_gradient_k, runtime);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
                goto cleanup;
            }
            break;
        case FLOAT64:
            error = tensor_constant_float64(2.0, &x_gradient_k, runtime);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
                goto cleanup;
            }
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
            goto cleanup;
        }

        error = tensor_as_tensor(result, &x_gradient_j, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor"), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_j, x_gradient_k, &x_gradient_i);
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

    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);
    tensor_destroy(x_gradient_j);
    tensor_destroy(x_gradient_k);

    return error;
}

static nw_error_t *reciprocal_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_reciprocal(x->buffer, result->buffer);
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
    tensor_t *x_gradient_k = NULL;

    if (x->requires_gradient)
    {
        error = tensor_as_tensor(result, &x_gradient_k, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_k, x_gradient_k, &x_gradient_i);
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
    tensor_destroy(x_gradient_k);

    return error;
}

static nw_error_t *contiguous_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_contiguous(x->buffer, result->buffer);
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

    error = runtime_negation(x->buffer, result->buffer);
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

    error = runtime_rectified_linear(x->buffer, result->buffer);
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
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;
    tensor_t *x_gradient_k = NULL;
    tensor_t *x_gradient_l = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    uint64_t *shape = x->buffer->view->shape;
    uint64_t rank = x->buffer->view->rank;

    if (x->requires_gradient)
    {
        switch (datatype)
        {
        case FLOAT32:
            error = tensor_constant_float32(0.0, &x_gradient_i, runtime);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
                goto cleanup;
            }
            break;
        case FLOAT64:
            error = tensor_constant_float64(0.0, &x_gradient_i, runtime);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
                goto cleanup;
            }
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
            goto cleanup;
        }

        error = tensor_as_tensor(x, &x_gradient_l, false);
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

        error = tensor_compare_greater(x_gradient_l, x_gradient_k, &x_gradient_j);
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

    tensor_destroy(x_gradient);
    tensor_destroy(x_gradient_i);
    tensor_destroy(x_gradient_j);
    tensor_destroy(x_gradient_k);
    tensor_destroy(x_gradient_l);

    return error;
}

static nw_error_t *sigmoid_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_sigmoid(x->buffer, result->buffer);
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
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;
    tensor_t *x_gradient_k = NULL;
    tensor_t *x_gradient_l = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;

    if (x->requires_gradient)
    {
        switch (datatype)
        {
        case FLOAT32:
            error = tensor_constant_float32(1.0, &x_gradient_i, runtime);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
                goto cleanup;
            }
            break;
        case FLOAT64:
            error = tensor_constant_float64(1.0, &x_gradient_i, runtime);
            if (error)
            {
                error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
                goto cleanup;
            }
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
            goto cleanup;
        }

        error = tensor_as_tensor(result, &x_gradient_j, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_subtraction(x_gradient_i, x_gradient_j, &x_gradient_k);
        if (error)
        {
            error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_j, x_gradient_k, &x_gradient_l);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_l, gradient, &x_gradient);
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
    tensor_destroy(x_gradient_k);
    tensor_destroy(x_gradient_l);

    return error;
}

nw_error_t *unary_operation_forward(unary_operation_t *unary_operation, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    tensor_t *x = unary_operation->x;
    unary_operation_type_t operation_type = unary_operation->operation_type;

    if (!*result)
    {
        error = tensor_empty_like(x, result, x->requires_gradient, false);    
        if (error)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        }
    }

    switch (operation_type)
    {
    case EXPONENTIAL_OPERATION:
        error = exponential_operation_forward(x, *result);
        break;
    case LOGARITHM_OPERATION:
        error = logarithm_operation_forward(x, *result);
        break;
    case SINE_OPERATION:
        error = sine_operation_forward(x, *result);
        break;
    case COSINE_OPERATION:
        error = cosine_operation_forward(x, *result);
        break;
    case SQUARE_ROOT_OPERATION:
        error = square_root_operation_forward(x, *result);
        break;
    case RECIPROCAL_OPERATION:
        error = reciprocal_operation_forward(x, *result);
        break;
    case CONTIGUOUS_OPERATION:
        error = contiguous_operation_forward(x, *result);
        break;
    case NEGATION_OPERATION:
        error = negation_operation_forward(x, *result);
        break;
    case RECTIFIED_LINEAR_OPERATION:
        error = rectified_linear_operation_forward(x, *result);
        break;
    case SIGMOID_OPERATION:
        error = sigmoid_operation_forward(x, *result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        tensor_destroy(*result);
        *result = NULL;
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply forward unary operation of type %s.",
                     unary_operation_type_string(operation_type)), error);
    }

    unary_operation->result = *result;
    
    return error;
}

nw_error_t *unary_operation_backward(unary_operation_t *unary_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x = unary_operation->x;
    tensor_t *result = unary_operation->result;
    unary_operation_type_t operation_type = unary_operation->operation_type;

    switch (operation_type)
    {
    case EXPONENTIAL_OPERATION:
        error = exponential_operation_backward(x, result, gradient);
        break;
    case LOGARITHM_OPERATION:
        error = logarithm_operation_backward(x, gradient);
        break;
    case SINE_OPERATION:
        error = sine_operation_backward(x, gradient);
        break;
    case COSINE_OPERATION:
        error = cosine_operation_backward(x, gradient);
        break;
    case SQUARE_ROOT_OPERATION:
        error = square_root_operation_backward(x, result, gradient);
        break;
    case RECIPROCAL_OPERATION:
        error = reciprocal_operation_backward(x, result, gradient);
        break;
    case CONTIGUOUS_OPERATION:
        error = contiguous_operation_backward(x, gradient);
        break;
    case NEGATION_OPERATION:
        error = negation_operation_backward(x, gradient);
        break;
    case RECTIFIED_LINEAR_OPERATION:
        error = rectified_linear_operation_backward(x, gradient);
        break;
    case SIGMOID_OPERATION:
        error = sigmoid_operation_backward(x, result, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_BACKWARD,
                     string_create("failed to apply backward unary operation of type %s.",
                     unary_operation_type_string(operation_type)), error);
    }
    
    return error;
}

nw_error_t *binary_operation_create(binary_operation_t **binary_operation,
                                    binary_operation_type_t binary_operation_type,
                                    const tensor_t *x,
                                    const tensor_t *y)
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
    (*binary_operation)->result = NULL;

    return NULL;
}

void binary_operation_destroy(binary_operation_t *binary_operation)
{
    if (binary_operation)
    {
        free(binary_operation);
    }
}

string_t binary_operation_type_string(binary_operation_type_t binary_operation_type)
{
    switch (binary_operation_type)
    {
    case ADDITION_OPERATION:
        return "ADDITION_OPERATION";
    case SUBTRACTION_OPERATION:
        return "SUBTRACTION_OPERATION";
    case MULTIPLICATION_OPERATION:
        return "MULTIPLICATION_OPERATION";
    case DIVISION_OPERATION:
        return "DIVISION_OPERATION";
    case POWER_OPERATION:
        return "POWER_OPERATION";
    case MATRIX_MULTIPLICATION_OPERATION:
        return "MATRIX_MULTIPLICATION_OPERATION";
    case COMPARE_EQUAL:
        return "COMPARE_EQUAL";
    case COMPARE_GREATER:
        return "COMPARE_GREATER";
    default:
        return "UKNOWN_OPERATION";
    }
}

static nw_error_t *addition_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_addition(x->buffer, y->buffer, result->buffer);
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

    error = runtime_subtraction(x->buffer, y->buffer, result->buffer);
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

    error = runtime_multiplication(x->buffer, y->buffer, result->buffer);
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
    tensor_t *x_gradient_i = NULL;
    tensor_t *y_gradient = NULL;
    tensor_t *y_gradient_i = NULL;

    if (x->requires_gradient)
    {
        error = tensor_as_tensor(y, &x_gradient_i, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_i, gradient, &x_gradient);
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
        error = tensor_as_tensor(x, &y_gradient_i, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(y_gradient_i, gradient, &y_gradient);
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
    tensor_destroy(x_gradient_i);
    tensor_destroy(y_gradient);
    tensor_destroy(y_gradient_i);

    return error;
}

static nw_error_t *division_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_division(x->buffer, y->buffer, result->buffer);
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
    tensor_t *x_gradient_i = NULL;
    tensor_t *y_gradient = NULL;
    tensor_t *y_gradient_i = NULL;
    tensor_t *y_gradient_j = NULL;
    tensor_t *y_gradient_k = NULL;
    tensor_t *y_gradient_l = NULL;
    tensor_t *y_gradient_m = NULL;
    tensor_t *y_gradient_n = NULL;

    if (x->requires_gradient)
    {
        error = tensor_as_tensor(y, &x_gradient_i, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_division(gradient, x_gradient_i, &x_gradient);
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
        error = tensor_as_tensor(x, &y_gradient_i, false); 
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_as_tensor(y, &y_gradient_j, false); 
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(y_gradient_j, y_gradient_j, &y_gradient_k);
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

        error = tensor_negation(y_gradient_l, &y_gradient_m);
        if (error)
        {
            error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(y_gradient_m, y_gradient_i, &y_gradient_n);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }
        
        error = tensor_multiplication(y_gradient_n, gradient, &y_gradient);
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
    tensor_destroy(y_gradient);
    tensor_destroy(y_gradient_i);
    tensor_destroy(y_gradient_j);
    tensor_destroy(y_gradient_k);
    tensor_destroy(y_gradient_l);
    tensor_destroy(y_gradient_m);
    tensor_destroy(y_gradient_n);

    return error;
}

static nw_error_t *power_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_power(x->buffer, y->buffer, result->buffer);
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
    tensor_t *x_gradient_k = NULL;
    tensor_t *x_gradient_l = NULL;
    tensor_t *x_gradient_m = NULL;
    tensor_t *y_gradient = NULL;
    tensor_t *y_gradient_i = NULL;
    tensor_t *y_gradient_j = NULL;
    tensor_t *y_gradient_k = NULL;
    tensor_t *y_gradient_l = NULL;

    if (x->requires_gradient)
    {
        error = tensor_as_tensor(x, &x_gradient_i, false); 
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_as_tensor(y, &x_gradient_j, false); 
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_as_tensor(result, &x_gradient_k, false); 
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_division(x_gradient_k, x_gradient_i, &x_gradient_l);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_l, x_gradient_j, &x_gradient_m);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(x_gradient_m, gradient, &x_gradient);
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
        error = tensor_as_tensor(x, &y_gradient_i, false); 
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensors."), error);
            goto cleanup;
        }

        error = tensor_as_tensor(result, &y_gradient_j, false); 
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensors."), error);
            goto cleanup;
        }

        error = tensor_logarithm(y_gradient_i, &y_gradient_k);
        if (error)
        {
            error = ERROR(ERROR_LOGARITHM, string_create("failed to log tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(y_gradient_k, y_gradient_j, &y_gradient_l);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }
        
        error = tensor_multiplication(y_gradient_l, gradient, &y_gradient);
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
    tensor_destroy(x_gradient_k);
    tensor_destroy(x_gradient_l);
    tensor_destroy(x_gradient_m);
    tensor_destroy(y_gradient);
    tensor_destroy(y_gradient_i);
    tensor_destroy(y_gradient_j);
    tensor_destroy(y_gradient_k);
    tensor_destroy(y_gradient_l);

    return error;
}

static nw_error_t *matrix_multiplication_operation_forward(tensor_t *x, tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_matrix_multiplication(x->buffer, y->buffer, result->buffer);
    if (error)
    {
        return ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to run matrix multiplication operation."), error);
    }

    return error;
}

nw_error_t *matrix_multiplication_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
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
    tensor_t *x_gradient_j = NULL;
    tensor_t *y_gradient = NULL;
    tensor_t *y_gradient_i = NULL;
    tensor_t *y_gradient_j = NULL;

    if (x->requires_gradient)
    {
        uint64_t rank = y->buffer->view->rank;

        error = tensor_as_tensor(y, &x_gradient_j, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_transpose(x_gradient_j, &x_gradient_i, rank - 2, rank - 1);
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
        uint64_t rank = x->buffer->view->rank;

        error = tensor_as_tensor(x, &y_gradient_j, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_transpose(y_gradient_j, &y_gradient_i, rank - 2, rank - 1);
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
    tensor_destroy(x_gradient_j);
    tensor_destroy(y_gradient);
    tensor_destroy(y_gradient_i);
    tensor_destroy(y_gradient_j);

    return error;
}

static nw_error_t *compare_equal_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_compare_equal(x->buffer, y->buffer, result->buffer);
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

    error = runtime_compare_greater(x->buffer, y->buffer, result->buffer);
    if (error)
    {
        return ERROR(ERROR_COMPARE_GREATER, string_create("failed to run compare greater operation."), error);
    }

    return error;
}

nw_error_t *binary_operation_forward(binary_operation_t *binary_operation, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(binary_operation->x, "binary_operation->x");
    CHECK_NULL_ARGUMENT(binary_operation->y, "binary_operation->y");
    CHECK_NULL_ARGUMENT(binary_operation->x->buffer, "binary_operation->x->buffer");
    CHECK_NULL_ARGUMENT(binary_operation->y->buffer, "binary_operation->y->buffer");
    CHECK_NULL_ARGUMENT(binary_operation->x->buffer->view, "binary_operation->x->buffer->view");
    CHECK_NULL_ARGUMENT(binary_operation->y->buffer->view, "binary_operation->y->buffer->view");
    CHECK_NULL_ARGUMENT(binary_operation->x->buffer->storage, "binary_operation->x->buffer->storage");
    CHECK_NULL_ARGUMENT(binary_operation->y->buffer->storage, "binary_operation->y->buffer->storage");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    tensor_t *x = binary_operation->x;
    tensor_t *y = binary_operation->y;
    uint64_t *x_shape = x->buffer->view->shape;
    uint64_t *y_shape = y->buffer->view->shape;
    uint64_t *shape;
    uint64_t x_rank = x->buffer->view->rank;
    uint64_t y_rank = y->buffer->view->rank;
    uint64_t rank;
    datatype_t x_datatype = x->buffer->storage->datatype;
    datatype_t y_datatype = y->buffer->storage->datatype;
    datatype_t datatype;
    runtime_t x_runtime = x->buffer->storage->runtime;
    runtime_t y_runtime = y->buffer->storage->runtime;
    runtime_t runtime;
    binary_operation_type_t operation_type = binary_operation->operation_type;
    bool_t requires_gradient = x->requires_gradient || y->requires_gradient;

    if (x_datatype != y_datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT, string_create("datatypes are incompatible."), NULL);
    }
    else
    {
        datatype = x_datatype;
    }

    if (x_runtime != y_runtime)
    {
        return ERROR(ERROR_RUNTIME_CONFLICT, string_create("runtimes are incompatible."), NULL);
    }
    else
    {
        runtime = x_runtime;
    }

    if (!*result)
    {
        if (operation_type != MATRIX_MULTIPLICATION_OPERATION)
        {
            if (!tensor_shapes_equal(x, y))
            {
                return ERROR(ERROR_SHAPE_CONFLICT, string_create("incompatible tensor shapes."), NULL);
            }
            else
            {
                shape = x_shape;
                rank = x_rank;
            }

            error = tensor_create_empty(shape, NULL, rank, result, requires_gradient, runtime, datatype);    
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            }
        }
        else
        {
            uint64_t rank = MAX(x_rank, y_rank);
            uint64_t shape[rank];

            error = matrix_multiplication_shape(x_shape, y_shape, shape, rank);
            if (error)
            {
                return ERROR(ERROR_SHAPE_CONFLICT, string_create("incompatible shapes for matrix multiplication."), error);
            }

            error = tensor_create_empty(shape, NULL, rank, result, requires_gradient, runtime, datatype);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            }
        }
    }

    switch (operation_type)
    {
    case ADDITION_OPERATION:
        error = addition_operation_forward(x, y, *result);
        break;
    case SUBTRACTION_OPERATION:
        error = subtraction_operation_forward(x, y, *result);
        break;
    case MULTIPLICATION_OPERATION:
        error = multiplication_operation_forward(x, y, *result);
        break;
    case DIVISION_OPERATION:
        error = division_operation_forward(x, y, *result);
        break;
    case POWER_OPERATION:
        error = power_operation_forward(x, y, *result);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        error = matrix_multiplication_operation_forward(x, y, *result);
        break;
    case COMPARE_EQUAL:
        error = compare_equal_operation_forward(x, y, *result);
        break;
    case COMPARE_GREATER:
        error = compare_greater_operation_forward(x, y, *result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unsupported binary operation type %d.",
                      (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        tensor_destroy(*result);
        *result = NULL;
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute binary operation forward pass of type %s.",
                     binary_operation_type_string(operation_type)), error);
    }

    binary_operation->result = *result;

    return error;
}

nw_error_t *binary_operation_backward(binary_operation_t *binary_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    binary_operation_type_t operation_type = binary_operation->operation_type;
    tensor_t *x = binary_operation->x;
    tensor_t *y = binary_operation->y;
    tensor_t *result = binary_operation->result;

    switch (operation_type)
    {
    case ADDITION_OPERATION:
        error = addition_operation_backward(x, y, gradient);
        break;
    case SUBTRACTION_OPERATION:
        error = subtraction_operation_backward(x, y, gradient);
        break;
    case MULTIPLICATION_OPERATION:
        error = multiplication_operation_backward(x, y, gradient);
        break;
    case DIVISION_OPERATION:
        error = division_operation_backward(x, y, gradient);
        break;
    case POWER_OPERATION:
        error = power_operation_backward(x, y, result, gradient);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        error = matrix_multiplication_operation_backward(x, y, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unsupported operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute binary operation backward pass of type %s.",
                     binary_operation_type_string(operation_type)), error);
    }
    
    return error;
}

nw_error_t *reduction_operation_create(reduction_operation_t **reduction_operation, 
                                       reduction_operation_type_t reduction_operation_type,
                                       const tensor_t *x,
                                       const uint64_t *axis,
                                       uint64_t length,
                                       bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");

    *reduction_operation = (reduction_operation_t *) malloc(sizeof(reduction_operation_t));
    if (!*reduction_operation)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                    string_create("failed to allocate %zu bytes.",
                    sizeof(reduction_operation_t)), NULL);
    }

    (*reduction_operation)->axis = (uint64_t *) malloc((size_t) (length * sizeof(uint64_t)));
    if (!(*reduction_operation)->axis)
    {
        free(*reduction_operation);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate size %zu bytes.",
                     (size_t) (length * sizeof(uint64_t))), NULL);
    }
    memcpy((*reduction_operation)->axis, axis, (size_t) (length * sizeof(uint64_t)));

    (*reduction_operation)->operation_type = reduction_operation_type;
    (*reduction_operation)->x = (tensor_t *) x; 
    (*reduction_operation)->length = length;
    (*reduction_operation)->keep_dimension = keep_dimension;
    (*reduction_operation)->result = NULL;

    return NULL;
}

void reduction_operation_destroy(reduction_operation_t *reduction_operation)
{
    if (reduction_operation)
    {
        free(reduction_operation->axis);
        free(reduction_operation);
    }
}

string_t reduction_operation_type_string(reduction_operation_type_t reduction_operation_type)
{
    switch (reduction_operation_type)
    {
    case SUMMATION_OPERATION:
        return "SUMMATION_OPERATION";
    case MAXIMUM_OPERATION:
        return "MAXIMUM_OPERATION";
    default:
        return "UNKNOWN_OPERATION";
    }
}


static nw_error_t *summation_operation_forward(tensor_t *x, uint64_t *axis, uint64_t length, tensor_t *result, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_summation(x->buffer, axis, length, result->buffer, keep_dimension);
    if (error)
    {
        return ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
    }

    return NULL; 
}

static nw_error_t *summation_operation_backward(tensor_t *x, uint64_t *axis, uint64_t length, tensor_t *gradient, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    uint64_t rank = x->buffer->view->rank;
    uint64_t *shape = x->buffer->view->shape;
    uint64_t recovered_shape[rank];
    uint64_t recovered_strides[rank];
    uint64_t reduced_rank = gradient->buffer->view->rank;
    uint64_t *reduced_shape = gradient->buffer->view->shape;
    uint64_t *reduced_strides = gradient->buffer->view->strides;

    if (x->requires_gradient)
    {
        if (!keep_dimension)
        {
            error = reduce_recover_dimensions(reduced_shape, reduced_rank, reduced_strides,
                                              recovered_shape, rank, recovered_strides, axis, length);
            if (error)
            {
                error = ERROR(ERROR_REDUCTION, string_create("failed to recover reduce dimensions."), error);
                goto cleanup;
            }

            error = tensor_reshape(gradient, &x_gradient_i, recovered_shape, rank);
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

        error = tensor_expand(x_gradient_i, shape, rank, &x_gradient);
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

    return error; 
}

static nw_error_t *maximum_operation_forward(tensor_t *x, uint64_t *axis, uint64_t length, tensor_t *result, bool_t keep_dimension)
{

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = runtime_maximum(x->buffer, axis, length, result->buffer, keep_dimension);
    if (error)
    {
        return ERROR(ERROR_MAXIMUM, string_create("failed to get maximum of tensor."), error);
    }

    return error; 

}

static nw_error_t *maximum_operation_backward(tensor_t *x, uint64_t *axis, uint64_t length, tensor_t *result, tensor_t *gradient, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;
    tensor_t *x_gradient_j = NULL;
    tensor_t *x_gradient_k = NULL;
    tensor_t *x_gradient_l = NULL;
    tensor_t *x_gradient_m = NULL;
    tensor_t *x_gradient_n = NULL;
    tensor_t *x_gradient_o = NULL;
    tensor_t *x_gradient_p = NULL;
    uint64_t rank = x->buffer->view->rank;
    uint64_t *shape = x->buffer->view->shape;

    if (x->requires_gradient)
    {
        error = tensor_as_tensor(x, &x_gradient_i, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_as_tensor(result, &x_gradient_j, false);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        if (!keep_dimension)
        {
            uint64_t recovered_result_shape[rank];
            uint64_t recovered_result_strides[rank];
            uint64_t recovered_gradient_shape[rank];
            uint64_t recovered_gradient_strides[rank];
            uint64_t reduced_result_rank = result->buffer->view->rank;
            uint64_t *reduced_result_shape = result->buffer->view->shape;
            uint64_t *reduced_result_strides = result->buffer->view->strides;
            uint64_t reduced_gradient_rank = gradient->buffer->view->rank;
            uint64_t *reduced_gradient_shape = gradient->buffer->view->shape;
            uint64_t *reduced_gradient_strides = gradient->buffer->view->strides;

            error = reduce_recover_dimensions(reduced_result_shape, reduced_result_rank, reduced_result_strides,
                                              recovered_result_shape, rank, recovered_result_strides, axis, length);
            if (error)
            {
                return ERROR(ERROR_REDUCTION, string_create("failed to recover from reduce dimensions."), error);
                goto cleanup;
            }

            error = reduce_recover_dimensions(reduced_gradient_shape, reduced_gradient_rank, reduced_gradient_strides,
                                              recovered_gradient_shape, rank, recovered_gradient_strides, axis, length);
            if (error)
            {
                return ERROR(ERROR_REDUCTION, string_create("failed to recover from reduce dimensions."), error);
                goto cleanup;
            }

            error = tensor_reshape(x_gradient_j, &x_gradient_k, recovered_result_shape, rank);
            if (error)
            {
                error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
                goto cleanup;
            }
            
            error = tensor_reshape(gradient, &x_gradient_l, recovered_gradient_shape, rank);
            if (error)
            {
                error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
                goto cleanup;
            }
        }
        else
        {
            x_gradient_k = x_gradient_j;
            x_gradient_l = gradient;
        }

        error = tensor_expand(x_gradient_k, shape, rank, &x_gradient_m);
        if (error)
        {
            error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
            goto cleanup;
        }

        error = tensor_compare_equal(x_gradient_m, x_gradient_i, &x_gradient_n);
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

    tensor_destroy(x_gradient_i);    
    if (x_gradient_j != x_gradient_k)
    {
        tensor_destroy(x_gradient_j);    
    }
    tensor_destroy(x_gradient_k);    
    if (x_gradient_l != gradient)
    {
        tensor_destroy(x_gradient_l);    
    }
    tensor_destroy(x_gradient_m);    
    tensor_destroy(x_gradient_n);    
    tensor_destroy(x_gradient_o);    
    tensor_destroy(x_gradient_p);    
    tensor_destroy(x_gradient);    

    return error; 
}

nw_error_t *reduction_operation_forward(reduction_operation_t *reduction_operation, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(reduction_operation->x, "reduction_operation->x");
    CHECK_NULL_ARGUMENT(reduction_operation->x->buffer, "reduction_operation->x->buffer");
    CHECK_NULL_ARGUMENT(reduction_operation->x->buffer->view, "reduction_operation->x->buffer->view");
    CHECK_NULL_ARGUMENT(reduction_operation->x->buffer->storage, "reduction_operation->x->buffer->storage");

    nw_error_t *error = NULL;
    tensor_t *x = reduction_operation->x;
    uint64_t rank = x->buffer->view->rank;
    uint64_t *shape = x->buffer->view->shape;
    uint64_t *strides = x->buffer->view->strides;
    uint64_t *axis = reduction_operation->axis;
    uint64_t length = reduction_operation->length;
    bool_t keep_dimension = reduction_operation->keep_dimension;
    bool_t requires_gradient = x->requires_gradient;
    uint64_t reduced_rank = (keep_dimension) ? rank : (rank - length); 
    uint64_t reduced_shape[reduced_rank];
    uint64_t reduced_strides[reduced_rank];
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    reduction_operation_type_t operation_type = reduction_operation->operation_type;

    if (rank < length)
    {
        return ERROR(ERROR_RANK_CONFLICT, string_create("reduction axis length greater than rank of tensor."), NULL);
    }

    error = reduce(shape, rank, strides, reduced_shape, reduced_rank, reduced_strides, axis, length, keep_dimension);
    if (error)
    {
        return ERROR(ERROR_REDUCTION, string_create("failed to reduce tensor."), error);
    }

    error = tensor_create_empty(reduced_shape, reduced_strides, reduced_rank, result, requires_gradient, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to reduce tensor."), error);
    }

    switch (operation_type)
    {
    case SUMMATION_OPERATION:
        error = summation_operation_forward(x, axis, length, *result, keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        error = maximum_operation_forward(x, axis, length, *result, keep_dimension);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        tensor_destroy(*result);
        *result = NULL;
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute reduction operation forward pass of type %s.",
                     reduction_operation_type_string(operation_type)), error);
    }

    reduction_operation->result = *result;

    return error;
}

nw_error_t *reduction_operation_backward(reduction_operation_t *reduction_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x = reduction_operation->x;
    tensor_t *result = reduction_operation->result;
    uint64_t *axis = reduction_operation->axis;
    uint64_t length = reduction_operation->length;
    bool_t keep_dimension = reduction_operation->keep_dimension;
    reduction_operation_type_t operation_type = reduction_operation->operation_type;

    switch (reduction_operation->operation_type)
    {
    case SUMMATION_OPERATION:
        error = summation_operation_backward(x, axis, length, gradient, keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        error = maximum_operation_backward(x, axis, length, result, gradient, keep_dimension);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute reduction operation backward pass %s.",
                     reduction_operation_type_string(operation_type)), error);
    }

    return error;
}

nw_error_t *structure_operation_create(structure_operation_t **structure_operation,
                                       structure_operation_type_t structure_operation_type,
                                       const tensor_t *x,
                                       const uint64_t *arguments,
                                       uint64_t length)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(x, "x");

    *structure_operation = (structure_operation_t *) malloc(sizeof(structure_operation_t));
    if (!*structure_operation)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(structure_operation_t)), NULL);
    }

    (*structure_operation)->arguments = (uint64_t *) malloc((size_t) (length * sizeof(uint64_t)));
    if (!(*structure_operation)->arguments)
    {
        free(*structure_operation);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate %zu bytes.",
                     (size_t) (length * sizeof(uint64_t))), NULL);
    }
    memcpy((*structure_operation)->arguments, arguments, (size_t) (length * sizeof(uint64_t)));

    (*structure_operation)->operation_type = structure_operation_type;
    (*structure_operation)->x = (tensor_t *) x; 
    (*structure_operation)->length = length;
    (*structure_operation)->result = NULL;

    return NULL;
}

void structure_operation_destroy(structure_operation_t *structure_operation)
{
    if (structure_operation)
    {
        free(structure_operation->arguments);
        free(structure_operation);
    }
}

string_t structure_operation_type_string(structure_operation_type_t structure_operation_type)
{
    switch (structure_operation_type)
    {
    case EXPAND_OPERATION:
        return "EXPAND_OPERATION";
    case PERMUTE_OPERATION:
        return "PERMUTE_OPERATION";
    case RESHAPE_OPERATION:
        return "RESHAPE_OPERATION";
    case SLICE_OPERATION:
        return "SLICE_OPERATION";
    case PADDING_OPERATION:
        return "PADDING_OPERATION";
    default:
        return "UNKNOWN_OPERATION";
    }
}

static nw_error_t *expand_operation_forward(tensor_t *x, uint64_t *shape, uint64_t length, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    uint64_t strides[length];
    view_t *view = NULL;

    error = broadcast_strides(x->buffer->view->shape, x->buffer->view->rank, x->buffer->view->strides, shape, length, strides);
    if (error)
    {
        return ERROR(ERROR_EXPAND, string_create("failed to expand strides"), error);
    }

    error = view_create(&view, x->buffer->view->offset, length, shape, strides);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&result->buffer, view, x->buffer->storage, false);
    if (error)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return error;
}

static nw_error_t *expand_operation_backward(tensor_t *x, uint64_t *shape, uint64_t length, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    uint64_t length_keep_dimension = 0;
    uint64_t length_remove_dimension = 0;
    tensor_t *x_gradient = NULL;
    tensor_t *x_gradient_i = NULL;

    if (x->requires_gradient)
    {
        error = reduce_axis_length(x->buffer->view->shape, x->buffer->view->rank, shape, length, &length_keep_dimension, &length_remove_dimension);
        if (error)
        {
            error = ERROR(ERROR_REDUCTION, string_create("failed to get reduction axis lengths."), error);
            goto cleanup;
        }

        uint64_t axis_keep_dimension[length_keep_dimension];
        uint64_t axis_remove_dimension[length_remove_dimension];

        error = reduce_axis(x->buffer->view->shape, x->buffer->view->rank, shape, length, axis_keep_dimension, axis_remove_dimension);
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

    return error;
}

static nw_error_t *permute_operation_forward(tensor_t *x, uint64_t *axis, uint64_t rank, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error;
    uint64_t shape[rank];
    uint64_t strides[rank];
    view_t *view;

    error = permute(x->buffer->view->shape, x->buffer->view->strides, shape, strides, axis, rank);
    if (error)
    {
        return ERROR(ERROR_PERMUTE, string_create("failed to permute shape and strides."), error);
    }

    error = view_create(&view, x->buffer->view->offset, rank, shape, strides);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&result->buffer, view, x->buffer->storage, false);
    if (error)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return error;
}

static nw_error_t *permute_operation_backward(tensor_t *x, uint64_t *axis, uint64_t rank, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    uint64_t gradient_axis[rank];
    tensor_t *x_gradient = NULL;

    if (x->requires_gradient)
    {
        error = reverse_permute(axis, rank, gradient_axis);
        if (error)
        {
            error = ERROR(ERROR_PERMUTE, string_create("failed to get permuted axis."), error);
            goto cleanup;
        }

        error = tensor_permute(gradient, &x_gradient, gradient_axis, rank);
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

static nw_error_t *reshape_operation_forward(tensor_t *x, uint64_t *shape, uint64_t rank, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    view_t *view = NULL;

    if (!tensor_is_contiguous(x))
    {
        return ERROR(ERROR_CONTIGUOUS, string_create("cannot reshape a non-contiguous tensor."), NULL);
    }

    error = view_create(&view, x->buffer->view->offset, rank, shape, NULL);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&result->buffer, view, x->buffer->storage, false);
    if (error)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
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

// static nw_error_t *slice_operation_forward(tensor_t *x, uint64_t *arguments, uint64_t length, tensor_t **result)
// {
//     CHECK_NULL_ARGUMENT(x, "x");
//     CHECK_NULL_ARGUMENT(arguments, "arguments");
//     CHECK_NULL_ARGUMENT(result, "result");

//     view_t *view;
//     uint64_t offset = 0;
//     uint64_t *shape = (uint64_t *) malloc(x->buffer->view->rank * sizeof(uint64_t));
//     if (shape == NULL)
//     {
//         return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed allocate shape of size %zu bytes.", x->buffer->view->rank * sizeof(uint64_t)), NULL);
//     }

//     nw_error_t *error = slice_offset(x->buffer->view->strides, x->buffer->view->rank, &offset, arguments, length);
//     if (shape == NULL)
//     {
//         return ERROR(ERROR_SLICE, string_create("failed to compute slice offset." ), NULL);
//     }

//     error = slice_shape(x->buffer->view->shape, x->buffer->view->rank, shape, length, arguments, length);
//     if (shape == NULL)
//     {
//         return ERROR(ERROR_SLICE, string_create("failed to compute slice shape." ), NULL);
//     }

//     error = view_create(&view, offset, x->buffer->view->rank, shape, x->buffer->view->strides);
//     if (error)
//     {
//         return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
//     }
//     free(shape);

//     error = buffer_create(&result->buffer, view, x->buffer->storage, false);
//     if (error)
//     {
//         view_destroy(view);
//         return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
//     }

//     return NULL;
// }

// static nw_error_t *slice_operation_backward(tensor_t *x, uint64_t *arguments, uint64_t length, tensor_t *gradient)
// {
//     CHECK_NULL_ARGUMENT(x, "x");
//     CHECK_NULL_ARGUMENT(gradient, "gradient");

//     if (x->requires_gradient)
//     {
//         tensor_t *x_gradient;

//         nw_error_t *error = tensor_create_default(&x_gradient);
//         if (error)
//         {
//             return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
//         }

//         uint64_t *new_arguments = (uint64_t *) malloc(length * sizeof(uint64_t));
//         if (new_arguments == NULL)
//         {
//             return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate new arguments of size %zu.", length * sizeof(uint64_t)), NULL);
//         }

//         error = reverse_slice(x->buffer->view->shape, x->buffer->view->rank, arguments, length, new_arguments, length);
//         if (error)
//         {
//             return ERROR(ERROR_SLICE, string_create("failed to compute padding arguments."), error);
//         }

//         error = tensor_padding(gradient, x_gradient, new_arguments, length);
//         if (error)
//         {
//             return ERROR(ERROR_PADDING, string_create("failed to successfully run padding operation."), error);
//         }
//         free(new_arguments);

//         error = tensor_accumulate_gradient(x, x_gradient);
//         if (error) 
//         {
//             return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
//         }
//     }

//     return NULL;
// }

// static nw_error_t *padding_operation_forward(tensor_t *x, uint64_t *arguments, uint64_t length, tensor_t **result)
// {
//     CHECK_NULL_ARGUMENT(x, "x");
//     CHECK_NULL_ARGUMENT(arguments, "arguments");
//     CHECK_NULL_ARGUMENT(result, "result");

//     uint64_t *padding_shape = (uint64_t *) malloc(x->buffer->view->rank * sizeof(uint64_t));
//     if (padding_shape == NULL)
//     {
//         return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate padding shape of size %zu bytes.", x->buffer->view->rank * sizeof(uint64_t)), NULL);
//     }

//     uint64_t *slice_arguments = (uint64_t *) malloc(length * sizeof(uint64_t));
//     if (slice_arguments == NULL)
//     {
//         return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate slice arguments of size %zu bytes.", length * sizeof(uint64_t)), NULL);
//     }

//     nw_error_t *error = padding(x->buffer->view->shape, x->buffer->view->rank, padding_shape, x->buffer->view->rank, arguments, length);
//     if (error)
//     {
//         return ERROR(ERROR_PADDING, string_create("failed to compute resultant padding shape."), error);
//     }

//     error = reverse_padding(x->buffer->view->shape, x->buffer->view->rank, arguments, length, slice_arguments, length);
//     if (error)
//     {
//         return ERROR(ERROR_PADDING, string_create("failed to compute slice arguments."), error);
//     }

//     uint64_t offset = 0;
//     error = slice_offset(x->buffer->view->strides, x->buffer->view->rank, &offset, slice_arguments, length);
//     if (error)
//     {
//         return ERROR(ERROR_SLICE, string_create("failed to compute slice offset."), error);
//     }

//     view_t *view;
//     storage_t *storage;
//     buffer_t *buffer;

//     error = view_create(&view, 0, x->buffer->view->rank, padding_shape, NULL);
//     if (error)
//     {
//         return ERROR(ERROR_CREATE, "failed to create view.", error);
//     }

//     error = storage_create(&storage, x->buffer->storage->runtime, x->buffer->storage->datatype, x->buffer->storage->n)

//     error = buffer_create(&buffer, x->buffer->runtime, x->buffer->datatype, view, NULL, 0, true);
//     if (error)
//     {
//         return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
//     }
//     memset(buffer->data, 0.0, buffer->size); 


//     view_t *sliced_view;
//     buffer_t *sliced_buffer;
//     error = view_create(&sliced_view, offset, x->buffer->view->rank, x->buffer->view->shape, NULL);
//     if (error)
//     {
//         return ERROR(ERROR_CREATE, "failed to create view.", error);
//     }

//     error = buffer_create(&sliced_buffer, x->buffer->runtime, x->buffer->datatype, sliced_view, buffer->data, buffer->n, false);
//     if (error)
//     {
//         return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
//     }

//     error = runtime_copy(x->buffer, sliced_buffer);
//     if (error)
//     {
//         return ERROR(ERROR_COPY, string_create("failed to copy tensor contents."), error);
//     }

//     result->buffer = buffer;

//     return NULL;
// }

// static nw_error_t *padding_operation_backward(tensor_t *x, uint64_t *arguments, uint64_t length, tensor_t *gradient)
// {
//     CHECK_NULL_ARGUMENT(x, "x");
//     CHECK_NULL_ARGUMENT(arguments, "arguments");
//     CHECK_NULL_ARGUMENT(gradient, "gradient");

//     if (x->requires_gradient)
//     {
//         nw_error_t *error = NULL;
//         tensor_t *x_gradient = NULL;
//         uint64_t *new_arguments = NULL;

//         error = tensor_create_default(&x_gradient);
//         if (error)
//         {
//             return ERROR(ERROR_CREATE,
//                          string_create("failed to create tensor x_gradient."),
//                          error);
//         }

//         new_arguments = (uint64_t *) malloc((size_t) (length * sizeof(uint64_t)));
//         if (new_arguments == NULL)
//         {
//             return ERROR(ERROR_MEMORY_ALLOCATION,
//                          string_create("failed to allocate new arguments of size %lu bytes.",
//                          (unsigned long) (length * sizeof(uint64_t))),
//                          NULL);
//         }

//         error = reverse_padding(x->buffer->view->shape,
//                                 x->buffer->view->rank,
//                                 arguments, length,
//                                 new_arguments, length);
//         if (error)
//         {
//             free(new_arguments);
//             return ERROR(ERROR_PADDING,
//                          string_create("cannot compute slice arguments from shape and padding arguments."),
//                          error);
//         }

//         error = tensor_slice(gradient, x_gradient, new_arguments, length);
//         if (error)
//         {
//             tensor_destroy(x_gradient);
//             free(new_arguments);
//             return ERROR(ERROR_SLICE,
//                          string_create("cannot slice tensor shape with arguments."),
//                          error);
//         }

//         error = tensor_accumulate_gradient(x, x_gradient);
//         if (error) 
//         {
//             free(new_arguments);
//             return ERROR(ERROR_ADDITION,
//                          string_create("failed to accumulate gradient."),
//                          error);
//         }

//         free(new_arguments);
//     }

//     return NULL;
// }

/**
 * @brief Apply structure operation forward.
 * @param structure_operation Structure operation to execute.
 * @return Error if `structure_operation` is NULL.
 *         Error if `structure_operation` failed to run.
 *         Error if operation type is unknown.
 *         NULL if `structure_operation` successfully executed.
 */
nw_error_t *structure_operation_forward(structure_operation_t *structure_operation, tensor_t **result)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(structure_operation->x, "structure_operation->x");
    CHECK_NULL_ARGUMENT(structure_operation->x->buffer, "structure_operation->x->buffer");
    CHECK_NULL_ARGUMENT(structure_operation->x->buffer->view, "structure_operation->x->buffer->view");

    nw_error_t *error = NULL;
    tensor_t *x = structure_operation->x;
    uint64_t *arguments = structure_operation->arguments;
    uint64_t length = structure_operation->length;
    structure_operation_type_t operation_type = structure_operation->operation_type;
    uint64_t *shape = x->buffer->view->shape;
    uint64_t rank = x->buffer->view->rank;

    switch (operation_type)
    {
    case EXPAND_OPERATION:
        if (shapes_equal(shape, rank, arguments, length))
        {
            error = tensor_as_tensor(x, result, x->requires_gradient);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor"), error);
            }
        }
        else
        {
            error = tensor_create(result, NULL, NULL, NULL, x->requires_gradient);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor"), error);
            }
            error = expand_operation_forward(x, arguments, length, *result);
        }
        break;
    case PERMUTE_OPERATION:
        error = tensor_create(result, NULL, NULL, NULL, x->requires_gradient);
        if (error)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create tensor"), error);
        }
        error = permute_operation_forward(x, arguments, length, *result);
        break;
    case RESHAPE_OPERATION:
        if (shapes_equal(shape, rank, arguments, length))
        {
            error = tensor_as_tensor(x, result, x->requires_gradient);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor"), error);
            }
        }
        else
        {
            error = tensor_create(result, NULL, NULL, NULL, x->requires_gradient);
            if (error)
            {
                return ERROR(ERROR_CREATE, string_create("failed to create tensor"), error);
            }
            error = reshape_operation_forward(x, arguments, length, *result);
        }
        break;
    // case SLICE_OPERATION:
    //     error = slice_operation_forward(x, arguments, length, *result);
    //     break;
    // case PADDING_OPERATION:
    //     error = padding_operation_forward(x, arguments, length, *result);
    //     break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        tensor_destroy(*result);
        *result = NULL;
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute structure operation forward pass of type %s.",
                     structure_operation_type_string(operation_type)), error);
    }

    structure_operation->result = *result;

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
nw_error_t *structure_operation_backward(structure_operation_t *structure_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;
    tensor_t *x = structure_operation->x;
    uint64_t *arguments = structure_operation->arguments;
    uint64_t length = structure_operation->length;
    structure_operation_type_t operation_type = structure_operation->operation_type;

    switch (operation_type)
    {
    case EXPAND_OPERATION:
        error = expand_operation_backward(x, arguments, length, gradient);
        break;
    case PERMUTE_OPERATION:
        error = permute_operation_backward(x, arguments, length, gradient);
        break;
    case RESHAPE_OPERATION:
        error = reshape_operation_backward(x, gradient);
        break;
    // case SLICE_OPERATION:
    //     error = slice_operation_backward(x, arguments, length, gradient);
    //     break;
    // case PADDING_OPERATION:
    //     error = padding_operation_backward(x, arguments, length, gradient);
    //     break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute structure operation backward pass of type %s.",
                     structure_operation_type_string(operation_type)), error);
    }

    return error;
}
