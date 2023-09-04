/**
 * @file function.c
 * @brief Mid-level Operations and Automatic Differentiation Engine
 */

#include <function.h>
#include <tensor.h>
#include <string.h>
#include <view.h>
#include <buffer.h>

/**
 * @brief The function constructor.
 * @param function The address of the pointer to the function being instantiated.
 * @param operation The operation the function applies.
 * @param operation_type The type of operation the function applies. 
 * @return Error in `function` or `operation` is NULL.
 *         Error if failed to allocate memory for `function`.
 *         NULL if function is created successfully.
 */
nw_error_t *function_create(function_t **function,
                            operation_t *operation,
                            operation_type_t operation_type)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(operation, "operation");

    *function = (function_t *) malloc(sizeof(function_t));
    if (*function == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, 
                     string_create("failed to allocate function of size %lu bytes.",
                     (unsigned long) sizeof(function_t)), 
                     NULL);
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
    if (function == NULL)
    {
        return;
    }

    operation_destroy(function->operation, function->operation_type);
    free(function);
}

/**
 * @brief Execute the operation of a generic function.
 * @param operation_type The type of operation being applied.
 * @param type_operation The generic operation being applied.
 * @return Error if `type_operation` is NULL.
 *         Error if function failed to execute.
 *         NULL if function executed successfully.
 */
static nw_error_t *apply_function(operation_type_t operation_type,
                                  void *type_operation)
{
    CHECK_NULL_ARGUMENT(type_operation, "type_operation");

    nw_error_t *error = NULL;
    operation_t *operation = NULL;
    function_t *function = NULL;

    error = operation_create(&operation, operation_type, type_operation);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create operation of type %s.",
                     operation_type_string(operation_type)),
                     error);
    }

    error = function_create(&function, operation, operation_type);
    if (error != NULL)
    {
        free(operation);
        return ERROR(ERROR_CREATE,
                     string_create("failed to create function operation of type %s.",
                     operation_type_string(operation_type)), 
                     error);
    }

    error = function_forward(function);
    if (error != NULL)
    {
        free(operation);
        free(function);
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute function forward pass of type %s.",
                     operation_type_string(operation_type)), 
                     error);
    }

    return NULL;
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
nw_error_t *apply_function_unary(unary_operation_type_t unary_operation_type,
                                 const tensor_t *x,
                                 tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    unary_operation_t *unary_operation = NULL;

    error = unary_operation_create(&unary_operation, unary_operation_type, x, result);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create unary operation of type %s.",
                     unary_operation_type_string(unary_operation_type)), 
                     error);
    }

    error = apply_function(UNARY_OPERATION, (void *) unary_operation);
    if (error != NULL)
    {
        unary_operation_destroy(unary_operation);
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply unary function of type %s.",
                     unary_operation_type_string(unary_operation_type)), 
                     error);
    }
    
    return NULL;
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
nw_error_t *apply_function_binary(binary_operation_type_t binary_operation_type,
                                  const tensor_t *x,
                                  const tensor_t *y,
                                  tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    tensor_t *x_brodcasted = NULL;
    tensor_t *y_brodcasted = NULL;
    binary_operation_t *binary_operation = NULL;

    error = tensor_create_empty(&x_brodcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create empty tensor x_broadcasted."),
                     error);
    }

    error = tensor_create_empty(&y_brodcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create empty tensor y_broadcasted."),
                     error);
    }
    
    error = tensor_broadcast(x, y, x_brodcasted, y_brodcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_BROADCAST,
                     string_create("failed to broadcast tensors x and y."),
                     error);
    }

    error = binary_operation_create(&binary_operation,
                                    binary_operation_type,
                                    x_brodcasted,
                                    y_brodcasted,
                                    result);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create binary operation of type %s.",
                     binary_operation_type_string(binary_operation_type)), 
                     error);
    }

    error = apply_function(BINARY_OPERATION, (void *) binary_operation);
    if (error != NULL)
    {
        binary_operation_destroy(binary_operation);
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply binary function of type %s.",
                     binary_operation_type_string(binary_operation_type)),
                     error);
    }
    
    return NULL;
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
                                     tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    reduction_operation_t *reduction_operation = NULL;

    error = reduction_operation_create(&reduction_operation,
                                       reduction_operation_type,
                                       x, axis, length, keep_dimension, result);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create reduction operation of type %s.",
                     reduction_operation_type_string(reduction_operation_type)),
                     error);
    }

    error = apply_function(REDUCTION_OPERATION, (void *) reduction_operation);
    if (error != NULL)
    {
        reduction_operation_destroy(reduction_operation);
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply reduction function of type %s.",
                     reduction_operation_type_string(reduction_operation_type)),
                     error);
    }
    
    return NULL;
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
                                     tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;
    structure_operation_t *structure_operation = NULL;

    error = structure_operation_create(&structure_operation,
                                       structure_operation_type,
                                       x, arguments, length, result);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create structure operation of type %s.",
                     structure_operation_type_string(structure_operation_type)),
                     error);
    }

    error = apply_function(STRUCTURE_OPERATION, (void *) structure_operation);
    if (error != NULL)
    {
        structure_operation_destroy(structure_operation);
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply structure function of type %s.",
                     structure_operation_type_string(structure_operation_type)),
                     error);
    }
    
    return NULL;
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
nw_error_t *function_forward(function_t *function)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(function->operation, "function->operation");

    nw_error_t *error = operation_forward(function->operation, function->operation_type);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, 
                     string_create("failed to execute operation forward pass of type %s.",
                     operation_type_string(function->operation_type)), 
                     error);
    }

    switch (function->operation_type)
    {
    case UNARY_OPERATION:
        CHECK_NULL_ARGUMENT(function->operation->unary_operation,
                            "function->operation->unary_operation");
        CHECK_NULL_ARGUMENT(function->operation->unary_operation->result,
                            "function->operation->unary_operation->result");
        function->operation->unary_operation->result->context = function;
        break;
    case BINARY_OPERATION:
        CHECK_NULL_ARGUMENT(function->operation->binary_operation,
                            "function->operation->binary_operation");
        CHECK_NULL_ARGUMENT(function->operation->binary_operation->result,
                            "function->operation->binary_operation->result");
        function->operation->binary_operation->result->context = function;
        break;
    case REDUCTION_OPERATION:
        CHECK_NULL_ARGUMENT(function->operation->reduction_operation,
                            "function->operation->reduction_operation");
        CHECK_NULL_ARGUMENT(function->operation->reduction_operation->result,
                            "function->operation->reduction_operation->result");
        function->operation->reduction_operation->result->context = function;
        break;
    case STRUCTURE_OPERATION:
        CHECK_NULL_ARGUMENT(function->operation->structure_operation,
                            "function->operation->structure_operation");
        CHECK_NULL_ARGUMENT(function->operation->structure_operation->result,
                            "function->operation->structure_operation->result");
        function->operation->structure_operation->result->context = function;
        break;
    default:
        return ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                     string_create("unknown operation type %d.",
                     (int) function->operation_type),
                     NULL);
    }
    
    return NULL;
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

    nw_error_t *error = operation_backward(function->operation,
                                           function->operation_type,
                                           gradient);
    if (error != NULL)
    {
        return ERROR(ERROR_BACKWARD,
                     string_create("failed to execute operation backward pass of type %s.",
                     operation_type_string(function->operation_type)),
                     error);
    }

    return NULL;
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
nw_error_t *operation_create(operation_t **operation,
                             operation_type_t operation_type,
                             void *type_operation)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(type_operation, "type_operation");

    *operation = (operation_t *) malloc(sizeof(operation_t));
    if (*operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate operation of size %lu bytes.",
                     (unsigned int) sizeof(operation_t)),
                     NULL);
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
        return ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                     string_create("unknown operation type %d.",
                     (int) operation_type),
                     NULL);
        break;
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
    if (operation == NULL)
    {
        return;
    }

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
nw_error_t *operation_forward(operation_t *operation, operation_type_t operation_type)
{
    CHECK_NULL_ARGUMENT(operation, "operation");

    nw_error_t *error = NULL;
    switch (operation_type)
    {
    case UNARY_OPERATION:
        error = unary_operation_forward(operation->unary_operation);
        break;
    case BINARY_OPERATION:
        error = binary_operation_forward(operation->binary_operation);
        break;
    case REDUCTION_OPERATION:
        error = reduction_operation_forward(operation->reduction_operation);
        break;
    case STRUCTURE_OPERATION:
        error = structure_operation_forward(operation->structure_operation);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown operation type %d.",
                      (int) operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute operation forward pass of type %s.", 
                     operation_type_string(operation_type)),
                     error);
    }

    return NULL;
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
nw_error_t *operation_backward(operation_t *operation,
                               operation_type_t operation_type,
                               tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error;
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
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown operation type %d.",
                      (int) operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_BACKWARD,
                     string_create("failed to execute operation backward pass of %s.",
                     operation_type_string(operation_type)),
                     error);
    }

    return NULL;
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
                                   const tensor_t *x,
                                   tensor_t *result)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    *unary_operation = (unary_operation_t *) malloc(sizeof(unary_operation_t));
    if (*unary_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate unary operation of size %lu bytes.",
                     (unsigned long) sizeof(unary_operation_t)),
                     NULL);
    }

    (*unary_operation)->operation_type = unary_operation_type;
    (*unary_operation)->x = (tensor_t *) x; 
    (*unary_operation)->result = result;

    return NULL;
}

/**
 * @brief Destroy a unary operation.
 * @param unary_operation The unary operation created with `unary_operation_create` to free.
 *                        Argument can be NULL.
 */
void unary_operation_destroy(unary_operation_t *unary_operation)
{
    if (unary_operation == NULL)
    {
        return;
    }

    free(unary_operation);
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
    case COPY_OPERATION:
        return "COPY_OPERATION";
    case CONTIGUOUS_OPERATION:
        return "CONTIGUOUS_OPERATION";
    case NEGATION_OPERATION:
        return "NEGATION_OPERATION";
    case RECTIFIED_LINEAR_OPERATION:
        return "RECTIFIED_LINEAR_OPERATION";
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

    error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create empty tensor."),
                     error);
    }

    error = runtime_exponential(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_EXPONENTIAL,
                     string_create("failed to successfully run exponential operation."),
                     error);
    }

    return NULL;
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

    if (x->requires_gradient)
    {
        nw_error_t *error = NULL;
        tensor_t *x_gradient = NULL;

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create empty tensor x_gradient."),
                         error);
        }

        error = tensor_multiplication(gradient, result, x_gradient);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            return ERROR(ERROR_MULTIPLICATION,
                         string_create("failed to multiply tensors gradient and result."),
                         error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            tensor_destroy(x_gradient);
            return ERROR(ERROR_ADDITION,
                         string_create("failed to accumulate gradient."),
                         error);
        }

        tensor_destroy(x_gradient);
    }

    return NULL;
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

    nw_error_t *error = error;

    error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create empty tensor."),
                     error);
    }

    error = runtime_logarithm(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_LOGARITHM,
                     string_create("failed to successfully run logarithm operation."),
                     error);
    }

    return NULL;
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

    if (x->requires_gradient)
    {
        nw_error_t *error = NULL;
        tensor_t *x_gradient = NULL;

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create tensor x_gradient."),
                         error);
        }

        error = tensor_division(gradient, x, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_DIVISION,
                         string_create("failed to successfully run division operation."),
                         error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION,
                         string_create("failed to accumulate gradient."),
                         error);
        }
    }

    return NULL;
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

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create empty tensor."),
                     error);
    }
 
    error = runtime_sine(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_SINE,
                     string_create("failed to successfully run sine operation."),
                     error);
    }

    return NULL;
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

    if (x->requires_gradient)
    {
        nw_error_t *error = NULL;
        tensor_t *x_gradient = NULL;
        tensor_t *x_gradient_i = NULL;

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create tensor x_gradient."),
                         error);
        }

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            return ERROR(ERROR_CREATE,
                         string_create("failed to create tensor x_gradient_i."),
                         error);
        }

        error = tensor_cosine(x, x_gradient_i);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            return ERROR(ERROR_COSINE,
                         string_create("failed to successfully run cosine operation."),
                         error);
        }

        error = tensor_multiplication(x_gradient_i, gradient, x_gradient);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            return ERROR(ERROR_MULTIPLICATION,
                         string_create("failed to successfully run multiplication operation."),
                         error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            return ERROR(ERROR_ADDITION,
                         string_create("failed to accumulate gradient."),
                         error);
        }

        tensor_destroy(x_gradient_i);
    }

    return NULL;
}

static nw_error_t *cosine_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create empty tensor."),
                     error);
    }
 
    error = runtime_cosine(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_COSINE,
                     string_create("failed to successfully run cosine operation."),
                     error);
    }

    return NULL;
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

    if (x->requires_gradient)
    {
        tensor_t *x_gradient;
        tensor_t *x_gradient_i;
        tensor_t *x_gradient_j;

        nw_error_t *error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_gradient tensor."),
                         error);
        }

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_gradient_i tensor."),
                         error);
        }

        error = tensor_create_empty(&x_gradient_j);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_gradient_j tensor."),
                         error);
        }

        error = tensor_sine(x, x_gradient_j);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(x_gradient_j);
            return ERROR(ERROR_SINE,
                         string_create("failed to successfully run sine operation."),
                         error);
        }

        error = tensor_negation(x_gradient_j, x_gradient_i);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(x_gradient_j);
            return ERROR(ERROR_NEGATION,
                         string_create("failed to successfully run negation operation."),
                         error);
        }

        error = tensor_multiplication(x_gradient_i, gradient, x_gradient);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(x_gradient_j);
            return ERROR(ERROR_MULTIPLICATION,
                         string_create("failed to successfully run multiplication operation."),
                         error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(x_gradient_j);
            return ERROR(ERROR_ADDITION,
                         string_create("failed to accumulate gradient."),
                         error);
        }

        tensor_destroy(x_gradient);
        tensor_destroy(x_gradient_i);
        tensor_destroy(x_gradient_j);
    }

    return NULL;
}

static nw_error_t *square_root_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create empty tensor."),
                     error);
    }

    error = runtime_square_root(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_SQUARE_ROOT,
                     string_create("failed to successfully run square root operation."),
                     error);
    }

    return NULL;
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
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient;
        tensor_t *x_gradient_i;
        tensor_t *constant;

        nw_error_t *error = tensor_create_empty(&constant);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create constant tensor."),
                         error);
        }

        switch (x->buffer->datatype)
        {
        case FLOAT32:
            float32_t constant_32 = 2.0;
            error = tensor_constant(&constant_32, x->buffer->datatype, x->buffer->runtime, constant);
            if (error != NULL)
            {
                return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize constant tensor."), error);
            }
            break;
        case FLOAT64:
            float64_t constant_64 = 2.0;
            error = tensor_constant(&constant_64, x->buffer->datatype, x->buffer->runtime, constant);
            if (error != NULL)
            {
                return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize constant tensor."), error);
            }
            break;
        default:
            break;
        }

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            tensor_destroy(constant);
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(constant);
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_i tensor."), error);
        }

        error = tensor_multiplication(result, constant, x_gradient_i);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(constant);
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_division(gradient, x_gradient_i, x_gradient);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(constant);
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(constant);
            return ERROR(ERROR_ADDITION, 
                         string_create("failed to accumulate gradient."),
                         error);
        }

        tensor_destroy(x_gradient);
        tensor_destroy(x_gradient_i);
        tensor_destroy(constant);
    }

    return NULL;
}

static nw_error_t *reciprocal_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_reciprocal(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_RECIPROCAL, string_create("failed to successfully run reciprocal operation."), error);
    }

    return NULL;
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

    if (x->requires_gradient)
    {
        nw_error_t *error = NULL;
        tensor_t *x_gradient = NULL;
        tensor_t *x_gradient_i = NULL;
        tensor_t *x_gradient_j = NULL;

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_gradient tensor."),
                         error);
        }

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_gradient_i tensor."),
                         error);
        }

        error = tensor_create_empty(&x_gradient_j);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_gradient_j tensor."),
                         error);
        }

        error = tensor_multiplication(result, result, x_gradient_i);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(x_gradient_j);
            return ERROR(ERROR_MULTIPLICATION,
                         string_create("failed to successfully run multiplication operation."),
                         error);
        }

        error = tensor_negation(x_gradient_i, x_gradient_j);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(x_gradient_j);
            return ERROR(ERROR_NEGATION,
                         string_create("failed to successfully run negation operation."),
                         error);
        }

        error = tensor_multiplication(x_gradient_j, gradient, x_gradient);
        if (error != NULL)
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(x_gradient_j);
            return ERROR(ERROR_MULTIPLICATION,
                         string_create("failed to successfully run multiplication operation."),
                         error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            tensor_destroy(x_gradient_j);
            return ERROR(ERROR_ADDITION,
                         string_create("failed to accumulate gradient."),
                         error);
        }

        tensor_destroy(x_gradient);
        tensor_destroy(x_gradient_i);
        tensor_destroy(x_gradient_j);
    }

    return NULL;
}

static nw_error_t *copy_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create empty tensor."),
                     error);
    }

    error = runtime_copy(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_COPY,
                     string_create("failed to successfully run copy operation."),
                     error);
    }

    return NULL;
}

static nw_error_t *copy_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        nw_error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION,
                         string_create("failed to accumulate gradient."),
                         error);
        }
    }

    return NULL;
}

static nw_error_t *contiguous_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_contiguous(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_COPY, string_create("failed to successfully run contiguous operation."), error);
    }

    return NULL;
}

static nw_error_t *contiguous_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        nw_error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    return NULL;
}

static nw_error_t *negation_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_negation(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_NEGATION, string_create("failed to successfully run negation operation."), error);
    }

    return NULL;
}

static nw_error_t *negation_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient;

        nw_error_t *error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_negation(gradient, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_NEGATION, string_create("failed to successfully run negation operation."), error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    return NULL;
}

static nw_error_t *rectified_linear_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_rectified_linear(x->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_RECTIFIED_LINEAR, string_create("failed to successfully run rectified linear operation."), error);
    }

    return NULL;
}

static nw_error_t *rectified_linear_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        nw_error_t *error = NULL;
        tensor_t *x_gradient = NULL;
        tensor_t *x_gradient_i = NULL;
        tensor_t *x_gradient_j = NULL;
        float64_t singularity = 0.0;

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_i tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_j tensor."), error);
        }

        error = tensor_constant(&singularity, x->buffer->datatype, x->buffer->runtime, x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_INITIALIZATION, string_create("failed to create constant tensor."), error);
        }

        error = tensor_as_empty(x, x_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize empty tensor."), error);
        }

        error = runtime_compare_greater(x->buffer, x_gradient_i->buffer, x_gradient_j->buffer);
        if (error != NULL)
        {
            return ERROR(ERROR_COMPARE_GREATER, string_create("failed to suvvessfully run compare greater operation."), error);
        }
        
        error = tensor_multiplication(x_gradient_j, gradient, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }

        tensor_destroy(x_gradient_i);
        tensor_destroy(x_gradient_j);
    }

    return NULL;
}

nw_error_t *unary_operation_forward(unary_operation_t *unary_operation)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");

    nw_error_t *error = NULL;

    switch (unary_operation->operation_type)
    {
    case EXPONENTIAL_OPERATION:
        error = exponential_operation_forward(unary_operation->x, unary_operation->result);
        break;
    case LOGARITHM_OPERATION:
        error = logarithm_operation_forward(unary_operation->x, unary_operation->result);
        break;
    case SINE_OPERATION:
        error = sine_operation_forward(unary_operation->x, unary_operation->result);
        break;
    case COSINE_OPERATION:
        error = cosine_operation_forward(unary_operation->x, unary_operation->result);
        break;
    case SQUARE_ROOT_OPERATION:
        error = square_root_operation_forward(unary_operation->x, unary_operation->result);
        break;
    case RECIPROCAL_OPERATION:
        error = reciprocal_operation_forward(unary_operation->x, unary_operation->result);
        break;
    case COPY_OPERATION:
        error = copy_operation_forward(unary_operation->x, unary_operation->result);
        break;
    case CONTIGUOUS_OPERATION:
        error = contiguous_operation_forward(unary_operation->x, unary_operation->result);
        break;
    case NEGATION_OPERATION:
        error = negation_operation_forward(unary_operation->x, unary_operation->result);
        break;
    case RECTIFIED_LINEAR_OPERATION:
        error = rectified_linear_operation_forward(unary_operation->x, unary_operation->result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown operation type %d.",
                      (int) unary_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply forward unary operation of type %s.",
                     unary_operation_type_string(unary_operation->operation_type)),
                     error);
    }

    unary_operation->result->requires_gradient = unary_operation->x->requires_gradient;
    
    return NULL;
}

nw_error_t *unary_operation_backward(unary_operation_t *unary_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    switch (unary_operation->operation_type)
    {
    case EXPONENTIAL_OPERATION:
        error = exponential_operation_backward(unary_operation->x, unary_operation->result, gradient);
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
        error = square_root_operation_backward(unary_operation->x, unary_operation->result, gradient);
        break;
    case RECIPROCAL_OPERATION:
        error = reciprocal_operation_backward(unary_operation->x, unary_operation->result, gradient);
        break;
    case COPY_OPERATION:
        error = copy_operation_backward(unary_operation->x, gradient);
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
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown operation type %d.",
                      (int) unary_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_BACKWARD,
                     string_create("failed to apply backward unary operation of type %s.",
                     unary_operation_type_string(unary_operation->operation_type)),
                     error);
    }
    
    return NULL;
}

nw_error_t *binary_operation_create(binary_operation_t **binary_operation,
                                    binary_operation_type_t binary_operation_type,
                                    const tensor_t *x,
                                    const tensor_t *y,
                                    tensor_t *result)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    *binary_operation = (binary_operation_t *) malloc(sizeof(binary_operation_t));
    if (*binary_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate binary operation of size %lu bytes.",
                     (unsigned long) sizeof(binary_operation_t)),
                     NULL);
    }

    (*binary_operation)->operation_type = binary_operation_type;
    (*binary_operation)->x = (tensor_t *) x; 
    (*binary_operation)->y = (tensor_t *) y;
    (*binary_operation)->result = result;

    return NULL;
}

void binary_operation_destroy(binary_operation_t *binary_operation)
{
    if (binary_operation == NULL)
    {
        return;
    }

    free(binary_operation);
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
    default:
        return "UKNOWN_OPERATION";
    }
}

static nw_error_t *addition_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_addition(x->buffer, y->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to successfully run addition operation."), error);
    }

    return NULL;
}

static nw_error_t *addition_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        nw_error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute addition operation backward pass x operand."), error);
        }
    }

    if (y->requires_gradient)
    {
        nw_error_t *error = tensor_accumulate_gradient(y, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute addition operation backward pass y operand."), error);
        }
    }

    return NULL;
}

static nw_error_t *subtraction_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_subtraction(x->buffer, y->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_SUBTRACTION, string_create("failed to successfully run subtraction operation."), error);
    }

    return NULL;
}

static nw_error_t *subtraction_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        nw_error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute subtraction operation backward pass x operand."), error);
        }
    }

    if (y->requires_gradient)
    {
        tensor_t *y_gradient;

        nw_error_t *error = tensor_create_empty(&y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient tensor."), error);
        }

        error = tensor_negation(gradient, y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_NEGATION, string_create("failed to successfully run negation operation."), error);
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    return NULL;
}

static nw_error_t *multiplication_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_multiplication(x->buffer, y->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
    }

    return NULL;
}

static nw_error_t *multiplication_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient;

        nw_error_t *error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_multiplication(y, gradient, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    if (y->requires_gradient)
    {
        tensor_t *y_gradient;

        nw_error_t *error = tensor_create_empty(&y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient tensor."), error);
        }

        error = tensor_multiplication(x, gradient, y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    return NULL;
}

static nw_error_t *division_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_division(x->buffer, y->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_DIVISION, string_create("failed to successfully run division operation."), error);
    }

    return NULL;
}

static nw_error_t *division_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient;
        tensor_t *x_gradient_i;

        nw_error_t *error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_i tensor."), error);
        }

        error = tensor_reciprocal(y, x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_RECIPROCAL, string_create("failed to successfully run reciprocal operation."), error);
        }

        error = tensor_multiplication(x_gradient_i, gradient, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    if (y->requires_gradient)
    {
        tensor_t *y_gradient;
        tensor_t *y_gradient_i;
        tensor_t *y_gradient_j;
        tensor_t *y_gradient_k;
        tensor_t *y_gradient_l;

        nw_error_t *error = tensor_create_empty(&y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient tensor."), error);
        }

        error = tensor_create_empty(&y_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient_i tensor."), error);
        }

        error = tensor_create_empty(&y_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient_j tensor."), error);
        }

        error = tensor_create_empty(&y_gradient_k);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient_k tensor."), error);
        }

        error = tensor_create_empty(&y_gradient_l);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient_l tensor."), error);
        }

        error = tensor_multiplication(y, y, y_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_reciprocal(y_gradient_i, y_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_RECIPROCAL, string_create("failed to successfully run reciprocal operation."), error);
        }

        error = tensor_negation(y_gradient_j, y_gradient_k);
        if (error != NULL)
        {
            return ERROR(ERROR_NEGATION, string_create("failed to successfully run negation operation."), error);
        }

        error = tensor_multiplication(y_gradient_k, x, y_gradient_l);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }
        
        error = tensor_multiplication(y_gradient_l, gradient, y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }

        tensor_destroy(y_gradient_i);
        tensor_destroy(y_gradient_j);
        tensor_destroy(y_gradient_k);
        tensor_destroy(y_gradient_l);
    }

    return NULL;
}

static nw_error_t *power_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_power(x->buffer, y->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_POWER, string_create("failed to successfully run power operation."), error);
    }

    return NULL;
}

static nw_error_t *power_operation_backward(tensor_t *x, tensor_t *y, tensor_t *result, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient;
        tensor_t *x_gradient_i;
        tensor_t *x_gradient_j;

        nw_error_t *error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_i tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_j tensor."), error);
        }

        error = tensor_division(result, x, x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_DIVISION, string_create("failed to successfully run division operation."), error);
        }

        error = tensor_multiplication(x_gradient_i, y, x_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_multiplication(x_gradient_j, gradient, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    if (y->requires_gradient)
    {
        tensor_t *y_gradient;
        tensor_t *y_gradient_i;
        tensor_t *y_gradient_j;

        nw_error_t *error = tensor_create_empty(&y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient tensor."), error);
        }

        error = tensor_create_empty(&y_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient_i tensor."), error);
        }

        error = tensor_create_empty(&y_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient_j tensor."), error);
        }

        error = tensor_logarithm(x, y_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_LOGARITHM, string_create("failed to successfully run logarithm operation."), error);
        }

        error = tensor_multiplication(y_gradient_i, result, y_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }
        
        error = tensor_multiplication(y_gradient_j, gradient, y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to successfully run multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }

        tensor_destroy(y_gradient_i);
        tensor_destroy(y_gradient_j);
    }

    return NULL;
}

static nw_error_t *matrix_multiplication_operation_forward(tensor_t *x, tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_matrix_multiplication(x->buffer, y->buffer, result->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_POWER, string_create("failed to successfully run power operation."), error);
    }

    return NULL;
}

nw_error_t *matrix_multiplication_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient;
        tensor_t *x_gradient_i;

        nw_error_t *error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_i tensor."), error);
        }

        uint64_t *axis = (uint64_t *) malloc(y->buffer->view->rank * sizeof(uint64_t));
        if (axis == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocated axis of size %zu bytes.", y->buffer->view->rank * sizeof(uint64_t)), NULL);
        }

        uint64_t rank = y->buffer->view->rank;
        for (uint64_t i = 0; i < rank; ++i)
        {
            axis[i] = i;
        }
        uint64_t temp = axis[rank - 1];
        axis[rank - 1] = axis[rank - 2];
        axis[rank - 2] = temp;

        error = tensor_permute(y, x_gradient_i, axis, rank);
        if (error != NULL)
        {
            return ERROR(ERROR_PERMUTE, string_create("failed to successfully run permute operation."), error);
        }

        error = tensor_matrix_multiplication(gradient, x_gradient_i, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to successfully run matrix multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    if (y->requires_gradient)
    {
        tensor_t *y_gradient;
        tensor_t *y_gradient_i;

        nw_error_t *error = tensor_create_empty(&y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient tensor."), error);
        }

        error = tensor_create_empty(&y_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create y_gradient_i tensor."), error);
        }

        uint64_t *axis = (uint64_t *) malloc(x->buffer->view->rank * sizeof(uint64_t));
        if (axis == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocated axis of size %zu bytes.", x->buffer->view->rank * sizeof(uint64_t)), NULL);
        }

        uint64_t rank = x->buffer->view->rank;
        for (uint64_t i = 0; i < rank; ++i)
        {
            axis[i] = i;
        }
        uint64_t temp = axis[rank - 1];
        axis[rank - 1] = axis[rank - 2];
        axis[rank - 2] = temp;

        error = tensor_permute(x, y_gradient_i, axis, rank);
        if (error != NULL)
        {
            return ERROR(ERROR_PERMUTE, string_create("failed to successfully run permute operation."), error);
        }

        error = tensor_matrix_multiplication(y_gradient_i, gradient, y_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to successfully run matrix multiplication operation."), error);
        }

        error = tensor_accumulate_gradient(y, y_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    return NULL;
}

nw_error_t *binary_operation_forward(binary_operation_t *binary_operation)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");

    nw_error_t *error = NULL;

    switch (binary_operation->operation_type)
    {
    case ADDITION_OPERATION:
        error = addition_operation_forward(binary_operation->x, binary_operation->y, binary_operation->result);
        break;
    case SUBTRACTION_OPERATION:
        error = subtraction_operation_forward(binary_operation->x, binary_operation->y, binary_operation->result);
        break;
    case MULTIPLICATION_OPERATION:
        error = multiplication_operation_forward(binary_operation->x, binary_operation->y, binary_operation->result);
        break;
    case DIVISION_OPERATION:
        error = division_operation_forward(binary_operation->x, binary_operation->y, binary_operation->result);
        break;
    case POWER_OPERATION:
        error = power_operation_forward(binary_operation->x, binary_operation->y, binary_operation->result);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        error = matrix_multiplication_operation_forward(binary_operation->x, binary_operation->y, binary_operation->result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown binary operation type %d.",
                      (int) binary_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute binary operation forward pass of type %s.",
                     binary_operation_type_string(binary_operation->operation_type)),
                     error);
    }

    binary_operation->result->requires_gradient = binary_operation->x->requires_gradient || binary_operation->y->requires_gradient;
    
    return NULL;
}

nw_error_t *binary_operation_backward(binary_operation_t *binary_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
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
        error = power_operation_backward(binary_operation->x, binary_operation->y, binary_operation->result, gradient);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        error = matrix_multiplication_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown binary operation type %d.",
                      (int) binary_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute binary operation backward pass of type %s.",
                     binary_operation_type_string(binary_operation->operation_type)),
                     error);
    }
    
    return NULL;
}

nw_error_t *reduction_operation_create(reduction_operation_t **reduction_operation, 
                                       reduction_operation_type_t reduction_operation_type,
                                       const tensor_t *x,
                                       const uint64_t *axis,
                                       uint64_t length,
                                       bool_t keep_dimension,
                                       tensor_t *result)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    *reduction_operation = (reduction_operation_t *) malloc(sizeof(reduction_operation_t));
    if (*reduction_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate reduction operation of size %lu bytes.",
                     (unsigned int) sizeof(reduction_operation_t)),
                     NULL);
    }

    (*reduction_operation)->axis = (uint64_t *) malloc((size_t) (length * sizeof(uint64_t)));
    if ((*reduction_operation)->axis == NULL)
    {
        free(*reduction_operation);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate reduction_operation->axis of size %lu bytes.",
                     (unsigned long) (length * sizeof(uint64_t))),
                     NULL);
    }
    memcpy((*reduction_operation)->axis, axis, (size_t) (length * sizeof(uint64_t)));

    (*reduction_operation)->operation_type = reduction_operation_type;
    (*reduction_operation)->x = (tensor_t *) x; 
    (*reduction_operation)->length = length;
    (*reduction_operation)->keep_dimension = keep_dimension;
    (*reduction_operation)->result = result;

    return NULL;
}

void reduction_operation_destroy(reduction_operation_t *reduction_operation)
{
    if (reduction_operation == NULL)
    {
        return;
    }

    free(reduction_operation->axis);
    free(reduction_operation);
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

static nw_error_t *reduction_forward(reduction_operation_type_t reduction_operation_type,
                                     tensor_t *x,
                                     uint64_t *axis,
                                     uint64_t length,
                                     tensor_t *result,
                                     bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    if (x->buffer->view->rank < length)
    {
        return ERROR(ERROR_RANK_CONFLICT,
                     string_create("rank of tensor being reduced (%lu) must be not be less than length of axis (%lu).",
                     (unsigned long) (x->buffer->view->rank), (unsigned long) length), 
                     NULL);
    }

    buffer_t *x_buffer = x->buffer;
    buffer_t *result_buffer = NULL;

    for (uint64_t i = 0; i < length; ++i)
    {
        uint64_t *reduced_shape = NULL;
        uint64_t *reduced_strides = NULL;
        uint64_t reduced_n = x->buffer->n;
        nw_error_t *error = NULL;
        view_t *view = NULL;

        reduced_shape = (uint64_t *) malloc((size_t) (x->buffer->view->rank * sizeof(uint64_t)));
        if (reduced_shape == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocated reduced_shape of size %lu bytes.",
                         (unsigned long) (x->buffer->view->rank * sizeof(uint64_t))),
                         NULL);
        }

        reduced_strides = (uint64_t *) malloc((size_t) (x->buffer->view->rank * sizeof(uint64_t)));
        if (reduced_strides == NULL)
        {
            free(reduced_shape);
            return ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocated reduced_strides of size %lu bytes.",
                         (unsigned long) (x->buffer->view->rank * sizeof(uint64_t))),
                         NULL);
        }

        error = reduce(x_buffer->view->shape,
                       x_buffer->view->rank,
                       x_buffer->view->strides,
                       reduced_shape,
                       x_buffer->view->rank,
                       reduced_strides,
                       &axis[i], 
                       (uint64_t) 1, 
                       true);
        if (error != NULL)
        {
            free(reduced_shape);
            free(reduced_strides);
            return ERROR(ERROR_REDUCTION,
                         string_create("failed to remove reduce tensor view along axis %lu.",
                         (unsigned long) axis[i]), 
                         error);
        }

        error = reduce_compute_buffer_size(x->buffer->view->shape,
                                           x_buffer->view->strides,
                                           x_buffer->view->rank,
                                           x_buffer->n,
                                           &axis[i],
                                           (uint64_t) 1,
                                           &reduced_n);
        if (error != NULL)
        {
            free(reduced_shape);
            free(reduced_strides);
            return ERROR(ERROR_REDUCTION,
                         string_create("failed to remove reduce buffer size %lu axis %lu.",
                         (unsigned long) x_buffer->n, (unsigned long) axis[i]), 
                         error);
        }

        error = view_create(&view,
                            x->buffer->view->offset,
                            x->buffer->view->rank,
                            reduced_shape,
                            reduced_strides);
        if (error != NULL)
        {
            free(reduced_shape);
            free(reduced_strides);
            return ERROR(ERROR_CREATE, 
                         string_create("failed to create view."),
                         error);
        }

        error = buffer_create(&result_buffer,
                              x->buffer->runtime,
                              x->buffer->datatype,
                              view,
                              NULL,
                              reduced_n,
                              true);
        if (error != NULL)
        {
            free(reduced_shape);
            free(reduced_strides);
            return ERROR(ERROR_CREATE, 
                         string_create("failed to create buffer."),
                         error);
        }

        switch (reduction_operation_type)
        {
        case SUMMATION_OPERATION:
            error = runtime_summation(x_buffer, result_buffer, axis[i]);
            if (error != NULL)
            {
                free(reduced_shape);
                free(reduced_strides);
                return ERROR(ERROR_SUMMATION,
                            string_create("failed to successfully run summation operation."),
                            error);
            }
            break;
        case MAXIMUM_OPERATION:
            error = runtime_maximum(x_buffer, result_buffer, axis[i]);
            if (error != NULL)
            {
                free(reduced_shape);
                free(reduced_strides);
                return ERROR(ERROR_MAXIMUM,
                            string_create("failed to successfully run maximum operation."),
                            error);
            }
            break;
        default:
            return ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                         string_create("unknown reduction operation type %d.",
                         (int) reduction_operation_type),
                         NULL);
        }

        free(reduced_shape);
        free(reduced_strides);

        if (i > 0)
        {
            buffer_destroy(x_buffer);
        }
        x_buffer = result_buffer;
    }

    if (!keep_dimension)
    {
        uint64_t *reduced_shape = NULL;
        uint64_t *reduced_strides = NULL;
        nw_error_t *error = NULL;

        if (result_buffer == NULL)
        {
            return ERROR(ERROR_NULL,
                         string_create("reduced_buffer is NULL."),
                         NULL);
        }

        reduced_shape = (uint64_t *) malloc((size_t)((result_buffer->view->rank - length) * sizeof(uint64_t)));
        if (reduced_shape == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocated reduced_shape of size %lu bytes.", 
                         (unsigned long) ((result_buffer->view->rank - length) * sizeof(uint64_t))),
                         NULL);
        }

        reduced_strides = (uint64_t *) malloc((size_t)((result_buffer->view->rank - length) * sizeof(uint64_t)));
        if (reduced_strides == NULL)
        {
            free(reduced_shape);
            return ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocated reduced_strides of size %lu bytes.", 
                         (unsigned long) ((result_buffer->view->rank - length) * sizeof(uint64_t))),
                         NULL);
        }

        error = reduce(result_buffer->view->shape,
                       result_buffer->view->rank,
                       result_buffer->view->strides, 
                       reduced_shape,
                       result_buffer->view->rank - length,
                       reduced_strides,
                       axis,
                       length,
                       keep_dimension);
        if (error != NULL)
        {
            free(reduced_shape);
            free(reduced_strides);
            return ERROR(ERROR_REDUCTION,
                         string_create("failed to remove dimensions from reduced tensor."),
                         error);
        }

        free(result_buffer->view->shape);
        free(result_buffer->view->strides);
        result_buffer->view->shape = reduced_shape;
        result_buffer->view->strides = reduced_strides;
        result_buffer->view->rank -= length;
    }

    result->buffer = result_buffer;

    return NULL; 
}

static nw_error_t *summation_operation_forward(tensor_t *x,
                                               uint64_t *axis,
                                               uint64_t length,
                                               tensor_t *result,
                                               bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = reduction_forward(SUMMATION_OPERATION, x, axis, length, result, keep_dimension);
    if (error != NULL)
    {
        return ERROR(ERROR_REDUCTION, 
                     string_create("failed to reduce tensor."),
                     error);
    }

    return NULL; 
}

static nw_error_t *summation_operation_backward(tensor_t *x,
                                                uint64_t *axis,
                                                uint64_t length,
                                                tensor_t *gradient,
                                                bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient = NULL;
        nw_error_t *error = NULL;

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_gradient tensor."),
                         error);
        }

        if (!keep_dimension)
        {
            uint64_t *reduced_shape = NULL;
            uint64_t *reduced_strides = NULL;

            reduced_shape = (uint64_t *) malloc((size_t) (x->buffer->view->rank * sizeof(uint64_t)));
            if (reduced_shape == NULL)
            {
                return ERROR(ERROR_MEMORY_ALLOCATION, 
                             string_create("failed to allocated reduced_shape of size %lu bytes.", 
                             (unsigned long) (x->buffer->view->rank * sizeof(uint64_t))),
                             NULL);
            }

            reduced_strides = (uint64_t *) malloc((size_t) (x->buffer->view->rank * sizeof(uint64_t)));
            if (reduced_strides == NULL)
            {
                free(reduced_shape);
                return ERROR(ERROR_MEMORY_ALLOCATION, 
                             string_create("failed to allocated reduced_shape of size %lu bytes.", 
                             (unsigned long) (x->buffer->view->rank * sizeof(uint64_t))),
                             NULL);
            }

            error = reduce_recover_dimensions(gradient->buffer->view->shape,
                                              gradient->buffer->view->rank,
                                              gradient->buffer->view->strides,
                                              reduced_shape,
                                              x->buffer->view->rank,
                                              reduced_strides,
                                              axis,
                                              length);
            if (error != NULL)
            {
                free(reduced_shape);
                free(reduced_strides);
                return ERROR(ERROR_REDUCTION, 
                             string_create("failed to recover reduce dimensions."),
                             error);
            }

            free(gradient->buffer->view->shape);
            free(gradient->buffer->view->strides);
            gradient->buffer->view->shape = reduced_shape;
            gradient->buffer->view->strides = reduced_strides;
            gradient->buffer->view->rank = x->buffer->view->rank;
        }

        error = tensor_expand(gradient,
                              x->buffer->view->shape,
                              x->buffer->view->rank,
                              x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_EXPAND,
                         string_create("failed to expand gradient."),
                         error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION,
                         string_create("failed to accumulate gradient."),
                         error);
        }
    }

    return NULL; 
}

static nw_error_t *maximum_operation_forward(tensor_t *x,
                                             uint64_t *axis,
                                             uint64_t length,
                                             tensor_t *result,
                                             bool_t keep_dimension)
{

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error = NULL;

    error = reduction_forward(MAXIMUM_OPERATION, x, axis, length, result, keep_dimension);
    if (error != NULL)
    {
        return ERROR(ERROR_REDUCTION, 
                     string_create("failed to reduce tensor."),
                     error);
    }

    return NULL; 

}

static nw_error_t *maximum_operation_backward(tensor_t *x, uint64_t *axis, uint64_t rank, tensor_t *result, tensor_t *gradient, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient_i;
        tensor_t *x_gradient_j;
        tensor_t *x_gradient_k;
        tensor_t *x_gradient_l;
        tensor_t *x_gradient;
        tensor_t *returned;

        nw_error_t *error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }
        
        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_i tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_j tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_k);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_k tensor."), error);
        }

        error = tensor_create_empty(&x_gradient_l);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_k tensor."), error);
        }

        error = tensor_create_empty(&returned);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create returned tensor."), error);
        }

        error = tensor_as_empty(x, x_gradient_j);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
        }

        error = tensor_as_tensor(result, returned);
        if (error != NULL)
        {
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tensor."), error);
        }

        if (!keep_dimension)
        {
            uint64_t *reduced_shape = (uint64_t *) malloc(x->buffer->view->rank * sizeof(uint64_t));
            if (reduced_shape == NULL)
            {
                return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocated reduced_shape of size %zu bytes.", x->buffer->view->rank * sizeof(uint64_t)), NULL);
            }

            uint64_t *reduced_strides = (uint64_t *) malloc(x->buffer->view->rank * sizeof(uint64_t));
            if (reduced_strides == NULL)
            {
                return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocated reduced_shape of size %zu bytes.", x->buffer->view->rank * sizeof(uint64_t)), NULL);
            }

            error = reduce_recover_dimensions(x->buffer->view->shape, x->buffer->view->rank, x->buffer->view->strides, reduced_shape, x->buffer->view->rank, reduced_strides, axis, rank);
            if (error != NULL)
            {
                return ERROR(ERROR_REDUCTION, string_create("failed to recover reduce dimensions."), error);
            }

            free(returned->buffer->view->shape);
            free(returned->buffer->view->strides);
            returned->buffer->view->shape = reduced_shape;
            returned->buffer->view->strides = reduced_strides;
            returned->buffer->view->rank = x->buffer->view->rank;
        }

        error = tensor_expand(returned, x->buffer->view->shape, x->buffer->view->rank, x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
        }

        error = runtime_compare_equal(x_gradient_i->buffer, x->buffer, x_gradient_j->buffer);
        if (error != NULL)
        {
            return ERROR(ERROR_COMPARE_EQUAL, string_create("failed to expand tensor."), error);
        }

        error = tensor_summation(x_gradient_j, x_gradient_k, NULL, 0, true);
        if (error != NULL)
        {
            return ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        }

        error = tensor_division(x_gradient_j, x_gradient_k, x_gradient_l);
        if (error != NULL)
        {
            return ERROR(ERROR_DIVISION, string_create("failed to divide tensor."), error);
        }

        error = tensor_multiplication(x_gradient_l, gradient, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensor."), error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    return NULL; 
}

nw_error_t *reduction_operation_forward(reduction_operation_t *reduction_operation)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");

    nw_error_t *error = NULL;

    switch (reduction_operation->operation_type)
    {
    case SUMMATION_OPERATION:
        error = summation_operation_forward(reduction_operation->x,
                                            reduction_operation->axis,
                                            reduction_operation->length,
                                            reduction_operation->result,
                                            reduction_operation->keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        error = maximum_operation_forward(reduction_operation->x,
                                          reduction_operation->axis,
                                          reduction_operation->length,
                                          reduction_operation->result,
                                          reduction_operation->keep_dimension);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown reduction operation type %d.",
                      (int) reduction_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute reduction operation forward pass of type %s.",
                     reduction_operation_type_string(reduction_operation->operation_type)),
                     error);
    }

    reduction_operation->result->requires_gradient = reduction_operation->x->requires_gradient;

    return NULL;
}

nw_error_t *reduction_operation_backward(reduction_operation_t *reduction_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    switch (reduction_operation->operation_type)
    {
    case SUMMATION_OPERATION:
        error = summation_operation_backward(reduction_operation->x,
                                             reduction_operation->axis,
                                             reduction_operation->length,
                                             gradient,
                                             reduction_operation->keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        error = maximum_operation_backward(reduction_operation->x,
                                           reduction_operation->axis,
                                           reduction_operation->length,
                                           reduction_operation->result,
                                           gradient,
                                           reduction_operation->keep_dimension);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown reduction operation type %d.",
                      (int) reduction_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute reduction operation backward pass %s.",
                     reduction_operation_type_string(reduction_operation->operation_type)),
                     error);
    }

    return NULL;
}

nw_error_t *structure_operation_create(structure_operation_t **structure_operation,
                                       structure_operation_type_t structure_operation_type,
                                       const tensor_t *x,
                                       const uint64_t *arguments,
                                       uint64_t length,
                                       tensor_t *result)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    *structure_operation = (structure_operation_t *) malloc(sizeof(structure_operation_t));
    if (*structure_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate structure operation of size %lu bytes.",
                     (unsigned long) sizeof(structure_operation_t)),
                     NULL);
    }

    (*structure_operation)->arguments = (uint64_t *) malloc((size_t) (length * sizeof(uint64_t)));
    if ((*structure_operation)->arguments == NULL)
    {
        free(*structure_operation);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate structure_operation->arguments of size %lu bytes.",
                     (unsigned long) (length * sizeof(uint64_t))), 
                     NULL);
    }
    memcpy((*structure_operation)->arguments, arguments, (size_t) (length * sizeof(uint64_t)));

    (*structure_operation)->operation_type = structure_operation_type;
    (*structure_operation)->x = (tensor_t *) x; 
    (*structure_operation)->length = length;
    (*structure_operation)->result = result;

    return NULL;
}

void structure_operation_destroy(structure_operation_t *structure_operation)
{
    if (structure_operation == NULL)
    {
        return;
    }

    free(structure_operation->arguments);
    free(structure_operation);
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
    uint64_t *strides = NULL;
    view_t *view = NULL;

    strides = (uint64_t *) malloc((size_t) (length * sizeof(uint64_t)));
    if (strides == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate strides of size %lu bytes.", 
                     (unsigned long) (length * sizeof(uint64_t))),
                     NULL);
    }

    error = broadcast_strides(x->buffer->view->shape,
                              x->buffer->view->rank,
                              x->buffer->view->strides,
                              shape,
                              length,
                              strides);
    if (error != NULL)
    {
        free(strides);
        return ERROR(ERROR_INITIALIZATION,
                     string_create("failed to initialize expand strides"),
                     error);
    }

    error = view_create(&view, x->buffer->view->offset, length, shape, strides);
    if (error != NULL)
    {
        free(strides);
        return ERROR(ERROR_CREATE,
                     string_create("failed to create view."),
                     error);
    }

    free(strides);

    error = buffer_create(&result->buffer,
                          x->buffer->runtime,
                          x->buffer->datatype,
                          view,
                          x->buffer->data,
                          x->buffer->n,
                          false);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, 
                     string_create("failed to create buffer."),
                     error);
    }

    return NULL;
}

static nw_error_t *expand_operation_backward(tensor_t *x,
                                             uint64_t *shape,
                                             uint64_t length,
                                             tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        nw_error_t *error = NULL;
        uint64_t length_keep_dimension = 0;
        uint64_t length_remove_dimension = 0;
        uint64_t *axis_keep_dimension = NULL;
        uint64_t *axis_remove_dimension = NULL;
        tensor_t *x_gradient_i = NULL;
        tensor_t *x_gradient = NULL;
        
        error = reverse_broadcast_length(x->buffer->view->shape,
                                         x->buffer->view->rank,
                                         shape,
                                         length,
                                         &length_keep_dimension,
                                         &length_remove_dimension);
        if (error != NULL)
        {
            return ERROR(ERROR_BROADCAST,
                         string_create("failed to get keep and remove dimension lengths."),
                         error);
        }

        axis_keep_dimension = (uint64_t *) malloc((size_t) (sizeof(uint64_t) * length_keep_dimension));
        if (axis_keep_dimension == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocate axis of size %lu bytes.",
                         (unsigned long) (sizeof(uint64_t) * length_keep_dimension)),
                         NULL);
        }

        axis_remove_dimension = (uint64_t *) malloc((size_t) (sizeof(uint64_t) * length_remove_dimension));
        if (axis_remove_dimension == NULL)
        {
            free(axis_keep_dimension);
            return ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocate axis of size %lu bytes.",
                         (unsigned long) (sizeof(uint64_t) * length_remove_dimension)),
                         NULL);
        }

        error = reverse_broadcast_axis(x->buffer->view->shape,
                                       x->buffer->view->rank,
                                       shape,
                                       length,
                                       axis_keep_dimension,
                                       axis_remove_dimension);
        if (error != NULL)
        {
            free(axis_keep_dimension);
            free(axis_remove_dimension);
            return ERROR(ERROR_BROADCAST,
                         string_create("failed to get corresponding reduce axis to reverse broadcast."),
                         error);
        }

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            free(axis_keep_dimension);
            free(axis_remove_dimension);
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_gradient_i tensor."),
                         error);
        }

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            free(axis_keep_dimension);
            free(axis_remove_dimension);
            tensor_destroy(x_gradient_i);
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_gradient tensor."),
                         error);
        }

        error = tensor_summation(gradient,
                                 x_gradient_i,
                                 axis_keep_dimension,
                                 length_keep_dimension,
                                 true);
        if (error != NULL)
        {
            free(axis_keep_dimension);
            free(axis_remove_dimension);
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            return ERROR(ERROR_SUMMATION,
                         string_create("failed to sum tensor."),
                         error);
        }

        error = tensor_summation(x_gradient_i,
                                 x_gradient,
                                 axis_remove_dimension,
                                 length_remove_dimension,
                                 false);
        if (error != NULL)
        {
            free(axis_keep_dimension);
            free(axis_remove_dimension);
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            return ERROR(ERROR_SUMMATION,
                         string_create("failed to sum tensor."),
                         error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL)
        {
            free(axis_keep_dimension);
            free(axis_remove_dimension);
            tensor_destroy(x_gradient);
            tensor_destroy(x_gradient_i);
            return ERROR(ERROR_ADDITION,
                         string_create("failed to add gradient."),
                         error);
        }

        free(axis_keep_dimension);
        free(axis_remove_dimension);
        tensor_destroy(x_gradient);
        tensor_destroy(x_gradient_i);
    }

    return NULL;
}

static nw_error_t *permute_operation_forward(tensor_t *x, uint64_t *axis, uint64_t rank, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error;
    uint64_t *shape;
    uint64_t *strides;
    view_t *view;

    strides = (uint64_t *) malloc(rank * sizeof(uint64_t));
    if (strides == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate strides of size %zu bytes.", rank * sizeof(uint64_t)), NULL);
    }

    shape = (uint64_t *) malloc(rank * sizeof(uint64_t));
    if (shape == NULL)
    {
        free(strides);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate shape of size %zu bytes.", rank * sizeof(uint64_t)), NULL);
    }

    error = permute(x->buffer->view->shape, x->buffer->view->rank, x->buffer->view->strides, shape, rank, strides, axis, rank);
    if (error != NULL)
    {
        free(shape);
        free(strides);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize permuted shape and strides."), error);
    }

    error = view_create(&view, x->buffer->view->offset, rank, shape, strides);
    if (error != NULL)
    {
        free(shape);
        free(strides);
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }
    free(shape);
    free(strides);

    error = buffer_create(&result->buffer, x->buffer->runtime, x->buffer->datatype, view, x->buffer->data, x->buffer->n, false);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return NULL;
}

static nw_error_t *permute_operation_backward(tensor_t *x, uint64_t *axis, uint64_t rank, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        nw_error_t *error;
        uint64_t *gradient_axis;
        tensor_t *x_gradient;

        gradient_axis = (uint64_t *) malloc(rank * sizeof(uint64_t));
        if (gradient_axis == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate strides of size %zu bytes.", rank * sizeof(uint64_t)), NULL);
        }

        error = reverse_permute(axis, rank, gradient_axis);
        if (error != NULL)
        {
            free(gradient_axis);
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize permuted axis."), error);
        }

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_permute(gradient, x_gradient, gradient_axis, rank);
        if (error != NULL)
        {
            return ERROR(ERROR_PERMUTE, string_create("failed to permute tensor."), error);
        }
        free(gradient_axis);

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add gradient."), error);
        }
    }

    return NULL;
}

static nw_error_t *reshape_operation_forward(tensor_t *x, uint64_t *shape, uint64_t rank, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    nw_error_t *error;
    view_t *view;

    if (!tensor_is_contiguous(x))
    {
        return ERROR(ERROR_CONTIGUOUS, string_create("cannot reshape a non-contiguous tensor."), NULL);
    }

    error = view_create(&view, x->buffer->view->offset, rank, shape, NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&result->buffer, x->buffer->runtime, x->buffer->datatype, view, x->buffer->data, x->buffer->n, false);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return NULL;
}

static nw_error_t *reshape_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient;

        nw_error_t *error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_reshape(gradient, x_gradient, x->buffer->view->shape, x->buffer->view->rank);        
        if (error != NULL)
        {
            return ERROR(ERROR_RESHAPE, string_create("failed to reshape gradient."), error);
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add gradient."), error);
        }
    }

    return NULL;
}

static nw_error_t *slice_operation_forward(tensor_t *x, uint64_t *arguments, uint64_t length, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(result, "result");

    view_t *view;
    uint64_t offset = 0;
    uint64_t *shape = (uint64_t *) malloc(x->buffer->view->rank * sizeof(uint64_t));
    if (shape == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed allocate shape of size %zu bytes.", x->buffer->view->rank * sizeof(uint64_t)), NULL);
    }

    nw_error_t *error = slice_offset(x->buffer->view->strides, x->buffer->view->rank, &offset, arguments, length);
    if (shape == NULL)
    {
        return ERROR(ERROR_SLICE, string_create("failed to compute slice offset." ), NULL);
    }

    error = slice_shape(x->buffer->view->shape, x->buffer->view->rank, shape, length, arguments, length);
    if (shape == NULL)
    {
        return ERROR(ERROR_SLICE, string_create("failed to compute slice shape." ), NULL);
    }

    error = view_create(&view, offset, x->buffer->view->rank, shape, x->buffer->view->strides);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }
    free(shape);

    error = buffer_create(&result->buffer, x->buffer->runtime, x->buffer->datatype, view, x->buffer->data, x->buffer->n, false);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return NULL;
}

static nw_error_t *slice_operation_backward(tensor_t *x, uint64_t *arguments, uint64_t length, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        tensor_t *x_gradient;

        nw_error_t *error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        uint64_t *new_arguments = (uint64_t *) malloc(length * sizeof(uint64_t));
        if (new_arguments == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate new arguments of size %zu.", length * sizeof(uint64_t)), NULL);
        }

        error = reverse_slice(x->buffer->view->shape, x->buffer->view->rank, arguments, length, new_arguments, length);
        if (error != NULL)
        {
            return ERROR(ERROR_SLICE, string_create("failed to compute padding arguments."), error);
        }

        error = tensor_padding(gradient, x_gradient, new_arguments, length);
        if (error != NULL)
        {
            return ERROR(ERROR_PADDING, string_create("failed to successfully run padding operation."), error);
        }
        free(new_arguments);

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to accumulate gradient."), error);
        }
    }

    return NULL;
}

static nw_error_t *padding_operation_forward(tensor_t *x, uint64_t *arguments, uint64_t length, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(result, "result");

    uint64_t *padding_shape = (uint64_t *) malloc(x->buffer->view->rank * sizeof(uint64_t));
    if (padding_shape == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate padding shape of size %zu bytes.", x->buffer->view->rank * sizeof(uint64_t)), NULL);
    }

    uint64_t *slice_arguments = (uint64_t *) malloc(length * sizeof(uint64_t));
    if (slice_arguments == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate slice arguments of size %zu bytes.", length * sizeof(uint64_t)), NULL);
    }

    nw_error_t *error = padding(x->buffer->view->shape, x->buffer->view->rank, padding_shape, x->buffer->view->rank, arguments, length);
    if (error != NULL)
    {
        return ERROR(ERROR_PADDING, string_create("failed to compute resultant padding shape."), error);
    }

    error = reverse_padding(x->buffer->view->shape, x->buffer->view->rank, arguments, length, slice_arguments, length);
    if (error != NULL)
    {
        return ERROR(ERROR_PADDING, string_create("failed to compute slice arguments."), error);
    }

    uint64_t offset = 0;
    error = slice_offset(x->buffer->view->strides, x->buffer->view->rank, &offset, slice_arguments, length);
    if (error != NULL)
    {
        return ERROR(ERROR_SLICE, string_create("failed to compute slice offset."), error);
    }

    view_t *view;
    buffer_t *buffer;

    error = view_create(&view, 0, x->buffer->view->rank, padding_shape, NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, "failed to create view.", error);
    }

    error = buffer_create(&buffer, x->buffer->runtime, x->buffer->datatype, view, NULL, 0, true);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }
    memset(buffer->data, 0.0, buffer->size); 


    view_t *sliced_view;
    buffer_t *sliced_buffer;
    error = view_create(&sliced_view, offset, x->buffer->view->rank, x->buffer->view->shape, NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, "failed to create view.", error);
    }

    error = buffer_create(&sliced_buffer, x->buffer->runtime, x->buffer->datatype, sliced_view, buffer->data, buffer->n, false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    error = runtime_copy(x->buffer, sliced_buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_COPY, string_create("failed to copy tensor contents."), error);
    }

    result->buffer = buffer;

    return NULL;
}

static nw_error_t *padding_operation_backward(tensor_t *x, uint64_t *arguments, uint64_t length, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        nw_error_t *error = NULL;
        tensor_t *x_gradient = NULL;
        uint64_t *new_arguments = NULL;

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create tensor x_gradient."),
                         error);
        }

        new_arguments = (uint64_t *) malloc((size_t) (length * sizeof(uint64_t)));
        if (new_arguments == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION,
                         string_create("failed to allocate new arguments of size %lu bytes.",
                         (unsigned long) (length * sizeof(uint64_t))),
                         NULL);
        }

        error = reverse_padding(x->buffer->view->shape,
                                x->buffer->view->rank,
                                arguments, length,
                                new_arguments, length);
        if (error != NULL)
        {
            string_t shape_string = uint64_array_to_string(x->buffer->view->shape,
                                                           x->buffer->view->rank);
            string_t arguments_string = uint64_array_to_string(arguments, length);
            nw_error_t *new_error = ERROR(ERROR_PADDING,
                                          string_create("cannot compute slice arguments from shape %s and padding arguments %s.",
                                          shape_string, arguments_string),
                                          error);
            string_destroy(shape_string);
            string_destroy(arguments_string);
            free(new_arguments);
            return new_error;
        }

        error = tensor_slice(gradient, x_gradient, new_arguments, length);
        if (error != NULL)
        {
            string_t shape_string = uint64_array_to_string(gradient->buffer->view->shape,
                                                           gradient->buffer->view->rank);
            string_t new_arguments_string = uint64_array_to_string(new_arguments, length);
            nw_error_t *new_error = ERROR(ERROR_SLICE,
                                          string_create("cannot slice tensor shape %s with arguments %s.",
                                          shape_string, new_arguments_string),
                                          error);
            string_destroy(shape_string);
            string_destroy(new_arguments_string);
            tensor_destroy(x_gradient);
            free(new_arguments);
            return new_error;
        }

        error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL) 
        {
            free(new_arguments);
            return ERROR(ERROR_ADDITION,
                         string_create("failed to accumulate gradient."),
                         error);
        }

        free(new_arguments);
    }

    return NULL;
}

/**
 * @brief Apply structure operation forward.
 * @param structure_operation Structure operation to execute.
 * @return Error if `structure_operation` is NULL.
 *         Error if `structure_operation` failed to run.
 *         Error if operation type is unknown.
 *         NULL if `structure_operation` successfully executed.
 */
nw_error_t *structure_operation_forward(structure_operation_t *structure_operation)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");

    nw_error_t *error = NULL;

    switch (structure_operation->operation_type)
    {
    case EXPAND_OPERATION:
        error = expand_operation_forward(structure_operation->x,
                                         structure_operation->arguments,
                                         structure_operation->length,
                                         structure_operation->result);
        break;
    case PERMUTE_OPERATION:
        error = permute_operation_forward(structure_operation->x,
                                          structure_operation->arguments,
                                          structure_operation->length,
                                          structure_operation->result);
        break;
    case RESHAPE_OPERATION:
        error = reshape_operation_forward(structure_operation->x,
                                          structure_operation->arguments,
                                          structure_operation->length,
                                          structure_operation->result);
        break;
    case SLICE_OPERATION:
        error = slice_operation_forward(structure_operation->x,
                                        structure_operation->arguments,
                                        structure_operation->length,
                                        structure_operation->result);
        break;
    case PADDING_OPERATION:
        error = padding_operation_forward(structure_operation->x,
                                          structure_operation->arguments,
                                          structure_operation->length,
                                          structure_operation->result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown structure operation type %d.",
                      (int) structure_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute structure operation forward pass of type %s.",
                     structure_operation_type_string(structure_operation->operation_type)),
                     error);
    }

    structure_operation->result->requires_gradient = structure_operation->x->requires_gradient;

    return NULL;
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

    switch (structure_operation->operation_type)
    {
    case EXPAND_OPERATION:
        error = expand_operation_backward(structure_operation->x,
                                          structure_operation->arguments,
                                          structure_operation->length,
                                          gradient);
        break;
    case PERMUTE_OPERATION:
        error = permute_operation_backward(structure_operation->x,
                                           structure_operation->arguments,
                                           structure_operation->length,
                                           gradient);
        break;
    case RESHAPE_OPERATION:
        error = reshape_operation_backward(structure_operation->x,
                                           gradient);
        break;
    case SLICE_OPERATION:
        error = slice_operation_backward(structure_operation->x,
                                         structure_operation->arguments,
                                         structure_operation->length,
                                         gradient);
        break;
    case PADDING_OPERATION:
        error = padding_operation_backward(structure_operation->x,
                                           structure_operation->arguments,
                                           structure_operation->length,
                                           gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown structure operation type %d.",
                      (int) structure_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute structure operation backward pass of type %s.",
                     structure_operation_type_string(structure_operation->operation_type)),
                     error);
    }

    return NULL;
}
