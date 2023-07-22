#include <function.h>

error_t *function_create(function_t **function, operation_t *operation, operation_type_t operation_type)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(operation, "operation");

    size_t size = sizeof(function_t);
    *function = (function_t *) malloc(size);
    if (*function == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate function of size %zu bytes.", size),
                     NULL);
    }

    // Initialize
    (*function)->operation = operation;
    (*function)->operation_type = operation_type;
    
    return NULL;
}

void function_destroy(function_t *function)
{
    if (function != NULL)
    {
        operation_destroy(function->operation, function->operation_type);
        free(function);
    }
}

error_t *function_forward(function_t *function, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = operation_forward(function->operation, function->operation_type, result);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute operation forward pass."),
                     error);
    }

    if (result->requires_gradient)
    {
        result->context = function;
    }
    
    return NULL;
}

error_t *function_backward(function_t *function, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error = operation_backward(function->operation, function->operation_type, gradient);
    if (error != NULL)
    {
        return ERROR(ERROR_BACKWARD,
                     string_create("failed to execute operation backward pass."),
                     error);
    }

    return NULL;
}

error_t *operation_create(operation_t **operation, operation_type_t operation_type, void *type_operation)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(type_operation, "type_operation");

    size_t size = sizeof(operation_t);
    *operation = (operation_t *) malloc(size);
    if (*operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate operation of size %zu bytes.", size),
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
                     string_create("unknown operation type %d.", operation_type),
                     NULL);
        break;
    }

    return NULL;
}

void operation_destroy(operation_t *operation, operation_type_t operation_type)
{
    if (operation != NULL)
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

error_t *operation_forward(operation_t *operation, operation_type_t operation_type, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error;
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
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown operation type %d", operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute operation type %d forward pass.", operation_type),
                     error);
    }

    return NULL;
}

error_t *operation_backward(operation_t *operation, operation_type_t operation_type, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;
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
                      string_create("unknown operation type %d.", operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_BACKWARD,
                     string_create("failed to execute operation type %d backward pass.", operation_type),
                     error);

    }

    return NULL;
}

error_t *unary_operation_create(unary_operation_t **unary_operation, unary_operation_type_t unary_operation_type, tensor_t *x)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");

    size_t size = sizeof(unary_operation_t);
    *unary_operation = (unary_operation_t *) malloc(size);
    if (*unary_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate unary operation of size %zu bytes.", size),
                     NULL);
    }

    (*unary_operation)->operation_type = unary_operation_type;
    (*unary_operation)->x = x; 

    return NULL;
}

void unary_operation_destroy(unary_operation_t *unary_operation)
{
    free(unary_operation);
}

error_t *unary_operation_forward(unary_operation_t *unary_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = NULL;
    switch (unary_operation->operation_type)
    {
    case EXPONENTIAL_OPERATION:
        // error = exponential_operation_forward(unary_operation->x, result);
        break;
    case LOGARITHM_OPERATION:
        // error = logarithm_operation_forward(unary_operation->x, result);
        break;
    case SIN_OPERATION:
        // error = sin_operation_forward(unary_operation->x, result);
        break;
    case SQUARE_ROOT_OPERATION:
        // error = square_root_operation_forward(unary_operation->x, result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown operation type %d.", unary_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply unary operation."),
                     error);
    }
    
    return NULL;
}

error_t *unary_operation_backward(unary_operation_t *unary_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error = NULL;
    switch (unary_operation->operation_type)
    {
    case EXPONENTIAL_OPERATION:
        // error = exponential_operation_backward(unary_operation->x, gradient);
        break;
    case LOGARITHM_OPERATION:
        // error = logarithm_operation_backward(unary_operation->x, gradient);
        break;
    case SIN_OPERATION:
        // error = sin_operation_backward(unary_operation->x, gradient);
        break;
    case SQUARE_ROOT_OPERATION:
        // error = square_root_operation_backward(unary_operation->x, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown operation type %d.", unary_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply unary operation."),
                     error);
    }
    
    return NULL;
}

error_t *binary_operation_create(binary_operation_t **binary_operation,
                                 binary_operation_type_t binary_operation_type,
                                 tensor_t *x,
                                 tensor_t *y)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    size_t size = sizeof(binary_operation_t);
    *binary_operation = (binary_operation_t *) malloc(size);
    if (*binary_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate binary operation of size %zu bytes.", size),
                     NULL);
    }

    // Initialize
    (*binary_operation)->operation_type = binary_operation_type;
    (*binary_operation)->x = x; 
    (*binary_operation)->y = y;

    return NULL;
}

void binary_operation_destroy(binary_operation_t *binary_operation)
{
    free(binary_operation);
}

error_t *addition_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    error_t *error = runtime_addition(x->buffer, y->buffer, z->buffer);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute addition operation forward pass."),
                     error);
    }

    return NULL;
}

error_t *addition_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD,
                         string_create("failed to execute addition operation backward pass x operand."),
                         error);
        }
    }

    if (y->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(y, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD,
                         string_create("failed to execute addition operation backward pass y operand."),
                         error);
        }
    }

    return NULL;
}

error_t *binary_operation_forward(const binary_operation_t *binary_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = NULL;
    switch (binary_operation->operation_type)
    {
    case ADDITION_OPERATION:
        error = addition_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case SUBTRACTION_OPERATION:
        // error = subtraction_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case MULTIPLICATION_OPERATION:
        // error = multiplication_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case DIVISION_OPERATION:
        // error = division_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        // error = matrix_multiplication_operation_forward(binary_operation->x, binary_operation->y, result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown binary operation type %d.", binary_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute binary operation forward pass."),
                     error);
    }
    
    return NULL;
}

error_t *binary_operation_backward(binary_operation_t *binary_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error = NULL;
    switch (binary_operation->operation_type)
    {
    case ADDITION_OPERATION:
        error = addition_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case SUBTRACTION_OPERATION:
        // error = subtraction_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case MULTIPLICATION_OPERATION:
        // error = multiplication_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case DIVISION_OPERATION:
        // error = division_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        // error = matrix_multiplication_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown binary operation type %d.", binary_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute binary operation backward pass."),
                     error);
    }
    
    return NULL;
}

error_t *reduction_operation_create(reduction_operation_t **reduction_operation,
                                    reduction_operation_type_t reduction_operation_type,
                                    tensor_t *x, uint32_t *axis, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(x, "x");

    size_t size = sizeof(reduction_operation_t);
    *reduction_operation = (reduction_operation_t *) malloc(size);
    if (*reduction_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate reduction operation of size %zu bytes.", size),
                     NULL);
    }

    // Initialize
    (*reduction_operation)->operation_type = reduction_operation_type;
    (*reduction_operation)->x = x; 
    (*reduction_operation)->axis = axis;
    (*reduction_operation)->keep_dimension = keep_dimension;

    return NULL;
}

void reduction_operation_destroy(reduction_operation_t *reduction_operation)
{
    free(reduction_operation);
}

error_t *reduction_operation_forward(reduction_operation_t *reduction_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = NULL;
    switch (reduction_operation->operation_type)
    {
    case SUMMATION_OPERATION:
        // error = summation_operation_forward(reduction_operation->x,
        //                                     reduction_operation->axis,
        //                                     result,
        //                                     reduction_operation->keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        // error = maximum_operation_forward(reduction_operation->x,
        //                                   reduction_operation->axis,
        //                                   result,
        //                                   reduction_operation->keep_dimension);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown reduction operation type %d.", reduction_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute reduction operation forward pass."),
                     error);
    }

    return NULL;
}

error_t *reduction_operation_backward(reduction_operation_t *reduction_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error = NULL;
    switch (reduction_operation->operation_type)
    {
    case SUMMATION_OPERATION:
        // error = summation_operation_backward(reduction_operation->x,
        //                                      reduction_operation->axis,
        //                                      gradient,
        //                                      reduction_operation->keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        // error = maximum_operation_backward(reduction_operation->x,
        //                                    reduction_operation->axis,
        //                                    gradient,
        //                                    reduction_operation->keep_dimension);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown reduction operation type %d.", reduction_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute reduction operation backward pass."),
                     error);
    }

    return NULL;
}

error_t *structure_operation_create(structure_operation_t **structure_operation,
                                    structure_operation_type_t structure_operation_type,
                                    tensor_t *x,
                                    void *arguments)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");

    size_t size = sizeof(structure_operation_t);
    *structure_operation = (structure_operation_t *) malloc(size);
    if (*structure_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate structure operation of size %zu bytes.", size),
                     NULL);
    }

    // Initialize
    (*structure_operation)->operation_type = structure_operation_type;
    (*structure_operation)->x = x; 
    (*structure_operation)->arguments = arguments;

    return NULL;
}

void structure_operation_destroy(structure_operation_t *structure_operation)
{
    free(structure_operation);
}

error_t *structure_operation_forward(structure_operation_t *structure_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = NULL;
    switch (structure_operation->operation_type)
    {
    case EXPAND_OPERATION:
        // error = expand_operation_forward(structure_operation->x, structure_operation->arguments, result);
        break;
    case PERMUTE_OPERATION:
        // error = permute_operation_forward(structure_operation->x, structure_operation->arguments, result);
        break;
    case RESHAPE_OPERATION:
        // error = reshape_operation_forward(structure_operation->x, structure_operation->arguments, result);
        break;
    case PADDING_OPERATION:
        // error = padding_operation_forward(structure_operation->x, structure_operation->arguments, result);
        break;
    case SLICE_OPERATION:
        // error = slice_operation_forward(structure_operation->x, structure_operation->arguments, result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown structure operation type %d.", structure_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute structure operation forward pass."),
                     error);
    }

    return NULL;
}

error_t *structure_operation_backward(structure_operation_t *structure_operation, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error = NULL;
    switch (structure_operation->operation_type)
    {
    case EXPAND_OPERATION:
        // error = expand_operation_forward(structure_operation->x, structure_operation->arguments, gradient);
        break;
    case PERMUTE_OPERATION:
        // error = permute_operation_forward(structure_operation->x, structure_operation->arguments, gradient);
        break;
    case RESHAPE_OPERATION:
        // error = reshape_operation_forward(structure_operation->x, structure_operation->arguments, gradient);
        break;
    case PADDING_OPERATION:
        // error = padding_operation_forward(structure_operation->x, structure_operation->arguments, gradient);
        break;
    case SLICE_OPERATION:
        // error = slice_operation_forward(structure_operation->x, structure_operation->arguments, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                      string_create("unknown structure operation type %d.", structure_operation->operation_type),
                      NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to execute structure operation backward pass."),
                     error);
    }

    return NULL;
}