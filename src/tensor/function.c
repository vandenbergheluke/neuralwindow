#include <function.h>

error_t *function_create(function_t **function, operation_t *operation, operation_type_t operation_type)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(operation, "operation");

    *function = (function_t *) malloc(sizeof(function_t));
    if (*function == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate function of size %zu bytes.", sizeof(function_t)), NULL);
    }

    (*function)->operation = operation;
    (*function)->operation_type = operation_type;
    
    return NULL;
}

void function_destroy(function_t *function)
{
    if (function == NULL)
    {
        return;
    }

    operation_destroy(function->operation, function->operation_type);
    free(function);
}

static error_t *apply_function(operation_type_t operation_type, void *type_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(type_operation, "x");
    CHECK_NULL_ARGUMENT(result, "y");

    error_t *error;
    operation_t *operation;
    function_t *function;

    error = operation_create(&operation, operation_type, type_operation);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create operation."), error);
    }

    error = function_create(&function, operation, type_operation);
    if (error != NULL)
    {
        operation_destroy(operation, type_operation);
        return ERROR(ERROR_CREATE, string_create("failed to create function operation."), error);
    }

    error = function_forward(function, result);
    if (error != NULL)
    {
        function_destroy(function);
        return ERROR(ERROR_CREATE, string_create("failed to execute function forward pass."), error);
    }
}

error_t *apply_function_unary(unary_operation_type_t unary_operation_type, tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    error_t *error;
    unary_operation_t *unary_operation;

    error = unary_operation_create(&unary_operation, unary_operation_type, x);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create unary operation."), error);
    }

    error = apply_function(UNARY_OPERATION, unary_operation, y);
    if (error != NULL)
    {
        unary_operation_destroy(unary_operation);
        return ERROR(ERROR_FORWARD, string_create("failed to apply unary function."), error);
    }
    
    return NULL;
}

error_t *apply_function_binary(binary_operation_type_t binary_operation_type, tensor_t *x, tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    error_t *error;
    tensor_t *x_brodcasted;
    tensor_t *y_brodcasted;
    binary_operation_t *binary_operation;

    error = tensor_create_empty(&x_brodcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to tensor."), error);
    }

    error = tensor_create_empty(&y_brodcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to tensor."), error);
    }
    
    error = tensor_broadcast(x, y, x_brodcasted, y_brodcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_BROADCAST, string_create("failed to broacast tensors."), error);
    }

    error = binary_operation_create(&binary_operation, binary_operation_type, x_brodcasted, y_brodcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create binary operation."), error);
    }

    error = apply_function(BINARY_OPERATION, binary_operation, z);
    if (error != NULL)
    {
        binary_operation_destroy(binary_operation);
        return ERROR(ERROR_FORWARD, string_create("failed to apply binary function."), error);
    }
    
    return NULL;
}

error_t *apply_function_reduction(reduction_operation_type_t reduction_operation_type, tensor_t *x, uint32_t *axis, uint32_t length, bool_t keep_dimension, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    error_t *error;
    reduction_operation_t *reduction_operation;

    error = reduction_operation_create(&reduction_operation, reduction_operation_type, x, axis, length, keep_dimension);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create reduction operation."), error);
    }

    error = apply_function(REDUCTION_OPERATION, reduction_operation, y);
    if (error != NULL)
    {
        reduction_operation_destroy(reduction_operation);
        return ERROR(ERROR_FORWARD, string_create("failed to apply reduction function."), error);
    }
    
    return NULL;
}

error_t *apply_function_structure(structure_operation_type_t structure_operation_type, tensor_t *x, uint32_t *arguments, uint32_t length, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    error_t *error;
    structure_operation_t *structure_operation;

    error = structure_operation_create(&structure_operation, structure_operation_type, x, arguments, length);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create structure operation."), error);
    }

    error = apply_function(STRUCTURE_OPERATION, structure_operation, y);
    if (error != NULL)
    {
        structure_operation_destroy(structure_operation);
        return ERROR(ERROR_FORWARD, string_create("failed to apply structure function."), error);
    }
    
    return NULL;
}

error_t *function_forward(function_t *function, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = operation_forward(function->operation, function->operation_type, result);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute operation forward pass."), error);
    }

    result->context = function;
    
    return NULL;
}

error_t *function_backward(function_t *function, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(function, "function");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error = operation_backward(function->operation, function->operation_type, gradient);
    if (error != NULL)
    {
        return ERROR(ERROR_BACKWARD, string_create("failed to execute operation backward pass."), error);
    }

    return NULL;
}

error_t *operation_create(operation_t **operation, operation_type_t operation_type, void *type_operation)
{
    CHECK_NULL_ARGUMENT(operation, "operation");
    CHECK_NULL_ARGUMENT(type_operation, "type_operation");

    *operation = (operation_t *) malloc(sizeof(operation_t));
    if (*operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate operation of size %zu bytes.", sizeof(operation_t)), NULL);
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
        return ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    return NULL;
}

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
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute operation type %d forward pass.", (int) operation_type), error);
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
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_BACKWARD, string_create("failed to execute operation type %d backward pass.", operation_type), error);
    }

    return NULL;
}

error_t *unary_operation_create(unary_operation_t **unary_operation, unary_operation_type_t unary_operation_type, tensor_t *x)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");

    *unary_operation = (unary_operation_t *) malloc(sizeof(unary_operation_t));
    if (*unary_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate unary operation of size %zu bytes.", sizeof(unary_operation_t)), NULL);
    }

    (*unary_operation)->operation_type = unary_operation_type;
    (*unary_operation)->x = x; 

    return NULL;
}

void unary_operation_destroy(unary_operation_t *unary_operation)
{
    if (unary_operation == NULL)
    {
        return;
    }

    free(unary_operation);
}

static error_t *exponential_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *exponential_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to combine gradient."), error);
        }
    }

    return NULL;
}

static error_t *logarithm_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *logarithm_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to combine gradient."), error);
        }
    }

    return NULL;
}

static error_t *sine_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *sine_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to combine gradient."), error);
        }
    }

    return NULL;
}

static error_t *cosine_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *cosine_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to combine gradient."), error);
        }
    }

    return NULL;
}

static error_t *square_root_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *square_root_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to combine gradient."), error);
        }
    }

    return NULL;
}

static error_t *reciprocal_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *reciprocal_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to combine gradient."), error);
        }
    }

    return NULL;
}

static error_t *copy_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error;

    if (x->buffer == NULL)
    {
        result->buffer = NULL;
    }
    else
    {
        view_t *view;
        buffer_t *buffer;

        error = view_create(&view, x->buffer->view->offset, x->buffer->view->rank,
                            x->buffer->view->shape, x->buffer->view->strides);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
        }

        error = buffer_create(&buffer, x->buffer->runtime, x->buffer->datatype, view, x->buffer->data);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
        }
        result->buffer = buffer;
    }

    if (x->gradient == NULL)
    {
        result->gradient = NULL;
    }
    else
    {
        error = tensor_create(result->gradient, NULL, NULL, NULL, false);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        }
        
        error = tensor_copy(x->gradient, result->gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to copy tensor."), error);
        }
    }
    result->requires_gradient = x->requires_gradient;
    
    return NULL;
}

static error_t *copy_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to combine gradient."), error);
        }
    }

    return NULL;
}

static error_t *contiguous_operation_forward(tensor_t *x, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *contiguous_operation_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;

    if (x->requires_gradient)
    {
        error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL) 
        {
            return ERROR(ERROR_ADDITION, string_create("failed to combine gradient."), error);
        }
    }

    return NULL;
}

error_t *unary_operation_forward(unary_operation_t *unary_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(unary_operation, "unary_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = NULL;
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
    case COPY_OPERATION:
        error = copy_operation_forward(unary_operation->x, result);
        break;
    case CONTIGUOUS_OPERATION:
        error = contiguous_operation_forward(unary_operation->x, result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) unary_operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply forward unary operation."), error);
    }

    result->requires_gradient = unary_operation->x->requires_gradient;
    
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
        error = exponential_operation_backward(unary_operation->x, gradient);
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
        error = square_root_backward(unary_operation->x, gradient);
        break;
    case RECIPROCAL_OPERATION:
        error = reciprocal_operation_backward(unary_operation->x, gradient);
        break;
    case COPY_OPERATION:
        error = copy_operation_backward(unary_operation->x, gradient);
        break;
    case CONTIGUOUS_OPERATION:
        error = contiguous_operation_backward(unary_operation->x, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) unary_operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_BACKWARD, string_create("failed to apply backward unary operation."), error);
    }
    
    return NULL;
}

error_t *binary_operation_create(binary_operation_t **binary_operation, binary_operation_type_t binary_operation_type, tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(binary_operation, "binary_operation");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    *binary_operation = (binary_operation_t *) malloc(sizeof(binary_operation_t));
    if (*binary_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate binary operation of size %zu bytes.", sizeof(binary_operation_t)), NULL);
    }

    (*binary_operation)->operation_type = binary_operation_type;
    (*binary_operation)->x = x; 
    (*binary_operation)->y = y;

    return NULL;
}

void binary_operation_destroy(binary_operation_t *binary_operation)
{
    if (binary_operation == NULL)
    {
        return NULL;
    }

    free(binary_operation);
}

static error_t *addition_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *addition_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute addition operation backward pass x operand."), error);
        }
    }

    if (y->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(y, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute addition operation backward pass y operand."), error);
        }
    }

    return NULL;
}

static error_t *subtraction_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *subtraction_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute subtraction operation backward pass x operand."), error);
        }
    }

    if (y->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(y, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute subtraction operation backward pass y operand."), error);
        }
    }

    return NULL;
}

static error_t *multiplication_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *multiplication_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute multiplication operation backward pass x operand."), error);
        }
    }

    if (y->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(y, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute multiplication operation backward pass y operand."), error);
        }
    }

    return NULL;
}

static error_t *division_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *division_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute division operation backward pass x operand."), error);
        }
    }

    if (y->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(y, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute division operation backward pass y operand."), error);
        }
    }

    return NULL;
}

static error_t *power_operation_forward(const tensor_t *x, const tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

static error_t *power_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute power operation backward pass x operand."), error);
        }
    }

    if (y->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(y, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute power operation backward pass y operand."), error);
        }
    }

    return NULL;
}

static error_t *matrix_multiplication_operation_forward(tensor_t *x, tensor_t *y, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = tensor_as_empty(x, result);    
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    return NULL;
}

error_t *matrix_multiplication_operation_backward(tensor_t *x, tensor_t *y, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(x, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute matrix multiplication operation backward pass x operand."), error);
        }
    }

    if (y->requires_gradient)
    {
        error_t *error = tensor_accumulate_gradient(y, gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_BACKWARD, string_create("failed to execute matrix multiplication operation backward pass y operand."), error);
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
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown binary operation type %d.", (int) binary_operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute binary operation forward pass."), error);
    }

    result->requires_gradient = binary_operation->x->requires_gradient || binary_operation->y->requires_gradient;
    
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
        error = subtraction_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case MULTIPLICATION_OPERATION:
        error = multiplication_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case DIVISION_OPERATION:
        error = division_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case POWER_OPERATION:
        error = power_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    case MATRIX_MULTIPLICATION_OPERATION:
        error = matrix_multiplication_operation_backward(binary_operation->x, binary_operation->y, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown binary operation type %d.", (int) binary_operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute binary operation backward pass."), error);
    }
    
    return NULL;
}

error_t *reduction_operation_create(reduction_operation_t **reduction_operation, 
                                    reduction_operation_type_t reduction_operation_type,
                                    tensor_t *x,
                                    uint32_t *axis,
                                    uint32_t rank,
                                    bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");

    *reduction_operation = (reduction_operation_t *) malloc(sizeof(reduction_operation_t));
    if (*reduction_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate reduction operation of size %zu bytes.", sizeof(reduction_operation_t)), NULL);
    }

    (*reduction_operation)->axis = (uint32_t *) malloc(rank * sizeof(uint32_t));
    if ((*reduction_operation)->axis == NULL)
    {
        free(*reduction_operation);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate reduction_operation->axis of size %zu bytes.", rank * sizeof(uint32_t)), NULL);
    }
    memcpy((*reduction_operation)->axis, axis, rank * sizeof(uint32_t));

    (*reduction_operation)->operation_type = reduction_operation_type;
    (*reduction_operation)->x = x; 
    (*reduction_operation)->rank = rank;
    (*reduction_operation)->keep_dimension = keep_dimension;

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

static error_t *summation_operation_forward(tensor_t *x, uint32_t *axis, uint32_t rank, tensor_t *result, bool_t keep_dimension)
{
   return NULL; 
}

static error_t *summation_operation_backward(tensor_t *x, uint32_t *axis, uint32_t rank, tensor_t *gradient, bool_t keep_dimension)
{
   return NULL; 
}

static error_t *maximum_operation_forward(tensor_t *x, uint32_t *axis, uint32_t rank, tensor_t *result, bool_t keep_dimension)
{
   return NULL; 
}

static error_t *maximum_operation_backward(tensor_t *x, uint32_t *axis, uint32_t rank, tensor_t *gradient, bool_t keep_dimension)
{
   return NULL; 
}

error_t *reduction_operation_forward(reduction_operation_t *reduction_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(reduction_operation, "reduction_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = NULL;
    switch (reduction_operation->operation_type)
    {
    case SUMMATION_OPERATION:
        error = summation_operation_forward(reduction_operation->x, reduction_operation->axis, reduction_operation->rank, result, reduction_operation->keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        error = maximum_operation_forward(reduction_operation->x, reduction_operation->axis, reduction_operation->rank, result, reduction_operation->keep_dimension);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown reduction operation type %d.", (int) reduction_operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute reduction operation forward pass."), error);
    }

    result->requires_gradient = reduction_operation->x->requires_gradient;

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
        error = summation_operation_backward(reduction_operation->x, reduction_operation->axis, reduction_operation->rank, gradient, reduction_operation->keep_dimension);
        break;
    case MAXIMUM_OPERATION:
        error = maximum_operation_backward(reduction_operation->x, reduction_operation->axis, reduction_operation->rank, gradient, reduction_operation->keep_dimension);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown reduction operation type %d.", (int) reduction_operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute reduction operation backward pass."), error);
    }

    return NULL;
}

error_t *structure_operation_create(structure_operation_t **structure_operation, structure_operation_type_t structure_operation_type, tensor_t *x, uint32_t *arguments, uint32_t length)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(x, "x");

    *structure_operation = (structure_operation_t *) malloc(sizeof(structure_operation_t));
    if (*structure_operation == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate structure operation of size %zu bytes.", sizeof(structure_operation_t)), NULL);
    }

    (*structure_operation)->arguments = (uint32_t *) malloc(length * sizeof(uint32_t));
    if ((*structure_operation)->arguments == NULL)
    {
        free(*structure_operation);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate structure_operation->arguments of size %zu bytes.", length * sizeof(uint32_t)), NULL);
    }
    memcpy((*structure_operation)->arguments, arguments, length * sizeof(uint32_t));

    (*structure_operation)->operation_type = structure_operation_type;
    (*structure_operation)->x = x; 
    (*structure_operation)->length = length;

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

static error_t *expand_operation_forward(tensor_t *x, uint32_t *shape, uint32_t rank, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error;
    uint32_t *strides;
    view_t *view;

    strides = (uint32_t *) malloc(rank * sizeof(uint32_t));
    if (strides == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate strides of size %zu bytes.", rank * sizeof(uint32_t)), NULL);
    }

    error = broadcast_strides(x->buffer->view->shape, x->buffer->view->rank, x->buffer->view->strides, shape, rank, strides);
    if (error != NULL)
    {
        free(strides);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize expand strides"), error);
    }

    error = view_create(&view, x->buffer->view->offset, shape, rank, strides);
    if (error != NULL)
    {
        free(strides);
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }
    free(strides);

    error = buffer_create(&result->buffer, x->buffer->runtime, x->buffer->datatype, view, x->buffer->data, x->buffer->size, false);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return NULL;
}

static error_t *expand_operation_backward(tensor_t *x, uint32_t *shape, uint32_t rank, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error;
        uint32_t length_keep_dimension;
        uint32_t length_remove_dimension;
        
        error = reverse_broadcast_length(x->buffer->view->shape, x->buffer->view->rank, shape, rank, &length_keep_dimension, &length_remove_dimension);
        uint32_t *axis_keep_dimension = (uint32_t *) malloc(sizeof(uint32_t) * length_keep_dimension);
        if (axis_keep_dimension == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate axis of size %zu bytes.", sizeof(uint32_t) * length_keep_dimension), NULL);
        }

        uint32_t *axis_remove_dimension = (uint32_t *) malloc(sizeof(uint32_t) * length_remove_dimension);
        if (axis_remove_dimension == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate axis of size %zu bytes.", sizeof(uint32_t) * length_remove_dimension), NULL);
        }

        error_t *error = reverse_broadcast_axis(x->buffer->view->shape, x->buffer->view->rank, shape, rank, axis_keep_dimension, axis_remove_dimension);
        if (error != NULL)
        {
            return ERROR(ERROR_BROADCAST, string_create("failed to get corresponding reduce axis to reverse broacast."), error);
        }

        tensor_t *x_gradient_i;
        tensor_t *x_gradient;

        error = tensor_create_empty(&x_gradient_i);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient_i tensor."), error);
        }

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_summation(gradient, x_gradient_i, axis_keep_dimension, length_keep_dimension, true);
        if (error != NULL)
        {
            return ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        }

        error = tensor_summation(x_gradient_i, x_gradient, axis_remove_dimension, length_remove_dimension, false);
        if (error != NULL)
        {
            return ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        }

        error_t *error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add gradient."), error);
        }

        tensor_destroy(x_gradient_i);
    }

    return NULL;
}

static error_t *permute_operation_forward(tensor_t *x, uint32_t *axis, uint32_t rank, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error;
    uint32_t *shape;
    uint32_t *strides;
    view_t *view;

    strides = (uint32_t *) malloc(rank * sizeof(uint32_t));
    if (strides == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate strides of size %zu bytes.", rank * sizeof(uint32_t)), NULL);
    }

    shape = (uint32_t *) malloc(rank * sizeof(uint32_t));
    if (shape == NULL)
    {
        free(strides);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate shape of size %zu bytes.", rank * sizeof(uint32_t)), NULL);
    }

    error = permute(x->buffer->view->shape, x->buffer->view->rank, x->buffer->view->strides, shape, rank, strides, axis, rank);
    if (error != NULL)
    {
        free(shape);
        free(strides);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize permuted shape and strides."), error);
    }

    error = view_create(&view, x->buffer->view->offset, shape, rank, strides);
    if (error != NULL)
    {
        free(shape);
        free(strides);
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }
    free(shape);
    free(strides);

    error = buffer_create(&result->buffer, x->buffer->runtime, x->buffer->datatype, view, x->buffer->data, x->buffer->size, false);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return NULL;
}

static error_t *permute_operation_backward(tensor_t *x, uint32_t *axis, uint32_t rank, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(axis, "axis");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error;
        uint32_t *gradient_axis;
        tensor_t *x_gradient;

        gradient_axis = (uint32_t *) malloc(rank * sizeof(uint32_t));
        if (gradient_axis == NULL)
        {
            return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate strides of size %zu bytes.", rank * sizeof(uint32_t)), NULL);
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

        error_t *error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add gradient."), error);
        }
    }

    return NULL;
}

static error_t *reshape_operation_forward(tensor_t *x, uint32_t *shape, uint32_t rank, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error;
    view_t *view;

    if (!tensor_is_contiguous(x))
    {
        return ERROR(ERROR_CONTIGUOUS, string_create("cannot reshape a non-contiguous tensor."), NULL);
    }

    error = view_create(&view, x->buffer->view->offset, shape, rank, NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&result->buffer, x->buffer->runtime, x->buffer->datatype, view, x->buffer->data, x->buffer->size, false);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return NULL;
}

static error_t *reshape_operation_backward(tensor_t *x, uint32_t *shape, uint32_t rank, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (x->requires_gradient)
    {
        error_t *error;
        tensor_t *x_gradient;

        error = tensor_create_empty(&x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_gradient tensor."), error);
        }

        error = tensor_reshape(gradient, x_gradient, x->buffer->view->shape, x->buffer->view->rank);        
        if (error != NULL)
        {
            return ERROR(ERROR_RESHAPE, string_create("failed to reshape gradient."), error);
        }

        error_t *error = tensor_accumulate_gradient(x, x_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add gradient."), error);
        }
    }

    return NULL;
}

static error_t *slice_operation_forward(tensor_t *x, uint32_t *arguments, uint32_t length, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(arguments, "arguments");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error;
    view_t *view;
    uint32_t offset = 0;
    uint32_t *shape = (uint32_t *) malloc(x->buffer->view->rank * sizeof(uint32_t));
    if (shape == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed allocate shape of size %zu bytes.", x->buffer->view->rank * sizeof(uint32_t)), NULL);
    }

    error = slice_offset(x->buffer->view->strides, x->buffer->view->rank, &offset, arguments, length);
    if (shape == NULL)
    {
        return ERROR(ERROR_SLICE, string_create("failed to compute slice offset." ), NULL);
    }

    error = slice_shape(x->buffer->view->shape, x->buffer->view->rank, shape, length, arguments, length);
    if (shape == NULL)
    {
        return ERROR(ERROR_SLICE, string_create("failed to compute slice shape." ), NULL);
    }

    error = view_create(&view, offset, shape, x->buffer->view->rank, x->buffer->view->strides);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }
    free(shape);

    error = buffer_create(&result->buffer, x->buffer->runtime, x->buffer->datatype, view, x->buffer->data, x->buffer->size, false);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    return NULL;
}

static error_t *slice_operation_backward(tensor_t *x, uint32_t *arguments, uint32_t length, tensor_t *gradient)
{
    return NULL;
}

static error_t *padding_operation_forward(tensor_t *x, uint32_t *arguments, uint32_t length, tensor_t *result)
{
    return NULL;
}

static error_t *padding_operation_backward(tensor_t *x, uint32_t *arguments, uint32_t length, tensor_t *gradient)
{
    return NULL;
}

error_t *structure_operation_forward(structure_operation_t *structure_operation, tensor_t *result)
{
    CHECK_NULL_ARGUMENT(structure_operation, "structure_operation");
    CHECK_NULL_ARGUMENT(result, "result");

    error_t *error = NULL;
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
    case SLICE_OPERATION:
        error = slice_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length, result);
        break;
    case PADDING_OPERATION:
        error = padding_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length, result);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown structure operation type %d.", (int) structure_operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute structure operation forward pass."), error);
    }

    result->requires_gradient = structure_operation->x->requires_gradient;

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
        error = expand_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length, gradient);
        break;
    case PERMUTE_OPERATION:
        error = permute_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length, gradient);
        break;
    case RESHAPE_OPERATION:
        error = reshape_operation_forward(structure_operation->x, structure_operation->arguments, structure_operation->length,  gradient);
        break;
    case SLICE_OPERATION:
        error = slice_operation_backward(structure_operation->x, structure_operation->arguments, structure_operation->length, gradient);
        break;
    case PADDING_OPERATION:
        error = padding_operation_backward(structure_operation->x, structure_operation->arguments, structure_operation->length, gradient);
        break;
    default:
        error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown structure operation type %d.", (int) structure_operation->operation_type), NULL);
        break;
    }

    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to execute structure operation backward pass."), error);
    }

    return NULL;
}