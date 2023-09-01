#include <tensor.h>
#include <stack.h>
#include <map.h>
#include <init.h>
#include <function.h>
#include <buffer.h>
#include <view.h>

nw_error_t *tensor_create(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient, bool_t lock)
{
    CHECK_NULL_ARGUMENT(tensor, "tensor");

    static uint64_t id = 0;

    *tensor = (tensor_t *) malloc(sizeof(tensor_t));
    if (*tensor == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate tensor of size %zu bytes.", sizeof(tensor_t)), NULL);
    }

    (*tensor)->id = id++;
    (*tensor)->buffer = buffer;
    (*tensor)->context = context;
    (*tensor)->gradient = gradient;
    (*tensor)->requires_gradient = requires_gradient;
    (*tensor)->lock = lock;

    return NULL;
}

void tensor_destroy(tensor_t *tensor)
{
    if (tensor == NULL)
    {
        return;
    }

    buffer_destroy(tensor->buffer);
    tensor_destroy(tensor->gradient);
    function_destroy(tensor->context);
    free(tensor);
}

nw_error_t *tensor_create_empty(tensor_t **tensor)
{
    CHECK_NULL_ARGUMENT(tensor, "tensor");

    nw_error_t *error = tensor_create(tensor, NULL, NULL, NULL, false, false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_copy(const tensor_t *source_tensor, tensor_t *destination_tensor)
{
    CHECK_NULL_ARGUMENT(source_tensor, "source_tensor");
    CHECK_NULL_ARGUMENT(destination_tensor, "destination_tensor");

    nw_error_t *error = apply_function_unary(COPY_OPERATION, source_tensor, destination_tensor);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to copy tensors."), error);
    }
    
    return NULL;
}

nw_error_t *tensor_broadcast(const tensor_t *x_original, const tensor_t *y_original, tensor_t *x_broadcasted, tensor_t *y_broadcasted)
{
    CHECK_NULL_ARGUMENT(x_original, "x_original");
    CHECK_NULL_ARGUMENT(y_original, "y_original");
    CHECK_NULL_ARGUMENT(x_broadcasted, "x_broadcasted");
    CHECK_NULL_ARGUMENT(y_broadcasted, "y_broadcasted");

    nw_error_t *error;
    uint64_t *x_shape = x_original->buffer->view->shape; 
    uint64_t x_rank = x_original->buffer->view->rank; 
    uint64_t *y_shape = y_original->buffer->view->shape; 
    uint64_t y_rank = y_original->buffer->view->rank; 
    uint64_t broadcasted_rank = MAX(x_rank, y_rank);
    uint64_t *broadcasted_shape = (uint64_t *) malloc(broadcasted_rank * sizeof(uint64_t));
    if (broadcasted_shape == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate broadcast shape of %zu bytes.", broadcasted_rank * sizeof(uint64_t)), NULL);
    }

    error = broadcast_shapes(x_shape, x_rank, y_shape, y_rank, broadcasted_shape, broadcasted_rank);
    if (error != NULL)
    {
        free(broadcasted_shape);
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor shapes."), error);
    }

    error = tensor_expand(x_original, broadcasted_shape, broadcasted_rank, x_broadcasted);
    if (error != NULL)
    {
        free(broadcasted_shape);
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor x."), error);
    }

    error = tensor_expand(y_original, broadcasted_shape, broadcasted_rank, y_broadcasted);
    if (error != NULL)
    {
        free(broadcasted_shape);
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor y."), error);
    }

    free(broadcasted_shape);

    return NULL;
}

nw_error_t *tensor_expand(const tensor_t *x, const uint64_t *shape, uint64_t rank, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = apply_function_structure(EXPAND_OPERATION, x, shape, rank, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to broadcast tensor x."), error);
    }

    return NULL;
}

nw_error_t *tensor_addition(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = apply_function_binary(ADDITION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to add tensors."), error);
    }

    return NULL;
}

nw_error_t *tensor_subtraction(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = apply_function_binary(SUBTRACTION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to subtract tensors."), error);
    }

    return NULL;
}

nw_error_t *tensor_division(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = apply_function_binary(DIVISION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to divide tensors."), error);
    }

    return NULL;
}

nw_error_t *tensor_multiplication(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = apply_function_binary(MULTIPLICATION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to multiply tensors."), error);
    }

    return NULL;
}

nw_error_t *tensor_power(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = apply_function_binary(POWER_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply power to tensors."), error);
    }

    return NULL;
}

nw_error_t *tensor_matrix_multiplication(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = apply_function_binary(MATRIX_MULTIPLICATION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to matrix multiply tensors."), error);
    }

    return NULL;
}

nw_error_t *tensor_summation(const tensor_t *x, tensor_t *y, const uint64_t *axis, uint64_t rank, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(axis, "axis");

    nw_error_t *error = apply_function_reduction(SUMMATION_OPERATION, x, axis, rank, keep_dimension, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to reduce tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_maximum(const tensor_t *x, tensor_t *y, const uint64_t *axis, uint64_t rank, bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(axis, "axis");

    nw_error_t *error = apply_function_reduction(MAXIMUM_OPERATION, x, axis, rank, keep_dimension, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to reduce tensor."), error);
    }

    return NULL;
}

bool_t tensor_is_contiguous(const tensor_t *x)
{
    if (x == NULL || x->buffer == NULL || x->buffer->view == NULL)
    {
        return false;
    }
    return is_contiguous(x->buffer->view->shape, x->buffer->view->rank, x->buffer->view->strides);
}

nw_error_t *tensor_reshape(const tensor_t *x, tensor_t *y, const uint64_t *shape, uint64_t rank)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error;

    if (!tensor_is_contiguous(x))
    {
        tensor_t *x_contiguous;
        error = tensor_create_empty(&x_contiguous);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create x_contiguous tensor."), error);
        }

        error = tensor_contiguous(x, x_contiguous);
        if (error != NULL)
        {
            return ERROR(ERROR_CONTIGUOUS, string_create("failed to apply contiguous operation to tensor."), error);
        }

        error = apply_function_structure(RESHAPE_OPERATION, x_contiguous, shape, rank, y);
        if (error != NULL)
        {
            return ERROR(ERROR_FORWARD, string_create("failed to reshape tensor."), error);
        }
    }
    else
    {
        error = apply_function_structure(RESHAPE_OPERATION, x, shape, rank, y);
        if (error != NULL)
        {
            return ERROR(ERROR_FORWARD, string_create("failed to reshape tensor."), error);
        }
    }

    return NULL;
}

nw_error_t *tensor_permute(const tensor_t *x, tensor_t *y, uint64_t *axis, uint64_t rank)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(axis, "axis");

    nw_error_t *error = apply_function_structure(PERMUTE_OPERATION, x, axis, rank, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to permute tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_slice(const tensor_t *x, tensor_t *y, uint64_t *arguments, uint64_t length)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    nw_error_t *error = apply_function_structure(SLICE_OPERATION, x, arguments, length, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to slice tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_padding(const tensor_t *x, tensor_t *y, uint64_t *arguments, uint64_t length)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    nw_error_t *error = apply_function_structure(PADDING_OPERATION, x, arguments, length, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to pad tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_contiguous(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = apply_function_unary(CONTIGUOUS_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to permute tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_logarithm(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = apply_function_unary(LOGARITHM_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply logarithm to tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_sine(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = apply_function_unary(SINE_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply logarithm to tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_cosine(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = apply_function_unary(COSINE_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply logarithm to tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_exponential(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = apply_function_unary(EXPONENTIAL_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply exp to tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_square_root(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = apply_function_unary(SQUARE_ROOT_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply squart root to tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_reciprocal(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = apply_function_unary(RECIPROCAL_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply reciprocal to tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_negation(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = apply_function_unary(NEGATION_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply negation to tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_constant(void *constant, datatype_t datatype, runtime_t runtime, tensor_t *x)
{
    if (!tensor_is_empty(x))
    {
        return ERROR(ERROR_CREATE, string_create("tensor x is not empty."), NULL);
    }

    view_t *view;
    buffer_t *buffer;
    nw_error_t *error;

    error = view_create(&view, 0, 1, (uint64_t[]){1}, NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&buffer, runtime, datatype, view, constant, datatype_size(datatype), true);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    x->buffer = buffer;

    return NULL;
}

static nw_error_t *topological_sort(tensor_t *tensor, map_t *visited, stack_t *tensors)
{
    if (tensor == NULL)
    {
        return NULL;
    }

    string_t id = string_create("%ld", tensor->id);
    if (map_contains(visited, id))
    {
        string_destroy(id);
        return NULL;
    }

    nw_error_t *error;
    if (tensor->context != NULL)
    {
        switch (tensor->context->operation_type)
        {
        case UNARY_OPERATION:
            error = topological_sort(tensor->context->operation->unary_operation->x, visited, tensors);
            break;
        case BINARY_OPERATION:
            error = topological_sort(tensor->context->operation->binary_operation->x, visited, tensors);
            error = topological_sort(tensor->context->operation->binary_operation->y, visited, tensors);
            break;
        case REDUCTION_OPERATION:
            error = topological_sort(tensor->context->operation->reduction_operation->x, visited, tensors);
            break;
        case STRUCTURE_OPERATION:
            error = topological_sort(tensor->context->operation->structure_operation->x, visited, tensors);
            break;
        default:
            error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) tensor->context->operation_type), NULL);
        }

        if (error != NULL)
        {
            return ERROR(ERROR_SQUARE_ROOT, string_create("failed to topologically sort computational graph."), error);
        }
    }

    error = map_set(visited, id, NULL);
    if (error != NULL)
    {
        string_destroy(id);
        return ERROR(ERROR_ADDITION, string_create("failed add tensor to map."), error);
    }

    error = stack_push(tensors, tensor);
    if (error != NULL)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to push tensor."), error);
    }

    return NULL;
}

nw_error_t *tensor_as_zeroes(const tensor_t *x, tensor_t *y)
{
    if (!tensor_is_empty(y))
    {
        return ERROR(ERROR_CREATE, string_create("tensor y is not empty."), NULL);
    }

    nw_error_t *error;

    error = tensor_as_empty(x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = init_zeroes(y);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to initialize gradient with zeroes."), error);
    }

    return NULL;
}

nw_error_t *tensor_as_tensor(const tensor_t *x, tensor_t *y)
{
    if (!tensor_is_empty(y))
    {
        return ERROR(ERROR_CREATE, string_create("tensor y is not empty."), NULL);
    }

    view_t *view;
    buffer_t *buffer;
    nw_error_t *error;

    error = view_create(&view, x->buffer->view->offset, x->buffer->view->rank, x->buffer->view->shape, x->buffer->view->strides);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&buffer, x->buffer->runtime, x->buffer->datatype, view, x->buffer->data, x->buffer->size, false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    y->buffer = buffer;

    return NULL;
}

nw_error_t *tensor_as_ones(const tensor_t *x, tensor_t *y)
{
    if (!tensor_is_empty(y))
    {
        return ERROR(ERROR_CREATE, string_create("tensor y is not empty."), NULL);
    }

    nw_error_t *error;

    error = tensor_as_empty(x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = init_ones(y);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to initialize gradient with ones."), error);
    }

    return NULL;
}

bool_t tensor_is_empty(const tensor_t *x)
{
    return !(x == NULL || x->buffer != NULL || x->gradient != NULL || x->context != NULL);
}

nw_error_t *tensor_as_empty(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    if (!tensor_is_empty(y))
    {
        return ERROR(ERROR_CREATE, string_create("tensor y is not empty."), NULL);
    }

    view_t *view;
    buffer_t *buffer;
    nw_error_t *error;

    error = view_create(&view, 0,  x->buffer->view->rank, x->buffer->view->shape, NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&buffer, x->buffer->runtime, x->buffer->datatype, view, NULL, x->buffer->size, true);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    y->buffer = buffer;

    return NULL;
}

nw_error_t *tensor_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");

    nw_error_t *error;
    stack_t *tensors;
    map_t *visited;

    if (gradient == NULL)
    {
        error = tensor_create_empty(&x->gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to gradient tensor."), error);
        }

        error = tensor_as_ones(x, x->gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize gradient tensor with ones."), error);
        }
    }

    error = stack_create(&tensors);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create stack."), error);
    }

    error = map_create(&visited);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create map."), error);
    }

    error = topological_sort(x, visited, tensors);
    if (error != NULL)
    {
        return ERROR(ERROR_SQUARE_ROOT, string_create("failed to topologically sort nodes."), error);
    }

    while (tensors->size > 0)
    {
        tensor_t *y;
        error = stack_pop(tensors, (void **) &y);
        if (error != NULL)
        {
            return ERROR(ERROR_DESTROY, string_create("failed to pop tensor from stack"), error);
        }
        function_backward(y->context, y->gradient);
        if (!y->lock)
        {
            tensor_destroy(y);
        }
    }
    map_destroy(visited);
    stack_destroy(tensors);
    
    return NULL;
}

nw_error_t *tensor_accumulate_gradient(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error;

    if (x->gradient == NULL)
    {
        error = tensor_create_empty(&x->gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
        }
        
        error = tensor_as_tensor(gradient, x->gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create add gradient."), error);
        }

        // Avoid destroying data in gradient when propogated to operands.
        gradient->buffer->copy = false;
        x->gradient->buffer->copy = true;
    }
    else
    {
        tensor_t *updated_gradient;

        error = tensor_create_empty(&updated_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create updated_gradient."), error);
        }
        error = tensor_addition(x->gradient, gradient, updated_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add gradient."), NULL);
        }
        tensor_destroy(x->gradient);
        x->gradient = updated_gradient;
    }

    return NULL;
}
