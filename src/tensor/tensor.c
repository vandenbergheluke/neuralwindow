#include <tensor.h>
#include <stack.h>
#include <map.h>
#include <init.h>

error_t *tensor_create(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient)
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

error_t *tensor_copy(const tensor_t *source_tensor, tensor_t *destination_tensor)
{
    CHECK_NULL_ARGUMENT(source_tensor, "source_tensor");
    CHECK_NULL_ARGUMENT(destination_tensor, "destination_tensor");

    error_t *error;
    error = apply_function_unary(COPY_OPERATION, source_tensor, destination_tensor);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to copy tensors."), error);
    }
    
    return NULL;
}

error_t *tensor_broadcast(const tensor_t *x_original, const tensor_t *y_original, tensor_t *x_broadcasted, tensor_t *y_broadcasted)
{
    CHECK_NULL_ARGUMENT(x_original, "x_original");
    CHECK_NULL_ARGUMENT(y_original, "y_original");
    CHECK_NULL_ARGUMENT(x_broadcasted, "x_broadcasted");
    CHECK_NULL_ARGUMENT(y_broadcasted, "y_broadcasted");

    error_t *error;
    uint32_t *x_shape = x_original->buffer->view->shape; 
    uint32_t *x_rank = x_original->buffer->view->rank; 
    uint32_t *y_shape = y_original->buffer->view->shape; 
    uint32_t *y_rank = y_original->buffer->view->rank; 
    uint32_t broadcasted_rank = MAX(x_rank, y_rank);
    uint32_t *broadcasted_shape = (uint32_t *) malloc(broadcasted_rank * sizeof(uint32_t));
    if (broadcasted_shape == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate broadcast shape of %zu bytes.", broadcasted_rank * sizeof(uint32_t)), NULL);
    }

    error = broadcast_shapes(x_shape, x_rank, y_shape, y_rank, broadcasted_shape, broadcasted_rank);
    if (error != NULL)
    {
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor shapes."), error);
    }

    error = tensor_expand(x_original, broadcasted_shape, broadcasted_rank, x_broadcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor x."), error);
    }

    error = tensor_expand(y_original, broadcasted_shape, broadcasted_rank, y_broadcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor y."), error);
    }

    return NULL;
}

error_t *tensor_expand(const tensor_t *x, const uint32_t *shape, uint32_t rank, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    error_t *error;
    error = apply_function_structure(EXPAND_OPERATION, x, shape, rank, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to broadcast tensor x."), error);
    }

    return NULL;
}

error_t *tensor_addition(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    error_t *error;
    tensor_t *x_brodcasted;
    tensor_t *y_brodcasted;

    error = tensor_create(&x_brodcasted, NULL, NULL, NULL, false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to tensor."), error);
    }

    error = tensor_create(&y_brodcasted, NULL, NULL, NULL, false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to tensor."), error);
    }
    
    error = tensor_broadcast(x, y, x_brodcasted, y_brodcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_BROADCAST, string_create("failed to broacast tensors."), error);
    }

    error = apply_function_binary(ADDITION_OPERATION, x_brodcasted, y_brodcasted, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to add tensors."), error);
    }

    return NULL;
}

error_t *tensor_matrix_multiplication(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    error_t *error;
    error = apply_function_binary(MATRIX_MULTIPLICATION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to matrix multiply tensors."), error);
    }

    return NULL;
}

static error_t *topological_sort(tensor_t *tensor, map_t *visited, stack_t *tensors)
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

    error_t *error;
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
            error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, string_create("unknown operation type %d.", (int) tensor->context->operation), NULL);
        }

        if (error != NULL)
        {
            return ERROR(ERROR_SORT, string_create("failed to topologically sort computational graph."), error);
        }
    }

    error = map_set(visited, id, tensor);
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

error_t *tensor_as_ones(const tensor_t *x, tensor_t *y)
{
    view_t *view;
    buffer_t *buffer;
    error_t *error;

    error = view_create(&view, x->buffer->view->offset,  x->buffer->view->rank, x->buffer->view->shape, x->buffer->view->strides);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&buffer, x->buffer->runtime, x->buffer->datatype, view, NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    y->buffer = buffer;

    error = init_ones(y);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to initialize gradient with ones."), error);
    }

    return NULL;
}

error_t *tensor_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");

    error_t *error;
    stack_t *tensors;
    map_t *visited;

    if (gradient == NULL)
    {
        error = tensor_create(&x->gradient, NULL, NULL, NULL, false);
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

    topological_sort(x, visited, tensors);
    while (tensors->size > 0)
    {
        tensor_t *y;
        error = stack_pop(tensors, &y);
        if (error != NULL)
        {
            return ERROR(ERROR_DESTROY, string_create("failed to pop tensor from stack"), error);
        }
        function_backward(y->context, y->gradient);
    }
    map_destroy(visited);
    stack_destroy(tensors);
    
    return NULL;
}

error_t *tensor_accumulate_gradient(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    error_t *error;
    tensor_t *updated_gradient;

    error = tensor_create(&updated_gradient, NULL, NULL, NULL, false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create updated_gradient."), error);
    }

    if (x->gradient == NULL)
    {
        error = tensor_copy(gradient, updated_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_COPY, string_create("failed to copy gradient."), NULL);
        }
    }
    else
    {
        error = tensor_addition(x->gradient, gradient, updated_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add gradient."), NULL);
        }
        tensor_destroy(x->gradient);
    }
    x->gradient = updated_gradient;

    return NULL;
}