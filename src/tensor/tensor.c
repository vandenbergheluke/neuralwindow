#include <tensor.h>
#include <queue.h>
#include <map.h>
#include <init.h>

error_t *tensor_create(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(tensor, "tensor");
    CHECK_NULL_ARGUMENT(buffer, "tensor");

    static uint64_t id = 0;
    size_t size = sizeof(tensor_t);

    *tensor = malloc(size);
    if (*tensor == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate tensor of size %zu bytes.", size),
                     NULL);
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
    if (tensor != NULL)
    {
        buffer_destroy(tensor->buffer);
        tensor_destroy(tensor->gradient);
        function_destroy(tensor->context);
        free(tensor);
    }
}

error_t *tensor_accumulate_gradient(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    if (!view_shape_equal(x->buffer->view, gradient->buffer->view))
    {
        return ERROR(ERROR_SHAPE_CONFLICT,
                     string_create("shape conflict between x and gradient."),
                     NULL);
    }

    if (x->buffer->datatype != gradient->buffer->datatype)
    {
        return ERROR(ERROR_DATATYPE_CONFLICT,
                     string_create("datatype conflict between tensor and gradient."),
                     NULL);
    }

    error_t *error;
    view_t *view;
    error = view_create(&view,
                        x->buffer->view->offset,
                        x->buffer->view->rank,
                        x->buffer->view->shape,
                        x->buffer->view->strides);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create view."),
                     error);
    }

    buffer_t *buffer;
    error = buffer_create(&buffer, x->buffer->runtime, x->buffer->datatype, view, NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create buffer."),
                     error);
    }
    
    tensor_t *new_gradient;
    error = tensor_create(&new_gradient, buffer, NULL, NULL, false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                        string_create("failed to create new_gradient."),
                        error);
    }
    if (x->gradient == NULL)
    {
        error = tensor_copy(gradient, new_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_COPY,
                         string_create("failed to copy gradient."),
                         NULL);
        }
        x->gradient = new_gradient;
    }
    else
    {
        error = tensor_addition(x->gradient, gradient, new_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_ADDITION,
                         string_create("failed to add gradient."),
                         NULL);
        }
        tensor_destroy(x->gradient);
        x->gradient = new_gradient;
    }


    return NULL;
}

error_t *tensor_addition(tensor_t *x, tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    error_t *error;

    binary_operation_t *binary_operation;
    error = binary_operation_create(&binary_operation, ADDITION_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create binary addition operation."),
                     error);
    }

    operation_t *operation;
    error = operation_create(&operation, BINARY_OPERATION, binary_operation);
    if (error != NULL)
    {
        binary_operation_destroy(binary_operation);
        return ERROR(ERROR_CREATE,
                     string_create("failed to create operation."),
                     error);
    }

    function_t *function;
    error = function_create(&function, operation, BINARY_OPERATION);
    if (error != NULL)
    {
        binary_operation_destroy(binary_operation);
        operation_destroy(operation, BINARY_OPERATION);
        return ERROR(ERROR_CREATE,
                     string_create("failed to create function operation."),
                     error);
    }

    error = function_forward(function, z);
    if (error != NULL)
    {
        binary_operation_destroy(binary_operation);
        operation_destroy(operation, BINARY_OPERATION);
        function_destroy(function);
        return ERROR(ERROR_CREATE,
                     string_create("failed to execute function forward pass."),
                     error);
    }
    
    return NULL;
}

error_t *tensor_matrix_multiplication(tensor_t *x, tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    error_t *error;

    binary_operation_t *binary_operation;
    error = binary_operation_create(&binary_operation, MATRIX_MULTIPLICATION_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create binary matrix multiplication operation."),
                     error);
    }

    operation_t *operation;
    error = operation_create(&operation, BINARY_OPERATION, binary_operation);
    if (error != NULL)
    {
        binary_operation_destroy(binary_operation);
        return ERROR(ERROR_CREATE,
                     string_create("failed to create operation."),
                     error);
    }

    function_t *function;
    error = function_create(&function, operation, BINARY_OPERATION);
    if (error != NULL)
    {
        binary_operation_destroy(binary_operation);
        operation_destroy(operation, BINARY_OPERATION);
        return ERROR(ERROR_CREATE,
                     string_create("failed to create function operation."),
                     error);
    }

    error = function_forward(function, z);
    if (error != NULL)
    {
        binary_operation_destroy(binary_operation);
        operation_destroy(operation, BINARY_OPERATION);
        function_destroy(function);
        return ERROR(ERROR_CREATE,
                     string_create("failed to execute function forward pass."),
                     error);
    }
    
    return NULL;
}

error_t *tensor_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");

    error_t *error;

    if (gradient == NULL)
    {
        view_t *view;
        error = view_create(&view,
                            x->buffer->view->offset, 
                            x->buffer->view->rank,
                            x->buffer->view->shape,
                            x->buffer->view->strides);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create view."),
                         error);
        }

        buffer_t *buffer;
        error = buffer_create(&buffer, x->buffer->runtime, x->buffer->datatype, view, NULL);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create buffer."),
                         error);
        }

        error = tensor_create(&gradient, buffer, NULL, NULL, false);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create tensor."),
                         error);
        }

        error = init_ones(gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to initialize gradient with ones."),
                         error);
        }
    }

    tensor_destroy(x->gradient);
    x->gradient = gradient;

    queue_t *queue;
    error = queue_create(&queue);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create queue."), 
                     error);
    }

    map_t *map;
    error = map_create(&map);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create map."), 
                     error);
    }

    error = queue_enqueue(queue, x);
    if (error != NULL)
    {
        return ERROR(ERROR_ADDITION,
                     string_create("failed to enqueue tensor."), 
                     error);
    }

    while (queue->size > 0)
    {
        tensor_t *y;
        queue_dequeue(queue, (void **) &y);
        string_t id = string_create("%ld", y->id);
        if (y->context != NULL && !map_contains(map, id))
        {
            error = function_backward(y->context, y->gradient);
            if (error != NULL)
            {
                return ERROR(ERROR_BACKWARD, 
                             string_create("failed to execute function backward pass."),
                             error);
            }

            tensor_t *operands[2];
            switch (y->context->operation_type)
            {
            case UNARY_OPERATION:
                operands[0] = x->context->operation->unary_operation->x;
                operands[1] = NULL;
                break;
            case BINARY_OPERATION:
                operands[0] = x->context->operation->binary_operation->x;
                operands[1] = x->context->operation->binary_operation->y;
                break;
            case REDUCTION_OPERATION:
                operands[0] = x->context->operation->reduction_operation->x;
                operands[1] = NULL;
                break;
            case STRUCTURE_OPERATION:
                operands[0] = x->context->operation->structure_operation->x;
                operands[1] = NULL;
                break;
            default:
                operands[0] = NULL;
                operands[1] = NULL;
            }

            for (int i = 0; i < 2; i++) 
            {
                if (operands[i] == NULL)
                {
                    continue;
                }

                error = queue_enqueue(queue, operands[i]);
                if (error != NULL)
                {
                    return ERROR(ERROR_ADDITION,
                                 string_create("failed to enqueue tensor."), 
                                 error);
                }
            }
            error = map_set(map, id, y);
            if (error != NULL)
            {
                return ERROR(ERROR_ADDITION,
                             string_create("failed add tensor to map."), 
                             error);
            }
            function_destroy(y->context);
        } else
        {
            string_destroy(id);
        }
    }
    map_destroy(map);
    queue_destroy(queue);
    
    return NULL;
}

error_t *tensor_copy(tensor_t *source_tensor, tensor_t *destination_tensor)
{
    CHECK_NULL_ARGUMENT(source_tensor, "source_tensor");
    CHECK_NULL_ARGUMENT(destination_tensor, "destination_tensor");

    memcpy(destination_tensor->buffer->data, 
           source_tensor->buffer->data,
           view_size(destination_tensor->buffer->view) * datatype_size(destination_tensor->buffer->datatype));
    
    return NULL;
}