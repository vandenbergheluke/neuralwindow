/**
 * @file tensor.c
 * @brief High Level Operations - Tensor Interface
 */

#include <tensor.h>
#include <stack.h>
#include <map.h>
#include <function.h>
#include <buffer.h>
#include <view.h>
#include <string.h>
#include <math.h>
#include <id_pool.h>

bool_t no_gradient = false;
static id_pool_t *id_pool = NULL;
static uint64_t id = 0;

/**
 * @brief Dynamically memory allocate and initialize a tensor.
 * 
 * @param[out] tensor Pointer to allocated tensor memory.
 * @param[in] buffer The underlying data storage of the tensor.
 * @param[in] context A record of the operation that generated the tensor. Used to 
 *                    build a directed acyclic graph (DAG) for the automatic
 *                    differentiation engine.  
 * @param[in] gradient The gradient associated with the tensor.
 * @param[in] requires_gradient Flag to indicate whether the tensor requires its 
 *                              corresponding gradient to be computed.
 * @return Error if received NULL argument for tensor.
 *         Error if no sufficient memory could be dynamically allocate for the tensor.
 *         NULL if tensor was created successfully.
 */
nw_error_t *tensor_create(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient, bool_t persist)
{
    CHECK_NULL_ARGUMENT(tensor, "tensor");

    if (!id_pool)
    {
        nw_error_t *error = id_pool_create(&id_pool);
        if (error)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create id pool."), error);
        }
    }

    *tensor = (tensor_t *) malloc(sizeof(tensor_t));
    if (!*tensor)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(tensor_t)), NULL);
    }

    if (id_pool_is_empty(id_pool))
    {
        (*tensor)->id = id++;
    }
    else
    {
        nw_error_t *error = id_pool_get(id_pool, &(*tensor)->id);
        if (error)
        {
            free(*tensor);
            return ERROR(ERROR_GET, string_create("failed to get id."), error);
        }
    }
    (*tensor)->buffer = buffer;
    (*tensor)->context = context;
    (*tensor)->gradient = gradient;
    (*tensor)->requires_gradient = requires_gradient;
    (*tensor)->persist = persist;

    return NULL;
}

/**
 * @brief Free dynamically allocated tensor instance. This destructor will
 *        also destroy the underlying buffer, context, and gradient of the
 *        given tensor.
 * @param[in] tensor The tensor to free from memory. 
 */
void tensor_destroy(tensor_t *tensor)
{
    if (tensor)
    {
        PRINTLN_DEBUG_LOCATION("input");
        PRINTLN_DEBUG_TENSOR("tensor", tensor);
        id_pool_put(id_pool, tensor->id);
        if (id_pool->size == id)
        {
            id_pool_destroy(id_pool);
            id = 0;
            id_pool = NULL;
        }

        buffer_destroy(tensor->buffer);
        tensor_destroy(tensor->gradient);
        function_destroy(tensor->context, true);
        free(tensor);
    }
}

nw_error_t *tensor_create_null(tensor_t **tensor)
{
    CHECK_NULL_ARGUMENT(tensor, "tensor");

    nw_error_t *error = NULL;

    error  = tensor_create(tensor, NULL, NULL, NULL, false, false);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

void with_no_gradient(bool_t flag)
{
    static uint64_t previous = 0;

    if (flag)
    {
        ++previous;
        no_gradient = true;
    }
    else
    {
        if (previous > 0)
        {
            --previous; 
            if (!previous)
            {
                no_gradient = false;
            }
        }
    }
}

nw_error_t *tensor_from_data(tensor_t **x, void *data, runtime_t runtime, datatype_t datatype, int64_t rank, 
                             const int64_t *shape, bool_t copy, bool_t requires_gradient, bool_t persist)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", *x);
    PRINTLN_DEBUG_INT64_ARRAY("shape", shape, rank);
    PRINTLN_DEBUG_BOOLEAN("copy", copy);
    PRINTLN_DEBUG_BOOLEAN("requires_gradient", requires_gradient);
    PRINTLN_DEBUG_BOOLEAN("persist", persist);
    PRINTF_DEBUG("runtime %s\n", runtime_string(runtime));
    PRINTF_DEBUG("datatype %s\n", datatype_string(datatype));
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(data, "data");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;

    creation_operation_type_t operation_type = (copy) ? COPY_OPERATION : FROM_OPERATION;
    error = apply_operation_creation(operation_type, shape, rank, runtime, datatype, requires_gradient, persist, NULL, 0, data, x);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", *x);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_concatenation(const tensor_t *x, const tensor_t *y, tensor_t **z, int64_t axis)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(y->buffer, "y->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(y->buffer->view, "y->buffer->view");
    CHECK_NULL_ARGUMENT(z, "z");

    if (x->buffer->view->rank != y->buffer->view->rank)
    {
        return ERROR(ERROR_RANK, string_create("tensors not the same rank."), NULL);
    }

    for (int64_t i = 0; i < x->buffer->view->rank; ++i)
    {
        if (i != axis && x->buffer->view->shape[i] != y->buffer->view->shape[i])
        {
            return ERROR(ERROR_SHAPE, string_create("tensors do not have same shape along non-axis dimensions."), NULL);
        }
    }

    axis = dimension_to_index(axis, x->buffer->view->rank);

    if (axis < 0 || axis >= x->buffer->view->rank)
    {
        return ERROR(ERROR_AXIS, string_create("axis is out of range of tensor."), NULL);
    }

    int64_t length = 2 * x->buffer->view->rank;
    int64_t x_arguments[length];
    int64_t y_arguments[length];
    tensor_t *x_padded = NULL;
    tensor_t *y_padded = NULL;
    nw_error_t *error = NULL;

    for (int64_t i = 0; i < x->buffer->view->rank; ++i)
    {
        x_arguments[2 * i] = 0;
        if (i == axis)
        {
            x_arguments[2 * i + 1] = y->buffer->view->shape[i];
        }
        else
        {
            x_arguments[2 * i + 1] = 0;
        }
    }

    for (int64_t i = 0; i < y->buffer->view->rank; ++i)
    {
        y_arguments[2 * i + 1] = 0;
        if (i == axis)
        {
            y_arguments[2 * i] = x->buffer->view->shape[i];
        }
        else
        {
            y_arguments[2 * i] = 0;
        }
    }

    error = tensor_padding(x, &x_padded, x_arguments, length);
    if (error)
    {
        error = ERROR(ERROR_PADDING, string_create("failed to pad tensor."), error);
        goto cleanup;
    }

    error = tensor_padding(y, &y_padded, y_arguments, length);
    if (error)
    {
        error = ERROR(ERROR_PADDING, string_create("failed to pad tensor."), error);
        goto cleanup;
    }

    error = tensor_addition(x_padded, y_padded, z);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

cleanup:

    if ((!x_padded->requires_gradient && !y_padded->requires_gradient) || no_gradient)
    {
        tensor_destroy(x_padded);
        tensor_destroy(y_padded);
    }

    return error;
}

nw_error_t *tensor_broadcast(const tensor_t *x_original, const tensor_t *y_original, tensor_t **x_broadcasted, tensor_t **y_broadcasted)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x_original, "x_original");
    CHECK_NULL_ARGUMENT(y_original, "y_original");
    CHECK_NULL_ARGUMENT(x_original->buffer, "x_original->buffer");
    CHECK_NULL_ARGUMENT(y_original->buffer, "y_original->buffer");
    CHECK_NULL_ARGUMENT(x_original->buffer->view, "x_original->buffer->view");
    CHECK_NULL_ARGUMENT(y_original->buffer->view, "y_original->buffer->view");
    CHECK_NULL_ARGUMENT(x_broadcasted, "x_broadcasted");
    CHECK_NULL_ARGUMENT(y_broadcasted, "y_broadcasted");

    nw_error_t *error = NULL;
    int64_t *broadcasted_shape = NULL;
    int64_t broadcasted_rank;

    error = view_broadcast(x_original->buffer->view, y_original->buffer->view, &broadcasted_shape, &broadcasted_rank);
    if (error)
    {
        error = ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor shapes."), error);
        goto cleanup;
    }

    error = tensor_expand(x_original, broadcasted_shape, broadcasted_rank, x_broadcasted);
    if (error)
    {
        error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
        goto cleanup;
    }

    error = tensor_expand(y_original, broadcasted_shape, broadcasted_rank, y_broadcasted);
    if (error)
    {
        error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINTLN_DEBUG_TENSOR("x_broadcasted", *x_broadcasted);
    PRINTLN_DEBUG_TENSOR("y_broadcasted", *y_broadcasted);
    PRINT_DEBUG_NEWLINE;

cleanup:

    free(broadcasted_shape);

    return error;
}

nw_error_t *tensor_broadcast_matrix_multiplication(const tensor_t *x_original,
                                                   const tensor_t *y_original,
                                                   tensor_t **x_broadcasted,
                                                   tensor_t **y_broadcasted)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x_original, "x_original");
    CHECK_NULL_ARGUMENT(y_original, "y_original");
    CHECK_NULL_ARGUMENT(x_original->buffer, "x_original->buffer");
    CHECK_NULL_ARGUMENT(y_original->buffer, "y_original->buffer");
    CHECK_NULL_ARGUMENT(x_original->buffer->view, "x_original->buffer->view");
    CHECK_NULL_ARGUMENT(y_original->buffer->view, "y_original->buffer->view");
    CHECK_NULL_ARGUMENT(x_broadcasted, "x_broadcasted");
    CHECK_NULL_ARGUMENT(y_broadcasted, "y_broadcasted");

    nw_error_t *error = NULL;
    int64_t broadcasted_rank;
    int64_t *x_broadcasted_shape = NULL;
    int64_t *y_broadcasted_shape = NULL;

    error = view_broadcast_matrix_multiplication(x_original->buffer->view, y_original->buffer->view, 
                                                 &x_broadcasted_shape, &y_broadcasted_shape, &broadcasted_rank);
    if (error)
    {
        error = ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor shapes."), error);
        goto cleanup;
    }

    error = tensor_expand(x_original, x_broadcasted_shape, broadcasted_rank, x_broadcasted);
    if (error)
    {
        error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
        goto cleanup;
    }

    error = tensor_expand(y_original, y_broadcasted_shape, broadcasted_rank, y_broadcasted);
    if (error)
    {
        error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINTLN_DEBUG_TENSOR("x_broadcasted", *x_broadcasted);
    PRINTLN_DEBUG_TENSOR("y_broadcasted", *y_broadcasted);
    PRINT_DEBUG_NEWLINE;

cleanup:

    free(x_broadcasted_shape);
    free(y_broadcasted_shape);

    return error;
}

nw_error_t *tensor_sigmoid(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(SIGMOID_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply sigmoid to tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_tanh(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *const_tensor_1 = NULL;
    tensor_t *const_tensor_2 = NULL;
    tensor_t *double_x = NULL;
    tensor_t *sigmoid_2x = NULL;
    tensor_t *double_sigmoid = NULL;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;

    switch(datatype)
    {
    case FLOAT32:
        float32_t scalar_1_32 = (float32_t) 1;
        float32_t scalar_2_32 = (float32_t) 2;
        error = tensor_constant(&scalar_1_32, datatype, runtime, false, false, &const_tensor_1);
        error = tensor_constant(&scalar_2_32, datatype, runtime, false, false, &const_tensor_2);
        break;
    case FLOAT64:
        float64_t scalar_1_64 = (float64_t) 1;
        float64_t scalar_2_64 = (float64_t) 2;
        error = tensor_constant(&scalar_1_64, datatype, runtime, false, false, &const_tensor_1);
        error = tensor_constant(&scalar_2_64, datatype, runtime, false, false, &const_tensor_2);
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int)datatype), NULL);
        break;
    }
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_multiplication(const_tensor_2, x, &double_x);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_sigmoid(double_x, &sigmoid_2x);
    if (error)
    {
        error = ERROR(ERROR_SIGMOID, string_create("failed to perform sigmoid operation."), error);
        goto cleanup;
    }

    error = tensor_multiplication(const_tensor_2, sigmoid_2x, &double_sigmoid);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_subtraction(double_sigmoid, const_tensor_1, y);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:
    if (!tensor_shapes_equal(const_tensor_1, x))
    {
        tensor_destroy(const_tensor_1);
    }

    if (!tensor_shapes_equal(const_tensor_2, x))
    {
        tensor_destroy(const_tensor_2);
    }

    if (!x->requires_gradient || no_gradient)
    {
        tensor_destroy(double_x);
        tensor_destroy(sigmoid_2x);
        tensor_destroy(double_sigmoid);
    }
    return error;
}

nw_error_t *tensor_absolute(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;

    error = tensor_rectified_linear(x, &x_i);
    if (error)
    {
        error = ERROR(ERROR_RECTIFIED_LINEAR, string_create("failed to apply rectified linear function on tensor."), error);
        goto cleanup;
    }

    error = tensor_negation(x, &x_j);
    if (error)
    {
        error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
        goto cleanup;
    }

    error = tensor_rectified_linear(x_j, &x_k);
    if (error)
    {
        error = ERROR(ERROR_RECTIFIED_LINEAR, string_create("failed to apply rectified linear function on tensor."), error);
        goto cleanup;
    }

    error = tensor_addition(x_i, x_k, y);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!x->requires_gradient || no_gradient)
    {
        tensor_destroy(x_i);
        tensor_destroy(x_j);
        tensor_destroy(x_k);
    }

    return error;
}

nw_error_t *tensor_expand(const tensor_t *x, const int64_t *shape, int64_t length, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("shape", shape, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;

    if (view_has_shape(x->buffer->view, shape, length))
    {
        *y = (tensor_t *) x;
    }
    else
    {
        error = apply_operation_structure(EXPAND_OPERATION, x, shape, length, y);
        if (error)
        {
            return ERROR(ERROR_FORWARD, string_create("failed to expand tensor."), error);
        }
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINTLN_DEBUG_INT64_ARRAY("shape", shape, length);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_addition(const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;

    error = apply_operation_binary(ADDITION_OPERATION, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to add tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_where(const tensor_t *w, const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("w", w);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(w, "w");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;

    error = apply_operation_ternary(WHERE_OPERATION, w, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to where tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("w", w);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_subtraction(const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;

    error = apply_operation_binary(SUBTRACTION_OPERATION, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to subtract tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_division(const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;

    error = apply_operation_binary(DIVISION_OPERATION, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to divide tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;
    
    return error;
}

nw_error_t *tensor_multiplication(const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;

    error = apply_operation_binary(MULTIPLICATION_OPERATION, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to multiply tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_compare_equal(const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;
    with_no_gradient(true);

    error = apply_operation_binary(COMPARE_EQUAL_OPERATION, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to compare equal tensors."), error);
    }

    with_no_gradient(false);

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_compare_greater(const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;
    with_no_gradient(true);

    error = apply_operation_binary(COMPARE_GREATER_OPERATION, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to compare greater tensors."), error);
    }

    with_no_gradient(false);

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_max(const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;
    tensor_t *compare_x_y = NULL;
    tensor_t *compare_y_x = NULL;
    tensor_t *x_max = NULL;
    tensor_t *y_max = NULL;

    with_no_gradient(true);
    
    error = tensor_compare_greater(x, y, &compare_x_y);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to compare greater tensors."), error);
        goto cleanup;
    }

    error = tensor_compare_greater(y, x, &compare_y_x);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to compare greater tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x, compare_x_y, &x_max);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(y, compare_y_x, &y_max);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(x_max, y_max, z);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    with_no_gradient(false);

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

cleanup:

    tensor_destroy(compare_x_y);
    tensor_destroy(compare_y_x);
    tensor_destroy(x_max);
    tensor_destroy(y_max);

    return error;
}

nw_error_t *tensor_power(const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_binary(POWER_OPERATION, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to power tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_matrix_multiplication(const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;
    tensor_t *x_contiguous = NULL;
    tensor_t *y_contiguous = NULL;

    error = tensor_contiguous(x, &x_contiguous);
    if (error)
    {
        error = ERROR(ERROR_CONTIGUOUS, string_create("failed make tensor contiguous."), error);
        goto cleanup;
    }

    error = tensor_contiguous(y, &y_contiguous);
    if (error)
    {
        error = ERROR(ERROR_CONTIGUOUS, string_create("failed make tensor contiguous."), error);
        goto cleanup;
    }

    error = apply_operation_binary(MATRIX_MULTIPLICATION_OPERATION, x_contiguous, y_contiguous, z);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (x != x_contiguous && (!(x->requires_gradient) || no_gradient))
    {
        tensor_destroy(x_contiguous);
    }

    if (y != y_contiguous && (!(y->requires_gradient) || no_gradient))
    {
        tensor_destroy(y_contiguous);
    }

    return error;
}

nw_error_t *tensor_summation(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    
    error = apply_operation_reduction(SUMMATION_OPERATION, x, axis, length, keep_dimension, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to sum tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_maximum(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_reduction(MAXIMUM_OPERATION, x, axis, length, keep_dimension, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to max tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_image_to_column(const tensor_t *x, tensor_t **y, int64_t kernel_size, int64_t stride, int64_t padding,
                                   int64_t channels, int64_t height, int64_t width)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("arguments", ((int64_t[]){kernel_size, stride, padding, channels, height, width}), 6);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_contiguous = NULL;

    error = tensor_contiguous(x, &x_contiguous);
    if (error)
    {
        error = ERROR(ERROR_CONTIGUOUS, string_create("failed to make tensor contiguous."), error);
        goto cleanup;
    }

    error = apply_operation_structure(IMAGE_TO_COLUMN_OPERATION, x_contiguous, (int64_t[]){kernel_size, stride, padding, channels, height, width}, 6, y);
    if (error)
    {
        error = ERROR(ERROR_IMAGE_TO_COLUMN, string_create("failed to convert image to column."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!x->requires_gradient || no_gradient)
    {
        if (x_contiguous != x)
        {
            tensor_destroy(x_contiguous);
        }
    }

    return error;
}

nw_error_t *tensor_column_to_image(const tensor_t *x, tensor_t **y, int64_t kernel_size, int64_t stride, int64_t padding,
                                   int64_t channels, int64_t height, int64_t width)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("arguments", ((int64_t[]){kernel_size, stride, padding, channels, height, width}), 6);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_contiguous = NULL;

    error = tensor_contiguous(x, &x_contiguous);
    if (error)
    {
        error = ERROR(ERROR_CONTIGUOUS, string_create("failed to make tensor contiguous."), error);
        goto cleanup;
    }

    error = apply_operation_structure(COLUMN_TO_IMAGE_OPERATION, x_contiguous, (int64_t[]){kernel_size, stride, padding, channels, height, width}, 6, y);
    if (error)
    {
        error = ERROR(ERROR_IMAGE_TO_COLUMN, string_create("failed to convert image to column."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!x->requires_gradient || no_gradient)
    {
        if (x_contiguous != x)
        {
            tensor_destroy(x_contiguous);
        }
    }

    return error;
}

nw_error_t *tensor_linear(const tensor_t *w, const tensor_t *x, const tensor_t *y, tensor_t **z)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("w", w);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(w, "w");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(z, "z");

    nw_error_t *error = NULL;
    tensor_t *u = NULL;

    error = tensor_matrix_multiplication(w, x, &u);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    if (y)
    {
        error = tensor_addition(u, y, z);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        *z = u;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("w", w);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!u->requires_gradient || no_gradient)
    {
        if (*z != u)
        {
            tensor_destroy(u);
        }
    }

    return error;
}

nw_error_t *tensor_batch_normalization_2d(const tensor_t *x, const tensor_t *weights, const tensor_t *bias, tensor_t *running_mean, 
                                          tensor_t *running_variance, tensor_t **y, bool_t inference, void *momentum, void *epsilon)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("weights", weights);
    PRINTLN_DEBUG_TENSOR("bias", bias);
    PRINTLN_DEBUG_TENSOR("running_mean", running_mean);
    PRINTLN_DEBUG_TENSOR("running_variance", running_variance);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");

    if (x->buffer->view->rank != 4)
    {
        return ERROR(ERROR_RANK, string_create("batch normalization 2d expects a rank 4 tensor."), NULL);
    }

    nw_error_t *error = NULL;
    tensor_t *mean = NULL;
    tensor_t *variance = NULL;
    tensor_t *mean_reshaped = NULL;
    tensor_t *variance_reshaped = NULL;
    tensor_t *running_mean_l = NULL;
    tensor_t *running_mean_r = NULL;
    tensor_t *running_variance_l = NULL;
    tensor_t *running_variance_r = NULL;
    tensor_t *variance_perturbed = NULL;
    tensor_t *epsilon_constant = NULL;
    void *momentum_complement = NULL;
    void *value = NULL;
    tensor_t *unbiased_variance = NULL;
    tensor_t *value_constant = NULL;
    tensor_t *momentum_constant = NULL;
    tensor_t *momentum_complement_constant = NULL;
    tensor_t *denominator = NULL;
    tensor_t *numerator = NULL;
    tensor_t *standard_normal_x = NULL;
    tensor_t *scaled_standard_normal_x = NULL;
    tensor_t *reshaped_weights = NULL;
    tensor_t *reshaped_bias = NULL;
    int64_t number_of_features = x->buffer->view->shape[1];
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;
    int64_t n;

    if (inference)
    {
        with_no_gradient(true);
    }

    error = tensor_constant(epsilon, datatype, runtime, false, false, &epsilon_constant);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    if (momentum && !inference)
    {
        momentum_complement = (void *) malloc(datatype_size(datatype));
        if (!momentum_complement)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
            goto cleanup;
        }

        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) momentum_complement = (float32_t) 1.0 - *(float32_t *) momentum;
            break;
        case FLOAT64:
            *(float64_t *) momentum_complement = (float64_t) 1.0 - *(float64_t *) momentum;
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype."), NULL);
            goto cleanup;
        }

        error = tensor_constant(momentum, datatype, runtime, false, false, &momentum_constant);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
        
        error = tensor_constant(momentum_complement, datatype, runtime, false, false, &momentum_complement_constant);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    if (!inference)
    {
        error = tensor_mean(x, &mean, (int64_t[]){0, 2, 3}, (int64_t) 3, false);
        if (error)
        {
            error = ERROR(ERROR_MEAN, string_create("failed to compute mean."), error);
            goto cleanup;
        }
    }
    else
    {
        mean = running_mean;
    }

    if (!inference)
    {
        error = tensor_variance(x, &variance, (int64_t[]){0, 2, 3}, (int64_t) 3, false, false);
        if (error)
        {
            error = ERROR(ERROR_VARIANCE, string_create("failed to compute variance."), error);
            goto cleanup;
        }
    }
    else
    {
        variance = running_variance;
    }

    error = tensor_reshape(mean, &mean_reshaped, (int64_t[]){1, number_of_features, 1, 1}, 4);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_reshape(variance, &variance_reshaped, (int64_t[]){1, number_of_features, 1, 1}, 4);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_addition(variance_reshaped, epsilon_constant, &variance_perturbed);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_square_root(variance_perturbed, &denominator);
    if (error)
    {
        error = ERROR(ERROR_SQUARE_ROOT, string_create("failed to square root tensor."), error);
        goto cleanup;
    }

    error = tensor_subtraction(x, mean_reshaped, &numerator);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    error = tensor_division(numerator, denominator, &standard_normal_x);
    if (error)
    {
        error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        goto cleanup;
    }

    if (weights)
    {
        error = tensor_reshape(weights, &reshaped_weights, (int64_t[]){1, number_of_features, 1, 1}, 4);
        if (error)
        {
            error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(reshaped_weights, standard_normal_x, &scaled_standard_normal_x);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        scaled_standard_normal_x = standard_normal_x;
    }

    if (bias)
    {
        error = tensor_reshape(bias, &reshaped_bias, (int64_t[]){1, number_of_features, 1, 1}, 4);
        if (error)
        {
            error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensors."), error);
            goto cleanup;
        }

        error = tensor_addition(reshaped_bias, scaled_standard_normal_x, y);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        *y = scaled_standard_normal_x;
    }

    with_no_gradient(true);

    if (running_mean && !inference)
    {
        error = tensor_multiplication(momentum_complement_constant, running_mean, &running_mean_l);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(momentum_constant, mean, &running_mean_r);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_addition(running_mean_l, running_mean_r, &running_mean);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }
    }

    if (running_variance && !inference)
    {
        error = tensor_number_of_elements(x, &n);
        if (error)
        {
            error = ERROR(ERROR_N, string_create("failed to get number of elements of tensor."), error);
            goto cleanup;
        }

        value = (void *) malloc(datatype_size(datatype));
        if (!value)
        {
            error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
            goto cleanup;
        }
        
        switch (datatype)
        {
        case FLOAT32:
            *(float32_t *) value = (float32_t) n / (float32_t) (n - number_of_features);
            break;
        case FLOAT64:
            *(float64_t *) value = (float64_t) n / (float64_t) (n - number_of_features);
            break;
        default:
            error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
            goto cleanup;
        }

        error = tensor_constant(value, datatype, runtime, x->requires_gradient, false, &value_constant);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
            goto cleanup;
        }

        error = tensor_multiplication(momentum_complement_constant, running_variance, &running_variance_l);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(value_constant, variance, &unbiased_variance);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_multiplication(momentum_constant, unbiased_variance, &running_variance_r);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }

        error = tensor_addition(running_variance_l, running_variance_r, &running_variance);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }
    }

    with_no_gradient(false);

    if (inference)
    {
        with_no_gradient(false);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("weights", weights);
    PRINTLN_DEBUG_TENSOR("bias", bias);
    PRINTLN_DEBUG_TENSOR("running_mean", running_mean);
    PRINTLN_DEBUG_TENSOR("running_variance", running_variance);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!x->requires_gradient || no_gradient || inference)
    {
        if (mean != mean_reshaped)
        {
            tensor_destroy(mean_reshaped);
        }

        if (running_mean != mean)
        {
            tensor_destroy(mean);
        }

        if (variance != variance_reshaped)
        {
            tensor_destroy(variance_reshaped);
        }

        if (running_variance != variance)
        {
            tensor_destroy(variance);
        }

        tensor_destroy(variance_perturbed);
        tensor_destroy(denominator);
        tensor_destroy(numerator);
    }

    if (!weights || !weights->requires_gradient || no_gradient || inference)
    {
        if (weights != reshaped_weights)
        {
            tensor_destroy(reshaped_weights);
        }
        if (standard_normal_x != scaled_standard_normal_x)
        {
            tensor_destroy(standard_normal_x);
        }
    }

    if (!bias || !bias->requires_gradient || no_gradient || inference)
    {
        if (bias != reshaped_bias)
        {
            tensor_destroy(reshaped_bias);
        }
        if (scaled_standard_normal_x != *y)
        {
            tensor_destroy(scaled_standard_normal_x);
        }
    }

    free(value);
    free(momentum_complement);
    tensor_destroy(unbiased_variance);
    tensor_destroy(value_constant);
    tensor_destroy(momentum_constant);
    tensor_destroy(momentum_complement_constant);
    tensor_destroy(running_mean_l);
    tensor_destroy(running_mean_r);
    tensor_destroy(running_variance_l);
    tensor_destroy(running_variance_r);
    tensor_destroy(epsilon_constant);

    return error;
}

nw_error_t *tensor_layer_normalization(const tensor_t *x, const tensor_t *weights, const tensor_t *bias, tensor_t **y, int64_t *normalized_shape, int64_t length, void *epsilon)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("weights", weights);
    PRINTLN_DEBUG_TENSOR("bias", bias);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(weights, "weights");
    CHECK_NULL_ARGUMENT(bias, "bias");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(normalized_shape, "normalized_shape");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");

    nw_error_t *error = NULL;
    tensor_t *epsilon_constant = NULL;
    tensor_t *mean = NULL;
    tensor_t *variance = NULL;
    tensor_t *variance_perturbed = NULL;
    tensor_t *denominator = NULL;
    tensor_t *numerator = NULL;
    tensor_t *standard_normal_x = NULL;
    tensor_t *scaled_standard_normal_x = NULL;
    int64_t axis[length];
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;

    for (int64_t i = 0; i < length; ++i)
    {
        axis[i] = -1 - i;
    }

    error = tensor_constant(epsilon, datatype, runtime, false, false, &epsilon_constant);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_mean(x, &mean, axis, length, true);
    if (error)
    {
        error = ERROR(ERROR_MEAN, string_create("failed to compute mean."), error);
        goto cleanup;
    }

    error = tensor_variance(x, &variance, axis, length, true, false);
    if (error)
    {
        error = ERROR(ERROR_VARIANCE, string_create("failed to compute variance."), error);
        goto cleanup;
    }

    error = tensor_addition(variance, epsilon_constant, &variance_perturbed);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_square_root(variance_perturbed, &denominator);
    if (error)
    {
        error = ERROR(ERROR_SQUARE_ROOT, string_create("failed to square root tensor."), error);
        goto cleanup;
    }

    error = tensor_subtraction(x, mean, &numerator);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    error = tensor_division(numerator, denominator, &standard_normal_x);
    if (error)
    {
        error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        goto cleanup;
    }

    if (weights)
    {
        error = tensor_multiplication(weights, standard_normal_x, &scaled_standard_normal_x);
        if (error)
        {
            error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        scaled_standard_normal_x = standard_normal_x;
    }

    if (bias)
    {
        error = tensor_addition(bias, scaled_standard_normal_x, y);
        if (error)
        {
            error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        *y = scaled_standard_normal_x;
    }


    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("weights", weights);
    PRINTLN_DEBUG_TENSOR("bias", bias);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:


    if (!x->requires_gradient || no_gradient || !tensor_shapes_equal(variance, epsilon_constant))
    {
        tensor_destroy(epsilon_constant);
    }

    if (!x->requires_gradient || no_gradient)
    {
        tensor_destroy(mean);
        tensor_destroy(variance);
        tensor_destroy(variance_perturbed);
        tensor_destroy(denominator);
        tensor_destroy(numerator);
    }

    if (!weights || !weights->requires_gradient || no_gradient)
    {
        if (standard_normal_x != scaled_standard_normal_x)
        {
            tensor_destroy(standard_normal_x);
        }
    }

    if (!bias || !bias->requires_gradient || no_gradient)
    {
        if (scaled_standard_normal_x != *y)
        {
            tensor_destroy(scaled_standard_normal_x);
        }
    }

    return error;

}

nw_error_t *tensor_convolution_2d(const tensor_t *w, const tensor_t *x, const tensor_t *y, tensor_t **z, int64_t stride, int64_t padding)
{
    CHECK_NULL_ARGUMENT(w, "w");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(z, "z");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("w", w);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_INT64_ARRAY("arguments", ((int64_t[]){stride, padding}), 2);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = NULL;
    tensor_t *w_toeplitz = NULL;
    tensor_t *x_reshape = NULL;
    tensor_t *y_reshape = NULL;
    tensor_t *v = NULL;
    tensor_t *u = NULL;
    int64_t batch_size = w->buffer->view->shape[0];
    int64_t in_channels = w->buffer->view->shape[1];
    int64_t height = w->buffer->view->shape[2];
    int64_t width = w->buffer->view->shape[3];
    int64_t out_channels = x->buffer->view->shape[0];
    int64_t kernel_size = x->buffer->view->shape[2];

    error = tensor_image_to_column(w, &w_toeplitz, kernel_size, stride, padding, in_channels, height, width);
    if (error)
    {
        error = ERROR(ERROR_IMAGE_TO_COLUMN, string_create("failed to convert image to column."), error);
        goto cleanup;
    }
   
    error = tensor_reshape(x, &x_reshape, (int64_t[]){out_channels, in_channels * kernel_size * kernel_size}, (int64_t) 2);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(x_reshape, w_toeplitz, &v);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    if (y)
    {
        error = tensor_reshape(y, &y_reshape, (int64_t[]){out_channels, 1}, 2);
        if (error)
        {
            error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
            goto cleanup;
        }

        error = tensor_addition(v, y_reshape, &u);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        u = v;
    }

    error = tensor_reshape(u, z, (int64_t[]){batch_size, out_channels, (height + 2 * padding - kernel_size) / stride + 1, (width + 2 * padding - kernel_size) / stride + 1}, 4);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("w", w);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINTLN_DEBUG_BOOLEAN("no_gradient", no_gradient);
    PRINT_DEBUG_NEWLINE;

cleanup:
    if (y)
    {
        if ((!v->requires_gradient && !y->requires_gradient) || no_gradient)
        {
            if (y != y_reshape)
            {
                tensor_destroy(y_reshape);
            }
            if (*z != u)
            {
                tensor_destroy(u);
            }
        }
    }

    if ((!x->requires_gradient && !w->requires_gradient) || no_gradient)
    {
        tensor_destroy(w_toeplitz);
        if (x != x_reshape)
        {
            tensor_destroy(x_reshape);
        }
        if (v != *z)
        {
            tensor_destroy(v);
        }
    }

    return error;
}

nw_error_t *tensor_convolution_transpose_2d(const tensor_t *w, const tensor_t *x, const tensor_t *y, tensor_t **z, int64_t stride, int64_t padding)
{
    CHECK_NULL_ARGUMENT(w, "w");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(z, "z");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("w", w);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_INT64_ARRAY("arguments", ((int64_t[]){stride, padding}), 2);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = NULL;
    tensor_t *w_reshape = NULL;
    tensor_t *x_reshape = NULL;
    tensor_t *x_transpose = NULL;
    tensor_t *y_reshape = NULL;
    tensor_t *v = NULL;
    tensor_t *u = NULL;
    int64_t batch_size = w->buffer->view->shape[0];
    int64_t in_height = w->buffer->view->shape[2];
    int64_t in_width = w->buffer->view->shape[3];
    int64_t in_channels = x->buffer->view->shape[0];
    int64_t out_channels = x->buffer->view->shape[1];
    int64_t kernel_size = x->buffer->view->shape[2];
    int64_t out_height = (in_height - 1) * stride - 2 * padding + (kernel_size - 1) + 1;
    int64_t out_width = (in_width - 1) * stride - 2 * padding + (kernel_size - 1) + 1;

    error = tensor_reshape(w, &w_reshape, (int64_t[]){batch_size, in_channels, in_height * in_width}, (int64_t) 3);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_reshape(x, &x_reshape, (int64_t[]){in_channels, out_channels * kernel_size * kernel_size}, (int64_t) 2);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_transpose(x_reshape, &x_transpose, 0, 1);
    if (error)
    {
        error = ERROR(ERROR_TRANSPOSE, string_create("failed to transpose tensor."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(x_transpose, w_reshape, &v);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_column_to_image(v, &u, kernel_size, stride, padding, out_channels, out_height, out_width);
    if (error)
    {
        error = ERROR(ERROR_COLUMN_TO_IMAGE, string_create("failed to covert columns to image."), error);
        goto cleanup;
    }

    if (y)
    {
        error = tensor_reshape(y, &y_reshape, (int64_t[]){out_channels, 1, 1}, 3);
        if (error)
        {
            error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
            goto cleanup;
        }

        error = tensor_addition(u, y_reshape, z);
        if (error)
        {
            error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
            goto cleanup;
        }
    }
    else
    {
        *z = u;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("w", w);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!x_transpose->requires_gradient || no_gradient)
    {
        if (x != x_reshape)
        {
            tensor_destroy(x_reshape);
        }
    }

    if ((!w_reshape->requires_gradient && !x_transpose->requires_gradient) || no_gradient)
    {
        if (w_reshape != w)
        {
            tensor_destroy(w_reshape);
        }
        tensor_destroy(x_transpose);
    }

    if (!v->requires_gradient || no_gradient)
    {
        tensor_destroy(v);
    }

    if (y)
    {
        if ((!u->requires_gradient && !y_reshape->requires_gradient) || no_gradient)
        {
            tensor_destroy(u);
            if (y != y_reshape)
            {
                tensor_destroy(y_reshape);
            }
        }
    }

    return error;
}

nw_error_t *tensor_item(const tensor_t *x, void *value)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(x->buffer->storage->data, "x->buffer->storage->data");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(value, "value");

    if (x->buffer->view->rank)
    {
        return ERROR(ERROR_RANK, string_create("tensor must be rank zero."), NULL);
    }

    switch (x->buffer->storage->datatype)
    {
    case FLOAT32:
        *(float32_t *) value = *(float32_t *) x->buffer->storage->data;
        break;
    case FLOAT64:
        *(float64_t *) value = *(float64_t *) x->buffer->storage->data;
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) x->buffer->storage->datatype), NULL);
    }

    return NULL;
}

nw_error_t *tensor_argument_maximum(const tensor_t *x, tensor_t **y, int64_t axis, bool_t keep_dimension)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTF_DEBUG("axis: %ld\n", axis);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(x->buffer->view->shape, "x->buffer->view->shape");
    CHECK_NULL_ARGUMENT(y, "y");
    nw_error_t *error = NULL;
    int64_t *shape = x->buffer->view->shape;
    int64_t rank = x->buffer->view->rank;
    axis = dimension_to_index(axis, rank);

    if ((!rank && axis) || (rank && axis >= rank))
    {
        return ERROR(ERROR_AXIS, string_create("axis out of range of tensor."), NULL);
    }

    with_no_gradient(true);
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    int64_t dimension = (rank) ? shape[axis] : 1;
    int64_t new_rank = rank - axis;
    int64_t new_shape[new_rank];
    int64_t *reduce_axis = (rank) ? ((int64_t[]) {axis}) : ((int64_t[]){});
    int64_t reduce_rank = (rank) ? 1 : 0;
    size_t size = datatype_size(datatype);
    void *value = NULL;
    void *start = NULL;
    void *stop = NULL;
    void *step = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;
    tensor_t *x_l = NULL;
    tensor_t *x_m = NULL;
    tensor_t *x_n = NULL;
    tensor_t *x_o = NULL;

    value = (void *) malloc(size);
    if (!value)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    start = (void *) malloc(size);
    if (!start)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    stop = (void *) malloc(size);
    if (!stop)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    step = (void *) malloc(size);
    if (!step)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    
    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) value = (float32_t) dimension;
        *(float32_t *) start = (float32_t) dimension;
        *(float32_t *) stop = (float32_t) 0;
        *(float32_t *) step = (float32_t) -1;
        break;
    case FLOAT64:
        *(float64_t *) value = (float64_t) dimension;
        *(float64_t *) start = (float64_t) dimension;
        *(float64_t *) stop = (float64_t) 0;
        *(float64_t *) step = (float64_t) -1;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    if (new_rank)
    {
        new_shape[0] = dimension;
        for (int64_t i = 1; i < new_rank; ++i)
        {
            new_shape[i] = (int64_t) 1;
        }
    }

    error = tensor_maximum(x, &x_i, reduce_axis, reduce_rank, true);
    if (error)
    {
        error = ERROR(ERROR_MAXIMUM, string_create("failed to get maximum of tensor."), error);
        goto cleanup;
    }

    error = tensor_compare_equal(x_i, x, &x_j);
    if (error)
    {
        error = ERROR(ERROR_COMPARE_EQUAL, string_create("failed to compare equal tensors."), error);
        goto cleanup;
    }

    error = tensor_arange(&x_k, start, stop, step, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_reshape(x_k, &x_l, new_shape, new_rank);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x_j, x_l, &x_m);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_maximum(x_m, &x_n, reduce_axis, reduce_rank, keep_dimension);
    if (error)
    {
        error = ERROR(ERROR_MAXIMUM, string_create("failed to get maximum of tensor."), error);
        goto cleanup;
    }

    error = tensor_constant(value, datatype, runtime, false, false, &x_o);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }
    
    error = tensor_subtraction(x_o, x_n, y);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    with_no_gradient(false);
    free(value);
    free(start);
    free(stop);
    free(step);
    if (x != x_i)
    {
        tensor_destroy(x_i);
    }
    tensor_destroy(x_j);
    if (x_k != x_l)
    {
        tensor_destroy(x_k);
    }
    tensor_destroy(x_l);
    if (x_m != x_n)
    {
        tensor_destroy(x_m);
    }
    tensor_destroy(x_n);
    tensor_destroy(x_o);

    return error;
}

nw_error_t *tensor_number_of_elements(const tensor_t *x, int64_t *n)
{
    CHECK_NULL_ARGUMENT(x, "x");    
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");    
    CHECK_NULL_ARGUMENT(n, "n");    

    nw_error_t *error = NULL;

    error = view_logical_size(x->buffer->view, n);
    if (error)
    {
        return ERROR(ERROR_N, string_create("failed to get logical size of view."), error);
    }

    return error;
}

nw_error_t *tensor_constant(void *constant, datatype_t datatype, runtime_t runtime, bool_t requires_gradient, bool_t persist, tensor_t **x)
{
    CHECK_NULL_ARGUMENT(constant, "constant");
    CHECK_NULL_ARGUMENT(x, "x");

    nw_error_t *error = NULL;

    error = tensor_from_data(x, constant, runtime, datatype, 0, (int64_t[]){}, true, requires_gradient, persist);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_mean(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    void *value = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    size_t size = datatype_size(datatype);

    value = (void *) malloc(size);
    if (!value)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    error = tensor_summation(x, &x_i, axis, length, keep_dimension);
    if (error)
    {
        error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        goto cleanup;
    }

    int64_t n, n_i;

    error = tensor_number_of_elements(x_i, &n_i);
    if (error)
    {
        error = ERROR(ERROR_N, string_create("failed to get number of elements of tensor."), error);
        goto cleanup;
    }

    error = tensor_number_of_elements(x, &n);
    if (error)
    {
        error = ERROR(ERROR_N, string_create("failed to get number of elements of tensor."), error);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) value = (float32_t) n_i / (float32_t) n;
        break;
    case FLOAT64:
        *(float64_t *) value = (float64_t) n_i / (float64_t) n;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    error = tensor_constant(value, datatype, runtime, x->requires_gradient, false, &x_j);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x_j, x_i, y);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    free(value);
    if (!x->requires_gradient || no_gradient)
    {
        if (x_i != x)
        {
            tensor_destroy(x_i);
        }
        tensor_destroy(x_j);
    }

    return error;
}

nw_error_t *tensor_variance(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension, bool_t unbiased)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    void *value = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;
    tensor_t *x_l = NULL;
    tensor_t *x_m = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    size_t size = datatype_size(datatype);
    int64_t n, n_i;
    view_t *view = NULL;

    error = view_reduce(x->buffer->view, &view, axis, length, keep_dimension);
    if (error)
    {
        error = ERROR(ERROR_REDUCTION, string_create("failed to reduce tensor."), error);
        goto cleanup;
    }

    error = view_logical_size(view, &n_i);
    if (error)
    {
        error = ERROR(ERROR_N, string_create("failed to get number of elements of tensor."), error);
        goto cleanup;
    }

    error = tensor_number_of_elements(x, &n);
    if (error)
    {
        error = ERROR(ERROR_N, string_create("failed to get number of elements of tensor."), error);
        goto cleanup;
    }

    if (n_i == n)
    {
        error = ERROR(ERROR_DIVISION, string_create("divide by zero error."), NULL);
        goto cleanup;
    }

    value = (void *) malloc(size);
    if (!value)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    error = tensor_mean(x, &x_i, axis, length, true);
    if (error)
    {
        error = ERROR(ERROR_MEAN, string_create("failed to average tensor."), error);
        goto cleanup;
    }

    error = tensor_subtraction(x, x_i, &x_j);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x_j, x_j, &x_k);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_summation(x_k, &x_l, axis, length, keep_dimension);
    if (error)
    {
        error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) value = (float32_t) n / (float32_t) n_i;
        if (unbiased)
        {
            *(float32_t *) value -= (float32_t) 1.0;
        }
        break;
    case FLOAT64:
        *(float64_t *) value = (float64_t) n / (float64_t) n_i;
        if (unbiased)
        {
            *(float64_t *) value -= (float64_t) 1.0;
        }
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    error = tensor_constant(value, datatype, runtime, x->requires_gradient, false, &x_m);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
        goto cleanup;
    }

    error = tensor_division(x_l, x_m, y);
    if (error)
    {
        error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    free(value);
    view_destroy(view);
    if (!x->requires_gradient || no_gradient)
    {
        if (x_i != x)
        {
            tensor_destroy(x_i);
        }
        tensor_destroy(x_j);
        if (x_k != x_l)
        {
            tensor_destroy(x_k);
        }
        tensor_destroy(x_l);
        tensor_destroy(x_m);
    }

    return error;
}

nw_error_t *tensor_standard_deviation(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension, bool_t unbiased)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;

    error = tensor_variance(x, &x_i, axis, length, keep_dimension, unbiased);
    if (error)
    {
        error = ERROR(ERROR_VARIANCE, string_create("failed to compute variance of tensor."), error);
        goto cleanup;
    }

    error = tensor_square_root(x_i, y);
    if (error)
    {
        error = ERROR(ERROR_SQUARE_ROOT, string_create("failed to square root tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!x->requires_gradient || no_gradient)
    {
        if (x != x_i)
        {
            tensor_destroy(x_i);
        }
    }

    return error;
}

static nw_error_t *softmax(const tensor_t *x, tensor_t **y_max, tensor_t **y_num, tensor_t **y_den, int64_t axis)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTF_DEBUG("axis %ld\n", axis);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(y_max, "y_max");
    CHECK_NULL_ARGUMENT(y_num, "y_num");
    CHECK_NULL_ARGUMENT(y_den, "y_den");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;
    int64_t rank = x->buffer->view->rank;
    int64_t *reduce_axis = (rank) ? ((int64_t[]) {axis}) : ((int64_t[]){});
    int64_t reduce_rank = (rank) ? 1 : 0;

    error = tensor_maximum(x, &x_i, reduce_axis, reduce_rank, true);
    if (error)
    {
        error = ERROR(ERROR_MAXIMUM, string_create("failed to get maximum of tensor."), error);
        goto cleanup;
    }

    error = tensor_subtraction(x, x_i, y_max);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    error = tensor_exponential(*y_max, y_num);
    if (error)
    {
        error = ERROR(ERROR_EXPONENTIAL, string_create("failed to exponentiate tensor."), error);
        goto cleanup;
    }

    error = tensor_summation(*y_num, y_den, reduce_axis, reduce_rank, true);
    if (error)
    {
        error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y_max", *y_max);
    PRINTLN_DEBUG_TENSOR("y_num", *y_num);
    PRINTLN_DEBUG_TENSOR("y_den", *y_den);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!x->requires_gradient || no_gradient)
    {
        if (x != x_i)
        {
            tensor_destroy(x_i);
        }
    }

    return error;
}

nw_error_t *tensor_softmax(const tensor_t *x, tensor_t **y, int64_t axis)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTF_DEBUG("axis %ld\n", axis);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;

    error = softmax(x, &x_i, &x_j, &x_k, axis);
    if (error)
    {
        error = ERROR(ERROR_SOFTMAX, string_create("failed to softmax tensor."), error);
        goto cleanup;
    }

    error = tensor_division(x_j, x_k, y);
    if (error)
    {
        error = ERROR(ERROR_DIVISION, string_create("failed to divide tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!x->requires_gradient || no_gradient)
    {
        tensor_destroy(x_i);
        if (x_j != x_k)
        {
            tensor_destroy(x_j);
        }
        tensor_destroy(x_k);
    }

    return error;
}

nw_error_t *tensor_logsoftmax(const tensor_t *x, tensor_t **y, int64_t axis)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTF_DEBUG("axis %ld\n", axis);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;
    tensor_t *x_l = NULL;

    error = softmax(x, &x_i, &x_j, &x_k, axis);
    if (error)
    {
        error = ERROR(ERROR_SOFTMAX, string_create("failed to softmax tensor."), error);
        goto cleanup;
    }
    
    error = tensor_logarithm(x_k, &x_l);
    if (error)
    {
        error = ERROR(ERROR_LOGARITHM, string_create("failed to log tensor."), error);
        goto cleanup;
    }

    error = tensor_subtraction(x_i, x_l, y);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (!x->requires_gradient || no_gradient)
    {
        tensor_destroy(x_i);
        tensor_destroy(x_j);
        tensor_destroy(x_k);
        tensor_destroy(x_l);
    }

    return error;
}

nw_error_t *tensor_is_contiguous(const tensor_t *x, bool_t *is_contiguous)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(is_contiguous, "is_contiguous");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");

    nw_error_t *error = NULL;

    error = view_is_contiguous(x->buffer->view, is_contiguous);
    if (error)
    {
        return ERROR(ERROR_CONTIGUOUS, string_create("failed to determin if view is contiguous."), error);
    }

    return error;
}

nw_error_t *tensor_reshape(const tensor_t *x, tensor_t **y, const int64_t *shape, int64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("shape", shape, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;
    tensor_t *x_contiguous = NULL;
    if (view_has_shape(x->buffer->view, shape, length))
    {
        *y = (tensor_t *) x;
    }
    else
    {
        error = tensor_contiguous(x, &x_contiguous);
        if (error)
        {
            error = ERROR(ERROR_CONTIGUOUS, string_create("failed to make tensor contiguous."), error);
            goto cleanup;
        }

        error = apply_operation_structure(RESHAPE_OPERATION, x_contiguous, shape, length, y);
        if (error)
        {
            error = ERROR(ERROR_FORWARD, string_create("failed to reshape tensor."), error);
            goto cleanup;
        }
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (x != x_contiguous && (!x->requires_gradient || no_gradient))
    {
        tensor_destroy(x_contiguous);
    }

    return error;
}

nw_error_t *tensor_permute(const tensor_t *x, tensor_t **y, int64_t *axis, int64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("axis", axis, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(axis, "axis");

    nw_error_t *error = NULL;

    error = apply_operation_structure(PERMUTE_OPERATION, x, axis, length, y);
    if (error)
    {
        return ERROR(ERROR_PERMUTE, string_create("failed to permute tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_slice(const tensor_t *x, tensor_t **y, int64_t *arguments, int64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("arguments", arguments, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    nw_error_t *error = NULL;

    error = apply_operation_structure(SLICE_OPERATION, x, arguments, length, y);
    if (error)
    {
        return ERROR(ERROR_SLICE, string_create("failed to slice tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_padding(const tensor_t *x, tensor_t **y, int64_t *arguments, int64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_INT64_ARRAY("arguments", arguments, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    nw_error_t *error = NULL;

    error = apply_operation_structure(PADDING_OPERATION, x, arguments, length, y);
    if (error)
    {
        return ERROR(ERROR_PADDING, string_create("failed to pad tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

bool_t tensor_shapes_equal(const tensor_t *x, const tensor_t *y)
{
    return x && y && x->buffer && y->buffer && x->buffer->view && y->buffer->view &&
           view_shapes_equal(x->buffer->view, y->buffer->view);
}

nw_error_t *tensor_transpose(const tensor_t *x, tensor_t **y, int64_t axis1, int64_t axis2)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTF_DEBUG("(axis1: %ld, axis2: %ld)\n", axis1, axis2);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    int64_t rank = x->buffer->view->rank;
    int64_t index1 = dimension_to_index(axis1, rank);
    int64_t index2 = dimension_to_index(axis2, rank);
    int64_t axis[rank];
    for (int64_t i = 0; i < rank; ++i)
    {
        axis[i] = i;
    }
    int64_t temp = axis[index2];
    axis[index2] = axis[index1];
    axis[index1] = temp;

    error = apply_operation_structure(PERMUTE_OPERATION, x, axis, rank, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to permute tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_contiguous(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    bool_t is_contiguous;

    error = tensor_is_contiguous(x, &is_contiguous);
    if (error)
    {
        return ERROR(ERROR_CONTIGUOUS, string_create("failed to determine if tensor is contiguous."), error);
    }

    if (is_contiguous)
    {
        *y = (tensor_t *) x;
    }
    else
    {
        error = apply_operation_unary(CONTIGUOUS_OPERATION, x, y);
        if (error)
        {
            return ERROR(ERROR_FORWARD, string_create("failed to permute tensor."), error);
        }
    }
    
    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_logarithm(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(LOGARITHM_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to log tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_sine(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(SINE_OPERATION, x, y);
    if (error)
    { 
        return ERROR(ERROR_FORWARD, string_create("failed to sine tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_cosine(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(COSINE_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to cosine tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_exponential(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(EXPONENTIAL_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to exponentiate tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_square_root(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(SQUARE_ROOT_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to square root tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_reciprocal(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(RECIPROCAL_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to get reciprocal of tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_negation(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(NEGATION_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to negate tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_rectified_linear(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(RECTIFIED_LINEAR_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to get rectified linear of tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_lower_triangular(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    void *start = NULL, *stop = NULL, *step = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    size_t size = datatype_size(datatype);
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;
    tensor_t *x_l = NULL;
    tensor_t *x_m = NULL;
    tensor_t *x_n = NULL;
    tensor_t *x_o = NULL;
    int64_t r = x->buffer->view->shape[x->buffer->view->rank - 2];
    int64_t c = x->buffer->view->shape[x->buffer->view->rank - 1];

    start = (void *) malloc(size);
    if (!start)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    stop = (void *) malloc(size);
    if (!stop)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    step = (void *) malloc(size);
    if (!stop)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) start = (float32_t) 0.0;
        *(float32_t *) stop = (float32_t) r;
        *(float32_t *) step = (float32_t) 1.0;
        break;
    case FLOAT64:
        *(float64_t *) start = (float64_t) 0.0;
        *(float64_t *) stop = (float64_t) r;
        *(float64_t *) step = (float64_t) 1.0;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype."), NULL);
        goto cleanup;
    }

    error = tensor_arange(&x_i, start, stop, step, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) start = (float32_t) -1.0;
        *(float32_t *) stop = (float32_t) (c - 1);
        *(float32_t *) step = (float32_t) 1.0;
        break;
    case FLOAT64:
        *(float64_t *) start = (float64_t) -1.0;
        *(float64_t *) stop = (float64_t) (c - 1);
        *(float64_t *) step = (float64_t) 1.0;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype."), NULL);
        goto cleanup;
    }

    error = tensor_arange(&x_j, start, stop, step, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_reshape(x_i, &x_k, (int64_t[]){r, 1}, 2);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_reshape(x_j, &x_l, (int64_t[]){1, c}, 2);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_expand(x_k, (int64_t[]){r, c}, 2, &x_m);
    if (error)
    {
        error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
        goto cleanup;
    }

    error = tensor_expand(x_l, (int64_t[]){r, c}, 2, &x_n);
    if (error)
    {
        error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
        goto cleanup;
    }

    error = tensor_compare_greater(x_m, x_n, &x_o);
    if (error)
    {
        error = ERROR(ERROR_COMPARE_GREATER, string_create("failed to compare greater tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x, x_o, y);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    free(start);
    free(stop);
    free(step);

    if (!x->requires_gradient || no_gradient || !tensor_shapes_equal(x_o, x))
    {
        tensor_destroy(x_o);
    }

    tensor_destroy(x_i);
    tensor_destroy(x_j);
    tensor_destroy(x_k);
    tensor_destroy(x_l);
    tensor_destroy(x_m);
    tensor_destroy(x_n);


    return error;
}

nw_error_t *tensor_leaky_rectified_linear(const tensor_t *x, void *c, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(c, "c");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *c_i = NULL;
    tensor_t *c_j = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;

    error = tensor_constant(c, x->buffer->storage->datatype, x->buffer->storage->runtime, false, false, &c_i);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_negation(c_i, &c_j);
    if (error)
    {
        error = ERROR(ERROR_NEGATION, string_create("failed to negate tensor."), error);
        goto cleanup;
    }

    error = tensor_rectified_linear(x, &x_i);
    if (error)
    {
        error = ERROR(ERROR_RECTIFIED_LINEAR, string_create("failed to rectified linear tensor."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x, c_j, &x_j);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensor."), error);
        goto cleanup;
    }

    error = tensor_rectified_linear(x_j, &x_k);
    if (error)
    {
        error = ERROR(ERROR_RECTIFIED_LINEAR, string_create("failed to rectified linear tensor."), error);
        goto cleanup;
    }

    error = tensor_subtraction(x_i, x_k, y);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensor."), error);
        goto cleanup;
    }
    
    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    tensor_destroy(c_i);
    if (!tensor_shapes_equal(x, c_j)) 
    {
        tensor_destroy(c_j);
    }
    if (!x->requires_gradient || no_gradient)
    {
        tensor_destroy(x_i);
        tensor_destroy(x_j);
        tensor_destroy(x_k);
    }

    return error;
}

nw_error_t *tensor_causal_multihead_self_attention(tensor_t *x, const tensor_t *input_weights, const tensor_t *input_bias, const tensor_t *output_weights, const tensor_t *output_bias,
                                                   int64_t number_of_heads, void *dropout_probability, bool_t inference, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("input_weights", input_weights);
    PRINTLN_DEBUG_TENSOR("input_bias", input_bias);
    PRINTLN_DEBUG_TENSOR("output_weights", output_weights);
    PRINTLN_DEBUG_TENSOR("output_bias", output_bias);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(input_weights, "input_weights");
    CHECK_NULL_ARGUMENT(input_bias, "input_bias");
    CHECK_NULL_ARGUMENT(output_weights, "output_weights");
    CHECK_NULL_ARGUMENT(output_bias, "output_bias");

    nw_error_t *error = NULL;
    tensor_t *input_projection = NULL;
    tensor_t *query = NULL;
    tensor_t *key = NULL;
    tensor_t *value = NULL;
    tensor_t *query_reshaped = NULL;
    tensor_t *key_reshaped = NULL;
    tensor_t *value_reshaped = NULL;
    tensor_t *query_transposed = NULL;
    tensor_t *key_transposed = NULL;
    tensor_t *value_transposed = NULL;
    tensor_t *attention = NULL;
    tensor_t *attention_transpose = NULL;
    tensor_t *attention_reshaped = NULL;
    tensor_t *output_projection = NULL;
    int64_t rank = x->buffer->view->rank;
    int64_t batch_size = x->buffer->view->shape[0];
    int64_t sequence_length = x->buffer->view->shape[1];
    int64_t embedding_size = x->buffer->view->shape[2];
    int64_t head_size = embedding_size / number_of_heads;

    error = tensor_linear(x, input_weights, input_bias, &input_projection);
    if (error)
    {
        error = ERROR(ERROR_LINEAR, string_create("failed to employ linear operation."), error);
        goto cleanup;
    }

    error = tensor_slice(input_projection, &query, (int64_t[]){0, batch_size, 0, sequence_length, 0, embedding_size}, 2 * rank);
    if (error)
    {
        error = ERROR(ERROR_SLICE, string_create("failed to slice tensor."), error);
        goto cleanup;
    }

    error = tensor_slice(input_projection, &key, (int64_t[]){0, batch_size, 0, sequence_length, embedding_size, 2 * embedding_size}, 2 * rank);
    if (error)
    {
        error = ERROR(ERROR_SLICE, string_create("failed to slice tensor."), error);
        goto cleanup;
    }

    error = tensor_slice(input_projection, &value, (int64_t[]){0, batch_size, 0, sequence_length, 2 * embedding_size, 3 * embedding_size}, 2 * rank);
    if (error)
    {
        error = ERROR(ERROR_SLICE, string_create("failed to slice tensor."), error);
        goto cleanup;
    }

    error = tensor_reshape(query, &query_reshaped, (int64_t[]){batch_size, sequence_length, number_of_heads, head_size}, rank + 1);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_reshape(key, &key_reshaped, (int64_t[]){batch_size, sequence_length, number_of_heads, head_size}, rank + 1);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_reshape(value, &value_reshaped, (int64_t[]){batch_size, sequence_length, number_of_heads, head_size}, rank + 1);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_transpose(query_reshaped, &query_transposed, 1, 2);
    if (error)
    {
        error = ERROR(ERROR_TRANSPOSE, string_create("failed to tranpose tensor."), error);
        goto cleanup;
    }

    error = tensor_transpose(key_reshaped, &key_transposed, 1, 2);
    if (error)
    {
        error = ERROR(ERROR_TRANSPOSE, string_create("failed to tranpose tensor."), error);
        goto cleanup;
    }

    error = tensor_transpose(value_reshaped, &value_transposed, 1, 2);
    if (error)
    {
        error = ERROR(ERROR_TRANSPOSE, string_create("failed to tranpose tensor."), error);
        goto cleanup;
    }

    error = tensor_scaled_dot_product_attention(query_transposed, key_transposed, value_transposed, &attention, dropout_probability, inference);
    if (error)
    {
        error = ERROR(ERROR_ATTENTION, string_create("failed to employ scaled dot product attention operation."), error);
        goto cleanup;
    }

    error = tensor_transpose(attention, &attention_transpose, 1, 2);
    if (error)
    {
        error = ERROR(ERROR_TRANSPOSE, string_create("failed to tranpose tensor."), error);
        goto cleanup;
    }

    error = tensor_reshape(attention_transpose, &attention_reshaped, (int64_t[]){batch_size, sequence_length, embedding_size}, rank);
    if (error)
    {
        error = ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    error = tensor_linear(attention_reshaped, output_weights, output_bias, &output_projection);
    if (error)
    {
        error = ERROR(ERROR_LINEAR, string_create("failed to employ linear operation."), error);
        goto cleanup;
    }

    error = tensor_dropout(output_projection, y, dropout_probability, inference);
    if (error)
    {
        error = ERROR(ERROR_DROPOUT, string_create("failed to dropout tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:
    
    if (!(x->requires_gradient || input_weights->requires_gradient || input_bias->requires_gradient) || no_gradient)
    {
        tensor_destroy(input_projection);
        tensor_destroy(query);
        tensor_destroy(key);
        tensor_destroy(value);
        tensor_destroy(query_reshaped);
        tensor_destroy(key_reshaped);
        tensor_destroy(value_reshaped);
        tensor_destroy(query_transposed);
        tensor_destroy(key_transposed);
        tensor_destroy(value_transposed);
        tensor_destroy(attention);
        if (attention_transpose != attention_reshaped)
        {
            tensor_destroy(attention_transpose);
        }
        if (!(output_weights->requires_gradient || output_bias->requires_gradient) || no_gradient)
        {
            tensor_destroy(attention_reshaped);
        }
        if (output_projection != *y)
        {
            tensor_destroy(output_projection);
        }
    }


    return error;
}

nw_error_t *tensor_scaled_dot_product_attention(const tensor_t *query, const tensor_t *key, const tensor_t *value, tensor_t **y, void *dropout_probability, bool_t inference)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("query", query);
    PRINTLN_DEBUG_TENSOR("key", key);
    PRINTLN_DEBUG_TENSOR("value", value);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(query, "query");
    CHECK_NULL_ARGUMENT(key, "key");
    CHECK_NULL_ARGUMENT(value, "value");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *key_transposed = NULL;
    void *scale = NULL;
    void *negative_infinity = NULL;
    tensor_t *zero_constant = NULL;
    tensor_t *scale_constant = NULL;
    tensor_t *negative_infinity_constant = NULL;
    tensor_t *similarity = NULL;
    tensor_t *scaled_similarity = NULL;
    tensor_t *bias = NULL;
    tensor_t *lower_triangular = NULL;
    tensor_t *upper_triangular = NULL;
    tensor_t *attention_i = NULL;
    tensor_t *attention_j = NULL;
    tensor_t *attention_k = NULL;
    tensor_t *attention_l = NULL;
    tensor_t *attention_m = NULL;
    datatype_t datatype = query->buffer->storage->datatype;
    runtime_t runtime = query->buffer->storage->runtime;
    int64_t d_k = key->buffer->view->shape[key->buffer->view->rank - 1];
    int64_t T = query->buffer->view->shape[query->buffer->view->rank - 2];
    size_t size = datatype_size(datatype);

    scale = (void *) malloc(size);
    if (!scale)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    negative_infinity = (void *) malloc(size);
    if (!negative_infinity)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) scale = (float32_t) 1.0 / sqrtf((float32_t) d_k);
        *(float32_t *) negative_infinity = -FLT_MAX;
        break;
    case FLOAT64:
        *(float64_t *) scale = (float64_t) 1.0 / sqrtf((float64_t) d_k);
        *(float64_t *) negative_infinity = -DBL_MAX;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype."), error);
        goto cleanup;
    }

    error = tensor_create_zeroes(&zero_constant, (int64_t[]){}, 0, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create zero tensor."), error);
        goto cleanup;
    }

    error = tensor_constant(negative_infinity, datatype, runtime, false, false, &negative_infinity_constant);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create constant tensor."), error);
        goto cleanup;
    }

    error = tensor_constant(scale, datatype, runtime, false, false, &scale_constant);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create constant tensor."), error);
        goto cleanup;
    }

    error = tensor_transpose(key, &key_transposed, -2, -1);
    if (error)
    {
        error = ERROR(ERROR_TRANSPOSE, string_create("failed to transpose tensor."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(query, key_transposed, &similarity);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(similarity, scale_constant, &scaled_similarity);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_create_ones(&bias, (int64_t[]){T, T}, 2, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_lower_triangular(bias, &lower_triangular);
    if (error)
    {
        error = ERROR(ERROR_LOWER_TRIANGULAR, string_create("failed to get lower triangular tensor."), error);
        goto cleanup;
    }

    error = tensor_compare_equal(lower_triangular, zero_constant, &upper_triangular);
    if (error)
    {
        error = ERROR(ERROR_COMPARE_EQUAL, string_create("failed to compare equal tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(upper_triangular, negative_infinity_constant, &attention_i);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    // error = tensor_where(upper_triangular, negative_infinity_constant, upper_triangular, &attention_i);
    // if (error)
    // {
    //     error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
    //     goto cleanup;
    // }

    error = tensor_multiplication(lower_triangular, scaled_similarity, &attention_j);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(attention_i, attention_j, &attention_k);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_softmax(attention_k, &attention_l, -1);
    if (error)
    {
        error = ERROR(ERROR_SOFTMAX, string_create("failed to softmax tensor."), error);
        goto cleanup;
    }
    
    error = tensor_dropout(attention_l, &attention_m, dropout_probability, inference);
    if (error)
    {
        error = ERROR(ERROR_DROPOUT, string_create("failed to dropout tensor."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(attention_m, value, y);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:
    free(scale);
    free(negative_infinity);
    tensor_destroy(bias);
    tensor_destroy(zero_constant);
    tensor_destroy(upper_triangular);
    tensor_destroy(negative_infinity_constant);

    if (!(key->requires_gradient || query->requires_gradient) || no_gradient || !tensor_shapes_equal(similarity, scale_constant))
    {
        tensor_destroy(scale_constant);
    }

    if (!(key->requires_gradient || query->requires_gradient) || no_gradient || !tensor_shapes_equal(attention_j, attention_i))
    {
        tensor_destroy(attention_i);
    }

    if (!(key->requires_gradient || query->requires_gradient) || no_gradient || !tensor_shapes_equal(scaled_similarity, lower_triangular))
    {
        tensor_destroy(lower_triangular);
    }

    if (!(key->requires_gradient || query->requires_gradient) || no_gradient)
    {
        tensor_destroy(key_transposed);
        tensor_destroy(similarity);
        tensor_destroy(scaled_similarity);
        tensor_destroy(attention_j);
        tensor_destroy(attention_k);
        if (attention_l != attention_m)
        {
            tensor_destroy(attention_l);
        }
    }

    if (!(value->requires_gradient) || no_gradient)
    {
        tensor_destroy(attention_m);
    }

    return error;
}


nw_error_t *tensor_dropout(const tensor_t *x, tensor_t **y, void *probability, bool_t inference)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *probability_constant = NULL;
    tensor_t *scale = NULL;
    tensor_t *mask = NULL;
    tensor_t *rand_tensor = NULL;
    tensor_t *x_i = NULL;
    void *min = NULL;
    void *max = NULL;
    void *scalar = NULL;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;
    size_t size = datatype_size(datatype);

    if (inference || !probability || is_zero(probability, datatype))
    {
        *y = (tensor_t *) x;
        return error;
    }

    min = (void *) malloc(size);
    if (!min) 
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    max = (void *) malloc(size);
    if (!max) 
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    scalar = (void *) malloc(size);
    if (!scalar) 
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) min = (float32_t) 0.0;
        *(float32_t *) max = (float32_t) 1.0;
        *(float32_t *) scalar = (float32_t) 1.0 / ((float32_t) 1.0 - *(float32_t *) probability);
        break;
    case FLOAT64:
        *(float64_t *) min = (float64_t) 0.0;
        *(float64_t *) max = (float64_t) 1.0;
        *(float64_t *) scalar = (float64_t) 1.0 / ((float64_t) 1.0 - *(float64_t *) probability);
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype."), NULL);
        goto cleanup;
    }

    error = tensor_constant(probability, datatype, runtime, false, false, &probability_constant);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = tensor_constant(scalar, datatype, runtime, false, false, &scale);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_create_uniform(&rand_tensor, x->buffer->view->shape, x->buffer->view->rank, runtime, datatype, false, false, min, max);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_compare_greater(rand_tensor, probability_constant, &mask);
    if (error)
    {
        error = ERROR(ERROR_COMPARE_GREATER, string_create("failed to compare greater tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x, mask, &x_i);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x_i, scale, y);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    free(scalar);
    free(min);
    free(max);
    
    tensor_destroy(probability_constant);
    tensor_destroy(rand_tensor);
    tensor_destroy(scale);

    if (!x->requires_gradient || no_gradient)
    {
        tensor_destroy(mask);
        tensor_destroy(x_i);
    }

    return error;
}

static nw_error_t *topological_sort(tensor_t *tensor, map_t *visited, stack_t *tensors)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("tensor", tensor);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(tensor, "tensor");
    CHECK_NULL_ARGUMENT(visited, "visited");
    CHECK_NULL_ARGUMENT(tensors, "tensors");

    nw_error_t *error = NULL;
    string_t id = string_create("%lu", tensor->id);
    function_t *context = tensor->context;

    if (map_contains(visited, id))
    {
        goto cleanup;
    }

    if (context)
    {
        operation_t *operation = tensor->context->operation;
        operation_type_t operation_type = tensor->context->operation_type;
        
        switch (operation_type)
        {
        case UNARY_OPERATION:
            if (operation->unary_operation)
            {
                error = topological_sort(operation->unary_operation->x, visited, tensors);
            }
            else
            {
                error = ERROR(ERROR_NULL, string_create("operation is null."), NULL);
            }
            break;
        case BINARY_OPERATION:
            if (operation->binary_operation)
            {
                error = topological_sort(operation->binary_operation->x, visited, tensors);
                if (!error)
                {
                    error = topological_sort(operation->binary_operation->y, visited, tensors);
                }
            }
            else
            {
                error = ERROR(ERROR_NULL, string_create("operation is null."), NULL);
            }
            break;
        case REDUCTION_OPERATION:
            if (operation->reduction_operation)
            {
                error = topological_sort(operation->reduction_operation->x, visited, tensors);
            }
            else
            {
                error = ERROR(ERROR_NULL, string_create("operation is null."), NULL);
            }
            break;
        case STRUCTURE_OPERATION:
            if (operation->structure_operation)
            {
                error = topological_sort(operation->structure_operation->x, visited, tensors);
            }
            else
            {
                error = ERROR(ERROR_NULL, string_create("operation is null."), NULL);
            }
            break;
        case CREATION_OPERATION:
            // Leaf node
            break;
        default:
            error = ERROR(ERROR_OPERATION_TYPE, string_create("unknown operation type %d.", (int) operation_type), NULL);
            break;
        }

        if (error)
        {
            error = ERROR(ERROR_SORT, string_create("failed to topologically sort computational graph."), error);
            goto cleanup;
        }
    }

    error = stack_push(tensors, tensor);
    if (error)
    {
        error = ERROR(ERROR_PUSH, string_create("failed to push tensor to stack."), error);
        goto cleanup;
    }

    error = map_set(visited, id, NULL);
    if (error)
    {
        error = ERROR(ERROR_SET, string_create("failed set tensor in map."), error);
        goto cleanup;
    }

cleanup:

    string_destroy(id);

    return error;
}

nw_error_t *tensor_arange(tensor_t **x, void *start, void *stop, void *step, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", *x);

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(start, "start");
    CHECK_NULL_ARGUMENT(stop, "stop");
    CHECK_NULL_ARGUMENT(step, "step");

    nw_error_t *error = NULL;
    int64_t rank = 1;
    int64_t shape[rank];
    const void *arguments[] = {start, stop, step};
    int64_t length = 3;
    
    switch (datatype)
    {
    case FLOAT32:
        *shape = (int64_t) ((*(float32_t *) stop - *(float32_t *) start) / *(float32_t *) step);
        break;
    case FLOAT64:
        *shape = (int64_t) ((*(float64_t *) stop - *(float64_t *) start) / *(float64_t *) step);
        break;
    default:
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
    } 

    error = apply_operation_creation(ARANGE_OPERATION, shape, rank, runtime, datatype, requires_gradient, persist, arguments, length, NULL, x);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", *x);

    return error;
}

nw_error_t *tensor_zeroes_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t persist)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    int64_t *shape = x->buffer->view->shape;
    int64_t rank = x->buffer->view->rank;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;

    error = tensor_create_zeroes(y, shape, rank, runtime, datatype, requires_gradient, persist);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_ones_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t persist)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    int64_t *shape = x->buffer->view->shape;
    int64_t rank = x->buffer->view->rank;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;

    error = tensor_create_ones(y, shape, rank, runtime, datatype, requires_gradient, persist);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_create_zeroes(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;

    error = apply_operation_creation(ZEROES_OPERATION, shape, rank, runtime, datatype, requires_gradient, persist, NULL, 0, NULL, x);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_create_ones(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;

    error = apply_operation_creation(ONES_OPERATION, shape, rank, runtime, datatype, requires_gradient, persist, NULL, 0, NULL, x);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_create_uniform(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, 
                                  bool_t requires_gradient, bool_t persist, void *lower_bound, void *upper_bound)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(lower_bound, "lower_bound");
    CHECK_NULL_ARGUMENT(upper_bound, "upper_bound");

    nw_error_t *error = NULL;
    const void *arguments[] = {lower_bound, upper_bound};
    int64_t length = 2;

    error = apply_operation_creation(UNIFORM_OPERATION, shape, rank, runtime, datatype, requires_gradient, persist, arguments, length, NULL, x);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_create_normal(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype,
                                 bool_t requires_gradient, bool_t persist, void *mean, void *standard_deviation)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(mean, "mean");
    CHECK_NULL_ARGUMENT(standard_deviation, "standard_deviation");

    nw_error_t *error = NULL;
    const void *arguments[] = {mean, standard_deviation};
    int64_t length = 2;

    error = apply_operation_creation(NORMAL_OPERATION, shape, rank, runtime, datatype, requires_gradient, persist, arguments, length, NULL, x);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

static nw_error_t *compute_fan(const int64_t *shape, int64_t rank, int64_t *fan, bool_t mode)
{
    if (!mode)
    {
        if (rank == 2)
        {
            *fan = shape[0];
        }
        else if (rank == 4)
        {
            *fan = shape[1] * shape[2] * shape[3];
        }
        else if (rank == 5)
        {
            *fan = shape[1] * shape[2] * shape[3] * shape[3];
        }
        else
        {
            return ERROR(ERROR_RANK, string_create("unable to compute fan_in for tensor."), NULL);
        }
    }
    else
    {
        if (rank == 2)
        {
            *fan = shape[1];
        }
        else if (rank == 4)
        {
            *fan = shape[0] * shape[2] * shape[3];
        }
        else if (rank == 5)
        {
            *fan = shape[0] * shape[2] * shape[3] * shape[3];
        }
        else
        {
            return ERROR(ERROR_RANK, string_create("unable to compute fan_out for tensor."), NULL);
        }
    }

    return NULL;
}

nw_error_t *tensor_create_kaiming_uniform(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime,
                                          datatype_t datatype, bool_t requires_gradient, bool_t persist, void *gain, bool_t mode)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");

    nw_error_t *error = NULL;
    void *lower_bound = NULL;
    void *upper_bound = NULL;
    size_t size = datatype_size(datatype);
    int64_t fan = 0;

    error = compute_fan(shape, rank, &fan, mode);
    if (error)
    {
        error = ERROR(ERROR_FAN, string_create("failed to compute fan."), error);
        goto cleanup;
    }

    lower_bound = (void *) malloc(size);
    if (!lower_bound)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    upper_bound = (void *) malloc(size);
    if (!upper_bound)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) upper_bound = *(float32_t *) gain * sqrtf(3.0 / (float32_t) fan);
        *(float32_t *) lower_bound = -*(float32_t *) upper_bound;
        break;
    case FLOAT64:
        *(float64_t *) upper_bound = *(float64_t *) gain * sqrtf(3.0 / (float64_t) fan);
        *(float64_t *) lower_bound = -*(float64_t *) upper_bound;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    error = tensor_create_uniform(x, shape, rank, runtime, datatype, requires_gradient, persist, lower_bound, upper_bound);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

cleanup:

    free(upper_bound);
    free(lower_bound);

    return error;
}

nw_error_t *tensor_create_kaiming_normal(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime,
                                         datatype_t datatype, bool_t requires_gradient, bool_t persist, void *gain, bool_t mode)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");

    nw_error_t *error = NULL;
    void *mean = NULL;
    void *standard_deviation = NULL;
    size_t size = datatype_size(datatype);
    int64_t fan = 0;

    error = compute_fan(shape, rank, &fan, mode);
    if (error)
    {
        error = ERROR(ERROR_FAN, string_create("failed to compute fan."), error);
        goto cleanup;
    }

    mean = (void *) malloc(size);
    if (!mean)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    standard_deviation = (void *) malloc(size);
    if (!standard_deviation)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) standard_deviation = *(float32_t *) gain / sqrtf((float32_t) fan);
        *(float32_t *) mean = (float32_t) 0.0;
        break;
    case FLOAT64:
        *(float64_t *) standard_deviation = *(float64_t *) gain / sqrt((float64_t) fan);
        *(float64_t *) mean = (float64_t) 0.0;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    error = tensor_create_normal(x, shape, rank, runtime, datatype, requires_gradient, persist, mean, standard_deviation);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

cleanup:

    free(mean);
    free(standard_deviation);

    return error;
}

nw_error_t *tensor_create_glorot_uniform(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype,
                                         bool_t requires_gradient, bool_t perist, void *gain)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");

    nw_error_t *error = NULL;
    void *lower_bound = NULL;
    void *upper_bound = NULL;
    size_t size = datatype_size(datatype);
    int64_t fan_in = 0, fan_out = 0;

    error = compute_fan(shape, rank, &fan_in, false);
    if (error)
    {
        error = ERROR(ERROR_FAN, string_create("failed to compute fan."), error);
        goto cleanup;
    }

    error = compute_fan(shape, rank, &fan_out, true);
    if (error)
    {
        error = ERROR(ERROR_FAN, string_create("failed to compute fan."), error);
        goto cleanup;
    }

    lower_bound = (void *) malloc(size);
    if (!lower_bound)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    upper_bound = (void *) malloc(size);
    if (!upper_bound)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) upper_bound = *(float32_t *) gain * sqrtf(6.0 / ((float32_t) fan_in + (float32_t) fan_out));
        *(float32_t *) lower_bound = -*(float32_t *) upper_bound;
        break;
    case FLOAT64:
        *(float64_t *) upper_bound =  *(float64_t *) gain * sqrt(6.0 / ((float64_t) fan_in + (float64_t) fan_out));
        *(float64_t *) lower_bound = -*(float64_t *) upper_bound;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    error = tensor_create_uniform(x, shape, rank, runtime, datatype, requires_gradient, perist, lower_bound, upper_bound);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

cleanup:

    free(upper_bound);
    free(lower_bound);

    return error;
}

nw_error_t *tensor_create_glorot_normal(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype,
                                        bool_t requires_gradient, bool_t persist, void *gain)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");

    nw_error_t *error = NULL;
    void *mean = NULL;
    void *standard_deviation = NULL;
    size_t size = datatype_size(datatype);
    int64_t fan_in = 0, fan_out = 0;

    error = compute_fan(shape, rank, &fan_in, false);
    if (error)
    {
        error = ERROR(ERROR_FAN, string_create("failed to compute fan."), error);
        goto cleanup;
    }

    error = compute_fan(shape, rank, &fan_out, true);
    if (error)
    {
        error = ERROR(ERROR_FAN, string_create("failed to compute fan."), error);
        goto cleanup;
    }

    mean = (void *) malloc(size);
    if (!mean)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    standard_deviation = (void *) malloc(size);
    if (!standard_deviation)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) standard_deviation = *(float32_t *) gain * sqrtf(2.0 / ((float32_t) fan_in + (float32_t) fan_out));
        *(float32_t *) mean = (float32_t) 0.0;
        break;
    case FLOAT64:
        *(float64_t *) standard_deviation = *(float64_t *) gain * sqrt(2.0 / ((float64_t) fan_in + (float64_t) fan_out));
        *(float64_t *) mean = (float64_t) 0.0;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        goto cleanup;
    }

    error = tensor_create_normal(x, shape, rank, runtime, datatype, requires_gradient, persist, mean, standard_deviation);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

cleanup:

    free(mean);
    free(standard_deviation);

    return error;
}

nw_error_t *tensor_empty_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t persist)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINTLN_DEBUG_BOOLEAN("requires_gradient", requires_gradient);
    PRINTLN_DEBUG_BOOLEAN("persist", persist);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    int64_t *shape = x->buffer->view->shape;
    int64_t rank = x->buffer->view->rank;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;

    error = tensor_create_empty(y, shape, rank, runtime, datatype, requires_gradient, persist);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_create_empty(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", *x);
    PRINTLN_DEBUG_INT64_ARRAY("shape", shape, rank);
    PRINTLN_DEBUG_BOOLEAN("requires_gradient", requires_gradient);
    PRINTLN_DEBUG_BOOLEAN("persist", persist);
    PRINTF_DEBUG("runtime %s\n", runtime_string(runtime));
    PRINTF_DEBUG("datatype %s\n", datatype_string(datatype));
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;

    error = apply_operation_creation(EMPTY_OPERATION, shape, rank, runtime, datatype, requires_gradient, persist, NULL, 0, NULL, x);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", *x);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_backward(tensor_t *x, tensor_t *gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("x->gradient", x->gradient);
    PRINTLN_DEBUG_TENSOR("gradient", gradient);
    PRINT_DEBUG_NEWLINE;

    with_no_gradient(true);
    nw_error_t *error = NULL;
    stack_t *tensors = NULL;
    map_t *visited = NULL;
    tensor_t *y = NULL;

    if (!gradient)
    {
        if (x->buffer->view->rank)
        {
            return ERROR(ERROR_RANK, string_create("gradient only implicitly created for scalars"), NULL);
        }

        error = tensor_ones_like(x, &x->gradient, false, false);
        if (error)
        {
            return ERROR(ERROR_CREATE, string_create("failed to create tensor of ones."), error);
        }
    }

    error = stack_create(&tensors);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create stack."), error);
        goto cleanup;
    }

    error = map_create(&visited);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create map."), error);
        goto cleanup;
    }

    error = topological_sort(x, visited, tensors);
    if (error)
    {
        error = ERROR(ERROR_SORT, string_create("failed to topologically sort tensors."), error);
        goto cleanup;
    }

    while (tensors->size > 0)
    {
        error = stack_pop(tensors, (void **) &y);
        if (error)
        {
            error = ERROR(ERROR_POP, string_create("failed to pop tensor from stack"), error);
            goto cleanup;
        }

        if (y->context)
        {
            error = apply_backward(y);
            if (error)
            {
                error = ERROR(ERROR_BACKWARD, string_create("failed to do backward pass."), error);
                goto cleanup;
            }

        }

        if (!y->persist)
        {
            tensor_destroy(y);
        }
    }

cleanup:

    map_destroy(visited);
    stack_destroy(tensors);
    with_no_gradient(false);
    
    return error;
}

nw_error_t *tensor_as_tensor(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    with_no_gradient(true);
    error = apply_operation_unary(AS_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }
    with_no_gradient(false);

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_accumulate_gradient(tensor_t *x, tensor_t *gradient)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("gradient", gradient);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(gradient, "gradient");

    nw_error_t *error = NULL;

    if (!x->gradient)
    {
        error = tensor_as_tensor(gradient, &(x->gradient));
        if (error)
        {
            return ERROR(ERROR_CREATE, string_create("failed create tensor."), error);
        }
    }
    else
    {
        tensor_t *updated_gradient = NULL;

        error = tensor_addition(x->gradient, gradient, &updated_gradient);
        if (error)
        {
            return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        }
        tensor_destroy(x->gradient);
        x->gradient = updated_gradient;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("x->gradient", x->gradient);
    PRINT_DEBUG_NEWLINE;

    return error;
}

