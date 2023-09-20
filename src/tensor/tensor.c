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
nw_error_t *tensor_create(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(tensor, "tensor");

    static uint64_t id = 0;

    *tensor = (tensor_t *) malloc(sizeof(tensor_t));
    if (!*tensor)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(tensor_t)), NULL);
    }

    (*tensor)->id = id++;
    (*tensor)->buffer = buffer;
    (*tensor)->context = context;
    (*tensor)->gradient = gradient;
    (*tensor)->requires_gradient = requires_gradient;

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
        buffer_destroy(tensor->buffer);
        tensor_destroy(tensor->gradient);
        function_destroy(tensor->context);
        free(tensor);
    }
}

nw_error_t *tensor_from_data(tensor_t **x, void *data, runtime_t runtime, datatype_t datatype, 
                             uint64_t rank, uint64_t *shape, bool_t copy, bool_t requires_gradient)
{
    nw_error_t *error = NULL;
    storage_t *storage = NULL;
    view_t *view = NULL;
    buffer_t *buffer = NULL;

    error = storage_create(&storage, runtime, datatype, shape_size(shape, rank), data, copy);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create storage."), error);
    }

    error = view_create(&view, 0, rank, shape, NULL);
    if (error)
    {
        storage_destroy(storage);
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = buffer_create(&buffer, view, storage, false);
    if (error)
    {
        view_destroy(view);
        storage_destroy(storage);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    error = tensor_create(x, buffer, NULL, NULL, requires_gradient);
    if (error)
    {
        buffer_destroy(buffer);
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_broadcast(const tensor_t *x_original,
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
    uint64_t *x_shape = x_original->buffer->view->shape; 
    uint64_t x_rank = x_original->buffer->view->rank; 
    uint64_t *y_shape = y_original->buffer->view->shape; 
    uint64_t y_rank = y_original->buffer->view->rank; 
    uint64_t broadcasted_rank = MAX(x_rank, y_rank);
    uint64_t broadcasted_shape[broadcasted_rank];

    error = broadcast_shapes(x_shape, x_rank, y_shape, y_rank, broadcasted_shape, broadcasted_rank);
    if (error)
    {
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor shapes."), error);
    }

    error = tensor_expand(x_original, broadcasted_shape, broadcasted_rank, x_broadcasted);
    if (error)
    {
        return ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
    }

    error = tensor_expand(y_original, broadcasted_shape, broadcasted_rank, y_broadcasted);
    if (error)
    {
        return ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINTLN_DEBUG_TENSOR("x_broadcasted", *x_broadcasted);
    PRINTLN_DEBUG_TENSOR("y_broadcasted", *y_broadcasted);
    PRINT_DEBUG_NEWLINE;

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
    uint64_t *x_shape = x_original->buffer->view->shape; 
    uint64_t x_rank = x_original->buffer->view->rank; 
    uint64_t *y_shape = y_original->buffer->view->shape; 
    uint64_t y_rank = y_original->buffer->view->rank; 
    uint64_t broadcasted_rank = MAX(x_rank, y_rank);
    uint64_t x_broadcasted_shape[broadcasted_rank];
    uint64_t y_broadcasted_shape[broadcasted_rank];

    error = matrix_multiplication_broadcast_shapes(x_shape, x_rank, y_shape, y_rank,
                                                   x_broadcasted_shape, y_broadcasted_shape, broadcasted_rank);
    if (error)
    {
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor shapes."), error);
    }

    error = tensor_expand(x_original, x_broadcasted_shape, broadcasted_rank, x_broadcasted);
    if (error)
    {
        return ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
    }

    error = tensor_expand(y_original, y_broadcasted_shape, broadcasted_rank, y_broadcasted);
    if (error)
    {
        return ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINTLN_DEBUG_TENSOR("x_broadcasted", *x_broadcasted);
    PRINTLN_DEBUG_TENSOR("y_broadcasted", *y_broadcasted);
    PRINT_DEBUG_NEWLINE;

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

    error = apply_function_unary(SIGMOID_OPERATION, x, y);
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

nw_error_t *tensor_expand(const tensor_t *x, const uint64_t *shape, uint64_t length, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("shape", shape, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(x->buffer->view->shape, "x->buffer->view->shape");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;

    error = apply_function_structure(EXPAND_OPERATION, x, shape, length, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to expand tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINTLN_DEBUG_UINT64_ARRAY("shape", shape, length);
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

    error = apply_function_binary(ADDITION_OPERATION, x, y, z);
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

    error = apply_function_binary(SUBTRACTION_OPERATION, x, y, z);
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

    error = apply_function_binary(DIVISION_OPERATION, x, y, z);
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

    error = apply_function_binary(MULTIPLICATION_OPERATION, x, y, z);
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

    error = apply_function_binary(COMPARE_EQUAL, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to compare equal tensors."), error);
    }

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

    error = apply_function_binary(COMPARE_GREATER, x, y, z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to compare greater tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

    return NULL;
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

    error = apply_function_binary(POWER_OPERATION, x, y, z);
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
    tensor_t *x_broadcasted = NULL;
    tensor_t *y_broadcasted = NULL;

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

    error = tensor_broadcast_matrix_multiplication(x_contiguous, y_contiguous, &x_broadcasted, &y_broadcasted);
    if (error)
    {
        error = ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensors."), error);
        goto cleanup;
    } 

    error = apply_function_binary(MATRIX_MULTIPLICATION_OPERATION, x_broadcasted, y_broadcasted, z);
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

    if (!(x->requires_gradient || y->requires_gradient))
    {
        if (x_contiguous != x_broadcasted)
        {
            tensor_destroy(x_broadcasted);
        }

        if (y_contiguous != y_broadcasted)
        {
            tensor_destroy(y_broadcasted);
        }
    }

    if (x != x_contiguous && !(x->requires_gradient))
    {
        tensor_destroy(x_contiguous);
    }

    if (y != y_contiguous && !(y->requires_gradient))
    {
        tensor_destroy(y_contiguous);
    }

    return error;
}

nw_error_t *tensor_summation(const tensor_t *x, tensor_t **y, const uint64_t *axis, uint64_t length, bool_t keep_dimension)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    
    error = apply_function_reduction(SUMMATION_OPERATION, x, axis, length, keep_dimension, y);
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

nw_error_t *tensor_maximum(const tensor_t *x, tensor_t **y, const uint64_t *axis, uint64_t length, bool_t keep_dimension)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_function_reduction(MAXIMUM_OPERATION, x, axis, length, keep_dimension, y);
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

nw_error_t *tensor_item(tensor_t *x, void *value)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(x->buffer->storage->data, "x->buffer->storage->data");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(value, "value");

    if (x->buffer->view->rank)
    {
        return ERROR(ERROR_RANK_CONFLICT, string_create("tensor must be rank zero."), NULL);
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

nw_error_t *tensor_argument_maximum(const tensor_t *x, tensor_t **y, uint64_t axis, bool_t keep_dimension)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTF_DEBUG("axis: %lu\n", axis);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(x->buffer->view->shape, "x->buffer->view->shape");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    uint64_t *shape = x->buffer->view->shape;
    uint64_t rank = x->buffer->view->rank;

    if (axis >= rank)
    {
        return ERROR(ERROR_AXIS, string_create("axis out of range of tensor."), NULL);
    }

    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;
    uint64_t new_rank = rank -axis;
    uint64_t new_shape [new_rank];
    new_shape[0] = shape[axis];
    for (uint64_t i = 1; i < new_rank; ++i)
    {
        new_shape[i] = (uint64_t) 1;
    }

    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;
    tensor_t *x_l = NULL;
    tensor_t *x_m = NULL;

    error = tensor_maximum(x, &x_i, (uint64_t[]){axis}, (uint64_t) 1, true);
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

    error = tensor_arange(&x_k, (uint64_t) 0, shape[axis], (uint64_t) 1, runtime, datatype, false);
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

    error = tensor_maximum(x_m, y, (uint64_t[]){axis}, (uint64_t) 1, keep_dimension);
    if (error)
    {
        error = ERROR(ERROR_MAXIMUM, string_create("failed to get maximum of tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    tensor_destroy(x_i);
    tensor_destroy(x_j);
    tensor_destroy(x_k);
    tensor_destroy(x_l);
    tensor_destroy(x_m);

    return error;
}

inline uint64_t tensor_number_of_elements(const tensor_t *x)
{
    return (x && x->buffer && x->buffer->view) ? shape_size(x->buffer->view->shape, x->buffer->view->rank) : 0;
}

nw_error_t *tensor_constant(void *constant, datatype_t datatype, runtime_t runtime, tensor_t **x)
{
    CHECK_NULL_ARGUMENT(constant, "constant");
    CHECK_NULL_ARGUMENT(x, "x");

    nw_error_t *error = NULL;
    view_t *view = NULL;
    storage_t *storage = NULL;
    buffer_t *buffer = NULL;

    error = view_create(&view, 0, 0, (uint64_t[]){}, (uint64_t[]){});
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = storage_create(&storage, runtime, datatype, 1, constant, true);
    if (error)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create storage."), error);
    }

    error = buffer_create(&buffer, view, storage, false);
    if (error)
    {
        view_destroy(view);
        storage_destroy(storage);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    error = tensor_create(x, buffer, NULL, NULL, false);
    if (error)
    {
        buffer_destroy(buffer);
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_constant_float32(float32_t constant, tensor_t **x, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(x, "x");

    nw_error_t *error = NULL;

    error = tensor_constant(&constant, FLOAT32, runtime, x);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
    }

    return error;
}

nw_error_t *tensor_constant_float64(float64_t constant, tensor_t **x, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(x, "x");

    nw_error_t *error = NULL;

    error = tensor_constant(&constant, FLOAT64, runtime, x);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
    }

    return error;
}

nw_error_t *tensor_mean(const tensor_t *x, tensor_t **y, const uint64_t *axis, uint64_t length, bool_t keep_dimension)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    runtime_t runtime = x->buffer->storage->runtime;
    datatype_t datatype = x->buffer->storage->datatype;

    error = tensor_summation(x, &x_i, axis, length, keep_dimension);
    if (error)
    {
        error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        error = tensor_constant_float32((float32_t) tensor_number_of_elements(x_i) / 
                                        (float32_t) tensor_number_of_elements(x), 
                                        &x_j, runtime);
        break;
    case FLOAT64:
        error = tensor_constant_float64((float64_t) tensor_number_of_elements(x_i) / 
                                        (float64_t) tensor_number_of_elements(x), 
                                        &x_j, runtime);
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        break;
    }

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

    if (!x_i->requires_gradient)
    {
        tensor_destroy(x_i);
    }
    tensor_destroy(x_j);

    return error;
}

static nw_error_t *softmax(const tensor_t *x, tensor_t **y_max, tensor_t **y_num, tensor_t **y_den, const uint64_t *axis, uint64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y_max, "y_max");
    CHECK_NULL_ARGUMENT(y_num, "y_num");
    CHECK_NULL_ARGUMENT(y_den, "y_den");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;

    error = tensor_maximum(x, &x_i, axis, length, true);
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

    error = tensor_summation(*y_num, y_den, axis, length, true);
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

    if (!x->requires_gradient)
    {
        tensor_destroy(x_i);
    }

    return error;
}

nw_error_t *tensor_softmax(const tensor_t *x, tensor_t **y, const uint64_t *axis, uint64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;

    error = softmax(x, &x_i, &x_j, &x_k, axis, length);
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

    tensor_destroy(x_i);
    if (!x->requires_gradient)
    {
        tensor_destroy(x_j);
        tensor_destroy(x_k);
    }

    return error;
}

nw_error_t *tensor_logsoftmax(const tensor_t *x, tensor_t **y, const uint64_t *axis, uint64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *x_i = NULL;
    tensor_t *x_j = NULL;
    tensor_t *x_k = NULL;
    tensor_t *x_l = NULL;

    error = softmax(x, &x_i, &x_j, &x_k, axis, length);
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

    tensor_destroy(x_j);
    if (!x->requires_gradient)
    {
        tensor_destroy(x_i);
        tensor_destroy(x_k);
        tensor_destroy(x_l);
    }

    return error;
}

bool_t tensor_is_contiguous(const tensor_t *x)
{
    return x && x->buffer && x->buffer->view &&
           is_contiguous(x->buffer->view->shape, x->buffer->view->rank, 
                         x->buffer->view->strides, x->buffer->view->offset);
}

nw_error_t *tensor_reshape(const tensor_t *x, tensor_t **y, const uint64_t *shape, uint64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("shape", shape, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;
    tensor_t *x_contiguous = NULL;

    error = tensor_contiguous(x, &x_contiguous);
    if (error)
    {
        error = ERROR(ERROR_CONTIGUOUS, string_create("failed to make tensor contiguous."), error);
        goto cleanup;
    }

    error = apply_function_structure(RESHAPE_OPERATION, x_contiguous, shape, length, y);
    if (error)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    if (x != x_contiguous && !x->requires_gradient)
    {
        tensor_destroy(x_contiguous);
    }

    return error;
}

nw_error_t *tensor_permute(const tensor_t *x, tensor_t **y, uint64_t *axis, uint64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(axis, "axis");

    nw_error_t *error = NULL;

    error = apply_function_structure(PERMUTE_OPERATION, x, axis, length, y);
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

bool_t tensor_shapes_equal(const tensor_t *x, const tensor_t *y)
{
    return x && y && x->buffer && y->buffer && x->buffer->view && y->buffer->view &&
           shapes_equal(x->buffer->view->shape, x->buffer->view->rank,
                        y->buffer->view->shape, y->buffer->view->rank);
}

nw_error_t *tensor_transpose(const tensor_t *x, tensor_t **y, uint64_t axis1, uint64_t axis2)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTF_DEBUG("(axis1: %lu, axis2: %lu)\n", axis1, axis2);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    uint64_t rank = x->buffer->view->rank;
    uint64_t axis[rank];
    for (uint64_t i = 0; i < rank; ++i)
    {
        axis[i] = i;
    }
    uint64_t temp = axis[axis2];
    axis[axis2] = axis[axis1];
    axis[axis1] = temp;

    error = apply_function_structure(PERMUTE_OPERATION, x, axis, rank, y);
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

nw_error_t *tensor_slice(const tensor_t *x, tensor_t **y, uint64_t *arguments, uint64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("arguments", arguments, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    nw_error_t *error = NULL;

    error = apply_function_structure(SLICE_OPERATION, x, arguments, length, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to slice tensor."), error);
    }
    
    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *tensor_padding(const tensor_t *x, tensor_t **y, uint64_t *arguments, uint64_t length)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_UINT64_ARRAY("arguments", arguments, length);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    nw_error_t *error = NULL;

    error = apply_function_structure(PADDING_OPERATION, x, arguments, length, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to pad tensor."), error);
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

    error = apply_function_unary(CONTIGUOUS_OPERATION, x, y);
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

nw_error_t *tensor_logarithm(const tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_function_unary(LOGARITHM_OPERATION, x, y);
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

    error = apply_function_unary(SINE_OPERATION, x, y);
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

    error = apply_function_unary(COSINE_OPERATION, x, y);
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

    error = apply_function_unary(EXPONENTIAL_OPERATION, x, y);
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

    error = apply_function_unary(SQUARE_ROOT_OPERATION, x, y);
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

    error = apply_function_unary(RECIPROCAL_OPERATION, x, y);
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

    error = apply_function_unary(NEGATION_OPERATION, x, y);
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

    error = apply_function_unary(RECTIFIED_LINEAR_OPERATION, x, y);
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


static nw_error_t *topological_sort(tensor_t *tensor, map_t *visited, stack_t *tensors)
{
    CHECK_NULL_ARGUMENT(tensor, "tensor");
    CHECK_NULL_ARGUMENT(visited, "visited");
    CHECK_NULL_ARGUMENT(tensors, "tensors");

    nw_error_t *error = NULL;
    string_t id = string_create("%ld", tensor->id);
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
        default:
            error = ERROR(ERROR_UKNOWN_OPERATION_TYPE, 
                          string_create("unknown operation type %d.", (int) operation_type), NULL);
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

    return error;

cleanup:

    string_destroy(id);

    return error;
}

nw_error_t *tensor_arange(tensor_t **x, uint64_t start, uint64_t stop, uint64_t step, runtime_t runtime, datatype_t datatype, bool_t requires_gradient)
{
    nw_error_t *error = NULL;
    uint64_t interval = 1 + (((stop - start) - 1 ) / step);
    buffer_t *buffer = NULL;

    error = buffer_create_empty(&buffer, (uint64_t[]) {interval}, NULL, 1, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    error = runtime_init_arange(buffer, start, stop, step);
    if (error)
    {
        buffer_destroy(buffer);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize buffer."), error);
    }

    error = tensor_create(x, buffer, NULL, NULL, requires_gradient);
    if (error)
    {
        buffer_destroy(buffer);
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_zeroes_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t preserve_memory_format)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_empty_like(x, y, requires_gradient, preserve_memory_format);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_init_zeroes((*y)->buffer);
    if (error)
    {
        tensor_destroy(*y);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tensor with zeroes."), error);
    }

    return error;
}

nw_error_t *tensor_ones_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t preserve_memory_format)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_empty_like(x, y, requires_gradient, preserve_memory_format);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = runtime_init_ones((*y)->buffer);
    if (error)
    {
        tensor_destroy(*y);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tensor with ones."), error);
    }

    return error;
}

nw_error_t *tensor_create_zeroes(tensor_t **x, const uint64_t *shape, uint64_t rank,
                                 runtime_t runtime, datatype_t datatype, bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;

    error = tensor_create_empty(shape, NULL, rank, x, requires_gradient, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = runtime_init_zeroes((*x)->buffer);
    {
        tensor_destroy(*x);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tensor."), error);
    }

    return error;
}

nw_error_t *tensor_create_ones(tensor_t **x, const uint64_t *shape, uint64_t rank,
                               runtime_t runtime, datatype_t datatype, bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;

    error = tensor_create_empty(shape, NULL, rank, x, requires_gradient, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = runtime_init_ones((*x)->buffer);
    {
        tensor_destroy(*x);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tensor."), error);
    }

    return error;
}

nw_error_t *tensor_create_uniform(tensor_t **x, const uint64_t *shape, uint64_t rank,
                                  runtime_t runtime, datatype_t datatype, bool_t requires_gradient,
                                  void *lower_bound, void *upper_bound)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(lower_bound, "lower_bound");
    CHECK_NULL_ARGUMENT(upper_bound, "upper_bound");

    nw_error_t *error = NULL;

    error = tensor_create_empty(shape, NULL, rank, x, requires_gradient, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = runtime_init_uniform((*x)->buffer, lower_bound, upper_bound);
    {
        tensor_destroy(*x);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tensor."), error);
    }

    return error;
}

nw_error_t *tensor_create_normal(tensor_t **x, const uint64_t *shape, uint64_t rank,
                                 runtime_t runtime, datatype_t datatype, bool_t requires_gradient,
                                 void *mean, void *standard_deviation)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(mean, "mean");
    CHECK_NULL_ARGUMENT(standard_deviation, "standard_deviation");

    nw_error_t *error = NULL;

    error = tensor_create_empty(shape, NULL, rank, x, requires_gradient, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = runtime_init_normal((*x)->buffer, mean, standard_deviation);
    {
        tensor_destroy(*x);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tensor."), error);
    }

    return error;
}

nw_error_t *tensor_create_kaiming_uniform(tensor_t **x, const uint64_t *shape, uint64_t rank,
                                          runtime_t runtime, datatype_t datatype, bool_t requires_gradient,
                                          void *gain, void *fan)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");
    CHECK_NULL_ARGUMENT(fan, "fan");

    nw_error_t *error = NULL;

    error = tensor_create_empty(shape, NULL, rank, x, requires_gradient, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    float32_t lower_bound_float32, upper_bound_float32;
    float64_t lower_bound_float64, upper_bound_float64;

    switch (datatype)
    {
    case FLOAT32:
        upper_bound_float32 = *(float32_t *) gain * sqrtf(3.0 / *(float32_t *) fan);
        lower_bound_float32 = -upper_bound_float32;
        error = runtime_init_uniform((*x)->buffer, &lower_bound_float32, &upper_bound_float32);
        break;
    case FLOAT64:
        upper_bound_float64 = *(float64_t *) gain * sqrt(3.0 / *(float64_t *) fan);
        lower_bound_float64 = -upper_bound_float64;
        error = runtime_init_uniform((*x)->buffer, &lower_bound_float64, &upper_bound_float64);
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        break;
    }

    if (error)
    {
        tensor_destroy(*x);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tenssor."), error);
    }

    return error;
}

nw_error_t *tensor_create_kaiming_normal(tensor_t **x, const uint64_t *shape, uint64_t rank,
                                         runtime_t runtime, datatype_t datatype, bool_t requires_gradient,
                                         void *gain, void *fan)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");
    CHECK_NULL_ARGUMENT(fan, "fan");

    nw_error_t *error = NULL;

    error = tensor_create_empty(shape, NULL, rank, x, requires_gradient, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    float32_t mean_float32, standard_deviation_float32;
    float64_t mean_float64, standard_deviation_float64;

    switch (datatype)
    {
    case FLOAT32:
        mean_float32 = (float32_t) 0.0;
        standard_deviation_float32 = *(float32_t *) gain / sqrtf(*(float32_t *) fan);
        error = runtime_init_normal((*x)->buffer, (void *) &mean_float32, (void *) &standard_deviation_float32);
        break;
    case FLOAT64:
        mean_float64 = (float64_t) 0.0;
        standard_deviation_float64 = *(float64_t *) gain / sqrt(*(float64_t *) fan);
        error = runtime_init_uniform((*x)->buffer, (void *) &mean_float64, (void *) &standard_deviation_float64);
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        break;
    }

    if (error)
    {
        tensor_destroy(*x);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tenssor."), error);
    }

    return error;
}

nw_error_t *tensor_create_glorot_uniform(tensor_t **x, const uint64_t *shape, uint64_t rank,
                                         runtime_t runtime, datatype_t datatype, bool_t requires_gradient,
                                         void *gain, void *fan_in, void *fan_out)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");
    CHECK_NULL_ARGUMENT(fan_in, "fan_in");
    CHECK_NULL_ARGUMENT(fan_out, "fan_out");

    nw_error_t *error = NULL;

    error = tensor_create_empty(shape, NULL, rank, x, requires_gradient, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    float32_t lower_bound_float32, upper_bound_float32;
    float64_t lower_bound_float64, upper_bound_float64;

    switch (datatype)
    {
    case FLOAT32:
        upper_bound_float32 = *(float32_t *) gain * sqrtf(6.0 / (*(float32_t *) fan_in + *(float32_t *) fan_out));
        lower_bound_float32 = -upper_bound_float32;
        error = runtime_init_uniform((*x)->buffer, &lower_bound_float32, &upper_bound_float32);
        break;
    case FLOAT64:
        upper_bound_float64 = *(float64_t *) gain * sqrt(6.0 / (*(float64_t *) fan_in + *(float64_t *) fan_out));
        lower_bound_float64 = -upper_bound_float64;
        error = runtime_init_uniform((*x)->buffer, &lower_bound_float64, &upper_bound_float64);
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        break;
    }

    if (error)
    {
        tensor_destroy(*x);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tenssor."), error);
    }

    return error;
}

nw_error_t *tensor_create_glorot_normal(tensor_t **x, const uint64_t *shape, uint64_t rank,
                                        runtime_t runtime, datatype_t datatype, bool_t requires_gradient,
                                        void *gain, void *fan_in, void *fan_out)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");
    CHECK_NULL_ARGUMENT(fan_in, "fan_in");
    CHECK_NULL_ARGUMENT(fan_in, "fan_in");

    nw_error_t *error = NULL;

    error = tensor_create_empty(shape, NULL, rank, x, requires_gradient, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    float32_t mean_float32, standard_deviation_float32;
    float64_t mean_float64, standard_deviation_float64;

    switch (datatype)
    {
    case FLOAT32:
        mean_float32 = (float32_t) 0.0;
        standard_deviation_float32 = *(float32_t *) gain * sqrtf(2.0 / (*(float32_t *) fan_in + *(float32_t *) fan_out));
        error = runtime_init_normal((*x)->buffer, (void *) &mean_float32, (void *) &standard_deviation_float32);
        break;
    case FLOAT64:
        mean_float64 = (float64_t) 0.0;
        standard_deviation_float64 = *(float64_t *) gain * sqrt(2.0 / (*(float64_t *) fan_in + *(float64_t *) fan_out));
        error = runtime_init_normal((*x)->buffer, (void *) &mean_float64, (void *) &standard_deviation_float64);
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        break;
    }

    if (error)
    {
        tensor_destroy(*x);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tenssor."), error);
    }

    return error;
}

nw_error_t *tensor_empty_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t preserve_memory_format)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    view_t *view = NULL;
    storage_t *storage = NULL;
    buffer_t *buffer = NULL;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;
    uint64_t rank = x->buffer->view->rank;
    uint64_t *shape = x->buffer->view->shape;
    uint64_t offset = preserve_memory_format ? x->buffer->view->offset : 0;
    uint64_t *strides = preserve_memory_format ? x->buffer->view->strides : NULL;
    uint64_t n = preserve_memory_format ? x->buffer->storage->n : shape_size(shape, rank);

    error = view_create(&view, offset, rank, shape, strides);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = storage_create(&storage, runtime, datatype, n, NULL, true);
    if (error)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create storage."), error);
    }

    error = buffer_create(&buffer, view, storage, false);
    if (error)
    {
        view_destroy(view);
        storage_destroy(storage);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    error = tensor_create(y, buffer, NULL, NULL, requires_gradient);
    if (error)
    {
        buffer_destroy(buffer);
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return error;
}

nw_error_t *tensor_create_empty(const uint64_t *shape, const uint64_t *strides, uint64_t rank, tensor_t **y, 
                                bool_t requires_gradient, runtime_t runtime, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    nw_error_t *error = NULL;
    buffer_t *buffer = NULL;

    error = buffer_create_empty(&buffer, shape, strides, rank, runtime, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    error = tensor_create(y, buffer, NULL, NULL, requires_gradient);
    if (error)
    {
        buffer_destroy(buffer);
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

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

    nw_error_t *error = NULL;
    stack_t *tensors = NULL;
    map_t *visited = NULL;
    tensor_t *y = NULL;

    if (!gradient)
    {
        if (x->buffer->view->rank)
        {
            return ERROR(ERROR_RANK_CONFLICT, string_create("gradient only implicitly created for scalars"), NULL);
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
            error = function_backward(y->context, y->gradient);
            if (error)
            {
                error = ERROR(ERROR_BACKWARD, string_create("failed to do backward pass."), error);
                goto cleanup;
            }

        }

        if (y->context || !y->requires_gradient)
        {
            tensor_destroy(y);
        }
    }

cleanup:

    map_destroy(visited);
    stack_destroy(tensors);
    
    return error;
}

nw_error_t *tensor_as_tensor(const tensor_t *x, tensor_t **y, bool_t requires_gradient)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    view_t *view = NULL;
    buffer_t *buffer = NULL;

    error = view_copy(x->buffer->view, &view);
    if (error)
    {
        return ERROR(ERROR_COPY, string_create("failed to copy view."), error);
    }

    error = buffer_create(&buffer, view, x->buffer->storage, false);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    error = tensor_create(y, buffer, NULL, NULL, requires_gradient);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    return NULL;
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
        error = tensor_as_tensor(gradient, &(x->gradient), gradient->requires_gradient);
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

