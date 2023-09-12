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
 * @param[in] lock Flag to indicate that the tensor persists after computational 
 *                 graph is incrementally destroyed during back propogation.
 * @return Error if received NULL argument for tensor.
 *         Error if no sufficient memory could be dynamically allocate for the tensor.
 *         NULL if tensor was created successfully.
 */
nw_error_t *tensor_create(tensor_t **tensor ,
                          buffer_t *buffer,
                          function_t *context,
                          tensor_t *gradient,
                          bool_t requires_gradient,
                          bool_t lock)
{
    CHECK_NULL_ARGUMENT(tensor, "tensor");

    static uint64_t id = 0;

    *tensor = (tensor_t *) malloc(sizeof(tensor_t));
    if (*tensor == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate tensor of size %zu bytes.",
                     sizeof(tensor_t)), NULL);
    }

    (*tensor)->id = id++;
    (*tensor)->buffer = buffer;
    (*tensor)->context = context;
    (*tensor)->gradient = gradient;
    (*tensor)->requires_gradient = requires_gradient;
    (*tensor)->lock = lock;

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
    if (error != NULL)
    {
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor shapes."), error);
    }

    error = tensor_expand(x_original, broadcasted_shape, broadcasted_rank, x_broadcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
    }

    error = tensor_expand(y_original, broadcasted_shape, broadcasted_rank, y_broadcasted);
    if (error != NULL)
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
    if (error != NULL)
    {
        return ERROR(ERROR_BROADCAST, string_create("failed to broadcast tensor shapes."), error);
    }

    error = tensor_expand(x_original, x_broadcasted_shape, broadcasted_rank, x_broadcasted);
    if (error != NULL)
    {
        return ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
    }

    error = tensor_expand(y_original, y_broadcasted_shape, broadcasted_rank, y_broadcasted);
    if (error != NULL)
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
    if (error != NULL)
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

    if (shapes_equal(x->buffer->view->shape, x->buffer->view->rank, shape, length))
    {
        *y = x;
    }
    else
    {
        error = apply_function_structure(EXPAND_OPERATION, x, shape, length, y);
        if (error != NULL)
        {
            return ERROR(ERROR_FORWARD, string_create("failed to expand tensor."), error);
        }
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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

    error = apply_function_binary(MATRIX_MULTIPLICATION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to matrix multiply tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", *z);
    PRINT_DEBUG_NEWLINE;

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
    if (error != NULL)
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
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to max tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

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
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create view."), error);
    }

    error = storage_create(&storage, runtime, datatype, 1, constant);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE, string_create("failed to create storage."), error);
    }

    error = buffer_create(&buffer, view, storage, false);
    if (error != NULL)
    {
        view_destroy(view);
        storage_destroy(storage);
        return ERROR(ERROR_CREATE, string_create("failed to create buffer."), error);
    }

    error = tensor_create(x, buffer, NULL, NULL, false, false);
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
    {
        error = ERROR(ERROR_SUMMATION, string_create("failed to sum tensor."), error);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        error = tensor_constant_float32((float32_t) tensor_number_of_elements(x_i) / 
                                        (float32_t) tensor_number_of_elements(x), 
                                        runtime, &x_j);
        break;
    case FLOAT64:
        error = tensor_constant_float64((float64_t) tensor_number_of_elements(x_i) / 
                                        (float64_t) tensor_number_of_elements(x), 
                                        runtime, &x_j);
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int) datatype), NULL);
        break;
    }

    if (error != NULL)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create scalar tensor."), error);
        goto cleanup;
    }

    error = tensor_multiplication(x_j, x_i, y);
    if (error != NULL)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    if (!x->requires_gradient)
    {
        goto cleanup;
    }

    return error;

cleanup:

    tensor_destroy(x_i);
    tensor_destroy(x_j);

    return error;
}

bool_t tensor_is_contiguous(const tensor_t *x)
{
    return !(x == NULL || x->buffer == NULL || x->buffer->view == NULL) &&
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
    bool_t x_is_contiguous = tensor_is_contiguous(x);

    if (x_is_contiguous)
    {
        x_contiguous = x;
    }
    else
    {
        error = tensor_contiguous(x, &x_contiguous);
        if (error != NULL)
        {
            error = ERROR(ERROR_CONTIGUOUS, string_create("failed to make tensor contiguous."), error);
            goto cleanup;
        }
    }

    error = apply_function_structure(RESHAPE_OPERATION, x_contiguous, shape, length, y);
    if (error != NULL)
    {
        error = ERROR(ERROR_FORWARD, string_create("failed to reshape tensor."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    if (!x->requires_gradient)
    {
        goto cleanup;
    }

    return error;

cleanup:

    if (!x_is_contiguous)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
    if (error != NULL)
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
            if (!operation->unary_operation)
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

        if (error != NULL)
        {
            error = ERROR(ERROR_SORT, string_create("failed to topologically sort computational graph."), error);
            goto cleanup;
        }
    }

    error = stack_push(tensors, tensor);
    if (error != NULL)
    {
        error = ERROR(ERROR_PUSH, string_create("failed to push tensor to stack."), error);
        goto cleanup;
    }

    error = map_set(visited, id, NULL);
    if (error != NULL)
    {
        error = ERROR(ERROR_SET, string_create("failed set tensor in map."), error);
        goto cleanup;
    }

    return error;

cleanup:

    string_destroy(id);

    return error;
}

nw_error_t *tensor_as_zeroes(const tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_as_empty(x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create empty tensor."), error);
    }

    error = init_zeroes(y);
    if (error != NULL)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize tensor with zeroes."), error);
    }

    return error;
}

nw_error_t *tensor_as_tensor(const tensor_t *x, tensor_t **y)
{
    if (!tensor_is_empty(y))
    {
        return ERROR(ERROR_CREATE,
                     string_create("tensor y is not empty."),
                     NULL);
    }

    view_t *view = NULL;
    buffer_t *buffer = NULL;
    nw_error_t *error = NULL;

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

    error = buffer_create(&buffer,
                          view,
                          x->buffer->storage,
                          false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create buffer."),
                     error);
    }

    y->buffer = buffer;

    return NULL;
}

nw_error_t *tensor_as_ones(const tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    if (!tensor_is_empty(y))
    {
        return ERROR(ERROR_CREATE,
                     string_create("tensor y is not empty."),
                     NULL);
    }

    nw_error_t *error = NULL;

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

nw_error_t *tensor_as_empty(const tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(y, "y");

    if (!tensor_is_empty(y))
    {
        return ERROR(ERROR_CREATE, 
                     string_create("tensor y is not empty."),
                     NULL);
    }

    view_t *view = NULL;
    storage_t *storage = NULL;
    buffer_t *buffer = NULL;
    nw_error_t *error = NULL;

    error = view_create(&view, 
                        0,
                        x->buffer->view->rank,
                        x->buffer->view->shape,
                        x->buffer->view->strides);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create view."),
                     error);
    }

    error = storage_create(&storage, 
                           x->buffer->storage->runtime,
                           x->buffer->storage->datatype,
                           x->buffer->storage->n,
                           NULL);

    error = buffer_create(&buffer,
                          view,
                          storage,
                          false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create buffer."),
                     error);
    }

    y->buffer = buffer;

    return NULL;
}

nw_error_t *tensor_as_empty_contiguous(const tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    if (!tensor_is_empty(y))
    {
        return ERROR(ERROR_CREATE, 
                     string_create("tensor y is not empty."),
                     NULL);
    }

    view_t *view = NULL;
    buffer_t *buffer = NULL;
    storage_t *storage = NULL;
    nw_error_t *error = NULL;
    uint64_t n = 0;

    error = view_create(&view, 
                        0,
                        x->buffer->view->rank,
                        x->buffer->view->shape,
                        NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create view."),
                     error);
    }

    error = n_from_shape_and_strides(view->shape, view->strides, view->rank, &n);
    if (error != NULL)
    {
        view_destroy(view);
        ERROR(ERROR_N,
              string_create("failed to compute storage size."),
              error);
    }

    error = storage_create(&storage, 
                           x->buffer->storage->runtime,
                           x->buffer->storage->datatype,
                           n,
                           NULL);
    if (error != NULL)
    {
        view_destroy(view);
        return ERROR(ERROR_CREATE,
                     string_create("failed to create storage."),
                     error);
    }

    error = buffer_create(&buffer,
                          view,
                          storage,
                          false);
    if (error != NULL)
    {
        view_destroy(view);
        storage_destroy(storage);
        return ERROR(ERROR_CREATE,
                     string_create("failed to create buffer."),
                     error);
    }

    y->buffer = buffer;

    return NULL;
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

    if (gradient == NULL)
    {
        if (x->buffer->view->rank > 0)
        {
            return ERROR(ERROR_RANK_CONFLICT,
                         string_create("x rank %lu should be a scalar tensor of rank 0.",
                         (unsigned long) x->buffer->view->rank),
                         NULL);
        }

        error = tensor_create_default(&x->gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to gradient tensor."),
                         error);
        }

        error = tensor_as_ones(x, x->gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_INITIALIZATION,
                         string_create("failed to initialize gradient tensor with ones."),
                         error);
        }
    }

    error = stack_create(&tensors);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create stack."),
                     error);
    }

    error = map_create(&visited);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create map."),
                     error);
    }

    error = topological_sort(x, visited, tensors);
    if (error != NULL)
    {
        return ERROR(ERROR_SORT,
                     string_create("failed to topologically sort nodes."),
                     error);
    }

    while (tensors->size > 0)
    {
        tensor_t *y = NULL;
        error = stack_pop(tensors, (void **) &y);
        if (error != NULL)
        {
            return ERROR(ERROR_POP, 
                         string_create("failed to pop tensor from stack"),
                         error);
        }

        if (y->context != NULL)
        {
            error = function_backward(y->context, y->gradient);
            if (error != NULL)
            {
                return ERROR(ERROR_BACKWARD, 
                             string_create("failed to run backward pass."),
                             error);
            }
        }

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

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("x->gradient", x->gradient);
    PRINTLN_DEBUG_TENSOR("gradient", gradient);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error;

    if (x->gradient == NULL)
    {
        error = tensor_create_default(&x->gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create empty tensor."),
                         error);
        }
        
        error = tensor_as_tensor(gradient, x->gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create add gradient."),
                         error);
        }
    }
    else
    {
        tensor_t *updated_gradient;

        error = tensor_create_default(&updated_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create updated_gradient."),
                         error);
        }
        error = tensor_addition(x->gradient, gradient, updated_gradient);
        if (error != NULL)
        {
            return ERROR(ERROR_ADDITION,
                         string_create("failed to add gradient."),
                         error);
        }
        tensor_destroy(x->gradient);
        x->gradient = updated_gradient;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("x->gradient", x->gradient);
    PRINTLN_DEBUG_TENSOR("gradient", gradient);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

