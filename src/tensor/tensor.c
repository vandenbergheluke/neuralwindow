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

bool_t no_gradient = false;

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
    int64_t *x_shape = x_original->buffer->view->shape; 
    int64_t x_rank = x_original->buffer->view->rank; 
    int64_t *y_shape = y_original->buffer->view->shape; 
    int64_t y_rank = y_original->buffer->view->rank; 
    int64_t broadcasted_rank = MAX(x_rank, y_rank);
    int64_t broadcasted_shape[broadcasted_rank];

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
    int64_t *x_shape = x_original->buffer->view->shape; 
    int64_t x_rank = x_original->buffer->view->rank; 
    int64_t *y_shape = y_original->buffer->view->shape; 
    int64_t y_rank = y_original->buffer->view->rank; 
    int64_t broadcasted_rank = MAX(x_rank, y_rank);
    int64_t x_broadcasted_shape[broadcasted_rank];
    int64_t y_broadcasted_shape[broadcasted_rank];

    error = matrix_multiplication_broadcast_shapes(x_shape, x_rank, y_shape, y_rank, x_broadcasted_shape, y_broadcasted_shape, broadcasted_rank);
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

    if (shapes_equal(x->buffer->view->shape, x->buffer->view->rank, shape, length))
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

inline int64_t tensor_number_of_elements(const tensor_t *x)
{
    return (x && x->buffer && x->buffer->view) ? shape_size(x->buffer->view->shape, x->buffer->view->rank) : 0;
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

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) value = (float32_t) tensor_number_of_elements(x_i) / (float32_t) tensor_number_of_elements(x);
        break;
    case FLOAT64:
        *(float64_t *) value = (float64_t) tensor_number_of_elements(x_i) / (float64_t) tensor_number_of_elements(x);
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
    if (!x_i->requires_gradient || no_gradient)
    {
        if (x_i != x)
        {
            tensor_destroy(x_i);
        }
        tensor_destroy(x_j);
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

bool_t tensor_is_contiguous(const tensor_t *x)
{
    return x && x->buffer && x->buffer->view &&
           is_contiguous(x->buffer->view->shape, x->buffer->view->rank, 
                         x->buffer->view->strides, x->buffer->view->offset);
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
    if (shapes_equal(x->buffer->view->shape, x->buffer->view->rank, shape, length))
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
    int64_t axis[rank];
    for (int64_t i = 0; i < rank; ++i)
    {
        axis[i] = i;
    }
    int64_t temp = axis[axis2];
    axis[axis2] = axis[axis1];
    axis[axis1] = temp;

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

    if (tensor_is_contiguous(x))
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

    return error;

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

nw_error_t *tensor_create_kaiming_uniform(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime,
                                          datatype_t datatype, bool_t requires_gradient, bool_t persist, void *gain, void *fan)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");
    CHECK_NULL_ARGUMENT(fan, "fan");

    nw_error_t *error = NULL;
    void *lower_bound = NULL;
    void *upper_bound = NULL;
    size_t size = datatype_size(datatype);

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
        *(float32_t *) upper_bound = *(float32_t *) gain * sqrtf(3.0 / *(float32_t *) fan);
        *(float32_t *) lower_bound = -*(float32_t *) upper_bound;
        break;
    case FLOAT64:
        *(float64_t *) upper_bound = *(float64_t *) gain * sqrtf(3.0 / *(float64_t *) fan);
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
                                         datatype_t datatype, bool_t requires_gradient, bool_t persist, void *gain, void *fan)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");
    CHECK_NULL_ARGUMENT(fan, "fan");

    nw_error_t *error = NULL;
    void *mean = NULL;
    void *standard_deviation = NULL;
    size_t size = datatype_size(datatype);

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
        
        *(float32_t *) standard_deviation = *(float32_t *) gain / sqrtf(*(float32_t *) fan);
        *(float32_t *) mean = (float32_t) 0.0;
        break;
    case FLOAT64:
        *(float64_t *) standard_deviation = *(float64_t *) gain / sqrt(*(float64_t *) fan);
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
                                         bool_t requires_gradient, bool_t perist, void *gain, void *fan_in, void *fan_out)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");
    CHECK_NULL_ARGUMENT(fan_in, "fan_in");
    CHECK_NULL_ARGUMENT(fan_out, "fan_out");

    nw_error_t *error = NULL;
    void *lower_bound = NULL;
    void *upper_bound = NULL;
    size_t size = datatype_size(datatype);

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
        *(float32_t *) upper_bound = *(float32_t *) gain * sqrtf(6.0 / (*(float32_t *) fan_in + *(float32_t *) fan_out));
        *(float32_t *) lower_bound = -*(float32_t *) upper_bound;
        break;
    case FLOAT64:
        *(float64_t *) upper_bound =  *(float64_t *) gain * sqrt(6.0 / (*(float64_t *) fan_in + *(float64_t *) fan_out));
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
                                        bool_t requires_gradient, bool_t persist, void *gain, void *fan_in, void *fan_out)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(shape, "shape");
    CHECK_NULL_ARGUMENT(gain, "gain");
    CHECK_NULL_ARGUMENT(fan_in, "fan_in");
    CHECK_NULL_ARGUMENT(fan_out, "fan_out");

    nw_error_t *error = NULL;
    void *mean = NULL;
    void *standard_deviation = NULL;

    size_t size = datatype_size(datatype);

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
        *(float32_t *) standard_deviation = *(float32_t *) gain * sqrtf(2.0 / (*(float32_t *) fan_in + *(float32_t *) fan_out));
        *(float32_t *) mean = (float32_t) 0.0;
        break;
    case FLOAT64:
        *(float64_t *) standard_deviation = *(float64_t *) gain * sqrt(2.0 / (*(float64_t *) fan_in + *(float64_t *) fan_out));
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
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = apply_operation_unary(AS_OPERATION, x, y);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

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

