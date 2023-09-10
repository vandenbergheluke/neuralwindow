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
                     string_create("failed to allocate tensor of size %lu bytes.",
                     (unsigned long) sizeof(tensor_t)),
                     NULL);
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
    if (tensor == NULL)
    {
        return;
    }

    buffer_destroy(tensor->buffer);
    tensor_destroy(tensor->gradient);
    function_destroy(tensor->context);
    free(tensor);
}

/**
 * @brief The default tensor constructor. Dynamically allocates memory for
 *         the tensor. Sets underlying storage `buffer`, `context`, and `gradient`
 *         to NULL. Sets the `requires_gradient` and `lock` flags to false.
 * @param[out] tensor The default tensor to dynamically allocate and initialize.
 * @return Error if the default tensor failed to be allocated.
 *         NULL if the default tensor was created sucessfully.
 */
nw_error_t *tensor_create_default(tensor_t **tensor)
{
    CHECK_NULL_ARGUMENT(tensor, "tensor");

    nw_error_t *error = NULL;

    error = tensor_create(tensor, NULL, NULL, NULL, false, false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create tensor."),
                     error);
    }

    return NULL;
}

nw_error_t *tensor_broadcast(const tensor_t *x_original,
                             const tensor_t *y_original,
                             tensor_t *x_broadcasted,
                             tensor_t *y_broadcasted)
{
    CHECK_NULL_ARGUMENT(x_original, "x_original");
    CHECK_NULL_ARGUMENT(y_original, "y_original");
    CHECK_NULL_ARGUMENT(x_broadcasted, "x_broadcasted");
    CHECK_NULL_ARGUMENT(y_broadcasted, "y_broadcasted");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINTLN_DEBUG_TENSOR("x_broadcasted", x_broadcasted);
    PRINTLN_DEBUG_TENSOR("y_broadcasted", y_broadcasted);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = NULL;
    uint64_t *x_shape = x_original->buffer->view->shape; 
    uint64_t x_rank = x_original->buffer->view->rank; 
    uint64_t *y_shape = y_original->buffer->view->shape; 
    uint64_t y_rank = y_original->buffer->view->rank; 
    uint64_t broadcasted_rank = MAX(x_rank, y_rank);

    uint64_t *broadcasted_shape = (uint64_t *) malloc((size_t) (broadcasted_rank * sizeof(uint64_t)));
    if (broadcasted_shape == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate broadcast shape of %lu bytes.",
                     (unsigned long) (broadcasted_rank * sizeof(uint64_t))),
                     NULL);
    }

    error = broadcast_shapes(x_shape,
                             x_rank,
                             y_shape,
                             y_rank,
                             broadcasted_shape,
                             broadcasted_rank);
    if (error != NULL)
    {
        free(broadcasted_shape);
        return ERROR(ERROR_BROADCAST,
                     string_create("failed to broadcast tensor shapes."),
                     error);
    }

    error = tensor_expand(x_original,
                          broadcasted_shape,
                          broadcasted_rank,
                          x_broadcasted);
    if (error != NULL)
    {
        free(broadcasted_shape);
        return ERROR(ERROR_EXPAND,
                     string_create("failed to expand tensor x."),
                     error);
    }

    error = tensor_expand(y_original,
                          broadcasted_shape,
                          broadcasted_rank,
                          y_broadcasted);
    if (error != NULL)
    {
        free(broadcasted_shape);
        return ERROR(ERROR_EXPAND,
                     string_create("failed to expand tensor y."),
                     error);
    }

    free(broadcasted_shape);

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINTLN_DEBUG_TENSOR("x_broadcasted", x_broadcasted);
    PRINTLN_DEBUG_TENSOR("y_broadcasted", y_broadcasted);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_broadcast_matrix_multiplication(const tensor_t *x_original,
                                                   const tensor_t *y_original,
                                                   tensor_t *x_broadcasted,
                                                   tensor_t *y_broadcasted)
{
    CHECK_NULL_ARGUMENT(x_original, "x_original");
    CHECK_NULL_ARGUMENT(y_original, "y_original");
    CHECK_NULL_ARGUMENT(x_broadcasted, "x_broadcasted");
    CHECK_NULL_ARGUMENT(y_broadcasted, "y_broadcasted");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINTLN_DEBUG_TENSOR("x_broadcasted", x_broadcasted);
    PRINTLN_DEBUG_TENSOR("y_broadcasted", y_broadcasted);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = NULL;
    uint64_t *x_shape = x_original->buffer->view->shape; 
    uint64_t x_rank = x_original->buffer->view->rank; 
    uint64_t *y_shape = y_original->buffer->view->shape; 
    uint64_t y_rank = y_original->buffer->view->rank; 
    uint64_t broadcasted_rank = MAX(x_rank, y_rank);

    uint64_t *x_broadcasted_shape = (uint64_t *) malloc((size_t) (broadcasted_rank * sizeof(uint64_t)));
    if (x_broadcasted_shape == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate broadcast shape of %lu bytes.",
                     (unsigned long) (broadcasted_rank * sizeof(uint64_t))),
                     NULL);
    }

    uint64_t *y_broadcasted_shape = (uint64_t *) malloc((size_t) (broadcasted_rank * sizeof(uint64_t)));
    if (y_broadcasted_shape == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate broadcast shape of %lu bytes.",
                     (unsigned long) (broadcasted_rank * sizeof(uint64_t))),
                     NULL);
    }

    error = matrix_multiplication_broadcast_shapes(x_shape,
                                                   x_rank,
                                                   y_shape,
                                                   y_rank,
                                                   x_broadcasted_shape,
                                                   y_broadcasted_shape,
                                                   broadcasted_rank);
    if (error != NULL)
    {
        free(x_broadcasted_shape);
        free(y_broadcasted_shape);
        return ERROR(ERROR_BROADCAST,
                     string_create("failed to broadcast tensor shapes."),
                     error);
    }

    error = tensor_expand(x_original,
                          x_broadcasted_shape,
                          broadcasted_rank,
                          x_broadcasted);
    if (error != NULL)
    {
        free(x_broadcasted_shape);
        free(y_broadcasted_shape);
        return ERROR(ERROR_EXPAND,
                     string_create("failed to expand tensor x."),
                     error);
    }

    error = tensor_expand(y_original,
                          y_broadcasted_shape,
                          broadcasted_rank,
                          y_broadcasted);
    if (error != NULL)
    {
        free(x_broadcasted_shape);
        free(y_broadcasted_shape);
        return ERROR(ERROR_EXPAND,
                     string_create("failed to expand tensor y."),
                     error);
    }

    free(x_broadcasted_shape);
    free(y_broadcasted_shape);

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x_original", x_original);
    PRINTLN_DEBUG_TENSOR("y_original", y_original);
    PRINTLN_DEBUG_TENSOR("x_broadcasted", x_broadcasted);
    PRINTLN_DEBUG_TENSOR("y_broadcasted", y_broadcasted);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_sigmoid(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(SIGMOID_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply sigmoid to tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_expand(const tensor_t *x,
                          const uint64_t *shape,
                          uint64_t length,
                          tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_UINT64_ARRAY("shape", shape, length);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_structure(EXPAND_OPERATION, x, shape, length, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to expand tensor x."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_UINT64_ARRAY("shape", shape, length);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_addition(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_binary(ADDITION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to add tensors."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_subtraction(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_binary(SUBTRACTION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to subtract tensors."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_division(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_binary(DIVISION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to divide tensors."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;
    
    return NULL;
}

nw_error_t *tensor_multiplication(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_binary(MULTIPLICATION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to multiply tensors."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_power(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_binary(POWER_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply power to tensors."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_matrix_multiplication(const tensor_t *x, const tensor_t *y, tensor_t *z)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(z, "z");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_binary(MATRIX_MULTIPLICATION_OPERATION, x, y, z);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to matrix multiply tensors."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_TENSOR("z", z);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_summation(const tensor_t *x,
                             tensor_t *y,
                             const uint64_t *axis,
                             uint64_t length,
                             bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = NULL;
    
    error = apply_function_reduction(SUMMATION_OPERATION,
                                     x,
                                     axis,
                                     length,
                                     keep_dimension,
                                     y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to reduce tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_maximum(const tensor_t *x,
                             tensor_t *y,
                             const uint64_t *axis,
                             uint64_t length,
                             bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = NULL;

    error = apply_function_reduction(MAXIMUM_OPERATION,
                                     x,
                                     axis,
                                     length,
                                     keep_dimension,
                                     y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to reduce tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_mean(const tensor_t *x,
                        tensor_t *y,
                        const uint64_t *axis,
                        uint64_t length,
                        bool_t keep_dimension)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->view, "x->buffer->view");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINTLN_DEBUG_BOOLEAN("keep_dimension", keep_dimension);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = NULL;
    tensor_t *x_i;
    tensor_t *x_j;

    error = tensor_create_default(&x_i);
    if (error != NULL)
    {
        ERROR(ERROR_CREATE,
              string_create("failed to create tensor x_i."),
              error);
    }

    error = tensor_create_default(&x_j);
    if (error != NULL)
    {
        tensor_destroy(x_i);
        ERROR(ERROR_CREATE,
              string_create("failed to create tensor x_j."),
              error);
    }

    error = tensor_summation(x, x_i, axis, length, keep_dimension);
    if (error != NULL)
    {
        tensor_destroy(x_i);
        tensor_destroy(x_j);
        ERROR(ERROR_SUMMATION,
              string_create("failed to sum tensor."),
              error);
    }

    switch (x->buffer->storage->datatype)
    {
    case FLOAT32:
        float32_t constant_32 = ((float32_t) shape_size(x_i->buffer->view->shape, x_i->buffer->view->rank) / 
                                 (float32_t) shape_size(x->buffer->view->shape, x->buffer->view->rank)) ;
        error = tensor_constant(&constant_32, 
                                x->buffer->storage->datatype,
                                x->buffer->storage->runtime,
                                x_j);
        if (error != NULL)
        {
            tensor_destroy(x_i);
            tensor_destroy(x_j);
            return ERROR(ERROR_INITIALIZATION,
                         string_create("failed to initialize constant tensor."),
                         error);
        }
        break;
    case FLOAT64:
        float64_t constant_64 = ((float64_t) shape_size(x_i->buffer->view->shape, x_i->buffer->view->rank) / 
                                 (float64_t) shape_size(x->buffer->view->shape, x->buffer->view->rank));
        error = tensor_constant(&constant_64,
                                x->buffer->storage->datatype,
                                x->buffer->storage->runtime,
                                x_j);
        if (error != NULL)
        {
            tensor_destroy(x_i);
            tensor_destroy(x_j);
            return ERROR(ERROR_INITIALIZATION,
                         string_create("failed to initialize constant tensor."),
                         error);
        }
        break;
    default:
        tensor_destroy(x_i);
        tensor_destroy(x_j);
        return ERROR(ERROR_DATATYPE,
                     string_create("unknown datatype %d",
                     (int) x->buffer->storage->datatype),
                     NULL);
    }

    error = tensor_multiplication(x_j, x_i, y);
    if (error != NULL)
    {
        tensor_destroy(x_i);
        tensor_destroy(x_j);
        return ERROR(ERROR_MULTIPLICATION,
                     string_create("failed to multiply tensor."),
                     error);
    }

    if (!x->requires_gradient)
    {
        tensor_destroy(x_i);
        tensor_destroy(x_j);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

bool_t tensor_is_contiguous(const tensor_t *x)
{
    if (x == NULL || x->buffer == NULL || x->buffer->view == NULL)
    {
        return false;
    }
    return is_contiguous(x->buffer->view->shape,
                         x->buffer->view->rank,
                         x->buffer->view->strides,
                         x->buffer->view->offset);
}

nw_error_t *tensor_reshape(const tensor_t *x,
                           tensor_t *y,
                           const uint64_t *shape,
                           uint64_t length)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(shape, "shape");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_UINT64_ARRAY("shape", shape, length);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error;

    if (!tensor_is_contiguous(x))
    {
        tensor_t *x_contiguous = NULL;

        error = tensor_create_default(&x_contiguous);
        if (error != NULL)
        {
            return ERROR(ERROR_CREATE,
                         string_create("failed to create x_contiguous tensor."),
                         error);
        }

        error = tensor_contiguous(x, x_contiguous);
        if (error != NULL)
        {
            return ERROR(ERROR_CONTIGUOUS,
                         string_create("failed to apply contiguous operation to tensor."),
                         error);
        }

        error = apply_function_structure(RESHAPE_OPERATION,
                                         x_contiguous,
                                         shape,
                                         length,
                                         y);
        if (error != NULL)
        {
            return ERROR(ERROR_FORWARD,
                         string_create("failed to reshape tensor."),
                         error);
        }
    }
    else
    {
        error = apply_function_structure(RESHAPE_OPERATION, x, shape, length, y);
        if (error != NULL)
        {
            return ERROR(ERROR_FORWARD,
                         string_create("failed to reshape tensor."),
                         error);
        }
    }
    
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_permute(const tensor_t *x,
                           tensor_t *y,
                           uint64_t *axis,
                           uint64_t length)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(axis, "axis");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_UINT64_ARRAY("axis", axis, length);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_structure(PERMUTE_OPERATION, x, axis, length, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to permute tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_slice(const tensor_t *x,
                         tensor_t *y,
                         uint64_t *arguments,
                         uint64_t length)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_UINT64_ARRAY("arguments", arguments, length);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_structure(SLICE_OPERATION, x, arguments, length, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to slice tensor."),
                     error);
    }
    
    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_padding(const tensor_t *x,
                           tensor_t *y,
                           uint64_t *arguments,
                           uint64_t length)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(arguments, "arguments");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINTLN_DEBUG_UINT64_ARRAY("arguments", arguments, length);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_structure(PADDING_OPERATION, x, arguments, length, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to pad tensor."),
                     error);
    }
    
    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_contiguous(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(CONTIGUOUS_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to permute tensor."),
                     error);
    }
    
    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_logarithm(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(LOGARITHM_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply logarithm to tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_sine(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(SINE_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply logarithm to tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_cosine(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(COSINE_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply logarithm to tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_exponential(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(EXPONENTIAL_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply exp to tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_square_root(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(SQUARE_ROOT_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply squart root to tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_reciprocal(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(RECIPROCAL_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply reciprocal to tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_negation(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(NEGATION_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply negation to tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_rectified_linear(const tensor_t *x, tensor_t *y)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error = apply_function_unary(RECTIFIED_LINEAR_OPERATION, x, y);
    if (error != NULL)
    {
        return ERROR(ERROR_FORWARD,
                     string_create("failed to apply rectified linear to tensor."),
                     error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("y", y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *tensor_constant(void *constant,
                            datatype_t datatype,
                            runtime_t runtime,
                            tensor_t *x)
{
    CHECK_NULL_ARGUMENT(constant, "constant");
    CHECK_NULL_ARGUMENT(x, "x");

    view_t *view = NULL;
    storage_t *storage = NULL;
    buffer_t *buffer = NULL;
    nw_error_t *error = NULL;

    if (!tensor_is_empty(x))
    {
        return ERROR(ERROR_CREATE, 
                     string_create("tensor x is not empty."),
                     NULL);
    }

    error = view_create(&view, 0, 0, (uint64_t[]){}, NULL);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create view."),
                     error);
    }

    error = storage_create(&storage, runtime, datatype, 1, constant);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create storage."),
                     error);
    }

    error = buffer_create(&buffer, view, storage, false);
    if (error != NULL)
    {
        return ERROR(ERROR_CREATE,
                     string_create("failed to create buffer."),
                     error);
    }

    x->buffer = buffer;

    return NULL;
}

static nw_error_t *topological_sort(tensor_t *tensor,
                                    map_t *visited,
                                    stack_t *tensors)
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
            error = topological_sort(tensor->context->operation->unary_operation->x,
                                     visited,
                                     tensors);
            break;
        case BINARY_OPERATION:
            error = topological_sort(tensor->context->operation->binary_operation->x,
                                     visited,
                                     tensors);
            error = topological_sort(tensor->context->operation->binary_operation->y,
                                     visited,
                                     tensors);
            break;
        case REDUCTION_OPERATION:
            error = topological_sort(tensor->context->operation->reduction_operation->x,
                                     visited,
                                     tensors);
            break;
        case STRUCTURE_OPERATION:
            error = topological_sort(tensor->context->operation->structure_operation->x,
                                     visited,
                                     tensors);
            break;
        default:
            error = ERROR(ERROR_UKNOWN_OPERATION_TYPE,
                          string_create("unknown operation type %d.",
                          (int) tensor->context->operation_type),
                          NULL);
        }

        if (error != NULL)
        {
            return ERROR(ERROR_SQUARE_ROOT,
                         string_create("failed to topologically sort computational graph."),
                         error);
        }
    }

    error = map_set(visited, id, NULL);
    if (error != NULL)
    {
        string_destroy(id);
        return ERROR(ERROR_ADDITION,
                     string_create("failed add tensor to map."),
                     error);
    }

    error = stack_push(tensors, tensor);
    if (error != NULL)
    {
        return ERROR(ERROR_ADDITION,
                     string_create("failed to push tensor."),
                     error);
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

nw_error_t *tensor_as_ones(const tensor_t *x, tensor_t *y)
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

nw_error_t *tensor_as_empty(const tensor_t *x, tensor_t *y)
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

nw_error_t *tensor_as_empty_contiguous(const tensor_t *x, tensor_t *y)
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

nw_error_t *init_zeroes(tensor_t *x)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(x->buffer->storage->data, "x->buffer->storage->data");

    for (uint64_t i = 0; i < x->buffer->storage->n; ++i)
    {
        switch (x->buffer->storage->datatype)
        {
        case FLOAT32:
            ((float32_t *) x->buffer->storage->data)[i] = (float32_t) 0.0;
            break;
        case FLOAT64:
            ((float64_t *) x->buffer->storage->data)[i] = (float64_t) 0.0;
            break;
        default:
            return ERROR(ERROR_DATATYPE,
                         string_create("unknown datatype %d.",
                         (int) x->buffer->storage->datatype),
                         NULL);
        }
    }


    return NULL;
}

nw_error_t *init_ones(tensor_t *x)
{
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(x->buffer, "x->buffer");
    CHECK_NULL_ARGUMENT(x->buffer->storage, "x->buffer->storage");
    CHECK_NULL_ARGUMENT(x->buffer->storage->data, "x->buffer->storage->data");

    for (uint64_t i = 0; i < x->buffer->storage->n; ++i)
    {
        switch (x->buffer->storage->datatype)
        {
        case FLOAT32:
            ((float32_t *) x->buffer->storage->data)[i] = (float32_t) 1.0;
            break;
        case FLOAT64:
            ((float64_t *) x->buffer->storage->data)[i] = (float64_t) 1.0;
            break;
        default:
            return ERROR(ERROR_DATATYPE,
                         string_create("unknown datatype %d.",
                         (int) x->buffer->storage->datatype),
                         NULL);
        }
    }

    return NULL;
}
