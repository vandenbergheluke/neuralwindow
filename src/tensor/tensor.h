/**@file tensor.h
 * @brief
 *
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <errors.h>
#include <datatype.h>

// Forward declarations
typedef struct function_t function_t;

typedef struct buffer_t buffer_t;
typedef enum runtime_t runtime_t;

typedef struct tensor_t
{
    uint64_t id;
    buffer_t *buffer;
    function_t *context;
    struct tensor_t *gradient;
    bool_t requires_gradient;
} tensor_t;

// typedef struct tensor_options_t
// {
//     uint64_t *shape;
//     uint64_t rank;
//     datatype_t datatype;
//     runtime_t runtime;
//     bool_t requires_gradient;
//     bool_t lock;
// } tensor_options_t;

// Creation Operations
nw_error_t *tensor_create(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient);
void tensor_destroy(tensor_t *tensor);
nw_error_t *tensor_constant(void *constant, datatype_t datatype, runtime_t runtime, tensor_t **x);
nw_error_t *tensor_constant_float64(float64_t constant, tensor_t **x, runtime_t runtime);
nw_error_t *tensor_constant_float32(float32_t constant, tensor_t **x, runtime_t runtime);
nw_error_t *tensor_zeroes_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t preserve_memory_format);
nw_error_t *tensor_ones_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t preserve_memory_format);
nw_error_t *tensor_empty_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t preserve_memory_format);
nw_error_t *tensor_as_tensor(const tensor_t *x, tensor_t **y, bool_t requires_gradient);

// Structure Operations
nw_error_t *tensor_broadcast(const tensor_t *x_original, const tensor_t *y_original, tensor_t **x_broadcasted, tensor_t **y_broadcasted);
nw_error_t *tensor_broadcast_matrix_multiplication(const tensor_t *x_original, const tensor_t *y_original, tensor_t **x_broadcasted, tensor_t **y_broadcasted);
nw_error_t *tensor_expand(const tensor_t *x, const uint64_t *shape, uint64_t length, tensor_t **y);
nw_error_t *tensor_reshape(const tensor_t *x, tensor_t **y, const uint64_t *shape, uint64_t length);
nw_error_t *tensor_permute(const tensor_t *x, tensor_t **y, uint64_t *axis, uint64_t length);
nw_error_t *tensor_slice(const tensor_t *x, tensor_t **y, uint64_t *arguments, uint64_t length);
nw_error_t *tensor_padding(const tensor_t *x, tensor_t **y, uint64_t *arguments, uint64_t length);
bool_t tensor_is_contiguous(const tensor_t *x);
uint64_t tensor_number_of_elements(const tensor_t *x);

// Binary Operations
nw_error_t *tensor_addition(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_subtraction(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_division(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_multiplication(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_power(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_matrix_multiplication(const tensor_t *x, const tensor_t *y, tensor_t **z);

// Reduction Operations
nw_error_t *tensor_summation(const tensor_t *x, tensor_t **y, const uint64_t *axis, uint64_t length, bool_t keep_dimension);
nw_error_t *tensor_maximum(const tensor_t *x, tensor_t **y, const uint64_t *axis, uint64_t length, bool_t keep_dimension);
nw_error_t *tensor_mean(const tensor_t *x, tensor_t **y, const uint64_t *axis, uint64_t length, bool_t keep_dimension);

// Unary Operations
nw_error_t *tensor_contiguous(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_logarithm(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_sine(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_cosine(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_exponential(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_square_root(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_reciprocal(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_negation(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_rectified_linear(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_sigmoid(const tensor_t *x, tensor_t **y);

// Back Propogation
nw_error_t *tensor_backward(tensor_t *x, tensor_t *gradient);
nw_error_t *tensor_accumulate_gradient(tensor_t *x, tensor_t *gradient);
#endif
