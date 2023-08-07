#ifndef TENSOR_H
#define TENSOR_H

#include <errors.h>
#include <datatype.h>
#include <buffer.h>
#include <function.h>

typedef struct function_t function_t;

typedef struct tensor_t
{
    uint64_t id;
    buffer_t *buffer;
    function_t *context;
    struct tensor_t *gradient;
    bool_t requires_gradient;
    bool_t lock; // Tensor persists after computational graph is destroyed.
} tensor_t;

error_t *tensor_create(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient, bool_t lock);
void tensor_destroy(tensor_t *tensor);
error_t *tensor_create_empty(tensor_t **tensor);
error_t *tensor_copy(const tensor_t *source_tensor, tensor_t *destination_tensor);
error_t *tensor_broadcast(const tensor_t *x_original, const tensor_t *y_original, tensor_t *x_broadcasted, tensor_t *y_broadcasted);
error_t *tensor_expand(const tensor_t *x, const uint32_t *shape, uint32_t rank, tensor_t *y);
error_t *tensor_addition(const tensor_t *x, const tensor_t *y, tensor_t *z);
error_t *tensor_subtraction(const tensor_t *x, const tensor_t *y, tensor_t *z);
error_t *tensor_division(const tensor_t *x, const tensor_t *y, tensor_t *z);
error_t *tensor_multiplication(const tensor_t *x, const tensor_t *y, tensor_t *z);
error_t *tensor_power(const tensor_t *x, const tensor_t *y, tensor_t *z);
error_t *tensor_matrix_multiplication(const tensor_t *x, const tensor_t *y, tensor_t *z);
error_t *tensor_summation(const tensor_t *x, tensor_t *y, const uint32_t *axis, uint32_t rank, bool_t keep_dimension);
error_t *tensor_maximum(const tensor_t *x, tensor_t *y, const uint32_t *axis, uint32_t rank, bool_t keep_dimension);
error_t *tensor_reshape(const tensor_t *x, tensor_t *y, const uint32_t *shape, uint32_t rank);
error_t *tensor_permute(const tensor_t *x, tensor_t *y, uint32_t *axis, uint32_t rank);
error_t *tensor_slice(const tensor_t *x, tensor_t *y, uint32_t *arguments, uint32_t length);
error_t *tensor_padding(const tensor_t *x, tensor_t *y, uint32_t *arguments, uint32_t length);
error_t *tensor_contiguous(const tensor_t *x, tensor_t *y);
error_t *tensor_logarithm(const tensor_t *x, tensor_t *y);
error_t *tensor_sine(const tensor_t *x, tensor_t *y);
error_t *tensor_cosine(const tensor_t *x, tensor_t *y);
error_t *tensor_exponential(const tensor_t *x, tensor_t *y);
error_t *tensor_square_root(const tensor_t *x, tensor_t *y);
error_t *tensor_reciprocal(const tensor_t *x, tensor_t *y);
error_t *tensor_as_zeroes(const tensor_t *x, tensor_t *y);
error_t *tensor_as_ones(const tensor_t *x, tensor_t *y);
bool_t tensor_is_empty(const tensor_t *x);
error_t *tensor_as_empty(const tensor_t *x, tensor_t *y);
error_t *tensor_backward(tensor_t *x, tensor_t *gradient);
error_t *tensor_accumulate_gradient(tensor_t *x, tensor_t *gradient);
bool_t tensor_is_contiguous(const tensor_t *x);

#endif