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

// Data Structure
typedef struct tensor_t
{
    uint64_t id;
    buffer_t *buffer;
    function_t *context;
    struct tensor_t *gradient;
    bool_t requires_gradient;
    bool_t persist;
} tensor_t;

// Constructor
nw_error_t *tensor_create(tensor_t **tensor, buffer_t *buffer, function_t *context, tensor_t *gradient, bool_t requires_gradient, bool_t persist);
nw_error_t *tensor_create_null(tensor_t **tensor);

// Destructor
void tensor_destroy(tensor_t *tensor);

// Creation Operations
nw_error_t *tensor_constant(void *constant, datatype_t datatype, runtime_t runtime, bool_t requires_gradient, bool_t persist, tensor_t **x);
nw_error_t *tensor_zeroes_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t persist);
nw_error_t *tensor_ones_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t persist);
nw_error_t *tensor_empty_like(const tensor_t *x, tensor_t **y, bool_t requires_gradient, bool_t persist);
nw_error_t *tensor_create_empty(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist);
nw_error_t *tensor_from_data(tensor_t **x, void *data, runtime_t runtime, datatype_t datatype, int64_t rank, const int64_t *shape,  bool_t copy, bool_t requires_gradient, bool_t persist);
nw_error_t *tensor_arange(tensor_t **x, void *start, void *stop, void *step, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist);
nw_error_t *tensor_create_zeroes(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist);
nw_error_t *tensor_create_ones(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist);
nw_error_t *tensor_create_uniform(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist, void *lower_bound, void *upper_bound);
nw_error_t *tensor_create_normal(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist, void *mean, void *standard_deviation);
nw_error_t *tensor_create_kaiming_uniform(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist, void *gain, bool_t mode);
nw_error_t *tensor_create_kaiming_normal(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist, void *gain, bool_t mode);
nw_error_t *tensor_create_glorot_uniform(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist, void *gain);
nw_error_t *tensor_create_glorot_normal(tensor_t **x, const int64_t *shape, int64_t rank, runtime_t runtime, datatype_t datatype, bool_t requires_gradient, bool_t persist, void *gain);
nw_error_t *tensor_item(const tensor_t *x, void *value);
nw_error_t *tensor_multinomial_sample(tensor_t *probabilities, void *sample);

// Structure Operations
nw_error_t *tensor_broadcast(const tensor_t *x_original, const tensor_t *y_original, tensor_t **x_broadcasted, tensor_t **y_broadcasted);
nw_error_t *tensor_broadcast_matrix_multiplication(const tensor_t *x_original, const tensor_t *y_original, tensor_t **x_broadcasted, tensor_t **y_broadcasted);
nw_error_t *tensor_expand(const tensor_t *x, const int64_t *shape, int64_t length, tensor_t **y);
nw_error_t *tensor_reshape(const tensor_t *x, tensor_t **y, const int64_t *shape, int64_t length);
nw_error_t *tensor_permute(const tensor_t *x, tensor_t **y, int64_t *axis, int64_t length);
nw_error_t *tensor_slice(const tensor_t *x, tensor_t **y, int64_t *arguments, int64_t length);
nw_error_t *tensor_padding(const tensor_t *x, tensor_t **y, int64_t *arguments, int64_t length);
nw_error_t *tensor_image_to_column(const tensor_t *x, tensor_t **y, int64_t kernel_size, int64_t stride, int64_t padding, int64_t channels, int64_t height, int64_t width);
nw_error_t *tensor_column_to_image(const tensor_t *x, tensor_t **y, int64_t kernel_size, int64_t stride, int64_t padding, int64_t channels, int64_t height, int64_t width);
nw_error_t *tensor_is_contiguous(const tensor_t *x, bool_t *is_contiguous);
nw_error_t *tensor_number_of_elements(const tensor_t *x, int64_t *n);
nw_error_t *tensor_transpose(const tensor_t *x, tensor_t **y, int64_t axis1, int64_t axis2);
bool_t tensor_shapes_equal(const tensor_t *x, const tensor_t *y);

// Binary Operations
nw_error_t *tensor_addition(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_subtraction(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_division(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_multiplication(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_power(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_matrix_multiplication(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_compare_equal(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_compare_greater(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_max(const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_concatenation(const tensor_t *x, const tensor_t *y, tensor_t **z, int64_t axis);

// Ternary Operations
nw_error_t *tensor_convolution_2d(const tensor_t *w, const tensor_t *x, const tensor_t *y, tensor_t **z, int64_t stride, int64_t padding);
nw_error_t *tensor_convolution_transpose_2d(const tensor_t *w, const tensor_t *x, const tensor_t *y, tensor_t **z, int64_t stride, int64_t padding);
nw_error_t *tensor_linear(const tensor_t *w, const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_batch_normalization_2d(const tensor_t *x, const tensor_t *weights, const tensor_t *bias, tensor_t *running_mean, 
                                          tensor_t *running_variance, tensor_t **y, bool_t inference, void *momentum, void *epsilon);
nw_error_t *tensor_layer_normalization(const tensor_t *x, const tensor_t *weights, const tensor_t *bias, tensor_t **y, int64_t *normalized_shape, int64_t length, void *epsilon);
nw_error_t *tensor_causal_multihead_self_attention(tensor_t *x, const tensor_t *input_weights, const tensor_t *input_bias, const tensor_t *output_weights, const tensor_t *output_bias,
                                                   int64_t number_of_heads, void *dropout_probability, bool_t inference, tensor_t **y);
nw_error_t *tensor_scaled_dot_product_attention(const tensor_t *query, const tensor_t *key, const tensor_t *value, tensor_t **y, void *dropout_probability, bool_t inference);
nw_error_t *tensor_where(const tensor_t *w, const tensor_t *x, const tensor_t *y, tensor_t **z);
nw_error_t *tensor_embedding(const tensor_t *x, const tensor_t *weights, const tensor_t *vocabulary_counter, tensor_t **z);

// Reduction Operations
nw_error_t *tensor_summation(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension);
nw_error_t *tensor_maximum(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension);
nw_error_t *tensor_mean(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension);
nw_error_t *tensor_softmax(const tensor_t *x, tensor_t **y, int64_t axis);
nw_error_t *tensor_logsoftmax(const tensor_t *x, tensor_t **y, int64_t axis);
nw_error_t *tensor_argument_maximum(const tensor_t *x, tensor_t **y, int64_t axis, bool_t keep_dimension);
nw_error_t *tensor_variance(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension, bool_t unbiased);
nw_error_t *tensor_standard_deviation(const tensor_t *x, tensor_t **y, const int64_t *axis, int64_t length, bool_t keep_dimension, bool_t unbiased);
nw_error_t *tensor_magnitude(const tensor_t *x, tensor_t **y);

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
nw_error_t *tensor_leaky_rectified_linear(const tensor_t *x, void *c, tensor_t **y);
nw_error_t *tensor_dropout(const tensor_t *x, tensor_t **y, void *probability, bool_t inference);
nw_error_t *tensor_sigmoid(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_tanh(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_gelu(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_absolute(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_as_tensor(const tensor_t *x, tensor_t **y);
nw_error_t *tensor_lower_triangular(const tensor_t *x, tensor_t **y);

// Back Propogation
nw_error_t *tensor_backward(tensor_t *x, tensor_t *gradient);
nw_error_t *tensor_accumulate_gradient(tensor_t *x, tensor_t *gradient);
void with_no_gradient(bool_t flag);
#endif
