#include <view.h>
#include <datatype.h>
#include <check.h>
#include <view.h>
#include <runtime.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <layer.h>
#include <test_helper.h>

void ck_assert_model_eq(const model_t *returned, const model_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);

    ck_assert_block_eq(returned->block, expected->block);
}

void ck_assert_block_eq(const block_t *returned, const block_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_int_eq(returned->depth, expected->depth);
    for (int64_t i = 0; i < expected->depth; ++i)
    {
        ck_assert_layer_eq(returned->layers[i], expected->layers[i]);
    }
}

void ck_assert_layer_eq(const layer_t *returned, const layer_t *expected)
{
    ck_assert_int_eq(returned->transform_type, expected->transform_type);
    ck_assert_ptr_nonnull(returned->transform);
    ck_assert_ptr_nonnull(expected->transform);

    switch (expected->transform_type)
    {
    case LINEAR:
        ck_assert_linear_eq(returned->transform->linear, expected->transform->linear);
        break;
    case CONVOLUTION_2D:
    case CONVOLUTION_TRANSPOSE_2D:
        ck_assert_convolution_2d_eq(returned->transform->convolution_2d, expected->transform->convolution_2d);
        break;
    case DROPOUT:
        ck_assert_dropout_eq(returned->transform->dropout, expected->transform->dropout);
        break;
    case BATCH_NORMALIZATION_2D:
        ck_assert_batch_normalization_2d_eq(returned->transform->batch_normalization_2d, expected->transform->batch_normalization_2d);
        break;
    case RESHAPE:
        ck_assert_reshape_eq(returned->transform->reshape, expected->transform->reshape);
        break;
    case LAYER_NORMALIZATION:
        ck_assert_layer_normalization_eq(returned->transform->layer_normalization, expected->transform->layer_normalization);
        break;
    case EMBEDDING:
        ck_assert_embedding_eq(returned->transform->embedding, expected->transform->embedding);
        break;
    case TRANSFORMER_EMBEDDING:
        ck_assert_transformer_embedding_eq(returned->transform->transformer_embedding, expected->transform->transformer_embedding);
        break;
    case CAUSAL_MULTIHEAD_SELF_ATTENTION:
        ck_assert_causal_multihead_self_attention_eq(returned->transform->causal_multihead_self_attention, expected->transform->causal_multihead_self_attention);
        break;
    case ACTIVATION:
        ck_assert_activation_eq(returned->transform->activation, expected->transform->activation);
        break;
    case RESIDUAL_BLOCK:
    case BLOCK:
        ck_assert_block_eq(returned->transform->block, expected->transform->block);
        break;
    default:
        ck_abort_msg("unknown transform type.");
        break;
    }
}

void ck_assert_linear_eq(linear_t *returned, linear_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_tensor_equiv(returned->weights, expected->weights);
    ck_assert_tensor_equiv(returned->bias, expected->bias);
}

void ck_assert_convolution_2d_eq(convolution_2d_t *returned, convolution_2d_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_tensor_equiv(returned->kernel, expected->kernel);
    ck_assert_tensor_equiv(returned->bias, expected->bias);
    ck_assert_int_eq(returned->padding, expected->padding);
    ck_assert_int_eq(returned->stride, expected->stride);
}

void ck_assert_dropout_eq(const dropout_t *returned, const dropout_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_int_eq(returned->datatype, expected->datatype);
    ck_assert_element_eq(returned->probability, 0, expected->probability, 0, expected->datatype, NULL);
    ck_assert(returned->inference == expected->inference);
}

void ck_assert_batch_normalization_2d_eq(batch_normalization_2d_t *returned, batch_normalization_2d_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_int_eq(returned->datatype, expected->datatype);
    ck_assert_tensor_equiv(returned->weights, expected->weights);
    ck_assert_tensor_equiv(returned->bias, expected->bias);
    ck_assert_tensor_equiv(returned->running_mean, expected->running_mean);
    ck_assert_tensor_equiv(returned->running_variance, expected->running_variance);
    ck_assert(returned->inference == expected->inference);
    ck_assert(returned->track_running_stats == expected->track_running_stats);
    ck_assert_element_eq(returned->epsilon, 0, expected->epsilon, 0, expected->datatype, NULL);
    ck_assert_element_eq(returned->momentum, 0, expected->momentum, 0, expected->datatype, NULL);
}

void ck_assert_reshape_eq(const reshape_t *returned, const reshape_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);

    ck_assert_int_eq(returned->length, expected->length);
    for (int64_t i = 0; i < expected->length; ++i)
    {
        ck_assert_int_eq(returned->shape[i], expected->shape[i]);
    }
}

void ck_assert_layer_normalization_eq(layer_normalization_t *returned, layer_normalization_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_int_eq(returned->datatype, expected->datatype);
    ck_assert_element_eq(returned->epsilon, 0, expected->epsilon, 0, expected->datatype, NULL);
    ck_assert_tensor_equiv(returned->weights, expected->weights);
    ck_assert_tensor_equiv(returned->bias, expected->bias);
    ck_assert_int_eq(returned->length, expected->length);
    for (int64_t i = 0; i < expected->length; ++i)
    {
        ck_assert_int_eq(returned->normalized_shape[i], expected->normalized_shape[i]);
    }
}

void ck_assert_embedding_eq(embedding_t *returned, embedding_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_tensor_equiv(returned->weights, expected->weights);
    ck_assert_tensor_equiv(returned->vocabulary_counter, expected->vocabulary_counter);
    ck_assert_int_eq(returned->embedding_size, expected->embedding_size);
    ck_assert_int_eq(returned->vocabulary_size, expected->vocabulary_size);
}

void ck_assert_transformer_embedding_eq(const transformer_embedding_t *returned, const transformer_embedding_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_embedding_eq(returned->position_embedding, expected->position_embedding);
    ck_assert_embedding_eq(returned->token_embedding, expected->token_embedding);
}

void ck_assert_causal_multihead_self_attention_eq(causal_multihead_self_attention_t *returned, causal_multihead_self_attention_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_tensor_equiv(returned->input_weights, expected->input_weights);
    ck_assert_tensor_equiv(returned->input_bias, expected->input_bias);
    ck_assert_tensor_equiv(returned->output_weights, expected->output_weights);
    ck_assert_tensor_equiv(returned->output_bias, expected->output_bias);
    ck_assert_int_eq(returned->embedding_size, expected->embedding_size);
    ck_assert_int_eq(returned->number_of_heads, expected->number_of_heads);
    ck_assert(returned->inference == expected->inference);
    ck_assert_int_eq(returned->datatype, expected->datatype);
    ck_assert_element_eq(returned->dropout_probability, 0, expected->dropout_probability, 0, expected->datatype, NULL);
}

void ck_assert_activation_eq(const activation_t *returned, const activation_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_int_eq(returned->activation_function_type, expected->activation_function_type);
    switch (expected->activation_function_type)
    {
    case ACTIVATION_RECTIFIED_LINEAR:
    case ACTIVATION_SIGMOID:
    case ACTIVATION_TANH:
    case ACTIVATION_GELU:
        break;
    case ACTIVATION_SOFTMAX:
    case ACTIVATION_LOGSOFTMAX:
        ck_assert_softmax_eq(returned->activation_function->softmax, expected->activation_function->softmax);
        break;
    case ACTIVATION_LEAKY_RECTIFIED_LINEAR:
        ck_assert_leaky_rectified_linear_eq(returned->activation_function->leaky_rectified_linear, expected->activation_function->leaky_rectified_linear);
        break;
    default:
        ck_abort_msg("unknown activaton function type");
        break;
    }
}

void ck_assert_softmax_eq(const softmax_t *returned, const softmax_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_int_eq(returned->axis, expected->axis);
}

void ck_assert_leaky_rectified_linear_eq(const leaky_rectified_linear_t *returned, const leaky_rectified_linear_t *expected)
{
    ck_assert_ptr_nonnull(returned);
    ck_assert_ptr_nonnull(expected);
    ck_assert_int_eq(returned->datatype, expected->datatype);
    ck_assert_element_eq(returned->c, 0, expected->c, 0, expected->datatype, NULL);
}

static inline float32_t get_epsilon_float(float32_t a, float32_t b, float32_t abs_epsilon_f)
{
    static float32_t epsilon_f = 128 * FLT_EPSILON;
    return MAX(abs_epsilon_f, epsilon_f * MIN((ABS(a) + ABS(b)), FLT_MAX));
}

static inline float64_t get_epsilon_double(float64_t a, float64_t b, float64_t abs_epsilon)
{
    static float64_t epsilon = 1e2 * FLT_EPSILON;
    return MAX(abs_epsilon, epsilon * MIN((ABS(a) + ABS(b)), FLT_MAX));
}

void ck_assert_element_eq(const void *returned_data, int64_t returned_index,
                          const void *expected_data, int64_t expected_index,
                          datatype_t datatype, void *epsilon)
{
    ck_assert_ptr_nonnull(returned_data);
    ck_assert_ptr_nonnull(expected_data);

    switch (datatype)
    {
    case FLOAT32:
        if (isnanf(((float32_t *) expected_data)[expected_index]))
        {
            ck_assert_float_nan(((float32_t *) returned_data)[returned_index]);
        }
        else
        {
            ck_assert_float_eq_tol(((float32_t *) returned_data)[returned_index],
                                   ((float32_t *) expected_data)[expected_index],
                                   (!epsilon) ? get_epsilon_float(((float32_t *) returned_data)[returned_index],
                                                                  ((float32_t *) expected_data)[expected_index], 1e-5) : 
                                                get_epsilon_float(((float32_t *) returned_data)[returned_index],
                                                                  ((float32_t *) expected_data)[expected_index], *(float32_t *) epsilon));
        }
        break;
    case FLOAT64:
        if (isnanf(((float64_t *) expected_data)[expected_index]))
        {
            ck_assert_double_nan(((float64_t *) returned_data)[returned_index]);
        }
        else
        {
            ck_assert_double_eq_tol(((float64_t *) returned_data)[returned_index],
                                    ((float64_t *) expected_data)[expected_index],
                                    (!epsilon) ? get_epsilon_double(((float64_t *) returned_data)[returned_index],
                                                                    ((float64_t *) expected_data)[expected_index], 1e-5) : 
                                                 get_epsilon_double(((float64_t *) returned_data)[returned_index],
                                                                    ((float64_t *) expected_data)[expected_index], *(float64_t *) epsilon));
        }
        break;
    default:
        ck_abort_msg("unknown datatype.");
    }
}

void ck_assert_storage_eq(storage_t *returned_storage, storage_t *expected_storage, void *epsilon)
{
    nw_error_t *error;

    ck_assert_ptr_nonnull(returned_storage);
    ck_assert_ptr_nonnull(expected_storage);
    
    ck_assert_int_eq(expected_storage->n, returned_storage->n);
    ck_assert_int_eq(expected_storage->datatype, returned_storage->datatype);
    ck_assert_int_eq(expected_storage->runtime, returned_storage->runtime);

    error = storage_dev_to_cpu(returned_storage);
    ck_assert_ptr_null(error);
    error_destroy(error);

    error = storage_dev_to_cpu(expected_storage);
    ck_assert_ptr_null(error);
    error_destroy(error);

    for (int64_t i = 0; i < expected_storage->n; ++i)
    {
        ck_assert_element_eq(returned_storage->data, i, expected_storage->data, i, expected_storage->datatype, epsilon);
    }
}

void ck_assert_buffer_eq(buffer_t *returned_buffer, buffer_t *expected_buffer, void *epsilon)
{
    ck_assert_ptr_nonnull(returned_buffer);
    ck_assert_ptr_nonnull(expected_buffer);

    ck_assert_view_eq(returned_buffer->view, expected_buffer->view);
    ck_assert_storage_eq(returned_buffer->storage, expected_buffer->storage, epsilon);
}

void ck_assert_function_eq(const tensor_t *returned_tensor, 
                           const function_t *returned_function,
                           const tensor_t *expected_tensor,
                           const function_t *expected_function)
{
    ck_assert_ptr_nonnull(returned_tensor);
    ck_assert_ptr_nonnull(expected_tensor);
    ck_assert_ptr_nonnull(returned_function);
    ck_assert_ptr_nonnull(expected_function);

    ck_assert_int_eq(expected_function->operation_type,
                     returned_function->operation_type);
    switch (expected_function->operation_type)
    {
    case UNARY_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->unary_operation->x,
                            returned_function->operation->unary_operation->x);
        ck_assert_int_eq(expected_function->operation->unary_operation->operation_type,
                         returned_function->operation->unary_operation->operation_type);
        break;
    case BINARY_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->binary_operation->x,
                            returned_function->operation->binary_operation->x);
        ck_assert_tensor_eq(expected_function->operation->binary_operation->y,
                            returned_function->operation->binary_operation->y);
        ck_assert_int_eq(expected_function->operation->binary_operation->operation_type,
                         returned_function->operation->binary_operation->operation_type);
        break;
    case REDUCTION_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->reduction_operation->x,
                            returned_function->operation->reduction_operation->x);
        ck_assert_int_eq(expected_function->operation->reduction_operation->operation_type,
                         returned_function->operation->reduction_operation->operation_type);
        ck_assert_int_eq(expected_function->operation->reduction_operation->length,
                          returned_function->operation->reduction_operation->length);
        ck_assert(expected_function->operation->reduction_operation->keep_dimension == 
                  returned_function->operation->reduction_operation->keep_dimension);
        for (int64_t j = 0; j < expected_function->operation->reduction_operation->length; ++j)
        {
            ck_assert_int_eq(expected_function->operation->reduction_operation->axis[j],
                              returned_function->operation->reduction_operation->axis[j]);
        }
        break;
    case STRUCTURE_OPERATION:
        ck_assert_tensor_eq(expected_function->operation->structure_operation->x,
                            returned_function->operation->structure_operation->x);
        ck_assert_int_eq(expected_function->operation->structure_operation->operation_type,
                         returned_function->operation->structure_operation->operation_type);
        ck_assert_int_eq(expected_function->operation->structure_operation->length,
                          returned_function->operation->structure_operation->length);
        for (int64_t j = 0; j < expected_function->operation->structure_operation->length; ++j)
        {
            ck_assert_int_eq(expected_function->operation->structure_operation->arguments[j],
                              returned_function->operation->structure_operation->arguments[j]);
        }
        break;
    default:
        ck_abort_msg("unknown operation type");
    } 
}

void ck_assert_tensor_eq(tensor_t *returned_tensor, tensor_t *expected_tensor)
{
    PRINTLN_DEBUG_LOCATION("test");
    PRINTLN_DEBUG_TENSOR("returned", returned_tensor);
    PRINTLN_DEBUG_TENSOR("expected", expected_tensor);
    PRINT_DEBUG_NEWLINE;

    if (!expected_tensor)
    {
        ck_assert_ptr_null(returned_tensor);
        return;
    }

    if (!expected_tensor->buffer)
    {
        ck_assert_ptr_null(expected_tensor->buffer);
    }
    else
    {
        ck_assert_buffer_eq(returned_tensor->buffer, expected_tensor->buffer, NULL);
    }

    ck_assert(returned_tensor->requires_gradient == expected_tensor->requires_gradient);
}

void ck_assert_data_equiv(const void *returned_data, const int64_t *returned_strides, int64_t returned_offset,
                          const void *expected_data, const int64_t *expected_strides, int64_t expected_offset,
                          const int64_t *shape, int64_t rank, datatype_t datatype, void *epsilon)
{
    switch (rank)
    {
    case 0:
        ck_assert_element_eq(returned_data, returned_offset, 
                             expected_data, expected_offset, 
                             datatype, epsilon);
        break;
    case 1:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0], 
                                 expected_data, expected_offset + i * expected_strides[0],
                                 datatype, epsilon);
        }
        break;
    case 2:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            for (int64_t j = 0; j < shape[1]; ++j)
            {
                ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1], 
                                     expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1],
                                     datatype, epsilon);
            }
        }
        break;
    case 3:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            for (int64_t j = 0; j < shape[1]; ++j)
            {
                for (int64_t k = 0; k < shape[2]; ++k)
                {
                    ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1] + k * returned_strides[2], 
                                         expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1] + k * expected_strides[2],
                                         datatype, epsilon);
                }
            }
        }
        break;
    case 4:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            for (int64_t j = 0; j < shape[1]; ++j)
            {
                for (int64_t k = 0; k < shape[2]; ++k)
                {
                    for (int64_t l = 0; l < shape[3]; ++l)
                    {
                        ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1] + k * returned_strides[2] + l * returned_strides[3], 
                                             expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1] + k * expected_strides[2] + l * expected_strides[3],
                                             datatype, epsilon);
                    }
                }
            }
        }
        break;
    case 5:
        for (int64_t i = 0; i < shape[0]; ++i)
        {
            for (int64_t j = 0; j < shape[1]; ++j)
            {
                for (int64_t k = 0; k < shape[2]; ++k)
                {
                    for (int64_t l = 0; l < shape[3]; ++l)
                    {
                        for (int64_t m = 0; m < shape[4]; ++m)
                        {
                            ck_assert_element_eq(returned_data, returned_offset + i * returned_strides[0] + j * returned_strides[1] + k * returned_strides[2] + l * returned_strides[3] + m * returned_strides[4], 
                                                 expected_data, expected_offset + i * expected_strides[0] + j * expected_strides[1] + k * expected_strides[2] + l * expected_strides[3] + m * expected_strides[4],
                                                 datatype, epsilon);
                        }
                    }
                }
            }
        }
        break;
    default:
        ck_abort_msg("unsupported rank.");
    }
}

static void _ck_assert_tensor_equiv(tensor_t *returned_tensor, tensor_t *expected_tensor, void *epsilon)
{
    PRINTLN_DEBUG_LOCATION("test");
    PRINTLN_DEBUG_TENSOR("returned", returned_tensor);
    PRINTLN_DEBUG_TENSOR("expected", expected_tensor);
    PRINT_DEBUG_NEWLINE;

    nw_error_t *error;

    if (!expected_tensor)
    {
        ck_assert_ptr_null(returned_tensor);
        return;
    }

    ck_assert_ptr_nonnull(expected_tensor->buffer);
    ck_assert_ptr_nonnull(expected_tensor->buffer->view);
    ck_assert_ptr_nonnull(expected_tensor->buffer->storage);
    ck_assert_ptr_nonnull(returned_tensor->buffer);
    ck_assert_ptr_nonnull(returned_tensor->buffer->view);
    ck_assert_ptr_nonnull(returned_tensor->buffer->storage);

    ck_assert_int_eq(returned_tensor->buffer->view->rank, expected_tensor->buffer->view->rank);
    for (int64_t i = 0; i < expected_tensor->buffer->view->rank; ++i)
    {
        ck_assert_int_eq(returned_tensor->buffer->view->shape[i], 
                          expected_tensor->buffer->view->shape[i]);
    }
    ck_assert_int_eq(returned_tensor->buffer->storage->datatype, expected_tensor->buffer->storage->datatype);

    error = storage_dev_to_cpu(returned_tensor->buffer->storage);
    ck_assert_ptr_null(error);
    error_destroy(error);

    error = storage_dev_to_cpu(expected_tensor->buffer->storage);
    ck_assert_ptr_null(error);
    error_destroy(error);

    ck_assert_data_equiv(returned_tensor->buffer->storage->data, returned_tensor->buffer->view->strides, returned_tensor->buffer->view->offset,
                         expected_tensor->buffer->storage->data, expected_tensor->buffer->view->strides, expected_tensor->buffer->view->offset,
                         expected_tensor->buffer->view->shape, expected_tensor->buffer->view->rank, expected_tensor->buffer->storage->datatype, epsilon);

}

void ck_assert_tensor_equiv(tensor_t *returned_tensor, tensor_t *expected_tensor)
{
    _ck_assert_tensor_equiv(returned_tensor, expected_tensor, NULL);
}

void ck_assert_tensor_equiv_flt(tensor_t *returned_tensor, tensor_t *expected_tensor, float32_t abs_epsilon)
{
    _ck_assert_tensor_equiv(returned_tensor, expected_tensor, &abs_epsilon);
}

void ck_assert_tensor_equiv_dbl(tensor_t *returned_tensor, tensor_t *expected_tensor, float64_t abs_epsilon)
{
    _ck_assert_tensor_equiv(returned_tensor, expected_tensor, &abs_epsilon);
}

void ck_assert_view_eq(const view_t *returned_view, const view_t *expected_view)
{
    ck_assert_ptr_nonnull(returned_view);
    ck_assert_ptr_nonnull(expected_view);

    ck_assert_int_eq(expected_view->rank, returned_view->rank);
    ck_assert_int_eq(expected_view->offset, returned_view->offset);
    for (int64_t i = 0; i < expected_view->rank; ++i)
    {
        ck_assert_int_eq(expected_view->shape[i], returned_view->shape[i]);
        
        if (expected_view->shape[i] == 1)
        {
            ck_assert(returned_view->strides[i] == (int64_t) 0 || expected_view->strides[i] == returned_view->strides[i]);
        }
        else
        {
            ck_assert_int_eq(expected_view->strides[i], returned_view->strides[i]);
        }
    }
}
