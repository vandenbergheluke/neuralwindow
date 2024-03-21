#ifndef TEST_HELPER_H
#define TEST_HELPER_H

typedef struct tensor_t tensor_t;
typedef struct function_t function_t;
typedef struct buffer_t buffer_t;
typedef struct view_t view_t;
typedef struct storage_t storage_t;
typedef struct model_t model_t;
typedef struct block_t block_t;
typedef struct layer_t layer_t;
typedef struct linear_t linear_t;
typedef struct convolution_2d_t convolution_2d_t;
typedef struct dropout_t dropout_t;
typedef struct batch_normalization_2d_t batch_normalization_2d_t;
typedef struct reshape_t reshape_t;
typedef struct layer_normalization_t layer_normalization_t;
typedef struct embedding_t embedding_t;
typedef struct transformer_embedding_t transformer_embedding_t;
typedef struct causal_multihead_self_attention_t causal_multihead_self_attention_t;
typedef struct activation_t activation_t;
typedef struct softmax_t softmax_t;
typedef struct leaky_rectified_linear_t leaky_rectified_linear_t;
typedef enum runtime_t runtime_t;
typedef enum datatype_t datatype_t;

void ck_assert_tensor_eq(const tensor_t *returned_tensor, const tensor_t *expected_tensor);
void ck_assert_tensor_equiv(const tensor_t *returned_tensor, const tensor_t *expected_tensor);
void ck_assert_view_eq(const view_t *returned_view, const view_t *expected_view);
void ck_assert_tensor_equiv_flt(const tensor_t *returned_tensor, const tensor_t *expected_tensor, float32_t abs_epsilon);
void ck_assert_tensor_equiv_dbl(const tensor_t *returned_tensor, const tensor_t *expected_tensor, float64_t abs_epsilon);
void ck_assert_model_eq(const model_t *returned, const model_t *expected);
void ck_assert_block_eq(const block_t *returned, const block_t *expected);
void ck_assert_layer_eq(const layer_t *returned, const layer_t *expected);
void ck_assert_linear_eq(const linear_t *returned, const linear_t *expected);
void ck_assert_convolution_2d_eq(const convolution_2d_t *returned, const convolution_2d_t *expected);
void ck_assert_dropout_eq(const dropout_t *returned, const dropout_t *expected);
void ck_assert_batch_normalization_2d_eq(const batch_normalization_2d_t *returned, const batch_normalization_2d_t *expected);
void ck_assert_reshape_eq(const reshape_t *returned, const reshape_t *expected);
void ck_assert_layer_normalization_eq(const layer_normalization_t *returned, const layer_normalization_t *expected);
void ck_assert_embedding_eq(const embedding_t *returned, const embedding_t *expected);
void ck_assert_transformer_embedding_eq(const transformer_embedding_t *returned, const transformer_embedding_t *expected);
void ck_assert_causal_multihead_self_attention_eq(const causal_multihead_self_attention_t *returned, const causal_multihead_self_attention_t *expected);
void ck_assert_activation_eq(const activation_t *returned, const activation_t *expected);
void ck_assert_softmax_eq(const softmax_t *returned, const softmax_t *expected);
void ck_assert_leaky_rectified_linear_eq(const leaky_rectified_linear_t *returned, const leaky_rectified_linear_t *expected);
void ck_assert_element_eq(const void *returned_data, int64_t returned_index,
                          const void *expected_data, int64_t expected_index,
                          datatype_t datatype, void *epsilon);

#endif