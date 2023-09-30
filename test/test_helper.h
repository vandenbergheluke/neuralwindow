#ifndef TEST_HELPER
#define TEST_HELPER

#include <torch/torch.h>

typedef struct tensor_t tensor_t;
typedef struct function_t function_t;
typedef struct buffer_t buffer_t;
typedef struct view_t view_t;

#define SEED 1234

tensor_t *torch_to_tensor(torch::Tensor torch_tensor, runtime_t runtime, datatype_t datatype);
void ck_assert_view_eq(const view_t *returned_view, const view_t *expected_view);
void ck_assert_storage_eq(const storage_t *returned_storage, const storage_t *expected_storage);
void ck_assert_buffer_eq(const buffer_t *returned_buffer, const buffer_t *expected_buffer);
void ck_assert_function_eq(const tensor_t *returned_tensor, const function_t *returned_function, const tensor_t *expected_tensor, const function_t *expected_function);
void ck_assert_tensor_eq(const tensor_t *returned_tensor, const tensor_t *expected_tensor);
void ck_assert_tensor_equiv(const tensor_t *returned_tensor, const tensor_t *expected_tensor);

#endif