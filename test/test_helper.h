#ifndef TEST_HELPER
#define TEST_HELPER

typedef struct tensor_t tensor_t;
typedef struct function_t function_t;
typedef struct buffer_t buffer_t;
typedef struct view_t view_t;

#define EPSILON 0.0001
#define SEED 1234

void ck_assert_view_eq(const view_t *returned_view, const view_t *expected_view);
void ck_assert_buffer_eq(const buffer_t *returned_buffer, const buffer_t *expected_buffer);
void ck_assert_function_eq(const tensor_t *returned_tensor, 
                           const function_t *returned_function,
                           const tensor_t *expected_tensor,
                           const function_t *expected_function);
void ck_assert_tensor_eq(const tensor_t *returned_tensor, const tensor_t *expected_tensor);
void ck_assert_tensor_equiv(const tensor_t *returned_tensor, const tensor_t *expected_tensor);

#endif