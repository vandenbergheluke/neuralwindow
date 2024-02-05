#ifndef TEST_HELPER_H
#define TEST_HELPER_H

typedef struct tensor_t tensor_t;
typedef struct function_t function_t;
typedef struct buffer_t buffer_t;
typedef struct view_t view_t;
typedef struct storage_t storage_t;
typedef enum runtime_t runtime_t;
typedef enum datatype_t datatype_t;

void ck_assert_tensor_eq(const tensor_t *returned_tensor, const tensor_t *expected_tensor);
void ck_assert_tensor_equiv(const tensor_t *returned_tensor, const tensor_t *expected_tensor);
void ck_assert_view_eq(const view_t *returned_view, const view_t *expected_view);
void ck_assert_tensor_equiv_flt(const tensor_t *returned_tensor, const tensor_t *expected_tensor, float32_t abs_epsilon);
void ck_assert_tensor_equiv_dbl(const tensor_t *returned_tensor, const tensor_t *expected_tensor, float64_t abs_epsilon);

#endif