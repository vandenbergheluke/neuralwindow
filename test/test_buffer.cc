#include <iostream>
extern "C"
{
#include <check.h>
#include <buffer.h>
}
#include <torch/torch.h>

#define CASES 2
#define EPSILON 0.0001
#define SEED 1234

bool_t set_seed = true;

nw_error_t *unary_error;

buffer_t *unary_buffers[CASES];
buffer_t *returned_unary_buffers[CASES];
buffer_t *expected_unary_buffers[CASES];

view_t *views[CASES];
view_t *returned_views[CASES];
view_t *expected_views[CASES];

void unary_setup(void)
{
    if (set_seed)
    {
        torch::manual_seed(SEED);
        set_seed = false;
    }

    for (int i = 0; i < CASES; i++)
    {
        unary_buffers[i] = NULL;
        returned_unary_buffers[i] = NULL;
        expected_unary_buffers[i] = NULL;

        views[i] = NULL;
        returned_views[i] = NULL;
        expected_views[i] = NULL;
    }
}

void unary_teardown(void)
{
    for (int i = 0; i < CASES; i++)
    {
        buffer_destroy(unary_buffers[i]);
        buffer_destroy(returned_unary_buffers[i]);
        buffer_destroy(expected_unary_buffers[i]);
    }
    error_destroy(unary_error);
}

START_TEST(test_exponential)
{
    for (int i = 0; i < CASES; ++i)
    {
        torch::Tensor tensor = torch::randn({3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
        torch::Tensor expected_tensor = torch::exp(tensor);

        unary_error = view_create(&views[i], (uint32_t) tensor.storage_offset(), (uint32_t) tensor.ndimension(), (uint32_t *) tensor.sizes().data(), NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&unary_buffers[i], OPENBLAS_RUNTIME, FLOAT32, views[i], (void *) tensor.data_ptr(), (uint32_t) tensor.numel(), true);
        ck_assert_ptr_null(unary_error);

        unary_error = view_create(&returned_views[i], (uint32_t) tensor.storage_offset(), (uint32_t) tensor.ndimension(), (uint32_t *) tensor.sizes().data(), NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&returned_unary_buffers[i], OPENBLAS_RUNTIME, FLOAT32, returned_views[i], NULL, (uint32_t) tensor.numel(), true);
        ck_assert_ptr_null(unary_error);

        unary_error = view_create(&expected_views[i], (uint32_t) expected_tensor.storage_offset(), (uint32_t) expected_tensor.ndimension(), (uint32_t *) expected_tensor.sizes().data(), NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i], OPENBLAS_RUNTIME, FLOAT32, expected_views[i], (void *) expected_tensor.data_ptr(), (uint32_t) expected_tensor.numel(), true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_exponential(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        for (int j = 0; j < expected_tensor.numel(); ++j)
        {
            ck_assert_double_eq_tol(((float32_t *) returned_unary_buffers[i]->data)[j],
                                    ((float32_t *) expected_unary_buffers[i]->data)[j], EPSILON);
        }
    }
}
END_TEST

Suite *make_buffer_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Unary Suite");
    tc_unary = tcase_create("Unary Case");
    tcase_add_checked_fixture(tc_unary, unary_setup, unary_teardown);
    tcase_add_test(tc_unary, test_exponential);
    suite_add_tcase(s, tc_unary);

    return s;
}

extern "C" int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
