#include <check.h>
#include <tensor.h>
#include <view.h>
#include <buffer.h>
#include <datatype.h>
#include <errors.h>

tensor_t *tensor;
nw_error_t *error;

void setup(void)
{
    view_t *view;
    error = view_create(&view, 0, 2, (uint64_t[]){1, 2}, NULL);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }

    buffer_t *buffer;
    error = buffer_create(&buffer, OPENBLAS_RUNTIME, FLOAT32, view, NULL, 0, true);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }

    error = tensor_create(&tensor, buffer, NULL, NULL, false, false);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }
}

void teardown(void)
{
    tensor_destroy(tensor);
}

START_TEST(test_memory_allocate)
{
    ck_assert_ptr_null(error);
    ck_assert_ptr_nonnull(tensor);
}
END_TEST

Suite *make_sample_creation_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Test Tensor Suite");
    tc_core = tcase_create("Test Tensor Case");
    tcase_add_checked_fixture(tc_core, setup, teardown);
    tcase_add_test(tc_core, test_memory_allocate);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_sample_creation_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
