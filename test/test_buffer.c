#include <check.h>
#include <buffer.h>

buffer_t *test_case_buffer_1;
buffer_t *test_case_buffer_2;
buffer_t *test_case_buffer_3;
error_t *error;

void setup(void)
{
    view_t *test_case_view_1;
    error = view_create(&test_case_view_1, 0, 2, (uint32_t[]){2, 2}, NULL);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }

    error = buffer_create(&test_case_buffer_1, MKL_RUNTIME, FLOAT32, test_case_view_1, (float32_t[]){1.0, 2.0, -3.0, 4.0}, 0, true);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }

    view_t *test_case_view_2;
    error = view_create(&test_case_view_2, 0, 2, (uint32_t[]){2, 2}, NULL);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }

    error = buffer_create(&test_case_buffer_2, MKL_RUNTIME, FLOAT32, test_case_view_2, (float32_t[]){1.0, 2.0, -3.0, 4.0}, 0, true);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }

    view_t *test_case_view_3;
    error = view_create(&test_case_view_3, 0, 2, (uint32_t[]){2, 2}, NULL);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }

    error = buffer_create(&test_case_buffer_3, MKL_RUNTIME, FLOAT32, test_case_view_3, NULL, 0, true);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }
}

void teardown(void)
{
    buffer_destroy(test_case_buffer_1);
    buffer_destroy(test_case_buffer_2);
    buffer_destroy(test_case_buffer_3);
}

START_TEST(test_addition)
{
    error = runtime_addition(test_case_buffer_1, test_case_buffer_2, test_case_buffer_3);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }
    float32_t *test_case_buffer_data_3 = (float32_t *) test_case_buffer_3->data;
    ck_assert_float_eq(test_case_buffer_data_3[0], 2.0);
    ck_assert_float_eq(test_case_buffer_data_3[1], 4.0);
    ck_assert_float_eq(test_case_buffer_data_3[2], -6.0);
    ck_assert_float_eq(test_case_buffer_data_3[3], 8.0);
}
END_TEST

Suite *make_buffer_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Test Suite");
    tc_core = tcase_create("Case 1");
    tcase_add_checked_fixture(tc_core, setup, teardown);
    tcase_add_test(tc_core, test_addition);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
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
