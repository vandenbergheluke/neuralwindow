
#include <check.h>
#include <mkl_runtime.h>

float32_t *test_case_data_float32_1;
float32_t *test_case_data_float32_2;
float32_t *test_case_data_float32_3;
size_t test_case_float32_size;
error_t *error;

void setup(void)
{
    test_case_float32_size = sizeof(float32_t) * 4;

    error = mkl_memory_allocate((void **) &test_case_data_float32_1, test_case_float32_size); 
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }
    for (int i = 0; i < 4; i++)
    {
        test_case_data_float32_1[i] = 1.0;
    }


    error = mkl_memory_allocate((void **) &test_case_data_float32_2, test_case_float32_size); 
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }
    for (int i = 0; i < 4; i++)
    {
        test_case_data_float32_2[i] = 1.0;
    }

    error = mkl_memory_allocate((void **) &test_case_data_float32_3, test_case_float32_size); 
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }
}

void teardown(void)
{
    mkl_memory_free(test_case_data_float32_1);
    mkl_memory_free(test_case_data_float32_2);
    mkl_memory_free(test_case_data_float32_3);
}

START_TEST(test_addition)
{
    error = mkl_addition(FLOAT32,
                         4,
                         test_case_data_float32_1,
                         test_case_data_float32_2,
                         test_case_data_float32_3);
    if (error != NULL)
    {
        error_print(error);
        error_destroy(error);
    }
    ck_assert_float_eq(test_case_data_float32_1[0], 1.0);
    ck_assert_float_eq(test_case_data_float32_1[1], 1.0);
    ck_assert_float_eq(test_case_data_float32_1[2], 1.0);
    ck_assert_float_eq(test_case_data_float32_1[3], 1.0);

    ck_assert_float_eq(test_case_data_float32_2[0], 1.0);
    ck_assert_float_eq(test_case_data_float32_2[1], 1.0);
    ck_assert_float_eq(test_case_data_float32_2[2], 1.0);
    ck_assert_float_eq(test_case_data_float32_2[3], 1.0);

    ck_assert_float_eq(test_case_data_float32_3[0], 2.0);
    ck_assert_float_eq(test_case_data_float32_3[1], 2.0);
    ck_assert_float_eq(test_case_data_float32_3[2], 2.0);
    ck_assert_float_eq(test_case_data_float32_3[3], 2.0);
}
END_TEST

Suite *make_mkl_runtime_suite(void)
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

    sr = srunner_create(make_mkl_runtime_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
