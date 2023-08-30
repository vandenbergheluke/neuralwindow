#include <check.h>
#include <function.h>

nw_error_t *error;
function_t *function;

void setup(void)
{
    error = NULL;
    function = NULL;
}

void teardown(void)
{
    error_destroy(error);
    function_destroy(function);
}

START_TEST(test_function_exponential)
{
    // error = apply
}
END_TEST

Suite *make_sample_creation_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Test Function Suite");
    tc_core = tcase_create("Test Function Case");
    tcase_add_checked_fixture(tc_core, setup, teardown);
    tcase_add_test(tc_core, test_function_exponential);
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