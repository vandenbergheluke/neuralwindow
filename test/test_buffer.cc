#include <iostream>
extern "C"
{
#include <check.h>
#include <buffer.h>
}
#include <torch/torch.h>

#define CASES 2
#define EPSILON 0.0001

nw_error_t *unary_error;
buffer_t *unary_buffers[CASES];

void unary_setup(void)
{
    // view_t *views[CASES];

    for (int i = 0; i < CASES; i++)
    {
        unary_buffers[i] = NULL;
        // views[i] = NULL;
    }
}

void unary_teardown(void)
{
    for (int i = 0; i < CASES; i++)
    {
        buffer_destroy(unary_buffers[i]);
    }
    error_destroy(unary_error);
}

// START_TEST(test_exponential)
// {
//     view_t *expected_view;
//     view_t *returned_view;
//     buffer_t *expected_unary_buffer;
//     buffer_t *returned_unary_buffer;
// }
// END_TEST

Suite *make_buffer_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Unary Suite");
    tc_unary = tcase_create("Unary Case");
    tcase_add_checked_fixture(tc_unary, unary_setup, unary_teardown);
    // tcase_add_test(tc_unary, test_exponential);
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
