#include <stdlib.h>
#include <stdint.h>
#include <check.h>
#include <tensor.h>

tensor_t *x;

void setup(void)
{
    x = construct_tensor(NULL, FLOAT32, (shape_t) {NULL, 0, NULL, 0});
}

void teardown(void)
{
    destroy_tensor(x);
}

START_TEST(test_tensor_construct)
{
    ck_assert_int_eq(x->datatype, FLOAT32);
}
END_TEST

Suite *make_sample_creation_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Tensor Construct Test Suite");
    tc_core = tcase_create("Case 1");
    tcase_add_checked_fixture(tc_core, setup, teardown);
    tcase_add_test(tc_core, test_tensor_construct);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_sample_creation_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    // srunner_add_suite(sr, suite());
    srunner_set_log(sr, "test_tensor.log");
    srunner_set_xml(sr, "test_tensor.xml");
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
