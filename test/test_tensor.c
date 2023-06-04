#include <stdlib.h>
#include <stdint.h>
#include <check.h>
#include <tensor.h>
#include <device.h>

tensor_t *tensor;

void setup(void)
{
    error_t *error = create_tensor(tensor, DEVICE_CUDA);
}

void teardown(void)
{
    error_t *error = memory_free(tensor, DEVICE_CUDA);
}

START_TEST(test_memory_allocate)
{
    ck_assert_ptr_nonnull(tensor);
}
END_TEST

Suite *make_sample_creation_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Test Suite");
    tc_core = tcase_create("Case 1");
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
    srunner_set_fork_status(sr, CK_FORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
