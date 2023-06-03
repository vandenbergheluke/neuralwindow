#include <stdlib.h>
#include <stdint.h>
#include <check.h>
#include <device.h>

buffer_t buffer;

void setup(void)
{
    error_t error = memory_allocate(&buffer, 10, DEVICE_CUDA);
}

void teardown(void)
{
    error_t error = memory_free(buffer, DEVICE_CUDA);
}

START_TEST(test_memory_allocate)
{
    ck_assert_ptr_nonnull(buffer);
}
END_TEST

Suite *make_sample_creation_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Device Test Suite");
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
    srunner_set_fork_status(sr, CK_NOFORK);
    // srunner_add_suite(sr, suite());
    srunner_set_log(sr, "test_device.log");
    srunner_set_xml(sr, "test_device.xml");
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
