#include <check.h>
#include <queue.h>
#include <errors.h>

queue_t *queue;
nw_error_t *error;

void setup(void)
{
    error = queue_create(&queue);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
}

void teardown(void)
{
    queue_destroy(queue);
}

START_TEST(test_null)
{
    void *ptr;
    error = queue_dequeue(queue, (void **) &ptr);
    ck_assert_ptr_nonnull(error);
    error_destroy(error);

    error = queue_dequeue(queue, NULL);
    ck_assert_ptr_nonnull(error);
    error_destroy(error);

    error = queue_dequeue(NULL, (void **) &ptr);
    ck_assert_ptr_nonnull(error);
    error_destroy(error);

    error = queue_dequeue(NULL, NULL);
    ck_assert_ptr_nonnull(error);
    error_destroy(error);

    error = queue_enqueue(NULL, ptr);
    ck_assert_ptr_nonnull(error);
    error_destroy(error);
}
END_TEST

START_TEST(test_queue)
{
    int test_case_1 = 1;
    int test_case_2 = 2;
    int test_case_3 = 3;
    error = queue_enqueue(queue, (void *) &test_case_1);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_uint_eq(queue->size, 1);

    error = queue_enqueue(queue, (void *) &test_case_2);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_uint_eq(queue->size, 2);

    error = queue_enqueue(queue, (void *) &test_case_3);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_uint_eq(queue->size, 3);

    int *test_case_ptr_1;
    int *test_case_ptr_2;
    int *test_case_ptr_3;
    int *test_case_ptr_4;

    error = queue_dequeue(queue, (void **) &test_case_ptr_1);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_uint_eq(queue->size, 2);
    ck_assert_int_eq(test_case_1, *test_case_ptr_1);

    error = queue_dequeue(queue, (void **) &test_case_ptr_2);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_uint_eq(queue->size, 1);
    ck_assert_int_eq(test_case_2, *test_case_ptr_2);

    error = queue_dequeue(queue, (void **) &test_case_ptr_3);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_uint_eq(queue->size, 0);
    ck_assert_int_eq(test_case_3, *test_case_ptr_3);
    
    error = queue_enqueue(queue, (void *) NULL);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_uint_eq(queue->size, 1);

    error = queue_dequeue(queue, (void **) &test_case_ptr_4);
    if (error != NULL)
    {
       error_print(error);
       error_destroy(error); 
    }
    ck_assert_uint_eq(queue->size, 0);
    ck_assert_ptr_null(test_case_ptr_4);
}
END_TEST

Suite *queue_creation_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Test Suite");
    tc_core = tcase_create("Case 1");
    tcase_add_checked_fixture(tc_core, setup, teardown);
    tcase_add_test(tc_core, test_queue);
    tcase_add_test(tc_core, test_null);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(queue_creation_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
