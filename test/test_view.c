#include <check.h>
#include <buffer.h>

error_t *test_case_view_error_0;
error_t *test_case_view_error_1;
error_t *test_case_view_error_2;
error_t *test_case_view_error_3;
error_t *test_case_view_error_4;
error_t *test_case_view_error_5;

view_t *test_case_view_1;
view_t *test_case_view_2;
view_t *test_case_view_3;
view_t *test_case_view_4;
view_t *test_case_view_5;

uint32_t *test_case_view_shape_0;
uint32_t *test_case_view_shape_1;
uint32_t *test_case_view_shape_2;
uint32_t *test_case_view_shape_3;
uint32_t *test_case_view_shape_4;
uint32_t *test_case_view_shape_5;

uint32_t *test_case_view_strides_0;
uint32_t *test_case_view_strides_1;
uint32_t *test_case_view_strides_2;
uint32_t *test_case_view_strides_3;
uint32_t *test_case_view_strides_4;
uint32_t *test_case_view_strides_5;

uint32_t test_case_view_offset_0;
uint32_t test_case_view_offset_1;
uint32_t test_case_view_offset_2;
uint32_t test_case_view_offset_3;
uint32_t test_case_view_offset_4;
uint32_t test_case_view_offset_5;

uint32_t test_case_view_rank_0;
uint32_t test_case_view_rank_1;
uint32_t test_case_view_rank_2;
uint32_t test_case_view_rank_3;
uint32_t test_case_view_rank_4;
uint32_t test_case_view_rank_5;

void view_setup(void)
{
    // NULL view argument.
    test_case_view_offset_0 = 0;
    test_case_view_rank_0 = 1;
    test_case_view_shape_0 = (uint32_t *) malloc(sizeof(uint32_t));
    ck_assert_ptr_nonnull(test_case_view_shape_0);
    test_case_view_shape_0[0] = (uint32_t) 1;
    test_case_view_strides_0 = (uint32_t *) malloc(sizeof(uint32_t));
    ck_assert_ptr_nonnull(test_case_view_strides_0);
    test_case_view_strides_0[0] = (uint32_t) 1;
    test_case_view_error_0 = view_create(NULL,
                                         test_case_view_offset_0,
                                         test_case_view_rank_0,
                                         test_case_view_shape_0,
                                         test_case_view_strides_0);

    // NULL shape argument.
    test_case_view_offset_1 = 0;
    test_case_view_rank_1 = 1;
    test_case_view_shape_1 = (uint32_t *) malloc(sizeof(uint32_t));
    ck_assert_ptr_nonnull(test_case_view_shape_1);
    test_case_view_shape_1[0] = (uint32_t) 1;
    test_case_view_strides_1 = (uint32_t *) malloc(sizeof(uint32_t));
    ck_assert_ptr_nonnull(test_case_view_strides_1);
    test_case_view_strides_1[0] = (uint32_t) 1;
    test_case_view_error_1 = view_create(&test_case_view_1,
                                         test_case_view_offset_1,
                                         test_case_view_rank_1,
                                         NULL,
                                         test_case_view_strides_1);

    // Rank 0 argument
    test_case_view_offset_2 = 0;
    test_case_view_rank_2 = 0;
    test_case_view_shape_2 = (uint32_t *) malloc(sizeof(uint32_t));
    ck_assert_ptr_nonnull(test_case_view_shape_2);
    test_case_view_shape_2[0] = (uint32_t) 1;
    test_case_view_strides_2 = (uint32_t *) malloc(sizeof(uint32_t));
    ck_assert_ptr_nonnull(test_case_view_strides_2);
    test_case_view_strides_2[0] = (uint32_t) 1;
    test_case_view_error_2 = view_create(&test_case_view_2,
                                         test_case_view_offset_2,
                                         test_case_view_rank_2,
                                         test_case_view_shape_2,
                                         test_case_view_strides_2);

    // Rank greater than max
    test_case_view_offset_3 = 0;
    test_case_view_rank_3 = MAX_RANK + 1;
    test_case_view_shape_3 = (uint32_t *) malloc(sizeof(uint32_t));
    ck_assert_ptr_nonnull(test_case_view_shape_3);
    test_case_view_shape_3[0] = (uint32_t) 1;
    test_case_view_strides_3 = (uint32_t *) malloc(sizeof(uint32_t));
    ck_assert_ptr_nonnull(test_case_view_strides_3);
    test_case_view_strides_3[0] = (uint32_t) 1;
    test_case_view_error_3 = view_create(&test_case_view_3,
                                         test_case_view_offset_3,
                                         test_case_view_rank_3,
                                         test_case_view_shape_3,
                                         test_case_view_strides_3);
    
    // Valid shape with strides
    test_case_view_offset_4 = 0;
    test_case_view_rank_4 = 3;
    test_case_view_shape_4 = (uint32_t *) malloc((size_t) (test_case_view_rank_4 * sizeof(uint32_t)));
    ck_assert_ptr_nonnull(test_case_view_shape_4);
    test_case_view_shape_4[0] = (uint32_t) 1;
    test_case_view_shape_4[1] = (uint32_t) 2;
    test_case_view_shape_4[2] = (uint32_t) 3;
    test_case_view_strides_4 = (uint32_t *) malloc((size_t) (test_case_view_rank_4 * sizeof(uint32_t)));
    ck_assert_ptr_nonnull(test_case_view_strides_4);
    test_case_view_strides_4[0] = (uint32_t) 6;
    test_case_view_strides_4[1] = (uint32_t) 3;
    test_case_view_strides_4[2] = (uint32_t) 1;
    test_case_view_error_4 = view_create(&test_case_view_4,
                                         test_case_view_offset_4,
                                         test_case_view_rank_4,
                                         test_case_view_shape_4,
                                         test_case_view_strides_4);

    // Valid shape without strides
    test_case_view_offset_5 = 0;
    test_case_view_rank_5 = 3;
    test_case_view_shape_5 = (uint32_t *) malloc((size_t) (test_case_view_rank_5 * sizeof(uint32_t)));
    ck_assert_ptr_nonnull(test_case_view_shape_5);
    test_case_view_shape_5[0] = (uint32_t) 1;
    test_case_view_shape_5[1] = (uint32_t) 2;
    test_case_view_shape_5[2] = (uint32_t) 3;
    test_case_view_strides_5 = (uint32_t *) malloc((size_t) (test_case_view_rank_5 * sizeof(uint32_t)));
    ck_assert_ptr_nonnull(test_case_view_strides_5);
    test_case_view_strides_5[0] = (uint32_t) 0;
    test_case_view_strides_5[1] = (uint32_t) 3;
    test_case_view_strides_5[2] = (uint32_t) 1;
    test_case_view_error_5 = view_create(&test_case_view_5,
                                         test_case_view_offset_5,
                                         test_case_view_rank_5,
                                         test_case_view_shape_5,
                                         NULL);
}

void view_teardown(void)
{
    error_destroy(NULL);
    error_destroy(test_case_view_error_0);
    error_destroy(test_case_view_error_1);
    error_destroy(test_case_view_error_2);
    error_destroy(test_case_view_error_3);

    view_destroy(NULL);
    view_destroy(test_case_view_1);
    view_destroy(test_case_view_2);
    view_destroy(test_case_view_3);

    free(test_case_view_shape_0);
    free(test_case_view_shape_1);
    free(test_case_view_shape_2);
    free(test_case_view_shape_3);
    free(test_case_view_shape_4);
    free(test_case_view_shape_5);

    free(test_case_view_strides_0);
    free(test_case_view_strides_1);
    free(test_case_view_strides_2);
    free(test_case_view_strides_3);
    free(test_case_view_strides_4);
    free(test_case_view_strides_5);
}

START_TEST(test_view_create_error)
{
    ck_assert_ptr_nonnull(test_case_view_error_0);
    ck_assert_int_eq(test_case_view_error_0->error_type, ERROR_NULL);

    ck_assert_ptr_nonnull(test_case_view_error_1);
    ck_assert_int_eq(test_case_view_error_1->error_type, ERROR_NULL);

    ck_assert_ptr_nonnull(test_case_view_error_2);
    ck_assert_int_eq(test_case_view_error_2->error_type, ERROR_RANK_CONFLICT);

    ck_assert_ptr_nonnull(test_case_view_error_3);
    ck_assert_int_eq(test_case_view_error_3->error_type, ERROR_RANK_CONFLICT);

    ck_assert_ptr_null(test_case_view_error_4);
    ck_assert_ptr_null(test_case_view_error_5);
}
END_TEST

START_TEST(test_view_create_shape)
{
    ck_assert_ptr_ne(test_case_view_4->shape, test_case_view_shape_4);
    ck_assert_ptr_nonnull(test_case_view_4->shape);
    for (uint32_t i = 0; i < test_case_view_rank_4; i++)
    {
        ck_assert_uint_eq(test_case_view_4->shape[i], test_case_view_shape_4[i]);
    }

    ck_assert_ptr_ne(test_case_view_5->shape, test_case_view_shape_5);
    ck_assert_ptr_nonnull(test_case_view_5->shape);
    for (uint32_t i = 0; i < test_case_view_rank_5; i++)
    {
        ck_assert_uint_eq(test_case_view_5->shape[i], test_case_view_shape_5[i]);
    }
}
END_TEST

START_TEST(test_view_create_strides)
{
    ck_assert_ptr_ne(test_case_view_4->strides, test_case_view_strides_4);
    ck_assert_ptr_nonnull(test_case_view_4->strides);
    for (uint32_t i = 0; i < test_case_view_rank_4; i++)
    {
        ck_assert_uint_eq(test_case_view_4->strides[i], test_case_view_strides_4[i]);
    }

    ck_assert_ptr_ne(test_case_view_5->strides, test_case_view_strides_5);
    ck_assert_ptr_nonnull(test_case_view_5->strides);
    for (uint32_t i = 0; i < test_case_view_rank_5; i++)
    {
        ck_assert_uint_eq(test_case_view_5->strides[i], test_case_view_strides_5[i]);
    }

    ck_assert_ptr_ne(test_case_view_4->strides, test_case_view_strides_4);
    ck_assert_ptr_ne(test_case_view_5->strides, test_case_view_strides_5);
}
END_TEST

START_TEST(test_view_create_offset)
{
    ck_assert_uint_eq(test_case_view_4->offset, test_case_view_offset_4);
    ck_assert_uint_eq(test_case_view_5->offset, test_case_view_offset_5);
}
END_TEST

START_TEST(test_view_create_rank)
{
    ck_assert_uint_eq(test_case_view_4->rank, test_case_view_rank_4);
    ck_assert_uint_eq(test_case_view_5->rank, test_case_view_rank_5);
}
END_TEST

Suite *make_view_suite(void)
{
    Suite *s;
    TCase *tc_view_create;

    s = suite_create("Test View Suite");

    tc_view_create = tcase_create("Test view_create");
    tcase_add_checked_fixture(tc_view_create, view_setup, view_teardown);
    tcase_add_test(tc_view_create, test_view_create_error);
    tcase_add_test(tc_view_create, test_view_create_shape);
    tcase_add_test(tc_view_create, test_view_create_strides);
    tcase_add_test(tc_view_create, test_view_create_offset);
    tcase_add_test(tc_view_create, test_view_create_rank);

    suite_add_tcase(s, tc_view_create);

    return s;
}

error_t *test_case_contiguous_error_0;
error_t *test_case_contiguous_error_1;
error_t *test_case_contiguous_error_2;
error_t *test_case_contiguous_error_3;
error_t *test_case_contiguous_error_4;
error_t *test_case_contiguous_error_5;

uint32_t *test_case_contiguous_shape_0;
uint32_t *test_case_contiguous_shape_1;
uint32_t *test_case_contiguous_shape_2;
uint32_t *test_case_contiguous_shape_3;
uint32_t *test_case_contiguous_shape_4;
uint32_t *test_case_contiguous_shape_5;

uint32_t *test_case_contiguous_expected_strides_0;
uint32_t *test_case_contiguous_expected_strides_1;
uint32_t *test_case_contiguous_expected_strides_2;
uint32_t *test_case_contiguous_expected_strides_3;
uint32_t *test_case_contiguous_expected_strides_4;
uint32_t *test_case_contiguous_expected_strides_5;

uint32_t *test_case_contiguous_returned_strides_0;
uint32_t *test_case_contiguous_returned_strides_1;
uint32_t *test_case_contiguous_returned_strides_2;
uint32_t *test_case_contiguous_returned_strides_3;
uint32_t *test_case_contiguous_returned_strides_4;
uint32_t *test_case_contiguous_returned_strides_5;

uint32_t test_case_contiguous_rank_0;
uint32_t test_case_contiguous_rank_1;
uint32_t test_case_contiguous_rank_2;
uint32_t test_case_contiguous_rank_3;
uint32_t test_case_contiguous_rank_4;
uint32_t test_case_contiguous_rank_5;

void contiguous_setup(void)
{
    // Test Case 0
    test_case_contiguous_rank_0 = 3;
    test_case_contiguous_shape_0 = (uint32_t *) malloc((size_t) (test_case_view_rank_0 * sizeof(uint32_t)));
    test_case_contiguous_returned_strides_0 = (uint32_t *) malloc((size_t) (test_case_view_rank_0 * sizeof(uint32_t)));
    test_case_contiguous_expected_strides_0 = (uint32_t *) malloc((size_t) (test_case_view_rank_0 * sizeof(uint32_t)));
    test_case_contiguous_error_0 = NULL;

    ck_assert_ptr_nonnull(test_case_contiguous_shape_0);
    ck_assert_ptr_nonnull(test_case_contiguous_returned_strides_0);
    ck_assert_ptr_nonnull(test_case_contiguous_expected_strides_0);

    test_case_contiguous_shape_0[0] = (uint32_t) 2;
    test_case_contiguous_shape_0[1] = (uint32_t) 2;
    test_case_contiguous_shape_0[2] = (uint32_t) 3;

    test_case_contiguous_expected_strides_0[0] = (uint32_t) 6;
    test_case_contiguous_expected_strides_0[1] = (uint32_t) 3;
    test_case_contiguous_expected_strides_0[2] = (uint32_t) 1;

    // Test Case 1
    test_case_contiguous_rank_1 = 4;
    test_case_contiguous_shape_1 = (uint32_t *) malloc((size_t) (test_case_view_rank_1 * sizeof(uint32_t)));
    test_case_contiguous_returned_strides_1 = (uint32_t *) malloc((size_t) (test_case_view_rank_1 * sizeof(uint32_t)));
    test_case_contiguous_expected_strides_1 = (uint32_t *) malloc((size_t) (test_case_view_rank_1 * sizeof(uint32_t)));
    test_case_contiguous_error_1 = NULL;

    ck_assert_ptr_nonnull(test_case_contiguous_shape_1);
    ck_assert_ptr_nonnull(test_case_contiguous_returned_strides_1);
    ck_assert_ptr_nonnull(test_case_contiguous_expected_strides_1);

    test_case_contiguous_shape_1[0] = (uint32_t) 10;
    test_case_contiguous_shape_1[1] = (uint32_t) 1;
    test_case_contiguous_shape_1[2] = (uint32_t) 2;
    test_case_contiguous_shape_1[3] = (uint32_t) 5;

    test_case_contiguous_expected_strides_1[0] = (uint32_t) 10;
    test_case_contiguous_expected_strides_1[1] = (uint32_t) 0;
    test_case_contiguous_expected_strides_1[2] = (uint32_t) 5;
    test_case_contiguous_expected_strides_1[3] = (uint32_t) 1;

    // Test Case 2
    test_case_contiguous_rank_2 = 1;
    test_case_contiguous_shape_2 = (uint32_t *) malloc((size_t) (test_case_view_rank_2 * sizeof(uint32_t)));
    test_case_contiguous_returned_strides_2 = (uint32_t *) malloc((size_t) (test_case_view_rank_2 * sizeof(uint32_t)));
    test_case_contiguous_expected_strides_2 = (uint32_t *) malloc((size_t) (test_case_view_rank_2 * sizeof(uint32_t)));
    test_case_contiguous_error_2 = NULL;

    ck_assert_ptr_nonnull(test_case_contiguous_shape_2);
    ck_assert_ptr_nonnull(test_case_contiguous_returned_strides_2);
    ck_assert_ptr_nonnull(test_case_contiguous_expected_strides_2);

    test_case_contiguous_shape_2[0] = (uint32_t) 10;

    test_case_contiguous_expected_strides_2[0] = (uint32_t) 1;

    // Test Case 3
    test_case_contiguous_rank_3 = 3;
    test_case_contiguous_shape_3 = (uint32_t *) malloc((size_t) (test_case_view_rank_3 * sizeof(uint32_t)));
    test_case_contiguous_returned_strides_3 = (uint32_t *) malloc((size_t) (test_case_view_rank_3 * sizeof(uint32_t)));
    test_case_contiguous_expected_strides_3 = (uint32_t *) malloc((size_t) (test_case_view_rank_3 * sizeof(uint32_t)));
    test_case_contiguous_error_3 = NULL;

    ck_assert_ptr_nonnull(test_case_contiguous_shape_3);
    ck_assert_ptr_nonnull(test_case_contiguous_returned_strides_3);
    ck_assert_ptr_nonnull(test_case_contiguous_expected_strides_3);

    test_case_contiguous_shape_3[0] = (uint32_t) 2;
    test_case_contiguous_shape_3[1] = (uint32_t) 1;
    test_case_contiguous_shape_3[2] = (uint32_t) 1;

    test_case_contiguous_expected_strides_3[0] = (uint32_t) 1;
    test_case_contiguous_expected_strides_3[1] = (uint32_t) 0;
    test_case_contiguous_expected_strides_3[2] = (uint32_t) 0;

    // Test Case 4
    test_case_contiguous_rank_4 = 2;
    test_case_contiguous_shape_4 = (uint32_t *) malloc((size_t) (test_case_view_rank_4 * sizeof(uint32_t)));
    test_case_contiguous_returned_strides_4 = (uint32_t *) malloc((size_t) (test_case_view_rank_4 * sizeof(uint32_t)));
    test_case_contiguous_expected_strides_4 = (uint32_t *) malloc((size_t) (test_case_view_rank_4 * sizeof(uint32_t)));
    test_case_contiguous_error_4 = NULL;

    ck_assert_ptr_nonnull(test_case_contiguous_shape_4);
    ck_assert_ptr_nonnull(test_case_contiguous_returned_strides_4);
    ck_assert_ptr_nonnull(test_case_contiguous_expected_strides_4);

    test_case_contiguous_shape_4[0] = (uint32_t) 1;
    test_case_contiguous_shape_4[1] = (uint32_t) 10;

    test_case_contiguous_expected_strides_4[0] = (uint32_t) 0;
    test_case_contiguous_expected_strides_4[1] = (uint32_t) 1;

    // Test Case 5
    test_case_contiguous_rank_5 = 4;
    test_case_contiguous_shape_5 = (uint32_t *) malloc((size_t) (test_case_view_rank_5 * sizeof(uint32_t)));
    test_case_contiguous_returned_strides_5 = (uint32_t *) malloc((size_t) (test_case_view_rank_5 * sizeof(uint32_t)));
    test_case_contiguous_expected_strides_5 = (uint32_t *) malloc((size_t) (test_case_view_rank_5 * sizeof(uint32_t)));
    test_case_contiguous_error_5 = NULL;

    ck_assert_ptr_nonnull(test_case_contiguous_shape_5);
    ck_assert_ptr_nonnull(test_case_contiguous_returned_strides_5);
    ck_assert_ptr_nonnull(test_case_contiguous_expected_strides_5);

    test_case_contiguous_shape_5[0] = (uint32_t) 2;
    test_case_contiguous_shape_5[1] = (uint32_t) 3;
    test_case_contiguous_shape_5[2] = (uint32_t) 4;
    test_case_contiguous_shape_5[3] = (uint32_t) 5;

    test_case_contiguous_expected_strides_5[0] = (uint32_t) 60;
    test_case_contiguous_expected_strides_5[1] = (uint32_t) 20;
    test_case_contiguous_expected_strides_5[2] = (uint32_t) 5;
    test_case_contiguous_expected_strides_5[3] = (uint32_t) 1;
}

void contiguous_teardown(void)
{
    error_destroy(test_case_contiguous_error_0);
    error_destroy(test_case_contiguous_error_1);
    error_destroy(test_case_contiguous_error_2);
    error_destroy(test_case_contiguous_error_3);
    error_destroy(test_case_contiguous_error_4);
    error_destroy(test_case_contiguous_error_5);

    free(test_case_contiguous_shape_0);
    free(test_case_contiguous_shape_1);
    free(test_case_contiguous_shape_2);
    free(test_case_contiguous_shape_3);
    free(test_case_contiguous_shape_4);
    free(test_case_contiguous_shape_5);

    free(test_case_contiguous_expected_strides_0);
    free(test_case_contiguous_expected_strides_1);
    free(test_case_contiguous_expected_strides_2);
    free(test_case_contiguous_expected_strides_3);
    free(test_case_contiguous_expected_strides_4);
    free(test_case_contiguous_expected_strides_5);

    free(test_case_contiguous_returned_strides_0);
    free(test_case_contiguous_returned_strides_1);
    free(test_case_contiguous_returned_strides_2);
    free(test_case_contiguous_returned_strides_3);
    free(test_case_contiguous_returned_strides_4);
    free(test_case_contiguous_returned_strides_5);
}

START_TEST(test_is_contiguous)
{
    ck_assert(is_contiguous(test_case_contiguous_shape_0, 
                            test_case_contiguous_rank_0, 
                            test_case_contiguous_expected_strides_0));
    ck_assert(is_contiguous(test_case_contiguous_shape_1, 
                            test_case_contiguous_rank_1, 
                            test_case_contiguous_expected_strides_1));
    ck_assert(is_contiguous(test_case_contiguous_shape_2, 
                            test_case_contiguous_rank_2, 
                            test_case_contiguous_expected_strides_2));
    ck_assert(is_contiguous(test_case_contiguous_shape_3, 
                            test_case_contiguous_rank_3, 
                            test_case_contiguous_expected_strides_3));
    ck_assert(is_contiguous(test_case_contiguous_shape_4, 
                            test_case_contiguous_rank_4, 
                            test_case_contiguous_expected_strides_4));
    ck_assert(is_contiguous(test_case_contiguous_shape_5, 
                            test_case_contiguous_rank_5, 
                            test_case_contiguous_expected_strides_5));

    ck_assert(!is_contiguous(NULL, 3, (uint32_t[]) {1, 2, 3}));
    ck_assert(!is_contiguous((uint32_t[]) {1, 2, 3}, 3, NULL));
    ck_assert(!is_contiguous(NULL, 3, NULL));
    ck_assert(!is_contiguous((uint32_t[]) {1}, 0, (uint32_t[]) {1}));
    ck_assert(is_contiguous((uint32_t[]) {1}, 1, (uint32_t[]) {1}));
    ck_assert(is_contiguous((uint32_t[]) {1}, 1, (uint32_t[]) {0}));
    ck_assert(is_contiguous((uint32_t[]) {1, 2, 1, 5}, 1, (uint32_t[]) {0, 5, 5, 1}));
    ck_assert(is_contiguous((uint32_t[]) {1, 2, 1, 5}, 1, (uint32_t[]) {10, 5, 0, 1}));
}
END_TEST

START_TEST(test_strides_from_shape)
{
    test_case_contiguous_error_0 = strides_from_shape(test_case_contiguous_returned_strides_0,
                                                      test_case_contiguous_shape_0,
                                                      test_case_contiguous_rank_0);
    ck_assert_ptr_null(test_case_contiguous_error_0);
    for (uint32_t i = 0; i < test_case_contiguous_rank_0; i++)
    {
        ck_assert_uint_eq(test_case_contiguous_expected_strides_0[i],
                          test_case_contiguous_returned_strides_0[i]);
    }

    test_case_contiguous_error_1 = strides_from_shape(test_case_contiguous_returned_strides_1,
                                                      test_case_contiguous_shape_1,
                                                      test_case_contiguous_rank_1);
    ck_assert_ptr_null(test_case_contiguous_error_1);
    for (uint32_t i = 0; i < test_case_contiguous_rank_1; i++)
    {
        ck_assert_uint_eq(test_case_contiguous_expected_strides_1[i],
                          test_case_contiguous_returned_strides_1[i]);
    }

    test_case_contiguous_error_2 = strides_from_shape(test_case_contiguous_returned_strides_2,
                                                      test_case_contiguous_shape_2,
                                                      test_case_contiguous_rank_2);
    ck_assert_ptr_null(test_case_contiguous_error_2);
    for (uint32_t i = 0; i < test_case_contiguous_rank_2; i++)
    {
        ck_assert_uint_eq(test_case_contiguous_expected_strides_2[i],
                          test_case_contiguous_returned_strides_2[i]);
    }

    test_case_contiguous_error_3 = strides_from_shape(test_case_contiguous_returned_strides_3,
                                                      test_case_contiguous_shape_3,
                                                      test_case_contiguous_rank_3);
    ck_assert_ptr_null(test_case_contiguous_error_3);
    for (uint32_t i = 0; i < test_case_contiguous_rank_3; i++)
    {
        ck_assert_uint_eq(test_case_contiguous_expected_strides_3[i],
                          test_case_contiguous_returned_strides_3[i]);
    }

    test_case_contiguous_error_4 = strides_from_shape(test_case_contiguous_returned_strides_4,
                                                      test_case_contiguous_shape_4,
                                                      test_case_contiguous_rank_4);
    ck_assert_ptr_null(test_case_contiguous_error_4);
    for (uint32_t i = 0; i < test_case_contiguous_rank_4; i++)
    {
        ck_assert_uint_eq(test_case_contiguous_expected_strides_4[i],
                          test_case_contiguous_returned_strides_4[i]);
    }

    test_case_contiguous_error_5 = strides_from_shape(test_case_contiguous_returned_strides_5,
                                                      test_case_contiguous_shape_5,
                                                      test_case_contiguous_rank_5);
    ck_assert_ptr_null(test_case_contiguous_error_5);
    for (uint32_t i = 0; i < test_case_contiguous_rank_5; i++)
    {
        ck_assert_uint_eq(test_case_contiguous_expected_strides_5[i],
                          test_case_contiguous_returned_strides_5[i]);
    }
}
END_TEST

START_TEST(test_strides_from_shape_error)
{
    test_case_contiguous_error_0 = strides_from_shape((uint32_t[]) {1}, NULL, 1);
    ck_assert_ptr_nonnull(test_case_contiguous_error_0);
    ck_assert_int_eq(test_case_contiguous_error_0->error_type, ERROR_NULL);
    test_case_contiguous_error_1 = strides_from_shape(NULL, (uint32_t[]) {1}, 1);
    ck_assert_ptr_nonnull(test_case_contiguous_error_1);
    ck_assert_int_eq(test_case_contiguous_error_1->error_type, ERROR_NULL);
    test_case_contiguous_error_2 = strides_from_shape(NULL, NULL, 1);
    ck_assert_ptr_nonnull(test_case_contiguous_error_2);
    ck_assert_int_eq(test_case_contiguous_error_2->error_type, ERROR_NULL);
    test_case_contiguous_error_3 = strides_from_shape((uint32_t[]) {1}, (uint32_t[]) {1}, 0);
    ck_assert_ptr_nonnull(test_case_contiguous_error_3);
    ck_assert_int_eq(test_case_contiguous_error_3->error_type, ERROR_RANK_CONFLICT);
    test_case_contiguous_error_4 = strides_from_shape((uint32_t[]) {1}, (uint32_t[]) {1}, MAX_RANK + 1);
    ck_assert_ptr_nonnull(test_case_contiguous_error_4);
    ck_assert_int_eq(test_case_contiguous_error_4->error_type, ERROR_RANK_CONFLICT);
}
END_TEST

Suite *make_contiguous_suite(void)
{
    Suite *s;
    TCase *tc_is_contiguous;
    TCase *tc_strides_from_shape;

    s = suite_create("Test Contiguous Suite");

    tc_is_contiguous = tcase_create("Test is_contiguous");
    tcase_add_checked_fixture(tc_is_contiguous, contiguous_setup, contiguous_teardown);
    tcase_add_test(tc_is_contiguous, test_is_contiguous);
    suite_add_tcase(s, tc_is_contiguous);

    tc_strides_from_shape = tcase_create("Test strides_from_shape");
    tcase_add_checked_fixture(tc_strides_from_shape, contiguous_setup, contiguous_teardown);
    tcase_add_test(tc_strides_from_shape, test_strides_from_shape);
    tcase_add_test(tc_strides_from_shape, test_strides_from_shape_error);
    suite_add_tcase(s, tc_strides_from_shape);

    return s;
}

error_t *test_case_permute_error_0;
error_t *test_case_permute_error_1;
error_t *test_case_permute_error_2;
error_t *test_case_permute_error_3;
error_t *test_case_permute_error_4;
error_t *test_case_permute_error_5;

uint32_t test_case_permute_rank_0;
uint32_t test_case_permute_rank_1;
uint32_t test_case_permute_rank_2;
uint32_t test_case_permute_rank_3;
uint32_t test_case_permute_rank_4;
uint32_t test_case_permute_rank_5;

uint32_t *test_case_permute_expected_shape_0;
uint32_t *test_case_permute_expected_shape_1;
uint32_t *test_case_permute_expected_shape_2;
uint32_t *test_case_permute_expected_shape_3;
uint32_t *test_case_permute_expected_shape_4;
uint32_t *test_case_permute_expected_shape_5;

uint32_t *test_case_permute_returned_shape_0;
uint32_t *test_case_permute_returned_shape_1;
uint32_t *test_case_permute_returned_shape_2;
uint32_t *test_case_permute_returned_shape_3;
uint32_t *test_case_permute_returned_shape_4;
uint32_t *test_case_permute_returned_shape_5;

uint32_t *test_case_permute_expected_strides_0;
uint32_t *test_case_permute_expected_strides_1;
uint32_t *test_case_permute_expected_strides_2;
uint32_t *test_case_permute_expected_strides_3;
uint32_t *test_case_permute_expected_strides_4;
uint32_t *test_case_permute_expected_strides_5;

uint32_t *test_case_permute_returned_strides_0;
uint32_t *test_case_permute_returned_strides_1;
uint32_t *test_case_permute_returned_strides_2;
uint32_t *test_case_permute_returned_strides_3;
uint32_t *test_case_permute_returned_strides_4;
uint32_t *test_case_permute_returned_strides_5;

void permute_setup(void)
{
    test_case_permute_error_0 = NULL;
    test_case_permute_error_1 = NULL;
    test_case_permute_error_2 = NULL;
    test_case_permute_error_3 = NULL;
    test_case_permute_error_4 = NULL;
    test_case_permute_error_5 = NULL;

    test_case_permute_rank_0 = 1;
    test_case_permute_rank_1 = 2;
    test_case_permute_rank_2 = 3;
    test_case_permute_rank_3 = 4;
    test_case_permute_rank_4 = 3;
    test_case_permute_rank_5 = 4;

    test_case_permute_expected_shape_0 = (uint32_t *) malloc((size_t) (test_case_permute_rank_0 * sizeof(uint32_t)));
    test_case_permute_expected_shape_1 = (uint32_t *) malloc((size_t) (test_case_permute_rank_1 * sizeof(uint32_t)));
    test_case_permute_expected_shape_2 = (uint32_t *) malloc((size_t) (test_case_permute_rank_2 * sizeof(uint32_t)));
    test_case_permute_expected_shape_3 = (uint32_t *) malloc((size_t) (test_case_permute_rank_3 * sizeof(uint32_t)));
    test_case_permute_expected_shape_4 = (uint32_t *) malloc((size_t) (test_case_permute_rank_4 * sizeof(uint32_t)));
    test_case_permute_expected_shape_5 = (uint32_t *) malloc((size_t) (test_case_permute_rank_5 * sizeof(uint32_t)));

    test_case_permute_returned_shape_0 = (uint32_t *) malloc((size_t) (test_case_permute_rank_0 * sizeof(uint32_t)));
    test_case_permute_returned_shape_1 = (uint32_t *) malloc((size_t) (test_case_permute_rank_1 * sizeof(uint32_t)));
    test_case_permute_returned_shape_2 = (uint32_t *) malloc((size_t) (test_case_permute_rank_2 * sizeof(uint32_t)));
    test_case_permute_returned_shape_3 = (uint32_t *) malloc((size_t) (test_case_permute_rank_3 * sizeof(uint32_t)));
    test_case_permute_returned_shape_4 = (uint32_t *) malloc((size_t) (test_case_permute_rank_4 * sizeof(uint32_t)));
    test_case_permute_returned_shape_5 = (uint32_t *) malloc((size_t) (test_case_permute_rank_5 * sizeof(uint32_t)));
    
    test_case_permute_expected_strides_0 = (uint32_t *) malloc((size_t) (test_case_permute_rank_0 * sizeof(uint32_t)));
    test_case_permute_expected_strides_1 = (uint32_t *) malloc((size_t) (test_case_permute_rank_1 * sizeof(uint32_t)));
    test_case_permute_expected_strides_2 = (uint32_t *) malloc((size_t) (test_case_permute_rank_2 * sizeof(uint32_t)));
    test_case_permute_expected_strides_3 = (uint32_t *) malloc((size_t) (test_case_permute_rank_3 * sizeof(uint32_t)));
    test_case_permute_expected_strides_4 = (uint32_t *) malloc((size_t) (test_case_permute_rank_4 * sizeof(uint32_t)));
    test_case_permute_expected_strides_5 = (uint32_t *) malloc((size_t) (test_case_permute_rank_5 * sizeof(uint32_t)));

    test_case_permute_returned_strides_0 = (uint32_t *) malloc((size_t) (test_case_permute_rank_0 * sizeof(uint32_t)));
    test_case_permute_returned_strides_1 = (uint32_t *) malloc((size_t) (test_case_permute_rank_1 * sizeof(uint32_t)));
    test_case_permute_returned_strides_2 = (uint32_t *) malloc((size_t) (test_case_permute_rank_2 * sizeof(uint32_t)));
    test_case_permute_returned_strides_3 = (uint32_t *) malloc((size_t) (test_case_permute_rank_3 * sizeof(uint32_t)));
    test_case_permute_returned_strides_4 = (uint32_t *) malloc((size_t) (test_case_permute_rank_4 * sizeof(uint32_t)));
    test_case_permute_returned_strides_5 = (uint32_t *) malloc((size_t) (test_case_permute_rank_5 * sizeof(uint32_t)));

    ck_assert_ptr_nonnull(test_case_permute_expected_shape_0);
    ck_assert_ptr_nonnull(test_case_permute_expected_shape_1);
    ck_assert_ptr_nonnull(test_case_permute_expected_shape_2);
    ck_assert_ptr_nonnull(test_case_permute_expected_shape_3);
    ck_assert_ptr_nonnull(test_case_permute_expected_shape_4);
    ck_assert_ptr_nonnull(test_case_permute_expected_shape_5);

    ck_assert_ptr_nonnull(test_case_permute_returned_shape_0);
    ck_assert_ptr_nonnull(test_case_permute_returned_shape_1);
    ck_assert_ptr_nonnull(test_case_permute_returned_shape_2);
    ck_assert_ptr_nonnull(test_case_permute_returned_shape_3);
    ck_assert_ptr_nonnull(test_case_permute_returned_shape_4);
    ck_assert_ptr_nonnull(test_case_permute_returned_shape_5);

    ck_assert_ptr_nonnull(test_case_permute_expected_strides_0);
    ck_assert_ptr_nonnull(test_case_permute_expected_strides_1);
    ck_assert_ptr_nonnull(test_case_permute_expected_strides_2);
    ck_assert_ptr_nonnull(test_case_permute_expected_strides_3);
    ck_assert_ptr_nonnull(test_case_permute_expected_strides_4);
    ck_assert_ptr_nonnull(test_case_permute_expected_strides_5);

    ck_assert_ptr_nonnull(test_case_permute_returned_strides_0);
    ck_assert_ptr_nonnull(test_case_permute_returned_strides_1);
    ck_assert_ptr_nonnull(test_case_permute_returned_strides_2);
    ck_assert_ptr_nonnull(test_case_permute_returned_strides_3);
    ck_assert_ptr_nonnull(test_case_permute_returned_strides_4);
    ck_assert_ptr_nonnull(test_case_permute_returned_strides_5);

    test_case_permute_expected_shape_0[0] = 1;

    test_case_permute_expected_shape_1[0] = 3;
    test_case_permute_expected_shape_1[1] = 5;

    test_case_permute_expected_shape_2[0] = 1;
    test_case_permute_expected_shape_2[1] = 2;
    test_case_permute_expected_shape_2[2] = 3;

    test_case_permute_expected_shape_3[0] = 4;
    test_case_permute_expected_shape_3[1] = 3;
    test_case_permute_expected_shape_3[2] = 2;
    test_case_permute_expected_shape_3[3] = 1;

    test_case_permute_expected_shape_4[0] = 2;
    test_case_permute_expected_shape_4[1] = 2;
    test_case_permute_expected_shape_4[2] = 2;

    test_case_permute_expected_shape_5[0] = 1;
    test_case_permute_expected_shape_5[1] = 1;
    test_case_permute_expected_shape_5[2] = 1;
    test_case_permute_expected_shape_5[3] = 1;

    test_case_permute_expected_strides_0[0] = 0;

    test_case_permute_expected_strides_1[0] = 1;
    test_case_permute_expected_strides_1[1] = 3;

    test_case_permute_expected_strides_2[0] = 1;
    test_case_permute_expected_strides_2[1] = 1;
    test_case_permute_expected_strides_2[2] = 2;

    test_case_permute_expected_strides_3[0] = 3;
    test_case_permute_expected_strides_3[1] = 1;
    test_case_permute_expected_strides_3[2] = 12;
    test_case_permute_expected_strides_3[3] = 1;

    test_case_permute_expected_strides_4[0] = 1;
    test_case_permute_expected_strides_4[1] = 4;
    test_case_permute_expected_strides_4[2] = 2;

    test_case_permute_expected_strides_5[0] = 0;
    test_case_permute_expected_strides_5[1] = 0;
    test_case_permute_expected_strides_5[2] = 0;
    test_case_permute_expected_strides_5[3] = 0;
}

void permute_teardown(void)
{
    error_destroy(test_case_permute_error_0);
    error_destroy(test_case_permute_error_1);
    error_destroy(test_case_permute_error_2);
    error_destroy(test_case_permute_error_3);
    error_destroy(test_case_permute_error_4);
    error_destroy(test_case_permute_error_5);

    free(test_case_permute_expected_shape_0);
    free(test_case_permute_expected_shape_1);
    free(test_case_permute_expected_shape_2);
    free(test_case_permute_expected_shape_3);
    free(test_case_permute_expected_shape_4);
    free(test_case_permute_expected_shape_5);

    free(test_case_permute_returned_shape_0);
    free(test_case_permute_returned_shape_1);
    free(test_case_permute_returned_shape_2);
    free(test_case_permute_returned_shape_3);
    free(test_case_permute_returned_shape_4);
    free(test_case_permute_returned_shape_5);

    free(test_case_permute_expected_strides_0);
    free(test_case_permute_expected_strides_1);
    free(test_case_permute_expected_strides_2);
    free(test_case_permute_expected_strides_3);
    free(test_case_permute_expected_strides_4);
    free(test_case_permute_expected_strides_5);

    free(test_case_permute_returned_strides_0);
    free(test_case_permute_returned_strides_1);
    free(test_case_permute_returned_strides_2);
    free(test_case_permute_returned_strides_3);
    free(test_case_permute_returned_strides_4);
    free(test_case_permute_returned_strides_5);
}

START_TEST(test_permute)
{
    test_case_permute_error_0 = permute((uint32_t[]) {1},
                                         test_case_permute_rank_0,
                                         (uint32_t[]) {0},
                                         test_case_permute_returned_shape_0,
                                         test_case_permute_rank_0,
                                         test_case_permute_returned_strides_0,
                                         (uint32_t[]) {0},
                                         test_case_permute_rank_0);
    ck_assert_ptr_null(test_case_permute_error_0);
    for (uint32_t i = 0; i < test_case_permute_rank_0; i++)
    {
        ck_assert_uint_eq(test_case_permute_expected_shape_0[i],
                          test_case_permute_returned_shape_0[i]);
        ck_assert_uint_eq(test_case_permute_expected_strides_0[i],
                          test_case_permute_returned_strides_0[i]);
    }
    test_case_permute_error_1 = permute((uint32_t[]) {5, 3},
                                         test_case_permute_rank_1,
                                         (uint32_t[]) {3, 1},
                                         test_case_permute_returned_shape_1,
                                         test_case_permute_rank_1,
                                         test_case_permute_returned_strides_1,
                                         (uint32_t[]) {1, 0},
                                         test_case_permute_rank_1);
    ck_assert_ptr_null(test_case_permute_error_1);
    for (uint32_t i = 0; i < test_case_permute_rank_1; i++)
    {
        ck_assert_uint_eq(test_case_permute_expected_shape_1[i],
                          test_case_permute_returned_shape_1[i]);
        ck_assert_uint_eq(test_case_permute_expected_strides_1[i],
                          test_case_permute_returned_strides_1[i]);
    }
    test_case_permute_error_2 = permute((uint32_t[]) {3, 2, 1},
                                         test_case_permute_rank_2,
                                         (uint32_t[]) {2, 1, 1},
                                         test_case_permute_returned_shape_2,
                                         test_case_permute_rank_2,
                                         test_case_permute_returned_strides_2,
                                         (uint32_t[]) {2, 1, 0},
                                         test_case_permute_rank_2);
    ck_assert_ptr_null(test_case_permute_error_2);
    for (uint32_t i = 0; i < test_case_permute_rank_2; i++)
    {
        ck_assert_uint_eq(test_case_permute_expected_shape_2[i],
                          test_case_permute_returned_shape_2[i]);
        ck_assert_uint_eq(test_case_permute_expected_strides_2[i],
                          test_case_permute_returned_strides_2[i]);
    }
    test_case_permute_error_3 = permute((uint32_t[]) {2, 4, 3, 1},
                                         test_case_permute_rank_3,
                                         (uint32_t[]) {12, 3, 1, 1},
                                         test_case_permute_returned_shape_3,
                                         test_case_permute_rank_3,
                                         test_case_permute_returned_strides_3,
                                         (uint32_t[]) {1, 2, 0, 3},
                                         test_case_permute_rank_3);
    ck_assert_ptr_null(test_case_permute_error_3);
    for (uint32_t i = 0; i < test_case_permute_rank_3; i++)
    {
        ck_assert_uint_eq(test_case_permute_expected_shape_3[i],
                          test_case_permute_returned_shape_3[i]);
        ck_assert_uint_eq(test_case_permute_expected_strides_3[i],
                          test_case_permute_returned_strides_3[i]);
    }
    test_case_permute_error_4 = permute((uint32_t[]) {2, 2, 2},
                                         test_case_permute_rank_4,
                                         (uint32_t[]) {4, 2, 1},
                                         test_case_permute_returned_shape_4,
                                         test_case_permute_rank_4,
                                         test_case_permute_returned_strides_4,
                                         (uint32_t[]) {2, 0, 1},
                                         test_case_permute_rank_4);
    ck_assert_ptr_null(test_case_permute_error_4);
    for (uint32_t i = 0; i < test_case_permute_rank_4; i++)
    {
        ck_assert_uint_eq(test_case_permute_expected_shape_4[i],
                          test_case_permute_returned_shape_4[i]);
        ck_assert_uint_eq(test_case_permute_expected_strides_4[i],
                          test_case_permute_returned_strides_4[i]);
    }
    test_case_permute_error_5 = permute((uint32_t[]) {1, 1, 1, 1},
                                         test_case_permute_rank_5,
                                         (uint32_t[]) {0, 0, 0, 0},
                                         test_case_permute_returned_shape_5,
                                         test_case_permute_rank_5,
                                         test_case_permute_returned_strides_5,
                                         (uint32_t[]) {0, 1, 3, 2},
                                         test_case_permute_rank_5);
    ck_assert_ptr_null(test_case_permute_error_5);
    for (uint32_t i = 0; i < test_case_permute_rank_5; i++)
    {
        ck_assert_uint_eq(test_case_permute_expected_shape_5[i],
                          test_case_permute_returned_shape_5[i]);
        ck_assert_uint_eq(test_case_permute_expected_strides_5[i],
                          test_case_permute_returned_strides_5[i]);
    }
}
END_TEST

START_TEST(test_permute_error)
{

}
END_TEST

void reverse_permute_setup(void)
{
    return;
}

void reverse_permute_teardown(void)
{
    return;
}

START_TEST(test_reverse_permute)
{

}
END_TEST

START_TEST(test_reverse_permute_error)
{

}
END_TEST

Suite *make_permute_suite(void)
{
    Suite *s;
    TCase *tc_permute;
    TCase *tc_reverse_permute;

    s = suite_create("Test Permute Suite");

    tc_permute = tcase_create("Test tc_permute");
    tcase_add_checked_fixture(tc_permute, permute_setup, permute_teardown);
    tcase_add_test(tc_permute, test_permute);
    tcase_add_test(tc_permute, test_permute_error);
    suite_add_tcase(s, tc_permute);

    tc_reverse_permute = tcase_create("Test tc_reverse_permute");
    tcase_add_checked_fixture(tc_reverse_permute, reverse_permute_setup, reverse_permute_teardown);
    tcase_add_test(tc_reverse_permute, test_reverse_permute);
    tcase_add_test(tc_reverse_permute, test_reverse_permute_error);
    suite_add_tcase(s, tc_reverse_permute);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_view_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_add_suite(sr, make_contiguous_suite());
    srunner_add_suite(sr, make_permute_suite());
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
