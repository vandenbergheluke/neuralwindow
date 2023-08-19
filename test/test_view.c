#include <check.h>
#include <view.h>

error_t *view_error;
view_t *view;

void view_setup(void)
{
    view_error = NULL;
    view = NULL;
}

void view_teardown(void)
{
    error_destroy(NULL);
    error_destroy(view_error);
    view_destroy(view);
}

START_TEST(test_view_create_error)
{
    uint32_t number_of_cases = 5;

    uint32_t offsets[] = {0, 0, 0, 0, 0};
    uint32_t ranks[] = {1, 1, 1, 0, MAX_RANK + 1};
    uint32_t *shapes[] = {
        (uint32_t[]) {1},
        NULL,
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
    };

    uint32_t *strides[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
    };

    error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
    };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        if (!i)
        {
            view_error = view_create(NULL, offsets[i], ranks[i], shapes[i], strides[i]);
        }
        else
        {
            view_error = view_create(&view, offsets[i], ranks[i], shapes[i], strides[i]);
        }
        ck_assert_ptr_nonnull(view_error);
        ck_assert_int_eq(view_error->error_type, error_types[i]);
        error_destroy(view_error);
        view_destroy(view);
        view_error = NULL;
        view = NULL;
    }
}
END_TEST

START_TEST(test_view_create)
{
    uint32_t number_of_cases = 5;

    uint32_t offsets[] = {0, 1, 10, 20, 30};
    uint32_t ranks[] = {1, 2, 3, 4, 5};
    uint32_t *expected_shapes[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {1, 2},
        (uint32_t[]) {1, 2, 3},
        (uint32_t[]) {1, 2, 3, 4},
        (uint32_t[]) {1, 2, 3, 4, 5},
    };

    uint32_t *expected_strides[] = {
        (uint32_t[]) {0},
        (uint32_t[]) {2, 1},
        (uint32_t[]) {6, 3, 1},
        (uint32_t[]) {0, 12, 4, 1},
        (uint32_t[]) {120, 60, 20, 5, 1},
    };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        if (i % 3 == 0)
        {
            view_error = view_create(&view, offsets[i], ranks[i], expected_shapes[i], NULL);
        }
        else
        {
            view_error = view_create(&view, offsets[i], ranks[i], expected_shapes[i], expected_strides[i]);
        }
        ck_assert_ptr_null(view_error);
        ck_assert_ptr_ne(view->shape, expected_shapes[i]);
        ck_assert_ptr_ne(view->strides, expected_strides[i]);
        ck_assert_ptr_nonnull(view->shape);
        ck_assert_ptr_nonnull(view->strides);
        for (uint32_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_uint_eq(view->shape[j], expected_shapes[i][j]);
            ck_assert_uint_eq(view->strides[j], expected_strides[i][j]);
        }
        ck_assert_uint_eq(view->offset, offsets[i]);
        ck_assert_uint_eq(view->rank, ranks[i]);
        error_destroy(view_error);
        view_destroy(view);
        view_error = NULL;
        view = NULL;
    }
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
    tcase_add_test(tc_view_create, test_view_create);

    suite_add_tcase(s, tc_view_create);

    return s;
}

error_t *is_contiguous_error;

void contiguous_setup(void)
{
    is_contiguous_error = NULL;
}

void contiguous_teardown(void)
{
    error_destroy(is_contiguous_error);
}

START_TEST(test_is_contiguous)
{
    ck_assert(is_contiguous((uint32_t[]) {2, 2, 3}, 3, (uint32_t[]) {6, 3, 1}));
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
    uint32_t number_of_cases = 6;
    uint32_t shapes[][MAX_RANK] = {
        {2, 3, 4, 5},
        {1, 10}, 
        {2, 1, 1},
        {10}, 
        {10, 1, 2, 5},
        {2, 2, 3},
    };
    uint32_t expected_strides[][MAX_RANK] = {
        {60, 20, 5, 1},
        {0, 1},
        {1, 0, 0},
        {1},
        {10, 0, 5, 1},
        {6, 3, 1},
    };
    uint32_t returned_strides[number_of_cases][MAX_RANK];
    uint32_t ranks[] = {
        4,
        2,
        3,
        1,
        4,
        3,
    };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        is_contiguous_error = strides_from_shape(returned_strides[i], shapes[i], ranks[i]);
        ck_assert_ptr_null(is_contiguous_error);
        for (uint32_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_uint_eq(expected_strides[i][j], returned_strides[i][j]);
        }
        error_destroy(is_contiguous_error);
        is_contiguous_error = NULL;
    }
}
END_TEST

START_TEST(test_strides_from_shape_error)
{
    uint32_t number_of_cases = 5;
    uint32_t *shapes[] = {
        NULL,
        (uint32_t[]) {1},
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
    };
    error_type_t expected_error_type[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
    };
    uint32_t *returned_strides[] = {
        NULL,
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1}
    };
    uint32_t ranks[] = {
        1,
        1,
        1,
        0,
        MAX_RANK + 1,
    };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        is_contiguous_error = strides_from_shape(returned_strides[i], shapes[i], ranks[i]);
        ck_assert_ptr_nonnull(is_contiguous_error);
        ck_assert_int_eq(is_contiguous_error->error_type, expected_error_type[i]);
        error_destroy(is_contiguous_error);
        is_contiguous_error = NULL;
    }
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

error_t *permute_error;

void permute_setup(void)
{
    permute_error = NULL;
}

void permute_teardown(void)
{
    error_destroy(permute_error);
}

START_TEST(test_permute)
{
    uint32_t number_of_cases = 6;

    uint32_t original_shapes[][MAX_RANK] = {
        {1},
        {5, 3},
        {3, 2, 1},
        {2, 4, 3, 1},
        {2, 2, 2},
        {1, 1, 1, 1},
    };

    uint32_t original_strides[][MAX_RANK] = {
        {0},
        {3, 1},
        {2, 1, 1},
        {12, 3, 1, 1},
        {4, 2, 1},
        {0, 0, 0, 0},
    };

    uint32_t axis[][MAX_RANK] = {
        {0},
        {1, 0},
        {2, 1, 0},
        {1, 2, 0, 3},
        {2, 0, 1},
        {0, 1, 3, 2},
    };

    uint32_t expected_shapes[][MAX_RANK] = {
        {1},
        {3, 5},
        {1, 2, 3},
        {4, 3, 2, 1},
        {2, 2, 2},
        {1, 1, 1, 1},
    };

    uint32_t expected_strides[][MAX_RANK] = {
        {0},
        {1, 3},
        {1, 1, 2},
        {3, 1, 12, 1},
        {1, 4, 2},
        {0, 0, 0, 0},
    };

    uint32_t ranks[] = {
        1,
        2,
        3,
        4,
        3,
        4,
    };

    uint32_t returned_shapes[number_of_cases][MAX_RANK];
    uint32_t returned_strides[number_of_cases][MAX_RANK];

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        permute_error = permute(original_shapes[i],
                                ranks[i],
                                original_strides[i],
                                returned_shapes[i],
                                ranks[i],
                                returned_strides[i],
                                axis[i],
                                ranks[i]);
        ck_assert_ptr_null(permute_error);
        for (uint32_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_uint_eq(expected_shapes[i][j], returned_shapes[i][j]);
            ck_assert_uint_eq(expected_strides[i][j], returned_strides[i][j]);
        }
        error_destroy(permute_error);
        permute_error = NULL;
    }
}
END_TEST

START_TEST(test_permute_error)
{
    uint32_t number_of_cases = 10;

    uint32_t *original_shapes[] = {
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
    };

    uint32_t *original_strides[] = {
        (uint32_t[]) {1},
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
    };

    uint32_t *axis[] = {
        (uint32_t[]) {0},
        (uint32_t[]) {0},
        NULL,
        (uint32_t[]) {0},
        (uint32_t[]) {0},
        (uint32_t[]) {0},
        (uint32_t[]) {0},
        (uint32_t[]) {0},
        (uint32_t[]) {0},
        (uint32_t[]) {0},
    };

    uint32_t *permuted_shapes[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
    };

    uint32_t *permuted_strides[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
    };

    uint32_t original_ranks[] = {
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
        0,
        MAX_RANK + 1,
    };
    
    uint32_t permuted_ranks[] = {
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        0,
        MAX_RANK + 1,
    };

    uint32_t axis_lengths[] = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        0,
        MAX_RANK + 1,
    };

    error_type_t expected_error_type[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
    };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        permute_error = permute(original_shapes[i],
                                original_ranks[i],
                                original_strides[i],
                                permuted_shapes[i],
                                permuted_ranks[i],
                                permuted_strides[i],
                                axis[i],
                                axis_lengths[i]);
        ck_assert_ptr_nonnull(permute_error);
        ck_assert_int_eq(permute_error->error_type, expected_error_type[i]);
        error_destroy(permute_error);
        permute_error = NULL;
    }
}
END_TEST


void reverse_permute_setup(void)
{
}

void reverse_permute_teardown(void)
{
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
