#include <check.h>
#include <view.h>

error_t *error;
view_t *view;

void setup(void)
{
    error = NULL;
    view = NULL;
}

void teardown(void)
{
    error_destroy(error);
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
            error = view_create(NULL, offsets[i], ranks[i], shapes[i], strides[i]);
        }
        else
        {
            error = view_create(&view, offsets[i], ranks[i], shapes[i], strides[i]);
        }
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        view_destroy(view);
        error = NULL;
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
            error = view_create(&view, offsets[i], ranks[i], expected_shapes[i], NULL);
        }
        else
        {
           error = view_create(&view, offsets[i], ranks[i], expected_shapes[i], expected_strides[i]);
        }
        ck_assert_ptr_null(error);
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
        error_destroy(error);
        view_destroy(view);
        error = NULL;
        view = NULL;
    }
}
END_TEST

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
    uint32_t *shapes[] = {
        (uint32_t[]) {2, 3, 4, 5},
        (uint32_t[]) {1, 10}, 
        (uint32_t[]) {2, 1, 1},
        (uint32_t[]) {10}, 
        (uint32_t[]) {10, 1, 2, 5},
        (uint32_t[]) {2, 2, 3},
    };
    uint32_t *expected_strides[] = {
        (uint32_t[]) {60, 20, 5, 1},
        (uint32_t[]) {0, 1},
        (uint32_t[]) {1, 0, 0},
        (uint32_t[]) {1},
        (uint32_t[]) {10, 0, 5, 1},
        (uint32_t[]) {6, 3, 1},
    };
    uint32_t returned_strides[number_of_cases][MAX_RANK];
    uint32_t ranks[] = { 4, 2, 3, 1, 4, 3 };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        error = strides_from_shape(returned_strides[i], shapes[i], ranks[i]);
        ck_assert_ptr_null(error);
        for (uint32_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_uint_eq(expected_strides[i][j], returned_strides[i][j]);
        }
        error_destroy(error);
        error = NULL;
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
    uint32_t ranks[] = { 1, 1, 1, 0, MAX_RANK + 1 };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        error = strides_from_shape(returned_strides[i], shapes[i], ranks[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, expected_error_type[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reverse_permute)
{
    uint32_t number_of_cases = 6;

    uint32_t *axis[] = {
        (uint32_t[]) {0},
        (uint32_t[]) {1, 0},
        (uint32_t[]) {2, 1, 0},
        (uint32_t[]) {1, 2, 0, 3},
        (uint32_t[]) {2, 0, 1},
        (uint32_t[]) {0, 1, 3, 2},
    };

    uint32_t returned_axis[number_of_cases][MAX_RANK];
    uint32_t *expected_axis[] = {
        (uint32_t[]) {0},
        (uint32_t[]) {1, 0},
        (uint32_t[]) {2, 1, 0},
        (uint32_t[]) {2, 0, 1, 3},
        (uint32_t[]) {1, 2, 0},
        (uint32_t[]) {0, 1, 3, 2},
    };
    uint32_t ranks[] = { 1, 2, 3, 4, 3, 4 };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        error = reverse_permute(axis[i], ranks[i], returned_axis[i]);
        ck_assert_ptr_null(error);
        for (uint32_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_uint_eq(expected_axis[i][j], returned_axis[i][j]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reverse_permute_error)
{
    uint32_t number_of_cases = 5;

    uint32_t *axis[] = {
        NULL,
        NULL,
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
    };
    uint32_t returned_axis[number_of_cases][MAX_RANK];
    uint32_t ranks[] = { 1, 1, 1, 0, MAX_RANK + 1 };
    error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
    };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        error = reverse_permute(axis[i], ranks[i], returned_axis[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_permute)
{
    uint32_t number_of_cases = 6;

    uint32_t *original_shapes[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {5, 3},
        (uint32_t[]) {3, 2, 1},
        (uint32_t[]) {2, 4, 3, 1},
        (uint32_t[]) {2, 2, 2},
        (uint32_t[]) {1, 1, 1, 1},
    };

    uint32_t *original_strides[] = {
        (uint32_t[]) {0},
        (uint32_t[]) {3, 1},
        (uint32_t[]) {2, 1, 1},
        (uint32_t[]) {12, 3, 1, 1},
        (uint32_t[]) {4, 2, 1},
        (uint32_t[]) {0, 0, 0, 0},
    };

    uint32_t *axis[] = {
        (uint32_t[]) {0},
        (uint32_t[]) {1, 0},
        (uint32_t[]) {2, 1, 0},
        (uint32_t[]) {1, 2, 0, 3},
        (uint32_t[]) {2, 0, 1},
        (uint32_t[]) {0, 1, 3, 2},
    };

    uint32_t *expected_shapes[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {3, 5},
        (uint32_t[]) {1, 2, 3},
        (uint32_t[]) {4, 3, 2, 1},
        (uint32_t[]) {2, 2, 2},
        (uint32_t[]) {1, 1, 1, 1},
    };

    uint32_t *expected_strides[] = {
        (uint32_t[]) {0},
        (uint32_t[]) {1, 3},
        (uint32_t[]) {1, 1, 2},
        (uint32_t[]) {3, 1, 12, 1},
        (uint32_t[]) {1, 4, 2},
        (uint32_t[]) {0, 0, 0, 0},
    };

    uint32_t ranks[] = { 1, 2, 3, 4, 3, 4 };

    uint32_t returned_shapes[number_of_cases][MAX_RANK];
    uint32_t returned_strides[number_of_cases][MAX_RANK];

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        error = permute(original_shapes[i],
                        ranks[i],
                        original_strides[i],
                        returned_shapes[i],
                        ranks[i],
                        returned_strides[i],
                        axis[i],
                        ranks[i]);
        ck_assert_ptr_null(error);
        for (uint32_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_uint_eq(expected_shapes[i][j], returned_shapes[i][j]);
            ck_assert_uint_eq(expected_strides[i][j], returned_strides[i][j]);
        }
        error_destroy(error);
        error = NULL;
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

    uint32_t original_ranks[] = { 1, 1, 1, 1, 1, 2, 1, 1, 0, MAX_RANK + 1 };
    uint32_t permuted_ranks[] = { 1, 1, 1, 1, 1, 1, 2, 1, 0, MAX_RANK + 1 };
    uint32_t axis_lengths[] = { 1, 1, 1, 1, 1, 1, 1, 2, 0, MAX_RANK + 1 };

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
        error = permute(original_shapes[i],
                        original_ranks[i],
                        original_strides[i],
                        permuted_shapes[i],
                        permuted_ranks[i],
                        permuted_strides[i],
                        axis[i],
                        axis_lengths[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, expected_error_type[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reduce_recover_dimension)
{
    uint32_t number_of_cases = 2;

    uint32_t *original_shapes[] = {
        (uint32_t[]) {2},
        (uint32_t[]) {1, 2, 3},
    };

    uint32_t original_ranks[] = {
        1,
        3,
    };

    uint32_t *original_strides[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {0, 3, 1},
    };

    uint32_t *expected_reduced_shapes[] = {
        (uint32_t[]) {2, 1},
        (uint32_t[]) {1, 2, 1, 1, 3},
    };

    uint32_t reduced_ranks[] = {
        2,
        5.
    };

    uint32_t *expected_reduced_strides[] = {
        (uint32_t[]) {1, 0},
        (uint32_t[]) {0, 3, 0, 0, 1},
    };

    uint32_t *axis[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {2, 3},
    };

    uint32_t lengths[] = {
        1,
        2,
    };
    
    uint32_t returned_reduced_shapes[number_of_cases][MAX_RANK];
    uint32_t returned_reduced_strides[number_of_cases][MAX_RANK];

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        error = reduce_recover_dimensions(original_shapes[i],
                                          original_ranks[i],
                                          original_strides[i],
                                          returned_reduced_shapes[i],
                                          reduced_ranks[i],
                                          returned_reduced_strides[i],
                                          axis[i],
                                          lengths[i]);
        ck_assert_ptr_null(error);
        for (uint32_t j = 0; j < reduced_ranks[i]; j++)
        {
            ck_assert_uint_eq(returned_reduced_shapes[i][j], expected_reduced_shapes[i][j]);
            ck_assert_uint_eq(returned_reduced_strides[i][j], expected_reduced_strides[i][j]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reduce_recover_dimension_error)
{
    uint32_t number_of_cases = 9;

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
    };

    uint32_t original_ranks[] = {
        1,
        1,
        1,
        1,
        1,
        0,
        MAX_RANK + 1,
        2,
        1,
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
    };

    uint32_t *reduced_shapes[] = {
        (uint32_t[]) {2, 1},
        (uint32_t[]) {2, 1},
        NULL,
        (uint32_t[]) {2, 1},
        (uint32_t[]) {2, 1},
        (uint32_t[]) {2, 1},
        (uint32_t[]) {2, 1},
        (uint32_t[]) {2, 1},
        (uint32_t[]) {2, 1},
    };

    uint32_t reduced_ranks[] = {
        2,
        2,
        2,
        2,
        2,
        0,
        MAX_RANK + 1,
        2,
        2,
    };

    uint32_t *reduced_strides[] = {
        (uint32_t[]) {2, 0},
        (uint32_t[]) {2, 0},
        (uint32_t[]) {2, 0},
        NULL,
        (uint32_t[]) {2, 0},
        (uint32_t[]) {2, 0},
        (uint32_t[]) {2, 0},
        (uint32_t[]) {2, 0},
        (uint32_t[]) {2, 0},
    };

    uint32_t *axis[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        NULL,
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {1},
        (uint32_t[]) {2},
    };

    uint32_t lengths[] = {
        1,
        1,
        1,
        1,
        1,
        0,
        MAX_RANK + 1,
        1,
        1,
    };

    error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
    };

    for (uint32_t i = 0; i < number_of_cases; i++)
    {
        error = reduce_recover_dimensions(original_shapes[i],
                                          original_ranks[i],
                                          original_strides[i],
                                          reduced_shapes[i],
                                          reduced_ranks[i],
                                          reduced_strides[i],
                                          axis[i],
                                          lengths[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

Suite *make_view_suite(void)
{
    Suite *s;
    TCase *tc;

    s = suite_create("Test View Suite");

    tc = tcase_create("Test View");
    tcase_add_checked_fixture(tc, setup, teardown);
    tcase_add_test(tc, test_view_create_error);
    tcase_add_test(tc, test_view_create);
    tcase_add_test(tc, test_permute);
    tcase_add_test(tc, test_permute_error);
    tcase_add_test(tc, test_reverse_permute);
    tcase_add_test(tc, test_reverse_permute_error);
    tcase_add_test(tc, test_is_contiguous);
    tcase_add_test(tc, test_strides_from_shape);
    tcase_add_test(tc, test_strides_from_shape_error);
    tcase_add_test(tc, test_reduce_recover_dimension);
    tcase_add_test(tc, test_reduce_recover_dimension_error);
    suite_add_tcase(s, tc);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_view_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
