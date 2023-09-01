#include <check.h>
#include <view.h>

nw_error_t *error;
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
    uint64_t number_of_cases = 5;

    uint64_t offsets[] = {0, 0, 0, 0, 0};
    uint64_t ranks[] = {1, 1, 1, 0, MAX_RANK + 1};
    uint64_t *shapes[] = {
        (uint64_t[]) {1},
        NULL,
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t *strides[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
    };

    for (uint64_t i = 0; i < number_of_cases; i++)
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
    uint64_t number_of_cases = 5;

    uint64_t offsets[] = {0, 1, 10, 20, 30};
    uint64_t ranks[] = {1, 2, 3, 4, 5};
    uint64_t *expected_shapes[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1, 2},
        (uint64_t[]) {1, 2, 3},
        (uint64_t[]) {1, 2, 3, 4},
        (uint64_t[]) {1, 2, 3, 4, 5},
    };

    uint64_t *expected_strides[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {6, 3, 1},
        (uint64_t[]) {0, 12, 4, 1},
        (uint64_t[]) {120, 60, 20, 5, 1},
    };

    for (uint64_t i = 0; i < number_of_cases; i++)
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
        for (uint64_t j = 0; j < ranks[i]; j++)
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
    ck_assert(is_contiguous((uint64_t[]) {2, 2, 3}, 3, (uint64_t[]) {6, 3, 1}));
    ck_assert(!is_contiguous(NULL, 3, (uint64_t[]) {1, 2, 3}));
    ck_assert(!is_contiguous((uint64_t[]) {1, 2, 3}, 3, NULL));
    ck_assert(!is_contiguous(NULL, 3, NULL));
    ck_assert(!is_contiguous((uint64_t[]) {1}, 0, (uint64_t[]) {1}));
    ck_assert(is_contiguous((uint64_t[]) {1}, 1, (uint64_t[]) {1}));
    ck_assert(is_contiguous((uint64_t[]) {1}, 1, (uint64_t[]) {0}));
    ck_assert(is_contiguous((uint64_t[]) {1, 2, 1, 5}, 1, (uint64_t[]) {0, 5, 5, 1}));
    ck_assert(is_contiguous((uint64_t[]) {1, 2, 1, 5}, 1, (uint64_t[]) {10, 5, 0, 1}));
}
END_TEST

START_TEST(test_strides_from_shape)
{
    uint64_t number_of_cases = 6;
    uint64_t *shapes[] = {
        (uint64_t[]) {2, 3, 4, 5},
        (uint64_t[]) {1, 10}, 
        (uint64_t[]) {2, 1, 1},
        (uint64_t[]) {10}, 
        (uint64_t[]) {10, 1, 2, 5},
        (uint64_t[]) {2, 2, 3},
    };
    uint64_t *expected_strides[] = {
        (uint64_t[]) {60, 20, 5, 1},
        (uint64_t[]) {0, 1},
        (uint64_t[]) {1, 0, 0},
        (uint64_t[]) {1},
        (uint64_t[]) {10, 0, 5, 1},
        (uint64_t[]) {6, 3, 1},
    };
    uint64_t returned_strides[number_of_cases][MAX_RANK];
    uint64_t ranks[] = { 4, 2, 3, 1, 4, 3 };

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = strides_from_shape(returned_strides[i], shapes[i], ranks[i]);
        ck_assert_ptr_null(error);
        for (uint64_t j = 0; j < ranks[i]; j++)
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
    uint64_t number_of_cases = 5;
    uint64_t *shapes[] = {
        NULL,
        (uint64_t[]) {1},
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };
    nw_error_type_t expected_error_type[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
    };
    uint64_t *returned_strides[] = {
        NULL,
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1}
    };
    uint64_t ranks[] = { 1, 1, 1, 0, MAX_RANK + 1 };

    for (uint64_t i = 0; i < number_of_cases; i++)
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
    uint64_t number_of_cases = 6;

    uint64_t *axis[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {1, 0},
        (uint64_t[]) {2, 1, 0},
        (uint64_t[]) {1, 2, 0, 3},
        (uint64_t[]) {2, 0, 1},
        (uint64_t[]) {0, 1, 3, 2},
    };

    uint64_t returned_axis[number_of_cases][MAX_RANK];
    uint64_t *expected_axis[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {1, 0},
        (uint64_t[]) {2, 1, 0},
        (uint64_t[]) {2, 0, 1, 3},
        (uint64_t[]) {1, 2, 0},
        (uint64_t[]) {0, 1, 3, 2},
    };
    uint64_t ranks[] = { 1, 2, 3, 4, 3, 4 };

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = reverse_permute(axis[i], ranks[i], returned_axis[i]);
        ck_assert_ptr_null(error);
        for (uint64_t j = 0; j < ranks[i]; j++)
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
    uint64_t number_of_cases = 5;

    uint64_t *axis[] = {
        NULL,
        NULL,
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };
    uint64_t returned_axis[number_of_cases][MAX_RANK];
    uint64_t ranks[] = { 1, 1, 1, 0, MAX_RANK + 1 };
    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
    };

    for (uint64_t i = 0; i < number_of_cases; i++)
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
    uint64_t number_of_cases = 6;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {5, 3},
        (uint64_t[]) {3, 2, 1},
        (uint64_t[]) {2, 4, 3, 1},
        (uint64_t[]) {2, 2, 2},
        (uint64_t[]) {1, 1, 1, 1},
    };

    uint64_t *original_strides[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {3, 1},
        (uint64_t[]) {2, 1, 1},
        (uint64_t[]) {12, 3, 1, 1},
        (uint64_t[]) {4, 2, 1},
        (uint64_t[]) {0, 0, 0, 0},
    };

    uint64_t *axis[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {1, 0},
        (uint64_t[]) {2, 1, 0},
        (uint64_t[]) {1, 2, 0, 3},
        (uint64_t[]) {2, 0, 1},
        (uint64_t[]) {0, 1, 3, 2},
    };

    uint64_t *expected_shapes[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {3, 5},
        (uint64_t[]) {1, 2, 3},
        (uint64_t[]) {4, 3, 2, 1},
        (uint64_t[]) {2, 2, 2},
        (uint64_t[]) {1, 1, 1, 1},
    };

    uint64_t *expected_strides[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {1, 3},
        (uint64_t[]) {1, 1, 2},
        (uint64_t[]) {3, 1, 12, 1},
        (uint64_t[]) {1, 4, 2},
        (uint64_t[]) {0, 0, 0, 0},
    };

    uint64_t ranks[] = { 1, 2, 3, 4, 3, 4 };

    uint64_t returned_shapes[number_of_cases][MAX_RANK];
    uint64_t returned_strides[number_of_cases][MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
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
        for (uint64_t j = 0; j < ranks[i]; j++)
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
    uint64_t number_of_cases = 10;

    uint64_t *original_shapes[] = {
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t *original_strides[] = {
        (uint64_t[]) {1},
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t *axis[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {0},
        NULL,
        (uint64_t[]) {0},
        (uint64_t[]) {0},
        (uint64_t[]) {0},
        (uint64_t[]) {0},
        (uint64_t[]) {0},
        (uint64_t[]) {0},
        (uint64_t[]) {0},
    };

    uint64_t *permuted_shapes[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t *permuted_strides[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t original_ranks[] = { 1, 1, 1, 1, 1, 2, 1, 1, 0, MAX_RANK + 1 };
    uint64_t permuted_ranks[] = { 1, 1, 1, 1, 1, 1, 2, 1, 0, MAX_RANK + 1 };
    uint64_t axis_lengths[] = { 1, 1, 1, 1, 1, 1, 1, 2, 0, MAX_RANK + 1 };

    nw_error_type_t expected_error_type[] = {
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

    for (uint64_t i = 0; i < number_of_cases; i++)
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
    uint64_t number_of_cases = 2;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {2},
        (uint64_t[]) {1, 2, 3},
    };

    uint64_t original_ranks[] = {
        1,
        3,
    };

    uint64_t *original_strides[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {0, 3, 1},
    };

    uint64_t *expected_reduced_shapes[] = {
        (uint64_t[]) {2, 1},
        (uint64_t[]) {1, 2, 1, 1, 3},
    };

    uint64_t reduced_ranks[] = {
        2,
        5.
    };

    uint64_t *expected_reduced_strides[] = {
        (uint64_t[]) {1, 0},
        (uint64_t[]) {0, 3, 0, 0, 1},
    };

    uint64_t *axis[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {2, 3},
    };

    uint64_t lengths[] = {
        1,
        2,
    };
    
    uint64_t returned_reduced_shapes[number_of_cases][MAX_RANK];
    uint64_t returned_reduced_strides[number_of_cases][MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
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
        for (uint64_t j = 0; j < reduced_ranks[i]; j++)
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
    uint64_t number_of_cases = 9;

    uint64_t *original_shapes[] = {
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t original_ranks[] = {
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

    uint64_t *original_strides[] = {
        (uint64_t[]) {1},
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t *reduced_shapes[] = {
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        NULL,
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
    };

    uint64_t reduced_ranks[] = {
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

    uint64_t *reduced_strides[] = {
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        NULL,
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
    };

    uint64_t *axis[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {2},
    };

    uint64_t lengths[] = {
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

    nw_error_type_t error_types[] = {
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

    for (uint64_t i = 0; i < number_of_cases; i++)
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

START_TEST(test_reduce)
{
    uint64_t number_of_cases = 8;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {2},
        (uint64_t[]) {2},
        (uint64_t[]) {2, 3},
        (uint64_t[]) {2, 3},
        (uint64_t[]) {2, 3},
        (uint64_t[]) {2, 3},
        (uint64_t[]) {2, 4, 3},
        (uint64_t[]) {2, 4, 3},
    };

    uint64_t original_ranks[] = {
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
    };

    uint64_t *original_strides[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {3, 1},
        (uint64_t[]) {3, 1},
        (uint64_t[]) {3, 1},
        (uint64_t[]) {3, 1},
        (uint64_t[]) {12, 3, 1},
        (uint64_t[]) {12, 3, 1},
    };

    uint64_t *expected_reduced_shapes[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {2},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {1},
        (uint64_t[]) {1, 1},
        (uint64_t[]) {2, 3},
        (uint64_t[]) {2, 1, 3},
    };

    uint64_t reduced_ranks[] = {
        1,
        1,
        1,
        2,
        1,
        2,
        2,
        3,
    };

    uint64_t *expected_reduced_strides[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {0},
        (uint64_t[]) {1},
        (uint64_t[]) {1, 0},
        (uint64_t[]) {0},
        (uint64_t[]) {0, 0},
        (uint64_t[]) {3, 1},
        (uint64_t[]) {3, 0, 1},
    };

    uint64_t *axis[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {0},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {0, 1},
        (uint64_t[]) {0, 1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t lengths[] = {
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
    };

    bool_t keep_dimensions[] = {
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        true,
    };
    
    uint64_t returned_reduced_shapes[number_of_cases][MAX_RANK];
    uint64_t returned_reduced_strides[number_of_cases][MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = reduce(original_shapes[i],
                       original_ranks[i],
                       original_strides[i],
                       returned_reduced_shapes[i],
                       reduced_ranks[i],
                       returned_reduced_strides[i],
                       axis[i],
                       lengths[i],
                       keep_dimensions[i]);
        ck_assert_ptr_null(error);
        for (uint64_t j = 0; j < reduced_ranks[i]; j++)
        {
            ck_assert_uint_eq(returned_reduced_shapes[i][j], expected_reduced_shapes[i][j]);
            ck_assert_uint_eq(returned_reduced_strides[i][j], expected_reduced_strides[i][j]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reduce_error)
{
    uint64_t number_of_cases = 9;

    uint64_t *original_shapes[] = {
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t original_ranks[] = {
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

    uint64_t *original_strides[] = {
        (uint64_t[]) {1},
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
    };

    uint64_t *reduced_shapes[] = {
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        NULL,
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
        (uint64_t[]) {2, 1},
    };

    uint64_t reduced_ranks[] = {
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

    uint64_t *reduced_strides[] = {
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        NULL,
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
        (uint64_t[]) {2, 0},
    };

    uint64_t *axis[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        NULL,
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {2},
    };

    uint64_t lengths[] = {
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

    nw_error_type_t error_types[] = {
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

    bool_t keep_dimensions[] = {
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
    };

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = reduce(original_shapes[i],
                       original_ranks[i],
                       original_strides[i],
                       reduced_shapes[i],
                       reduced_ranks[i],
                       reduced_strides[i],
                       axis[i],
                       lengths[i],
                       keep_dimensions[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_shapes_equal)
{
    ck_assert(!shapes_equal(NULL, 1, (uint64_t[]) {1}, 1));
    ck_assert(!shapes_equal((uint64_t[]) {1}, 1, NULL, 1));
    ck_assert(!shapes_equal(NULL, 1, NULL, 1));
    ck_assert(!shapes_equal((uint64_t[]) {1}, 2, (uint64_t[]) {1}, 1));
    ck_assert(!shapes_equal((uint64_t[]) {1}, 1, (uint64_t[]) {1}, 2));
    ck_assert(shapes_equal((uint64_t[]) {1}, 1, (uint64_t[]) {1}, 1));
    ck_assert(shapes_equal((uint64_t[]) {1, 2, 3}, 3, (uint64_t[]) {1, 2, 3}, 3));
    ck_assert(!shapes_equal((uint64_t[]) {1, 2, 4}, 3, (uint64_t[]) {1, 2, 3}, 3));
}
END_TEST

START_TEST(test_shapes_size)
{
    ck_assert_uint_eq(shape_size(NULL, 0), 0);
    ck_assert_uint_eq(shape_size((uint64_t[]) {1}, 1), 1);
    ck_assert_uint_eq(shape_size((uint64_t[]) {2}, 1), 2);
    ck_assert_uint_eq(shape_size((uint64_t[]) {1, 2, 1}, 3), 2);
    ck_assert_uint_eq(shape_size((uint64_t[]) {1, 2, 3}, 3), 6);
    ck_assert_uint_eq(shape_size((uint64_t[]) {4, 2, 3}, 3), 24);
}
END_TEST

START_TEST(test_broadcast_strides)
{
    uint64_t number_of_cases = 3;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1},
        (uint64_t[]) {2, 1},
    };

    uint64_t original_ranks[] = {
        1,
        1,
        2,
    };

    uint64_t *original_strides[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {0},
        (uint64_t[]) {1, 0},
    };

    uint64_t *broadcasted_shapes[] = {
        (uint64_t[]) {2},
        (uint64_t[]) {2, 2},
        (uint64_t[]) {3, 2, 2},
    };

    uint64_t broadcasted_ranks[] = {
        1,
        2,
        3,
    };

    uint64_t *expected_broadcasted_strides[] = {
        (uint64_t[]) {0},
        (uint64_t[]) {0, 0},
        (uint64_t[]) {0, 1, 0},
    };

    uint64_t returned_broadcasted_strides[number_of_cases][MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = broadcast_strides(original_shapes[i],
                                  original_ranks[i],
                                  original_strides[i],
                                  broadcasted_shapes[i],
                                  broadcasted_ranks[i],
                                  returned_broadcasted_strides[i]);
        ck_assert_ptr_null(error);
        for (uint64_t j = 0; j < broadcasted_ranks[i]; j++)
        {
            ck_assert_uint_eq(returned_broadcasted_strides[i][broadcasted_ranks[i] - (j + 1)], expected_broadcasted_strides[i][broadcasted_ranks[i] - (j + 1)]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_broadcast_shapes)
{
    uint64_t number_of_cases = 5;

    uint64_t *x_original_shapes[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {1, 2},
        (uint64_t[]) {5, 9, 4, 1, 8},
        (uint64_t[]) {1, 1, 1, 1},
        (uint64_t[]) {9, 8, 7, 6, 5},
    };

    uint64_t x_original_ranks[] = {
        1,
        2,
        5,
        4,
        5,
    };

    uint64_t *y_original_shapes[] = {
        (uint64_t[]) {2},
        (uint64_t[]) {4, 3, 1},
        (uint64_t[]) {4, 3, 1},
        (uint64_t[]) {5, 4, 3, 2, 1},
        (uint64_t[]) {1},
    };

    uint64_t y_original_ranks[] = {
        1,
        3,
        3,
        5,
        1,
    };

    uint64_t *expected_broadcasted_shapes[] = {
        (uint64_t[]) {2},
        (uint64_t[]) {4, 3, 2},
        (uint64_t[]) {5, 9, 4, 3, 8},
        (uint64_t[]) {5, 4, 3, 2, 1},
        (uint64_t[]) {9, 8, 7, 6, 5},
    };

    uint64_t broadcasted_ranks[] = {
        1,
        3,
        5,
        5,
        5,
    };

    uint64_t returned_broadcasted_shapes[number_of_cases][MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = broadcast_shapes(x_original_shapes[i],
                                 x_original_ranks[i],
                                 y_original_shapes[i],
                                 y_original_ranks[i],
                                 returned_broadcasted_shapes[i],
                                 broadcasted_ranks[i]);
        ck_assert_ptr_null(error);
        for (uint64_t j = 0; j < broadcasted_ranks[i]; j++)
        {
            printf("Test Subcase: %lu\n", j);
            ck_assert_uint_eq(returned_broadcasted_shapes[i][broadcasted_ranks[i] - (j + 1)], expected_broadcasted_shapes[i][broadcasted_ranks[i] - (j + 1)]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_is_broadcastable)
{
    ck_assert(is_broadcastable((uint64_t[]) {1}, 1, (uint64_t[]) {1}, 1));
    ck_assert(is_broadcastable((uint64_t[]) {1}, 1, (uint64_t[]) {2, 1}, 2));
    ck_assert(!is_broadcastable((uint64_t[]) {3, 1}, 2, (uint64_t[]) {2, 1}, 2));
    ck_assert(is_broadcastable((uint64_t[]) {2}, 1, (uint64_t[]) {2, 2}, 2));
    ck_assert(!is_broadcastable(NULL, 1, (uint64_t[]) {2, 2}, 2));
    ck_assert(!is_broadcastable((uint64_t[]) {1}, 1, NULL, 1));
    ck_assert(!is_broadcastable(NULL, 1, NULL, 1));
    ck_assert(is_broadcastable((uint64_t[]) {5, 1, 3}, 3, (uint64_t[]) {7, 6, 5, 4, 3}, 5));
    ck_assert(!is_broadcastable((uint64_t[]) {5, 2, 3}, 3, (uint64_t[]) {7, 6, 5, 4, 3}, 5));
}
END_TEST

START_TEST(test_reverse_broadcast_length)
{
    uint64_t number_of_cases = 3;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {3, 1},
        (uint64_t[]) {3, 1},
        (uint64_t[]) {5, 1, 1, 3},
    };

    uint64_t original_ranks[] = {
        2,
        2,
        4,
    };

    uint64_t *broadcasted_shapes[] = {
        (uint64_t[]) {3, 3},
        (uint64_t[]) {3, 3, 3, 3},
        (uint64_t[]) {9, 5, 4, 1, 3},
    };

    uint64_t broadcasted_ranks[] = {
        2,
        4,
        5,
    };

    uint64_t expected_length_keep_dimensions[] = {
        1,
        1,
        1,
    };

    uint64_t expected_length_remove_dimensions[] = {
        0,
        2,
        1,
    };

    uint64_t returned_length_keep_dimensions[number_of_cases];
    uint64_t returned_length_remove_dimensions[number_of_cases];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = reverse_broadcast_length(original_shapes[i],
                                         original_ranks[i],
                                         broadcasted_shapes[i],
                                         broadcasted_ranks[i],
                                         &returned_length_keep_dimensions[i],
                                         &returned_length_remove_dimensions[i]);
        ck_assert_ptr_null(error);
        ck_assert_uint_eq(returned_length_keep_dimensions[i], expected_length_keep_dimensions[i]);
        ck_assert_uint_eq(returned_length_remove_dimensions[i], expected_length_remove_dimensions[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reverse_broadcast_axis)
{
    uint64_t number_of_cases = 3;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {3, 1},
        (uint64_t[]) {3, 1},
        (uint64_t[]) {5, 1, 1, 3},
    };

    uint64_t original_ranks[] = {
        2,
        2,
        4,
    };

    uint64_t *broadcasted_shapes[] = {
        (uint64_t[]) {3, 3},
        (uint64_t[]) {3, 3, 3, 3},
        (uint64_t[]) {9, 5, 4, 1, 3},
    };

    uint64_t broadcasted_ranks[] = {
        2,
        4,
        5,
    };

    uint64_t *expected_axis_keep_dimensions[] = {
        (uint64_t[]) {1},
        (uint64_t[]) {3},
        (uint64_t[]) {2},
    };

    uint64_t *expected_axis_remove_dimensions[] = {
        NULL,
        (uint64_t[]) {1, 0},
        (uint64_t[]) {0},
    };

    uint64_t returned_axis_keep_dimensions[number_of_cases][MAX_RANK];
    uint64_t returned_axis_remove_dimensions[number_of_cases][MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        uint64_t length_keep_dimension, length_remove_dimension;
        error = reverse_broadcast_length(original_shapes[i],
                                         original_ranks[i],
                                         broadcasted_shapes[i],
                                         broadcasted_ranks[i],
                                         &length_keep_dimension,
                                         &length_remove_dimension);
        ck_assert_ptr_null(error);
        error_destroy(error);
        error = NULL;
        error = reverse_broadcast_axis(original_shapes[i],
                                       original_ranks[i],
                                       broadcasted_shapes[i],
                                       broadcasted_ranks[i],
                                       returned_axis_keep_dimensions[i],
                                       returned_axis_remove_dimensions[i]);
        ck_assert_ptr_null(error);
        error_destroy(error);
        error = NULL;

        for (uint64_t j = 0; j < length_keep_dimension; j++)
        {
            ck_assert_uint_eq(returned_axis_keep_dimensions[i][j], expected_axis_keep_dimensions[i][j]);
        }

        for (uint64_t j = 0; j < length_remove_dimension; j++)
        {
            ck_assert_uint_eq(returned_axis_remove_dimensions[i][j], expected_axis_remove_dimensions[i][j]);
        }
    }
}
END_TEST

START_TEST(test_slice_shape)
{
    uint64_t number_of_cases = 2;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {2, 2},
        (uint64_t[]) {9, 8, 7, 6, 5},
    };

    uint64_t original_ranks[] = {
        2,
        5,
    };

    uint64_t *expected_slice_shapes[] = {
        (uint64_t[]) {1, 1},
        (uint64_t[]) {3, 5, 6, 1, 2},
    };

    uint64_t slice_ranks[] = {
        2,
        5,
    };

    uint64_t *arguments[] = {
        (uint64_t[]) {1, 2, 1, 2},
        (uint64_t[]) {2, 5, 0, 5, 1, 7, 0, 1, 2, 4},
    };

    uint64_t length[] = {
        4,
        10,
    };

    uint64_t returned_slice_shapes[number_of_cases][MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = slice_shape(original_shapes[i],
                            original_ranks[i],
                            returned_slice_shapes[i],
                            slice_ranks[i],
                            arguments[i],
                            length[i]);
        ck_assert_ptr_null(error);
        for (uint64_t j = 0; j < slice_ranks[i]; j++)
        {
            ck_assert_uint_eq(returned_slice_shapes[i][j], expected_slice_shapes[i][j]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_slice_offset)
{
    uint64_t number_of_cases = 1;

    uint64_t *original_strides[] = {
        (uint64_t[]) {6, 3, 1},
    };

    uint64_t original_ranks[] = {
        3,
    };

    uint64_t expected_offset[] = {
        6
    };

    uint64_t *arguments[] = {
        (uint64_t[]) {1, 3, 0, 2, 0, 1},
    };

    uint64_t length[] = {
        6
    };

    uint64_t returned_offset[number_of_cases];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = slice_offset(original_strides[i],
                             original_ranks[i],
                             &returned_offset[i],
                             arguments[i],
                             length[i]);
        ck_assert_ptr_null(error);
        ck_assert_uint_eq(returned_offset[i], expected_offset[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reverse_slice)
{
    uint64_t number_of_cases = 1;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {2, 2},
    };

    uint64_t original_ranks[] = {
        2,
    };

    uint64_t *arguments[] = {
        (uint64_t[]) {1, 2, 1, 2},
    };

    uint64_t length[] = {
        4
    };

    uint64_t *expected_new_arguments[] = {
        (uint64_t[]) {1, 0, 1, 0},
    };

    uint64_t new_length[] = {
        4,
    };

    uint64_t returned_new_arguments[number_of_cases][2 * MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = reverse_slice(original_shapes[i],
                              original_ranks[i],
                              arguments[i],
                              length[i],
                              returned_new_arguments[i],
                              new_length[i]);
        ck_assert_ptr_null(error);
        for (uint64_t j = 0; j < new_length[i]; j++)
        {
            ck_assert_uint_eq(returned_new_arguments[i][j], expected_new_arguments[i][j]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_padding)
{
    uint64_t number_of_cases = 2;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {2, 2},
        (uint64_t[]) {9, 8, 7, 6, 5},
    };

    uint64_t original_ranks[] = {
        2,
        5,
    };

    uint64_t *expected_padding_shapes[] = {
        (uint64_t[]) {3, 4},
        (uint64_t[]) {11, 9, 10, 10, 5},
    };

    uint64_t padding_ranks[] = {
        2,
        5,
    };

    uint64_t *arguments[] = {
        (uint64_t[]) {0, 1, 1, 1},
        (uint64_t[]) {0, 2, 1, 0, 1, 2, 2, 2, 0, 0},
    };

    uint64_t length[] = {
        4,
        10,
    };

    uint64_t returned_padding_shapes[number_of_cases][MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = padding(original_shapes[i],
                        original_ranks[i],
                        returned_padding_shapes[i],
                        padding_ranks[i],
                        arguments[i],
                        length[i]);
        ck_assert_ptr_null(error);
        for (uint64_t j = 0; j < padding_ranks[i]; j++)
        {
            ck_assert_uint_eq(returned_padding_shapes[i][j], expected_padding_shapes[i][j]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reverse_padding)
{
    uint64_t number_of_cases = 1;

    uint64_t *original_shapes[] = {
        (uint64_t[]) {2, 2},
    };

    uint64_t original_ranks[] = {
        2,
    };

    uint64_t *arguments[] = {
        (uint64_t[]) {0, 1, 1, 0},
    };

    uint64_t length[] = {
        4,
    };

    uint64_t *expected_new_arguments[] = {
        (uint64_t[]) {0, 2, 1, 3}
    };

    uint64_t new_length[] = {
        4
    };

    uint64_t returned_new_arguments[number_of_cases][2 * MAX_RANK];

    for (uint64_t i = 0; i < number_of_cases; i++)
    {
        error = reverse_padding(original_shapes[i],
                                original_ranks[i],
                                arguments[i],
                                length[i],
                                returned_new_arguments[i],
                                new_length[i]);
        ck_assert_ptr_null(error);
        for (uint64_t j = 0; j < new_length[i]; j++)
        {
            ck_assert_uint_eq(returned_new_arguments[i][j], expected_new_arguments[i][j]);
        }
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
    tcase_add_test(tc, test_reduce);
    tcase_add_test(tc, test_reduce_error);
    tcase_add_test(tc, test_shapes_equal);
    tcase_add_test(tc, test_shapes_size);
    tcase_add_test(tc, test_broadcast_strides);
    tcase_add_test(tc, test_broadcast_shapes);
    tcase_add_test(tc, test_is_broadcastable);
    tcase_add_test(tc, test_reverse_broadcast_length);
    tcase_add_test(tc, test_reverse_broadcast_axis);
    tcase_add_test(tc, test_slice_shape);
    tcase_add_test(tc, test_slice_offset);
    tcase_add_test(tc, test_reverse_slice);
    tcase_add_test(tc, test_padding);
    tcase_add_test(tc, test_reverse_padding);
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
