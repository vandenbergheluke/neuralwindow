#include <check.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>

nw_error_t *error;
view_t *view;

void setup(void)
{
    error = NULL;
    view = NULL;
}

void teardown(void)
{
    error_print(error);
    error_destroy(error);
    view_destroy(view);
}

START_TEST(test_view_create_error)
{
    int64_t number_of_cases = 9;

    int64_t offsets[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int64_t ranks[] = {1, 1, 1, MAX_RANK + 1, 5, 5, 5, 5, 5};
    int64_t *shapes[] = {
        (int64_t[]) {1},
        NULL,
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1, 2, 0, 4, 5},
        (int64_t[]) {1, 0, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 0, 5},
        (int64_t[]) {1, 2, 3, 4, 0},
        (int64_t[]) {0, 2, 3, 4, 5},
    };

    int64_t *strides[] = {
        (int64_t[]) {1},
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
    };

    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_SHAPE_CONFLICT,
        ERROR_SHAPE_CONFLICT,
        ERROR_SHAPE_CONFLICT,
        ERROR_SHAPE_CONFLICT,
        ERROR_SHAPE_CONFLICT,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
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

        // Teardown
        error_destroy(error);
        view_destroy(view);
        error = NULL;
        view = NULL;
    }
}
END_TEST

START_TEST(test_view_create)
{
    int64_t number_of_cases = 6;

    int64_t offsets[] = {0, 0, 0, 0, 0, 0};
    int64_t ranks[] = {0, 1, 2, 3, 4, 5};
    int64_t *shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {1, 2},
        (int64_t[]) {2, 2, 3},
        (int64_t[]) {1, 2, 3, 4},
        (int64_t[]) {1, 2, 3, 1, 5},
    };

    int64_t *strides[] = {
        (int64_t[]) {},
        (int64_t[]) {1},
        NULL,
        NULL,
        (int64_t[]) {24, 12, 4, 1},
        NULL,
    };

    int64_t *expected_strides[] = {
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {0, 1},
        (int64_t[]) {6, 3, 1},
        (int64_t[]) {24, 12, 4, 1},
        (int64_t[]) {0, 15, 5, 0, 1},
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = view_create(&view, offsets[i], ranks[i], shapes[i], strides[i]);
        ck_assert_ptr_null(error);

        // Ranks and offsets must be the same as arguments
        ck_assert_int_eq(view->offset, offsets[i]);
        ck_assert_int_eq(view->rank, ranks[i]);

        // Ensure the shape and strides are not NULL.
        ck_assert_ptr_nonnull(view->shape);
        ck_assert_ptr_nonnull(view->strides);

        // Shapes and strides need to be copied and not directly assigned
        ck_assert_ptr_ne(view->shape, shapes[i]);
        ck_assert_ptr_ne(view->strides, expected_strides[i]);

        // Compare shape and strides with expected
        for (int64_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_int_eq(view->shape[j], shapes[i][j]);
            ck_assert_int_eq(view->strides[j], expected_strides[i][j]);
        }

        // Destroy view
        view_destroy(view);
        view = NULL;
    }
}
END_TEST

START_TEST(test_is_contiguous)
{
    ck_assert(is_contiguous((int64_t[]) {2, 2, 3}, 3, (int64_t[]) {6, 3, 1}, 0));
    ck_assert(!is_contiguous(NULL, 3, (int64_t[]) {1, 2, 3}, 0));
    ck_assert(!is_contiguous((int64_t[]) {1, 2, 3}, 3, NULL, 0));
    ck_assert(!is_contiguous(NULL, 3, NULL, 0));
    ck_assert(is_contiguous((int64_t[]) {}, 0, (int64_t[]) {}, 0));
    ck_assert(!is_contiguous(NULL, 0, NULL, 0));
    ck_assert(is_contiguous((int64_t[]) {1}, 1, (int64_t[]) {1}, 0));
    ck_assert(is_contiguous((int64_t[]) {1}, 1, (int64_t[]) {0}, 0));
    ck_assert(is_contiguous((int64_t[]) {1, 2, 1, 5}, 4, (int64_t[]) {0, 5, 5, 1}, 0));
    ck_assert(is_contiguous((int64_t[]) {1, 2, 1, 5}, 4, (int64_t[]) {10, 5, 0, 1}, 0));
    ck_assert(is_contiguous((int64_t[]) {1, 2, 1, 5}, 4, (int64_t[]) {0, 5, 0, 1}, 0));
    ck_assert(is_contiguous((int64_t[]) {5, 1, 2, 1, 5}, 5, (int64_t[]) {10, 0, 5, 0, 1}, 0));
    ck_assert(is_contiguous((int64_t[]) {1, 2, 3, 4, 5}, 5, (int64_t[]) {120, 60, 20, 5, 1}, 0));
    ck_assert(is_contiguous((int64_t[]) {1, 2, 3, 4, 5}, 5, (int64_t[]) {0, 60, 20, 5, 1}, 0));
    ck_assert(!is_contiguous((int64_t[]) {1, 2, 3, 4, 5}, 5, (int64_t[]) {0, 60, 20, 5, 1}, 10));
}
END_TEST

START_TEST(test_strides_from_shape)
{
    int64_t number_of_cases = 9;
    int64_t *shapes[] = {
        (int64_t[]) {2, 3, 4, 5},
        (int64_t[]) {1, 10}, 
        (int64_t[]) {2, 1, 1},
        (int64_t[]) {10}, 
        (int64_t[]) {10, 1, 2, 5},
        (int64_t[]) {2, 2, 3},
        (int64_t[]) {},
        (int64_t[]) {10, 1, 2, 5, 1},
        (int64_t[]) {1, 2, 3, 4, 5},
    };
    int64_t *expected_strides[] = {
        (int64_t[]) {60, 20, 5, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {1, 0, 0},
        (int64_t[]) {1},
        (int64_t[]) {10, 0, 5, 1},
        (int64_t[]) {6, 3, 1},
        (int64_t[]) {},
        (int64_t[]) {10, 0, 5, 1, 0},
        (int64_t[]) {0, 60, 20, 5, 1},
    };
    int64_t returned_strides[number_of_cases][MAX_RANK];
    int64_t ranks[] = { 4, 2, 3, 1, 4, 3, 0, 5, 5 };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = strides_from_shape(returned_strides[i], shapes[i], ranks[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_int_eq(expected_strides[i][j], returned_strides[i][j]);
        }
    }
}
END_TEST

START_TEST(test_strides_from_shape_error)
{
    int64_t number_of_cases = 7;
    int64_t *shapes[] = {
        NULL,
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {0},
        (int64_t[]) {1, 2, 3, 4, 0},
        (int64_t[]) {1, 2, 0, 4, 5},
    };
    nw_error_type_t expected_error_type[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_SHAPE_CONFLICT,
        ERROR_SHAPE_CONFLICT,
        ERROR_SHAPE_CONFLICT,
    };
    int64_t *returned_strides[] = {
        NULL,
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
    };
    int64_t ranks[] = { 1, 1, 1, MAX_RANK + 1, 1, 5, 5};

    for (int64_t i = 0; i < number_of_cases; i++)
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
    int64_t number_of_cases = 8;

    int64_t *axis[] = {
        (int64_t[]) {0},
        (int64_t[]) {1, 0},
        (int64_t[]) {2, 1, 0},
        (int64_t[]) {1, 2, 0, 3},
        (int64_t[]) {2, 0, 1},
        (int64_t[]) {0, 1, 3, 2},
        (int64_t[]) {},
        (int64_t[]) {4, 2, 3, 0, 1},
    };

    int64_t returned_axis[number_of_cases][MAX_RANK];
    int64_t *expected_axis[] = {
        (int64_t[]) {0},
        (int64_t[]) {1, 0},
        (int64_t[]) {2, 1, 0},
        (int64_t[]) {2, 0, 1, 3},
        (int64_t[]) {1, 2, 0},
        (int64_t[]) {0, 1, 3, 2},
        (int64_t[]) {},
        (int64_t[]) {3, 4, 1, 2, 0},
    };
    int64_t ranks[] = { 1, 2, 3, 4, 3, 4, 0, 5 };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = reverse_permute(axis[i], ranks[i], returned_axis[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_int_eq(expected_axis[i][j], returned_axis[i][j]);
        }
    }
}
END_TEST

START_TEST(test_reverse_permute_error)
{
    int64_t number_of_cases = 8;

    int64_t *axis[] = {
        NULL,
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {0, 1, 2, 3, 4, 5},
        (int64_t[]) {1},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 5, 4, 0},
        (int64_t[]) {1, 2, 5, 0, 0},
    };
    int64_t *returned_axis[] = {
        (int64_t[]) {1},
        NULL,
        NULL,
        (int64_t[]) {0, 1},
        (int64_t[]) {1},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 5, 4, 0},
        (int64_t[]) {1, 2, 5, 4, 0},
    };
    int64_t ranks[] = { 1, 1, 1, MAX_RANK + 1, 1, 5, 5, 5};
    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_UNIQUE,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
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
    int64_t number_of_cases = 7;

    int64_t *original_shapes[] = {
        (int64_t[]) {1},
        (int64_t[]) {5, 3},
        (int64_t[]) {3, 2, 1},
        (int64_t[]) {2, 4, 3, 1},
        (int64_t[]) {2, 2, 2},
        (int64_t[]) {1, 2, 3, 5, 4},
        (int64_t[]) {},
    };

    int64_t *original_strides[] = {
        (int64_t[]) {0},
        (int64_t[]) {3, 1},
        (int64_t[]) {2, 1, 1},
        (int64_t[]) {12, 3, 1, 1},
        (int64_t[]) {4, 2, 1},
        (int64_t[]) {0, 60, 20, 4, 1},
        (int64_t[]) {},
    };

    int64_t *axis[] = {
        (int64_t[]) {0},
        (int64_t[]) {1, 0},
        (int64_t[]) {2, 1, 0},
        (int64_t[]) {1, 2, 0, 3},
        (int64_t[]) {2, 0, 1},
        (int64_t[]) {4, 2, 3, 0, 1},
        (int64_t[]) {},
    };

    int64_t *expected_shapes[] = {
        (int64_t[]) {1},
        (int64_t[]) {3, 5},
        (int64_t[]) {1, 2, 3},
        (int64_t[]) {4, 3, 2, 1},
        (int64_t[]) {2, 2, 2},
        (int64_t[]) {4, 3, 5, 1, 2},
        (int64_t[]) {},
    };

    int64_t *expected_strides[] = {
        (int64_t[]) {0},
        (int64_t[]) {1, 3},
        (int64_t[]) {1, 1, 2},
        (int64_t[]) {3, 1, 12, 1},
        (int64_t[]) {1, 4, 2},
        (int64_t[]) {1, 20, 4, 0, 60},
        (int64_t[]) {},
    };

    int64_t lengths[] = { 1, 2, 3, 4, 3, 5, 0 };

    int64_t returned_shapes[number_of_cases][MAX_RANK];
    int64_t returned_strides[number_of_cases][MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = permute(original_shapes[i],
                        original_strides[i],
                        returned_shapes[i],
                        returned_strides[i],
                        axis[i],
                        lengths[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < lengths[i]; j++)
        {
            ck_assert_int_eq(expected_shapes[i][j], returned_shapes[i][j]);
            ck_assert_int_eq(expected_strides[i][j], returned_strides[i][j]);
        }
    }
}
END_TEST

START_TEST(test_permute_error)
{
    int64_t number_of_cases = 11;

    int64_t *original_shapes[] = {
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {0},
        (int64_t[]) {1},
        (int64_t[]) {1, 2},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
    };

    int64_t *original_strides[] = {
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {2, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
    };

    int64_t *axis[] = {
        (int64_t[]) {0},
        (int64_t[]) {0},
        NULL,
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0, 1, 2, 3, 4, 5},
        (int64_t[]) {0, 2},
        (int64_t[]) {5, 4, 0, 1, 2},
        (int64_t[]) {2, 4, 3, 1, 6},
        (int64_t[]) {2, 4, 3, 1, 1},
    };

    int64_t *permuted_shapes[] = {
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1, 2},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
    };

    int64_t *permuted_strides[] = {
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1, 2},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
    };

    int64_t axis_lengths[] = { 1, 1, 1, 1, 1, 1, MAX_RANK + 1, 2, 5, 5, 5};

    nw_error_type_t expected_error_type[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_SHAPE_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_PERMUTE, 
        ERROR_PERMUTE, 
        ERROR_PERMUTE, 
        ERROR_UNIQUE,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = permute(original_shapes[i],
                        original_strides[i],
                        permuted_shapes[i],
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
    int64_t number_of_cases = 13;

    int64_t *reduced_shapes[] = {
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {1, 2},
        (int64_t[]) {3, 2},
        (int64_t[]) {3, 2, 1},
        (int64_t[]) {7, 6, 4, 8},
        (int64_t[]) {2, 2, 2, 2},
        (int64_t[]) {7, 6, 4, 8, 9},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
    };

    int64_t reduced_ranks[] = { 1, 1, 2, 2, 3, 4, 4, 5, 0, 0, 0, 0, 0 };

    int64_t *reduced_strides[] = {
        (int64_t[]) {0},
        (int64_t[]) {1},
        (int64_t[]) {0, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1, 0},
        (int64_t[]) {0, 24, 8, 1},
        (int64_t[]) {0, 0, 0, 0},
        (int64_t[]) {0, 0, 72, 9, 1},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
    };

    int64_t *expected_recovered_shapes[] = {
        (int64_t[]) {1},
        (int64_t[]) {1, 2, 1},
        (int64_t[]) {1, 1, 2, 1, 1},
        (int64_t[]) {3, 1, 2},
        (int64_t[]) {1, 1, 3, 2, 1},
        (int64_t[]) {7, 6, 4, 1, 8},
        (int64_t[]) {2, 2, 1, 2, 2},
        (int64_t[]) {7, 6, 4, 8, 9},
        (int64_t[]) {1},
        (int64_t[]) {1, 1},
        (int64_t[]) {1, 1, 1},
        (int64_t[]) {1, 1, 1, 1},
        (int64_t[]) {1, 1, 1, 1, 1},
    };

    int64_t recovered_ranks[] = { 1, 3, 5, 3, 5, 5, 5, 5, 1, 2, 3, 4, 5};

    int64_t *expected_recovered_strides[] = {
        (int64_t[]) {0},
        (int64_t[]) {0, 1, 0},
        (int64_t[]) {0, 0, 1, 0, 0},
        (int64_t[]) {2, 0, 1},
        (int64_t[]) {0, 0, 2, 1, 0},
        (int64_t[]) {0, 24, 8, 0, 1},
        (int64_t[]) {0, 0, 0, 0, 0},
        (int64_t[]) {0, 0, 72, 9, 1},
        (int64_t[]) {0},
        (int64_t[]) {0, 0},
        (int64_t[]) {0, 0, 0},
        (int64_t[]) {0, 0, 0, 0},
        (int64_t[]) {0, 0, 0, 0, 0},
    };

    int64_t *axis[] = {
        (int64_t[]) {},
        (int64_t[]) {0, 2},
        (int64_t[]) {0, 3, 4},
        (int64_t[]) {1},
        (int64_t[]) {0, 1},
        (int64_t[]) {3},
        (int64_t[]) {2},
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {0, 1},
        (int64_t[]) {2, 0, 1},
        (int64_t[]) {0, 3, 1, 2},
        (int64_t[]) {0, 1, 4, 3, 2},
    };

    int64_t lengths[] = { 0, 2, 3, 1, 2, 1, 1, 0, 1, 2, 3, 4, 5 };
    
    int64_t returned_recovered_shapes[number_of_cases][MAX_RANK];
    int64_t returned_recovered_strides[number_of_cases][MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = reduce_recover_dimensions(reduced_shapes[i],
                                          reduced_ranks[i],
                                          reduced_strides[i],
                                          returned_recovered_shapes[i],
                                          recovered_ranks[i],
                                          returned_recovered_strides[i],
                                          axis[i],
                                          lengths[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < recovered_ranks[i]; j++)
        {
            ck_assert_int_eq(returned_recovered_shapes[i][j], expected_recovered_shapes[i][j]);
            ck_assert_int_eq(returned_recovered_strides[i][j], expected_recovered_strides[i][j]);
        }
    }
}
END_TEST

START_TEST(test_reduce_recover_dimension_error)
{
    int64_t number_of_cases = 11;

    int64_t *reduced_shapes[] = {
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {0},
    };

    int64_t reduced_ranks[] = {
        1,
        1,
        1,
        1,
        1,
        MAX_RANK + 1,
        1,
        1,
        2,
        1,
        1,
    };

    int64_t *reduced_strides[] = {
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
    };

    int64_t *recovered_shapes[] = {
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        NULL,
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
    };

    int64_t recovered_ranks[] = {
        2,
        2,
        2,
        2,
        2,
        2,
        MAX_RANK + 1,
        2,
        2,
        2,
        2,
    };

    int64_t *recovered_strides[] = {
        (int64_t[]) {2, 0},
        (int64_t[]) {2, 0},
        (int64_t[]) {2, 0},
        NULL,
        (int64_t[]) {2, 0},
        (int64_t[]) {2, 0},
        (int64_t[]) {2, 0},
        (int64_t[]) {2, 0},
        (int64_t[]) {2, 0},
        (int64_t[]) {2, 0},
        (int64_t[]) {2, 0},
    };

    int64_t *axis[] = {
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {0, 1, 2, 3, 4, 5, 6},
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {1},
    };

    int64_t lengths[] = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        MAX_RANK + 1,
        1,
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
        ERROR_RANK_CONFLICT,
        ERROR_SHAPE_CONFLICT,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = reduce_recover_dimensions(reduced_shapes[i],
                                          reduced_ranks[i],
                                          reduced_strides[i],
                                          recovered_shapes[i],
                                          recovered_ranks[i],
                                          recovered_strides[i],
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
    int64_t number_of_cases = 34;

    int64_t *original_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {2},
        (int64_t[]) {1, 2},
        (int64_t[]) {1, 2},
        (int64_t[]) {1, 2},
        (int64_t[]) {1, 2},
        (int64_t[]) {1, 2},
        (int64_t[]) {1, 2},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {2, 3, 4},
        (int64_t[]) {3, 2, 4},
        (int64_t[]) {5, 1, 3, 2, 4},
        (int64_t[]) {5, 1, 3, 2, 4},
        (int64_t[]) {5, 1, 3, 2, 4},
        (int64_t[]) {5, 1, 3, 2, 4},
        (int64_t[]) {5, 1, 3, 2, 4},
        (int64_t[]) {5, 1, 3, 2, 4},
    };

    int64_t original_ranks[] = {
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        5,
        5,
        5,
        5,
        5,
        5,
};

    int64_t *original_strides[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 4, 0},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 0, 1},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {0, 4, 1},
        (int64_t[]) {12, 4, 0},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 0, 1},
        (int64_t[]) {4, 12, 1},
        (int64_t[]) {1, 120, 20, 60, 5},
        (int64_t[]) {1, 120, 20, 60, 5},
        (int64_t[]) {1, 120, 0, 60, 5},
        (int64_t[]) {1, 120, 20, 60, 0},
        (int64_t[]) {0, 120, 20, 60, 5},
        (int64_t[]) {1, 0, 20, 0, 5},
    };

    int64_t *expected_reduced_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {},
        (int64_t[]) {1, 1},
        (int64_t[]) {2},
        (int64_t[]) {1, 2},
        (int64_t[]) {1},
        (int64_t[]) {1, 1},
        (int64_t[]) {},
        (int64_t[]) {1, 1, 1},
        (int64_t[]) {4},
        (int64_t[]) {1, 1, 4},
        (int64_t[]) {3},
        (int64_t[]) {1, 3, 1},
        (int64_t[]) {2},
        (int64_t[]) {2, 1, 1},
        (int64_t[]) {2, 3},
        (int64_t[]) {2, 3, 1},
        (int64_t[]) {2, 4},
        (int64_t[]) {2, 1, 4},
        (int64_t[]) {3, 4},
        (int64_t[]) {1, 3, 4},
        (int64_t[]) {3, 4},
        (int64_t[]) {2, 4},
        (int64_t[]) {5, 3, 4},
        (int64_t[]) {5, 1, 3, 1, 4},
        (int64_t[]) {5, 3, 4},
        (int64_t[]) {5, 3, 4},
        (int64_t[]) {5, 3, 4},
        (int64_t[]) {5, 1, 2, 4},
    };

    int64_t reduced_ranks[] = {
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        2,
        1,
        2,
        1,
        2,
        0,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
        2,
        3,
        2,
        3,
        2,
        3,
        2,
        2,
        3,
        5,
        3,
        3,
        3,
        4,
};

    int64_t *expected_reduced_strides[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {},
        (int64_t[]) {0, 0},
        (int64_t[]) {1},
        (int64_t[]) {0, 1},
        (int64_t[]) {0},
        (int64_t[]) {0, 0},
        (int64_t[]) {},
        (int64_t[]) {0, 0, 0},
        (int64_t[]) {1},
        (int64_t[]) {0, 0, 0},
        (int64_t[]) {1},
        (int64_t[]) {0, 1, 0},
        (int64_t[]) {1},
        (int64_t[]) {1, 0, 0},
        (int64_t[]) {1, 0},
        (int64_t[]) {3, 1, 0},
        (int64_t[]) {4, 1},
        (int64_t[]) {0, 0, 1},
        (int64_t[]) {1, 0},
        (int64_t[]) {0, 4, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {4, 1},
        (int64_t[]) {12, 4, 1},
        (int64_t[]) {12, 0, 4, 0, 1},
        (int64_t[]) {4, 0, 1},
        (int64_t[]) {3, 1, 0},
        (int64_t[]) {0, 4, 1},
        (int64_t[]) {4, 0, 0, 1},
    };

    int64_t *axis[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {0, 1, 2},
        (int64_t[]) {0, 1, 2},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, 2},
        (int64_t[]) {0, 2},
        (int64_t[]) {1, 2},
        (int64_t[]) {1, 2},
        (int64_t[]) {2},
        (int64_t[]) {2},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {1, 3},
        (int64_t[]) {1, 3},
        (int64_t[]) {1, 3},
        (int64_t[]) {1, 3},
        (int64_t[]) {1, 3},
        (int64_t[]) {2},
    };

    int64_t lengths[] = {
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
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
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        true,
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
    };
    
    int64_t returned_reduced_shapes[number_of_cases][MAX_RANK];
    int64_t returned_reduced_strides[number_of_cases][MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
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
        for (int64_t j = 0; j < reduced_ranks[i]; j++)
        {
            ck_assert_int_eq(returned_reduced_shapes[i][j], expected_reduced_shapes[i][j]);
            ck_assert_int_eq(returned_reduced_strides[i][j], expected_reduced_strides[i][j]);
        }
    }
}
END_TEST

START_TEST(test_reduce_error)
{
    int64_t number_of_cases = 12;

    int64_t *original_shapes[] = {
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        NULL,
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1, 2},
    };

    int64_t original_ranks[] = {
        2,
        2,
        2,
        2,
        2,
        2,
        MAX_RANK + 1,
        2,
        2,
        2,
        2,
        3,
    };

    int64_t *original_strides[] = {
        (int64_t[]) {1, 0},
        (int64_t[]) {1, 0},
        (int64_t[]) {1, 0},
        NULL,
        (int64_t[]) {1, 0},
        (int64_t[]) {1, 0},
        (int64_t[]) {1, 0},
        (int64_t[]) {1, 0},
        (int64_t[]) {1, 0},
        (int64_t[]) {1, 0},
        (int64_t[]) {1, 0},
        (int64_t[]) {1, 0, 0},
    };

    int64_t *reduced_shapes[] = {
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
    };

    int64_t reduced_ranks[] = {
        1,
        1,
        1,
        1,
        1,
        MAX_RANK + 1,
        1,
        1,
        2,
        1,
        1,
        1,
    };

    int64_t *reduced_strides[] = {
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
    };

    int64_t *axis[] = {
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {0, 1, 2, 3, 4, 5, 6},
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {0},
        (int64_t[]) {0, 0},
    };

    int64_t lengths[] = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        MAX_RANK + 1,
        1,
        1,
        1,
        2,
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
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_UNIQUE,
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
        false,
        true,
        false,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
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
    ck_assert(!shapes_equal(NULL, 1, (int64_t[]) {1}, 1));
    ck_assert(!shapes_equal((int64_t[]) {1}, 1, NULL, 1));
    ck_assert(!shapes_equal(NULL, 1, NULL, 1));
    ck_assert(!shapes_equal((int64_t[]) {1}, 2, (int64_t[]) {1}, 1));
    ck_assert(!shapes_equal((int64_t[]) {1}, 1, (int64_t[]) {1}, 2));
    ck_assert(shapes_equal((int64_t[]) {1}, 1, (int64_t[]) {1}, 1));
    ck_assert(shapes_equal((int64_t[]) {1, 2, 3}, 3, (int64_t[]) {1, 2, 3}, 3));
    ck_assert(shapes_equal((int64_t[]) {1, 2, 3, 4, 5}, 5, (int64_t[]) {1, 2, 3, 4, 5}, 5));
    ck_assert(!shapes_equal((int64_t[]) {1, 2, 4}, 3, (int64_t[]) {1, 2, 3}, 3));
    ck_assert(!shapes_equal((int64_t[]) {2, 2, 4}, 3, (int64_t[]) {1, 2, 3}, 3));
    ck_assert(!shapes_equal((int64_t[]) {2, 2, 4}, 3, (int64_t[]) {2, 3, 3}, 3));
    ck_assert(shapes_equal((int64_t[]) {}, 0, (int64_t[]) {}, 0));
}
END_TEST

START_TEST(test_shapes_size)
{
    ck_assert_int_eq(shape_size(NULL, 0), 1);
    ck_assert_int_eq(shape_size((int64_t[]){}, 0), 1);
    ck_assert_int_eq(shape_size((int64_t[]) {1}, 1), 1);
    ck_assert_int_eq(shape_size((int64_t[]) {2}, 1), 2);
    ck_assert_int_eq(shape_size((int64_t[]) {1, 2, 1}, 3), 2);
    ck_assert_int_eq(shape_size((int64_t[]) {1, 2, 3}, 3), 6);
    ck_assert_int_eq(shape_size((int64_t[]) {4, 2, 3}, 3), 24);
    ck_assert_int_eq(shape_size((int64_t[]) {5, 4, 3, 2, 1}, 5), 120);
}
END_TEST

START_TEST(test_broadcast_strides)
{
    int64_t number_of_cases = 14;

    int64_t *original_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {5, 1, 3, 2},
        (int64_t[]) {5, 1, 3, 2},
        (int64_t[]) {1, 1, 1, 1, 1},
        (int64_t[]) {1, 1, 1, 1, 1},
        (int64_t[]) {6, 5, 4, 3, 2},
        (int64_t[]) {1, 5, 1, 3, 1},
        (int64_t[]) {6, 1, 4, 1, 2},
        (int64_t[]) {1, 5, 1, 3, 1},
        (int64_t[]) {6, 1, 4, 1, 2},
        (int64_t[]) {4, 1, 2},
    };

    int64_t original_ranks[] = {
        0,
        0,
        1,
        1,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        3,
    };

    int64_t *original_strides[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {1},
        (int64_t[]) {6, 0, 2, 1},
        (int64_t[]) {6, 0, 2, 0},
        (int64_t[]) {0, 0, 0, 0, 0},
        (int64_t[]) {1, 1, 1, 1, 1},
        (int64_t[]) {120, 24, 6, 2, 1},
        (int64_t[]) {1, 3, 1, 1, 1},
        (int64_t[]) {8, 1, 2, 1, 1},
        (int64_t[]) {0, 3, 0, 1, 0},
        (int64_t[]) {8, 0, 2, 0, 1},
        (int64_t[]) {2, 0, 1},
    };

    int64_t *broadcasted_shapes[] = {
        (int64_t[]) {1},
        (int64_t[]) {10, 9, 8, 7, 6},
        (int64_t[]) {10, 9, 8, 7, 6},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2, 1},
        (int64_t[]) {5, 4, 3, 2, 1},
        (int64_t[]) {6, 5, 4, 3, 2},
        (int64_t[]) {6, 5, 4, 3, 2},
        (int64_t[]) {6, 5, 4, 3, 2},
        (int64_t[]) {6, 5, 4, 3, 2},
        (int64_t[]) {6, 5, 4, 3, 2},
        (int64_t[]) {6, 5, 4, 3, 2},
    };

    int64_t broadcasted_ranks[] = {
        1,
        5,
        5,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
    };

    int64_t *expected_broadcasted_strides[] = {
        (int64_t[]) {0},
        (int64_t[]) {0, 0, 0, 0, 0},
        (int64_t[]) {0, 0, 0, 0, 0},
        (int64_t[]) {0, 0, 0, 1},
        (int64_t[]) {6, 0, 2, 1},
        (int64_t[]) {6, 0, 2, 0},
        (int64_t[]) {0, 0, 0, 0, 0},
        (int64_t[]) {0, 0, 0, 0, 0},
        (int64_t[]) {120, 24, 6, 2, 1},
        (int64_t[]) {0, 3, 0, 1, 0},
        (int64_t[]) {8, 0, 2, 0, 1},
        (int64_t[]) {0, 3, 0, 1, 0},
        (int64_t[]) {8, 0, 2, 0, 1},
        (int64_t[]) {0, 0, 2, 0, 1},
    };

    int64_t returned_broadcasted_strides[number_of_cases][MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = broadcast_strides(original_shapes[i],
                                  original_ranks[i],
                                  original_strides[i],
                                  broadcasted_shapes[i],
                                  broadcasted_ranks[i],
                                  returned_broadcasted_strides[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < broadcasted_ranks[i]; j++)
        {
            ck_assert_int_eq(returned_broadcasted_strides[i][j],
                              expected_broadcasted_strides[i][j]);
        }
    }
}
END_TEST

START_TEST(test_broadcast_strides_error)
{
    int64_t number_of_cases = 7;

    int64_t *original_shapes[] = {
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1, 2, 3, 4, 5},
    };

    int64_t original_ranks[] = {
        0,
        0,
        0,
        0,
        MAX_RANK + 1,
        0,
        5,
    };

    int64_t *original_strides[] = {
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {120, 60, 20, 5, 1},
    };

    int64_t *broadcasted_shapes[] = {
        (int64_t[]) {1},
        (int64_t[]) {1},
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {5, 4, 3, 2, 1},
    };

    int64_t broadcasted_ranks[] = {
        1,
        1,
        1,
        1,
        1,
        MAX_RANK + 1,
        5,
    };

    int64_t *broadcasted_strides[] = {
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0},
        NULL,
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {24, 6, 2, 1, 1},
    };

    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_BROADCAST,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = broadcast_strides(original_shapes[i],
                                  original_ranks[i],
                                  original_strides[i],
                                  broadcasted_shapes[i],
                                  broadcasted_ranks[i],
                                  broadcasted_strides[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_broadcast_shapes)
{
    int64_t number_of_cases = 21;

    int64_t *x_original_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {256, 256, 3},
        (int64_t[]) {3},
        (int64_t[]) {8, 1, 6, 1},
        (int64_t[]) {7, 1, 5},
        (int64_t[]) {5, 4},
        (int64_t[]) {1},
        (int64_t[]) {5, 4},
        (int64_t[]) {4},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 1, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {3, 1},
        (int64_t[]) {1},
        (int64_t[]) {4, 4, 4, 4, 4},
    };

    int64_t x_original_ranks[] = {
        0,
        0,
        1,
        0,
        5,
        3,
        1,
        4,
        3,
        2,
        1,
        2,
        1,
        3,
        3,
        3,
        2,
        3,
        2,
        1,
        5,
    };

    int64_t *y_original_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {},
        (int64_t[]) {3},
        (int64_t[]) {256, 256, 3},
        (int64_t[]) {7, 1, 5},
        (int64_t[]) {8, 1, 6, 1},
        (int64_t[]) {1},
        (int64_t[]) {5, 4},
        (int64_t[]) {4},
        (int64_t[]) {5, 4},
        (int64_t[]) {15, 1, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {3, 1},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {4, 4, 4, 4, 4},
        (int64_t[]) {1},
    };

    int64_t y_original_ranks[] = {
        0,
        1,
        0,
        5,
        0,
        1,
        3,
        3,
        4,
        1,
        2,
        1,
        2,
        3,
        3,
        2,
        3,
        2,
        3,
        5,
        1,
    };

    int64_t *expected_broadcasted_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {256, 256, 3},
        (int64_t[]) {256, 256, 3},
        (int64_t[]) {8, 7, 6, 5},
        (int64_t[]) {8, 7, 6, 5},
        (int64_t[]) {5, 4},
        (int64_t[]) {5, 4},
        (int64_t[]) {5, 4},
        (int64_t[]) {5, 4},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {4, 4, 4, 4, 4},
        (int64_t[]) {4, 4, 4, 4, 4},
    };

    int64_t broadcasted_ranks[] = {
        0,
        1,
        1,
        5,
        5,
        3,
        3,
        4,
        4,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        5,
        5,
    };

    int64_t returned_broadcasted_shapes[number_of_cases][MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = broadcast_shapes(x_original_shapes[i],
                                 x_original_ranks[i],
                                 y_original_shapes[i],
                                 y_original_ranks[i],
                                 returned_broadcasted_shapes[i],
                                 broadcasted_ranks[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < broadcasted_ranks[i]; j++)
        {
            ck_assert_int_eq(returned_broadcasted_shapes[i][j],
                              expected_broadcasted_shapes[i][j]);
        }
    }
}
END_TEST

START_TEST(test_broadcast_shapes_error)
{
    int64_t number_of_cases = 11;

    int64_t *x_original_shapes[] = {
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {3},
        (int64_t[]) {4},
        (int64_t[]) {2, 1},
        (int64_t[]) {8, 4, 3},
    };

    int64_t x_original_ranks[] = {
        0,
        0,
        0,
        MAX_RANK + 1,
        0,
        0,
        5,
        1,
        1,
        2,
        3,
    };

    int64_t *y_original_shapes[] = {
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {5, 4, 3, 2, 1},
        (int64_t[]) {4},
        (int64_t[]) {3},
        (int64_t[]) {8, 4, 3},
        (int64_t[]) {2, 1},
    };

    int64_t y_original_ranks[] = {
        0,
        0,
        0,
        0,
        MAX_RANK + 1,
        0,
        5,
        1,
        1,
        3,
        2,
    };

    int64_t *broadcasted_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {1, 2, 3},
        (int64_t[]) {1, 2, 3},
    };

    int64_t broadcasted_ranks[] = {
        0,
        0,
        0,
        0,
        0,
        1,
        5,
        1,
        1,
        3,
        3,
    };

    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_BROADCAST,
        ERROR_BROADCAST,
        ERROR_BROADCAST,
        ERROR_BROADCAST,
        ERROR_BROADCAST,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = broadcast_shapes(x_original_shapes[i],
                                 x_original_ranks[i],
                                 y_original_shapes[i],
                                 y_original_ranks[i],
                                 broadcasted_shapes[i],
                                 broadcasted_ranks[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_is_broadcastable)
{
    ck_assert(is_broadcastable((int64_t[]) {1}, 1, (int64_t[]) {1}, 1));
    ck_assert(is_broadcastable((int64_t[]) {1}, 1, (int64_t[]) {2, 1}, 2));
    ck_assert(!is_broadcastable((int64_t[]) {3, 1}, 2, (int64_t[]) {2, 1}, 2));
    ck_assert(is_broadcastable((int64_t[]) {2}, 1, (int64_t[]) {2, 2}, 2));
    ck_assert(!is_broadcastable(NULL, 1, (int64_t[]) {2, 2}, 2));
    ck_assert(!is_broadcastable((int64_t[]) {1}, 1, NULL, 1));
    ck_assert(!is_broadcastable(NULL, 1, NULL, 1));
    ck_assert(is_broadcastable((int64_t[]) {5, 1, 3}, 3, (int64_t[]) {7, 6, 5, 4, 3}, 5));
    ck_assert(!is_broadcastable((int64_t[]) {5, 2, 3}, 3, (int64_t[]) {7, 6, 5, 4, 3}, 5));
    ck_assert(is_broadcastable((int64_t[]) {}, 0, (int64_t[]) {7, 6, 5, 4, 3}, 5));
    ck_assert(is_broadcastable((int64_t[]) {}, 0, (int64_t[]) {}, 0));
    ck_assert(!is_broadcastable((int64_t[]) {3}, 1, (int64_t[]) {4}, 1));
    ck_assert(!is_broadcastable((int64_t[]) {2, 1}, 2, (int64_t[]) {8, 4, 3}, 3));
}
END_TEST

START_TEST(test_reduce_axis_length)
{
    int64_t number_of_cases = 17;

    int64_t *original_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {1},
        (int64_t[]) {3},
        (int64_t[]) {1},
        (int64_t[]) {8, 1, 6, 1},
        (int64_t[]) {7, 1, 5},
        (int64_t[]) {4},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 1, 5},
        (int64_t[]) {3, 5},
        (int64_t[]) {3, 1},
    };

    int64_t original_ranks[] = {
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        4,
        3,
        1,
        3,
        3,
        2,
        2,
    };

    int64_t *broadcasted_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {1, 1, 1, 1, 1},
        (int64_t[]) {6, 5, 4, 3, 2},
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {2},
        (int64_t[]) {5, 4},
        (int64_t[]) {256, 256, 3},
        (int64_t[]) {256, 256, 3},
        (int64_t[]) {8, 7, 6, 5},
        (int64_t[]) {8, 7, 6, 5},
        (int64_t[]) {5, 4},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
    };

    int64_t broadcasted_ranks[] = {
        0,
        1,
        5,
        5,
        1,
        1,
        1,
        2,
        3,
        3,
        4,
        4,
        2,
        3,
        3,
        3,
        3,
    };

    int64_t expected_length_keep_dimensions[] = {
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        2,
        1,
        0,
        0,
        1,
        0,
        1,
    };

    int64_t expected_length_remove_dimensions[] = {
        0,
        1,
        5,
        5,
        0,
        0,
        0,
        1,
        2,
        2,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
    };

    int64_t returned_length_keep_dimensions[number_of_cases];
    int64_t returned_length_remove_dimensions[number_of_cases];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = reduce_axis_length(original_shapes[i],
                                   original_ranks[i],
                                   broadcasted_shapes[i],
                                   broadcasted_ranks[i],
                                   &returned_length_keep_dimensions[i],
                                   &returned_length_remove_dimensions[i]);
        ck_assert_ptr_null(error);
        ck_assert_int_eq(returned_length_keep_dimensions[i],
                          expected_length_keep_dimensions[i]);
        ck_assert_int_eq(returned_length_remove_dimensions[i],
                          expected_length_remove_dimensions[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reduce_axis_length_error)
{
    int64_t number_of_cases = 8;

    int64_t *original_shapes[] = {
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {3},
        (int64_t[]) {2, 1},
    };

    int64_t original_ranks[] = {
        0,
        0,
        0,
        0,
        MAX_RANK + 1,
        0,
        1,
        2,
    };

    int64_t *broadcasted_shapes[] = {
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {4},
        (int64_t[]) {8, 4, 3},
    };

    int64_t broadcasted_ranks[] = {
        0,
        0,
        0,
        0,
        0,
        MAX_RANK + 1,
        1,
        3,
    };

    int64_t *length_keep_dimensions[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
    };

    int64_t *length_remove_dimensions[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
    };

    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_BROADCAST,
        ERROR_BROADCAST,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = reduce_axis_length(original_shapes[i],
                                   original_ranks[i],
                                   broadcasted_shapes[i],
                                   broadcasted_ranks[i],
                                   length_keep_dimensions[i],
                                   length_remove_dimensions[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

static int compare_uint64(const void * a, const void * b)
{
    return ( *((int64_t *) a) > *((int64_t *) b) );
}

START_TEST(test_reduce_axis)
{
    int64_t number_of_cases = 11;

    int64_t *original_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {5},
        (int64_t[]) {15, 1, 5},
        (int64_t[]) {3, 5},
        (int64_t[]) {3, 1},
        (int64_t[]) {8, 1, 6, 1},
        (int64_t[]) {7, 1, 5},
    };

    int64_t original_ranks[] = {
        0,
        0,
        0,
        0,
        1,
        1,
        3,
        2,
        2,
        4,
        3,
    };

    int64_t *broadcasted_shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {15, 3, 5},
        (int64_t[]) {8, 7, 6, 5},
        (int64_t[]) {8, 7, 6, 5},
    };

    int64_t broadcasted_ranks[] = {
        0,
        1,
        1,
        5,
        5,
        5,
        3,
        3,
        3,
        4,
        4,
    };

    int64_t *expected_axis_keep_dimensions[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {4},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {},
        (int64_t[]) {2},
        (int64_t[]) {1, 3},
        (int64_t[]) {2},
    };

    int64_t *expected_axis_remove_dimensions[] = {
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0, 1, 2, 3, 4},
        (int64_t[]) {0, 1, 2, 3},
        (int64_t[]) {0, 1, 2, 3},
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {},
        (int64_t[]) {0},
    };

    int64_t length_keep_dimensions[] = {
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        2,
        1,
    };

    int64_t length_remove_dimensions[] = {
        0,
        1,
        1,
        5,
        4,
        4,
        0,
        1,
        1,
        0,
        1,
    };

    int64_t returned_axis_keep_dimensions[number_of_cases][MAX_RANK];
    int64_t returned_axis_remove_dimensions[number_of_cases][MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = reduce_axis(original_shapes[i],
                            original_ranks[i],
                            broadcasted_shapes[i],
                            broadcasted_ranks[i],
                            returned_axis_keep_dimensions[i],
                            returned_axis_remove_dimensions[i]);
        ck_assert_ptr_null(error);
        qsort(returned_axis_keep_dimensions[i],
              (size_t) length_keep_dimensions[i],
              sizeof(int64_t),
              compare_uint64);
        qsort(returned_axis_remove_dimensions[i],
              (size_t) length_remove_dimensions[i],
              sizeof(int64_t),
              compare_uint64);
        for (int64_t j = 0; j < length_keep_dimensions[i]; j++)
        {
            ck_assert_int_eq(returned_axis_keep_dimensions[i][j],
                              expected_axis_keep_dimensions[i][j]);
        }

        for (int64_t j = 0; j < length_remove_dimensions[i]; j++)
        {
            ck_assert_int_eq(returned_axis_remove_dimensions[i][j],
                              expected_axis_remove_dimensions[i][j]);
        }
    }
}
END_TEST

START_TEST(test_reduce_axis_error)
{
    int64_t number_of_cases = 9;

    int64_t *original_shapes[] = {
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {3},
        (int64_t[]) {2, 1},
        (int64_t[]) {8, 3},
    };

    int64_t original_ranks[] = {
        0,
        0,
        0,
        0,
        MAX_RANK + 1,
        0,
        1,
        2,
        2,
    };

    int64_t *broadcasted_shapes[] = {
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {4},
        (int64_t[]) {8, 4, 3},
        (int64_t[]) {8, 4, 3},
    };

    int64_t broadcasted_ranks[] = {
        0,
        0,
        0,
        0,
        0,
        MAX_RANK + 1,
        1,
        3,
        3,
    };

    int64_t *axis_keep_dimensions[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
    };

    int64_t *axis_remove_dimensions[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {},
    };

    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK_CONFLICT,
        ERROR_RANK_CONFLICT,
        ERROR_BROADCAST,
        ERROR_BROADCAST,
        ERROR_BROADCAST,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = reduce_axis(original_shapes[i],
                            original_ranks[i],
                            broadcasted_shapes[i],
                            broadcasted_ranks[i],
                            axis_keep_dimensions[i],
                            axis_remove_dimensions[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_n_from_shape_and_strides)
{
    int64_t number_of_cases = 15;

    int64_t *shapes[] = {
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {2},
        (int64_t[]) {2, 1},
        (int64_t[]) {2, 1},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2},
        (int64_t[]) {5, 4, 3, 2},
    };

    int64_t ranks[] = {
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    };

    int64_t *strides[] = {
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {1},
        (int64_t[]) {0},
        (int64_t[]) {0, 0},
        (int64_t[]) {1, 0},
        (int64_t[]) {6, 0, 2, 1},
        (int64_t[]) {24, 6, 2, 1},
        (int64_t[]) {12, 3, 1, 0},
        (int64_t[]) {0, 6, 2, 1},
        (int64_t[]) {0, 3, 1, 0},
        (int64_t[]) {0, 0, 1, 0},
        (int64_t[]) {0, 0, 0, 0},
        (int64_t[]) {1, 0, 0, 0},
    };

    int64_t expected_n[] = {
        1,
        1,
        1,
        2,
        1,
        1,
        2,
        30,
        120,
        60,
        24,
        12,
        3,
        1,
        5,
    };
    
    int64_t returned_n[number_of_cases];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = n_from_shape_and_strides(shapes[i],
                                         strides[i],
                                         ranks[i],
                                         &returned_n[i]);
        ck_assert_ptr_null(error);
        ck_assert_int_eq(returned_n[i], expected_n[i]);
    }
}
END_TEST

START_TEST(test_n_from_shape_and_strides_error)
{
    int64_t number_of_cases = 5;

    int64_t *shapes[] = {
        NULL,
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {1},
    };

    int64_t ranks[] = {
        0,
        0,
        0,
        1,
        MAX_RANK + 1,
    };

    int64_t *strides[] = {
        (int64_t[]) {},
        NULL,
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {0},
    };

    int64_t *n[] = {
        (int64_t[]) {},
        (int64_t[]) {},
        NULL,
        (int64_t[]) {0},
        (int64_t[]) {0},
    };
    
    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_SHAPE_CONFLICT,
        ERROR_RANK_CONFLICT,
    };


    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = n_from_shape_and_strides(shapes[i],
                                         strides[i],
                                         ranks[i],
                                         n[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_slice_shape)
{
    int64_t number_of_cases = 3;

    int64_t *original_shapes[] = {
        (int64_t[]) {2, 2},
        (int64_t[]) {9, 8, 7, 6, 5},
        (int64_t[]) {},
    };

    int64_t original_ranks[] = {
        2,
        5,
        0,
    };

    int64_t *expected_slice_shapes[] = {
        (int64_t[]) {1, 1},
        (int64_t[]) {3, 5, 6, 1, 2},
        (int64_t[]) {},
    };

    int64_t slice_ranks[] = {
        2,
        5,
        0,
    };

    int64_t *arguments[] = {
        (int64_t[]) {1, 2, 1, 2},
        (int64_t[]) {2, 5, 0, 5, 1, 7, 0, 1, 2, 4},
        (int64_t[]) {},
    };

    int64_t length[] = {
        4,
        10,
        0,
    };

    int64_t returned_slice_shapes[number_of_cases][MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = slice_shape(original_shapes[i],
                            original_ranks[i],
                            returned_slice_shapes[i],
                            slice_ranks[i],
                            arguments[i],
                            length[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < slice_ranks[i]; j++)
        {
            ck_assert_int_eq(returned_slice_shapes[i][j], expected_slice_shapes[i][j]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_slice_offset)
{
    int64_t number_of_cases = 2;

    int64_t *original_strides[] = {
        (int64_t[]) {6, 3, 1},
        (int64_t[]) {},
    };

    int64_t original_ranks[] = {
        3,
        0,
    };

    int64_t expected_offset[] = {
        6,
        0,
    };

    int64_t *arguments[] = {
        (int64_t[]) {1, 3, 0, 2, 0, 1},
        (int64_t[]) {},
    };

    int64_t length[] = {
        6,
        0,
    };

    int64_t returned_offset[number_of_cases];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = slice_offset(original_strides[i],
                             original_ranks[i],
                             &returned_offset[i],
                             arguments[i],
                             length[i]);
        ck_assert_ptr_null(error);
        ck_assert_int_eq(returned_offset[i], expected_offset[i]);
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reverse_slice)
{
    int64_t number_of_cases = 2;

    int64_t *original_shapes[] = {
        (int64_t[]) {2, 2},
        (int64_t[]) {},
    };

    int64_t original_ranks[] = {
        2,
        0,
    };

    int64_t *arguments[] = {
        (int64_t[]) {1, 2, 1, 2},
        (int64_t[]) {},
    };

    int64_t length[] = {
        4,
        0,
    };

    int64_t *expected_new_arguments[] = {
        (int64_t[]) {1, 0, 1, 0},
        (int64_t[]) {},
    };

    int64_t new_length[] = {
        4,
        0,
    };

    int64_t returned_new_arguments[number_of_cases][2 * MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = reverse_slice(original_shapes[i],
                              original_ranks[i],
                              arguments[i],
                              length[i],
                              returned_new_arguments[i],
                              new_length[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < new_length[i]; j++)
        {
            ck_assert_int_eq(returned_new_arguments[i][j], expected_new_arguments[i][j]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_padding)
{
    int64_t number_of_cases = 3;

    int64_t *original_shapes[] = {
        (int64_t[]) {2, 2},
        (int64_t[]) {9, 8, 7, 6, 5},
        (int64_t[]) {},
    };

    int64_t original_ranks[] = {
        2,
        5,
        0,
    };

    int64_t *expected_padding_shapes[] = {
        (int64_t[]) {3, 4},
        (int64_t[]) {11, 9, 10, 10, 5},
        (int64_t[]) {},
    };

    int64_t padding_ranks[] = {
        2,
        5,
        0,
    };

    int64_t *arguments[] = {
        (int64_t[]) {0, 1, 1, 1},
        (int64_t[]) {0, 2, 1, 0, 1, 2, 2, 2, 0, 0},
        (int64_t[]) {},
    };

    int64_t length[] = {
        4,
        10,
        0,
    };

    int64_t returned_padding_shapes[number_of_cases][MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = padding(original_shapes[i],
                        original_ranks[i],
                        returned_padding_shapes[i],
                        padding_ranks[i],
                        arguments[i],
                        length[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < padding_ranks[i]; j++)
        {
            ck_assert_int_eq(returned_padding_shapes[i][j], expected_padding_shapes[i][j]);
        }
        error_destroy(error);
        error = NULL;
    }
}
END_TEST

START_TEST(test_reverse_padding)
{
    int64_t number_of_cases = 2;

    int64_t *original_shapes[] = {
        (int64_t[]) {2, 2},
        (int64_t[]) {},
    };

    int64_t original_ranks[] = {
        2,
        0,
    };

    int64_t *arguments[] = {
        (int64_t[]) {0, 1, 1, 0},
        (int64_t[]) {},
    };

    int64_t length[] = {
        4,
        0,
    };

    int64_t *expected_new_arguments[] = {
        (int64_t[]) {0, 2, 1, 3},
        (int64_t[]) {},
    };

    int64_t new_length[] = {
        4,
        0,
    };

    int64_t returned_new_arguments[number_of_cases][2 * MAX_RANK];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        error = reverse_padding(original_shapes[i],
                                original_ranks[i],
                                arguments[i],
                                length[i],
                                returned_new_arguments[i],
                                new_length[i]);
        ck_assert_ptr_null(error);
        for (int64_t j = 0; j < new_length[i]; j++)
        {
            ck_assert_int_eq(returned_new_arguments[i][j], expected_new_arguments[i][j]);
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
    tcase_add_test(tc, test_broadcast_strides_error);
    tcase_add_test(tc, test_broadcast_shapes);
    tcase_add_test(tc, test_broadcast_shapes_error);
    tcase_add_test(tc, test_is_broadcastable);
    tcase_add_test(tc, test_reduce_axis_length);
    tcase_add_test(tc, test_reduce_axis_length_error);
    tcase_add_test(tc, test_reduce_axis);
    tcase_add_test(tc, test_reduce_axis_error);
    tcase_add_test(tc, test_n_from_shape_and_strides);
    tcase_add_test(tc, test_n_from_shape_and_strides_error);
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
