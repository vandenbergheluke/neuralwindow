#include <check.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <test_helper.h>

nw_error_t *error;
view_t *original_view;
view_t *returned_view;
view_t *expected_view;

void setup(void)
{
    error = NULL;
    original_view = NULL;
    returned_view = NULL;
    expected_view = NULL;
}

void teardown(void)
{
    error_print(error);
    error_destroy(error);
    view_destroy(original_view);
    view_destroy(returned_view);
    view_destroy(expected_view);
}

START_TEST(test_view_create_error)
{
    int64_t number_of_cases = 8;

    int64_t offsets[] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t ranks[] = {1, 1, 1, MAX_RANK + 1, 5, 5, 5, 5};
    int64_t *shapes[] = {
        (int64_t[]) {1},
        NULL,
        NULL,
        (int64_t[]) {1},
        (int64_t[]) {1, 2, 0, 4, 5},
        (int64_t[]) {1, 0, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 0, 5},
        (int64_t[]) {1, 2, 3, 4, 0},
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
    };
    
    nw_error_type_t error_types[] = {
        ERROR_NULL,
        ERROR_NULL,
        ERROR_NULL,
        ERROR_RANK,
        ERROR_SHAPE,
        ERROR_SHAPE,
        ERROR_SHAPE,
        ERROR_SHAPE,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        if (!i)
        {
            error = view_create(NULL, offsets[i], ranks[i], shapes[i], strides[i]);
        }
        else
        {
            error = view_create(&returned_view, offsets[i], ranks[i], shapes[i], strides[i]);
        }
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);

        error_destroy(error);
        error = NULL;
        view_destroy(returned_view);
        returned_view = NULL;
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
        error = view_create(&returned_view, offsets[i], ranks[i], shapes[i], strides[i]);
        ck_assert_ptr_null(error);

        // Ranks and offsets must be the same as arguments
        ck_assert_int_eq(returned_view->offset, offsets[i]);
        ck_assert_int_eq(returned_view->rank, ranks[i]);

        // Ensure the shape and strides are not NULL.
        ck_assert_ptr_nonnull(returned_view->shape);
        ck_assert_ptr_nonnull(returned_view->strides);

        // Shapes and strides need to be copied and not directly assigned
        ck_assert_ptr_ne(returned_view->shape, shapes[i]);
        ck_assert_ptr_ne(returned_view->strides, strides[i]);

        // Compare shape and strides with expected
        for (int64_t j = 0; j < ranks[i]; j++)
        {
            ck_assert_int_eq(returned_view->shape[j], shapes[i][j]);
            ck_assert_int_eq(returned_view->strides[j], expected_strides[i][j]);
        }

        // Destroy view
        view_destroy(returned_view);
        returned_view = NULL;
    }
}
END_TEST

START_TEST(test_view_is_contiguous)
{
    int64_t number_of_cases = 12;
    int64_t offsets[] = {
        0, 
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -120,
        120,
        0
    };
    int64_t ranks[] = {
        3,
        0,
        1,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
    };
    int64_t *shapes[] = {
        (int64_t[]) {2, 2, 3},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {1, 2, 1, 5},
        (int64_t[]) {1, 2, 1, 5},
        (int64_t[]) {1, 2, 1, 5},
        (int64_t[]) {5, 1, 2, 1, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 4, 5},
        (int64_t[]) {2, 2, 3, 4, 5},
        (int64_t[]) {1, 1, 3, 4, 5},
        (int64_t[]) {1, 2, 3, 5, 4},
    };
    int64_t *strides[] = {
        (int64_t[]) {6, 3, 1},
        (int64_t[]) {},
        (int64_t[]) {1},
        (int64_t[]) {0, 5, 5, 1},
        (int64_t[]) {10, 5, 0, 1},
        (int64_t[]) {0, 5, 0, 1},
        (int64_t[]) {10, 0, 5, 0, 1}, 
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {0, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 5, 1},
        (int64_t[]) {120, 60, 20, 1, 5},
    };
    bool_t expected_is_contiguous[] = {
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        printf("%ld\n", i);
        bool_t returned_is_contiguous;

        error = view_create(&returned_view, offsets[i], ranks[i], shapes[i], strides[i]);
        ck_assert_ptr_null(error);

        error = view_is_contiguous(returned_view, &returned_is_contiguous);
        ck_assert_ptr_null(error);

        ck_assert(expected_is_contiguous[i] == returned_is_contiguous);

        view_destroy(returned_view);
        returned_view = NULL;
    }
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
        ERROR_RANK,
        ERROR_SHAPE,
        ERROR_SHAPE,
        ERROR_SHAPE,
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

START_TEST(test_view_permute)
{
    int64_t number_of_cases = 10;

    int64_t *original_shapes[] = {
        (int64_t[]) {1},
        (int64_t[]) {5, 3},
        (int64_t[]) {3, 2, 1},
        (int64_t[]) {2, 4, 3, 1},
        (int64_t[]) {2, 2, 2},
        (int64_t[]) {1, 2, 3, 5, 4},
        (int64_t[]) {1, 2, 3, 5, 4},
        (int64_t[]) {1, 2, 3, 5, 4},
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
        (int64_t[]) {0, 60, 20, 4, 1},
        (int64_t[]) {0, 60, 20, 4, 1},
        (int64_t[]) {0, 60, 20, 4, 1},
        (int64_t[]) {},
    };

    int64_t ranks[] = {
        1,
        2,
        3,
        4,
        3,
        5,
        5,
        5,
        5,
        0,
    };

    int64_t offsets[] = {
        0,
        1,
        0,
        0,
        0,
        4,
        4,
        4,
        4,
        0,
    };

    int64_t *axes[] = {
        (int64_t[]) {0},
        (int64_t[]) {1, 0},
        (int64_t[]) {2, 1, 0},
        (int64_t[]) {1, 2, 0, 3},
        (int64_t[]) {2, 0, 1},
        (int64_t[]) {4, 2, 3, 0, 1},
        (int64_t[]) {-1, -3, -2, -5, -4},
        (int64_t[]) {-1, 2, -2, 0, -4},
        (int64_t[]) {4, -3, 3, -5, 1},
        (int64_t[]) {},
    };

    int64_t *expected_shapes[] = {
        (int64_t[]) {1},
        (int64_t[]) {3, 5},
        (int64_t[]) {1, 2, 3},
        (int64_t[]) {4, 3, 2, 1},
        (int64_t[]) {2, 2, 2},
        (int64_t[]) {4, 3, 5, 1, 2},
        (int64_t[]) {4, 3, 5, 1, 2},
        (int64_t[]) {4, 3, 5, 1, 2},
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
        (int64_t[]) {1, 20, 4, 0, 60},
        (int64_t[]) {1, 20, 4, 0, 60},
        (int64_t[]) {1, 20, 4, 0, 60},
        (int64_t[]) {},
    };

    int64_t lengths[] = { 1, 2, 3, 4, 3, 5, 5, 5, 5, 0 };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        view_t *original_view = NULL;
        view_t *expected_view = NULL;
        view_t *returned_view = NULL;

        error = view_create(&original_view, offsets[i], ranks[i], original_shapes[i], original_strides[i]);
        ck_assert_ptr_null(error);
        error = view_create(&expected_view, offsets[i], ranks[i], expected_shapes[i], expected_strides[i]);
        ck_assert_ptr_null(error);
        error = view_permute(original_view, &returned_view, axes[i], lengths[i]);
        ck_assert_ptr_null(error);
        ck_assert_view_eq(returned_view, expected_view);

        view_destroy(original_view);
        view_destroy(expected_view);
        view_destroy(returned_view);
    }
}
END_TEST

START_TEST(test_view_recover_dimensions)
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

    int64_t reduced_offsets[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

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

    int64_t recovered_offsets[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        view_t *original_view = NULL;
        view_t *expected_view = NULL;
        view_t *returned_view = NULL;

        error = view_create(&original_view, reduced_offsets[i], reduced_ranks[i], reduced_shapes[i], reduced_strides[i]);
        ck_assert_ptr_null(error);
        error = view_create(&expected_view, recovered_offsets[i], recovered_ranks[i], expected_recovered_shapes[i], expected_recovered_strides[i]);
        ck_assert_ptr_null(error);
        error = view_recover_dimensions(original_view, &returned_view, axis[i], lengths[i]);
        ck_assert_ptr_null(error);
        ck_assert_view_eq(returned_view, expected_view);

        view_destroy(original_view);
        view_destroy(expected_view);
        view_destroy(returned_view);
    }
}
END_TEST

START_TEST(test_view_reduce)
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
        (int64_t[]) {-3, -1, -2},
        (int64_t[]) {0, 2, -2},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, 1},
        (int64_t[]) {0, -1},
        (int64_t[]) {-3, 2},
        (int64_t[]) {-1, -2},
        (int64_t[]) {1, 2},
        (int64_t[]) {-1},
        (int64_t[]) {2},
        (int64_t[]) {1},
        (int64_t[]) {1},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {1, -2},
        (int64_t[]) {1, 3},
        (int64_t[]) {1, 3},
        (int64_t[]) {1, 3},
        (int64_t[]) {1, 3},
        (int64_t[]) {-3},
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

    int64_t original_offsets[] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };

    int64_t reduced_offsets[] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
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
    
    for (int64_t i = 0; i < number_of_cases; i++)
    {
        view_t *original_view = NULL;
        view_t *expected_view = NULL;
        view_t *returned_view = NULL;

        error = view_create(&original_view, original_offsets[i], original_ranks[i], original_shapes[i], original_strides[i]);
        ck_assert_ptr_null(error);
        error = view_create(&expected_view, reduced_offsets[i], reduced_ranks[i], expected_reduced_shapes[i], expected_reduced_strides[i]);
        ck_assert_ptr_null(error);
        error = view_reduce(original_view, &returned_view, axis[i], lengths[i], keep_dimensions[i]);
        ck_assert_ptr_null(error);
        ck_assert_view_eq(returned_view, expected_view);

        view_destroy(original_view);
        view_destroy(expected_view);
        view_destroy(returned_view);
    }
}
END_TEST

START_TEST(test_view_shapes_equal)
{
    int64_t number_of_cases = 6;

    int64_t *x_shapes[] = {
        (int64_t[]) {1, 2, 3},
        (int64_t[]) {1, 2, 3},
        (int64_t[]) {1, 2},
        (int64_t[]) {},
        (int64_t[]) {},
        (int64_t[]) {1},
    };

    int64_t x_ranks[] = {
        3,
        3,
        2,
        0,
        0,
        1,
    };

    int64_t *y_shapes[] = {
        (int64_t[]) {1, 2, 3},
        (int64_t[]) {1, 2, 4},
        (int64_t[]) {1, 2, 3},
        (int64_t[]) {1},
        (int64_t[]) {},
        (int64_t[]) {1},
    };

    int64_t y_ranks[] = {
        3,
        3,
        3,
        1,
        0,
        1,
    };

    bool_t expected[] = {
        true,
        false,
        false,
        false,
        true,
        true,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        view_t *x_view = NULL;
        view_t *y_view = NULL;

        error = view_create(&x_view, 0, x_ranks[i], x_shapes[i], NULL);
        ck_assert_ptr_null(error);
        error = view_create(&y_view, 0, y_ranks[i], y_shapes[i], NULL);
        ck_assert_ptr_null(error);
        ck_assert(expected[i] == view_shapes_equal(x_view, y_view));
        ck_assert(expected[i] == view_has_shape(x_view, y_view->shape, y_view->rank));
        ck_assert(expected[i] == view_has_shape(y_view, x_view->shape, x_view->rank));

        view_destroy(x_view);
        view_destroy(y_view);
    }
}
END_TEST

START_TEST(test_view_logical_size)
{
    int64_t number_of_cases = 7;

    int64_t *shapes[] = {
        (int64_t[]){},
        (int64_t[]) {1},
        (int64_t[]) {2},
        (int64_t[]) {1, 2, 1},
        (int64_t[]) {1, 2, 3},
        (int64_t[]) {4, 2, 3},
        (int64_t[]) {5, 4, 3},
    };

    int64_t ranks[] = {
        0,
        1,
        1,
        3,
        3,
        3,
        3,
    };

    int64_t offsets[] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };

    int64_t *strides[] = {
        (int64_t[]) {},
        (int64_t[]) {0},
        (int64_t[]) {0},
        (int64_t[]) {0, 1, 0},
        (int64_t[]) {6, 3, 1},
        (int64_t[]) {0, 3, 1},
        (int64_t[]) {4, 1, 0},
    };

    int64_t expected_n[] = {
        1,
        1,
        2,
        2,
        6,
        24,
        60,
    };
    
    int64_t returned_n[number_of_cases];

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        view_t *view = NULL;

        error = view_create(&view, offsets[i], ranks[i], shapes[i], strides[i]);
        ck_assert_ptr_null(error);
        error = view_logical_size(view, &returned_n[i]);
        ck_assert_ptr_null(error);
        ck_assert_int_eq(returned_n[i], expected_n[i]);

        view_destroy(view);
    }
}
END_TEST

START_TEST(test_view_expand)
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

    int64_t original_offsets[] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
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

    int64_t *expanded_shapes[] = {
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

    int64_t expanded_ranks[] = {
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

    int64_t expanded_offsets[] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };

    int64_t *expanded_strides[] = {
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

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        view_t *original_view = NULL;
        view_t *expected_view = NULL;
        view_t *returned_view = NULL;

        error = view_create(&original_view, original_offsets[i], original_ranks[i], original_shapes[i], original_strides[i]);
        ck_assert_ptr_null(error);
        error = view_create(&expected_view, expanded_offsets[i], expanded_ranks[i], expanded_shapes[i], expanded_strides[i]);
        ck_assert_ptr_null(error);
        error = view_expand(original_view, &returned_view, expanded_shapes[i], expanded_ranks[i]);
        ck_assert_ptr_null(error);
        ck_assert_view_eq(returned_view, expected_view);

        view_destroy(original_view);
        view_destroy(expected_view);
        view_destroy(returned_view);
    }
}
END_TEST

START_TEST(test_view_expand_error)
{
    int64_t number_of_cases = 1;

    int64_t *original_shapes[] = {
        (int64_t[]) {1, 2, 3, 4, 5},
    };

    int64_t original_ranks[] = {
        5,
    };

    int64_t original_offsets[] = {
        0,
    };

    int64_t *original_strides[] = {
        (int64_t[]) {120, 60, 20, 5, 1},
    };

    int64_t *expanded_shapes[] = {
        (int64_t[]) {5, 4, 3, 2, 1},
    };

    int64_t expanded_ranks[] = {
        5,
    };

    int64_t expanded_offsets[] = {
        0,
    };

    int64_t *expanded_strides[] = {
        (int64_t[]) {24, 6, 2, 1, 1},
    };

    nw_error_type_t error_types[] = {
        ERROR_EXPAND,
    };

    for (int64_t i = 0; i < number_of_cases; i++)
    {
        view_t *original_view = NULL;
        view_t *returned_view = NULL;

        error = view_create(&original_view, original_offsets[i], original_ranks[i], original_shapes[i], original_strides[i]);
        ck_assert_ptr_null(error);
        error = view_create(&expected_view, expanded_offsets[i], expanded_ranks[i], expanded_shapes[i], expanded_strides[i]);
        ck_assert_ptr_null(error);
        error = view_expand(original_view, &returned_view, expanded_shapes[i], expanded_ranks[i]);
        ck_assert_ptr_nonnull(error);
        ck_assert_int_eq(error->error_type, error_types[i]);
        error_destroy(error);
        error = NULL;

        view_destroy(original_view);
        view_destroy(returned_view);
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
        ERROR_RANK,
        ERROR_RANK,
        ERROR_RANK,
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
        ERROR_RANK,
        ERROR_RANK,
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
        ERROR_RANK,
        ERROR_RANK,
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

START_TEST(test_view_physical_size)
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

    int64_t offsets[] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
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
        view_t *view = NULL;

        error = view_create(&view, offsets[i], ranks[i], shapes[i], strides[i]);
        ck_assert_ptr_null(error);
        error = view_physical_size(view, &returned_n[i]);
        ck_assert_ptr_null(error);
        ck_assert_int_eq(returned_n[i], expected_n[i]);

        view_destroy(view);
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
    tcase_add_test(tc, test_view_create);
    tcase_add_test(tc, test_view_create_error);
    tcase_add_test(tc, test_view_permute);
    tcase_add_test(tc, test_view_is_contiguous);
    tcase_add_test(tc, test_strides_from_shape);
    tcase_add_test(tc, test_strides_from_shape_error);
    tcase_add_test(tc, test_view_recover_dimensions);
    tcase_add_test(tc, test_view_reduce);
    tcase_add_test(tc, test_view_shapes_equal);
    tcase_add_test(tc, test_view_logical_size);
    tcase_add_test(tc, test_view_expand);
    tcase_add_test(tc, test_view_expand_error);
    tcase_add_test(tc, test_broadcast_shapes);
    tcase_add_test(tc, test_broadcast_shapes_error);
    tcase_add_test(tc, test_reduce_axis_length);
    tcase_add_test(tc, test_reduce_axis_length_error);
    tcase_add_test(tc, test_reduce_axis);
    tcase_add_test(tc, test_reduce_axis_error);
    tcase_add_test(tc, test_view_physical_size);
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
