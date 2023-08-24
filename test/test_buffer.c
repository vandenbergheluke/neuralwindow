#include <check.h>
#include <buffer.h>

#define CASES 2
#define EPSILON 0.0001

error_t *unary_error;
buffer_t *unary_buffers[CASES];

void unary_setup(void)
{
    view_t *views[CASES];

    for (int i = 0; i < CASES; i++)
    {
        unary_buffers[i] = NULL;
        views[i] = NULL;
    }

    uint32_t *shapes[] = {
        (uint32_t[]) {1},
        (uint32_t[]) {1},
    };

    uint32_t *strides[] = {
        (uint32_t[]) {0},
        (uint32_t[]) {0},
    };

    uint32_t ranks[] = {
        1,
        1,
    };

    uint32_t offsets[] = {
        0,
        0,
    };

    runtime_t runtimes[] = {
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
    };

    datatype_t datatypes[] = {
        FLOAT32,
        FLOAT64,
    };

    float32_t *data_float32[] = {
        (float32_t[]) {0.0},
        NULL,
    };

    float64_t *data_float64[] = {
        NULL,
        (float64_t[]) {0.0},
    };

    uint32_t n[] = {
        1,
        1,
    };

    for (int i = 0; i < CASES; i++)
    {
        unary_error = view_create(&views[i], offsets[i], ranks[i], shapes[i], strides[i]);
        ck_assert_ptr_null(unary_error);
        error_destroy(unary_error);
        switch (datatypes[i])
        {
        case FLOAT32:
            unary_error = buffer_create(&unary_buffers[i],
                                        runtimes[i],
                                        datatypes[i],
                                        views[i],
                                        data_float32[i],
                                        n[i],
                                        true);
            break;
        case FLOAT64:
            unary_error = buffer_create(&unary_buffers[i],
                                        runtimes[i],
                                        datatypes[i],
                                        views[i],
                                        data_float64[i],
                                        n[i],
                                        true);
            break;
        default:
            break;
        }
        ck_assert_ptr_null(unary_error);
        error_destroy(unary_error);
    }
}

void unary_teardown(void)
{
    for (int i = 0; i < CASES; i++)
    {
        buffer_destroy(unary_buffers[i]);
    }
    error_destroy(unary_error);
}

START_TEST(test_exponential)
{
    view_t *expected_view;
    view_t *returned_view;
    buffer_t *expected_unary_buffer;
    buffer_t *returned_unary_buffer;

    float32_t *expected_data_float32[] = {
        (float32_t[]) {1.0},
        NULL,
    };

    float64_t *expected_data_float64[] = {
        NULL,
        (float64_t[]) {1.0},
    };

    for (int i = 0; i < CASES; i++)
    {
        expected_unary_buffer = NULL;
        returned_unary_buffer = NULL;
        expected_view = NULL;
        returned_view = NULL;

        unary_error = view_create(&expected_view, 
                                  unary_buffers[i]->view->offset,
                                  unary_buffers[i]->view->rank,
                                  unary_buffers[i]->view->shape,
                                  unary_buffers[i]->view->strides);
        ck_assert_ptr_null(unary_error);
        error_destroy(unary_error);
        switch (unary_buffers[i]->datatype)
        {
        case FLOAT32:
            unary_error = buffer_create(&expected_unary_buffer,
                                        unary_buffers[i]->runtime,
                                        unary_buffers[i]->datatype,
                                        expected_view,
                                        expected_data_float32[i],
                                        unary_buffers[i]->n,
                                        true);
            break;
        case FLOAT64:
            unary_error = buffer_create(&expected_unary_buffer,
                                        unary_buffers[i]->runtime,
                                        unary_buffers[i]->datatype,
                                        expected_view,
                                        expected_data_float64[i],
                                        unary_buffers[i]->n,
                                        true);
            break;
        default:
            break;
        }
        ck_assert_ptr_null(unary_error);
        error_destroy(unary_error);
        unary_error = view_create(&returned_view, 
                                  unary_buffers[i]->view->offset,
                                  unary_buffers[i]->view->rank,
                                  unary_buffers[i]->view->shape,
                                  unary_buffers[i]->view->strides);
        ck_assert_ptr_null(unary_error);
        error_destroy(unary_error);
        unary_error = buffer_create(&returned_unary_buffer,
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    returned_view,
                                    NULL,
                                    unary_buffers[i]->n,
                                    true);
        ck_assert_ptr_null(unary_error);
        error_destroy(unary_error);

        unary_error = runtime_exponential(unary_buffers[i], returned_unary_buffer);
        ck_assert_ptr_null(unary_error);
        error_destroy(unary_error);

        for (uint32_t j = 0; j < expected_unary_buffer->n; j++)
        {
            switch (expected_unary_buffer->datatype)
            {
            case FLOAT32:
                ck_assert_float_eq_tol(((float32_t *) expected_unary_buffer->data)[j],
                                       ((float32_t *) returned_unary_buffer->data)[j], EPSILON);
                break;
            case FLOAT64:
                ck_assert_double_eq_tol(((float64_t *) expected_unary_buffer->data)[j], 
                                        ((float64_t *) returned_unary_buffer->data)[j], EPSILON);
                break;
            default:
                break;
            }
        }

        buffer_destroy(expected_unary_buffer);
        buffer_destroy(returned_unary_buffer);
    }
}
END_TEST

Suite *make_buffer_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Unary Suite");
    tc_unary = tcase_create("Unary Case");
    tcase_add_checked_fixture(tc_unary, unary_setup, unary_teardown);
    tcase_add_test(tc_unary, test_exponential);
    suite_add_tcase(s, tc_unary);

    return s;
}

int main(void)
{
    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
