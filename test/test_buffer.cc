#include <iostream>
extern "C"
{
#include <check.h>
#include <buffer.h>
}
#include <torch/torch.h>

#define CASES 6
#define EPSILON 0.0001
#define SEED 1234

bool_t set_seed = true;

nw_error_t *unary_error;

buffer_t *unary_buffers[CASES];
buffer_t *returned_unary_buffers[CASES];
buffer_t *expected_unary_buffers[CASES];

view_t *views[CASES];
view_t *returned_views[CASES];
view_t *expected_views[CASES];

torch::Tensor tensors[CASES];

void unary_setup(void)
{
    if (set_seed)
    {
        torch::manual_seed(SEED);
        set_seed = false;
    }

    for (int i = 0; i < CASES; i++)
    {
        unary_buffers[i] = NULL;
        returned_unary_buffers[i] = NULL;
        expected_unary_buffers[i] = NULL;

        views[i] = NULL;
        returned_views[i] = NULL;
        expected_views[i] = NULL;
    }

    std::vector<int64_t> shapes[] = {
        {1},
        {1},
        {1},
        {1},
        {3, 4, 5},
        {3, 4, 5},
    };
    
    runtime_t runtimes[] = {
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
        OPENBLAS_RUNTIME,
        MKL_RUNTIME,
    };

    datatype_t datatypes[] = {
        FLOAT32,
        FLOAT32,
        FLOAT64,
        FLOAT64,
        FLOAT32,
        FLOAT32,
    };

    torch::ScalarType torch_datatypes[] = {
        torch::kFloat32,
        torch::kFloat32,
        torch::kFloat64,
        torch::kFloat64,
        torch::kFloat32,
        torch::kFloat32,
    };

    for (int i = 0; i < CASES; ++i)
    {
        tensors[i] = torch::randn(shapes[i], torch::TensorOptions().dtype(torch_datatypes[i]));

        unary_error = view_create(&views[i], 
                                  (uint64_t) tensors[i].storage_offset(),
                                  (uint64_t) tensors[i].ndimension(),
                                  (uint64_t *) tensors[i].sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&unary_buffers[i],
                                    runtimes[i],
                                    datatypes[i],
                                    views[i],
                                    (void *) tensors[i].data_ptr(),
                                    (uint64_t) tensors[i].numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = view_create(&returned_views[i],
                                  (uint64_t) tensors[i].storage_offset(),
                                  (uint64_t) tensors[i].ndimension(),
                                  (uint64_t *) tensors[i].sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&returned_unary_buffers[i],
                                    runtimes[i],
                                    datatypes[i],
                                    returned_views[i],
                                    NULL,
                                    (uint64_t) tensors[i].numel(),
                                    true);
        ck_assert_ptr_null(unary_error);
    }
}

void unary_teardown(void)
{
    for (int i = 0; i < CASES; i++)
    {
        buffer_destroy(unary_buffers[i]);
        buffer_destroy(returned_unary_buffers[i]);
        buffer_destroy(expected_unary_buffers[i]);
    }
    error_destroy(unary_error);
}

void ck_assert_buffer_eq(const buffer_t *returned_buffer, const buffer_t *expected_buffer)
{
    ck_assert_uint_eq(expected_buffer->view->rank, returned_buffer->view->rank);
    ck_assert_uint_eq(expected_buffer->view->offset, returned_buffer->view->offset);
    ck_assert_uint_eq(expected_buffer->n, returned_buffer->n);
    ck_assert_uint_eq(expected_buffer->size, returned_buffer->size);
    ck_assert_int_eq(expected_buffer->datatype, returned_buffer->datatype);
    ck_assert_int_eq(expected_buffer->runtime, returned_buffer->runtime);

    for (uint64_t i = 0; i < expected_buffer->view->rank; ++i)
    {
        ck_assert_uint_eq(expected_buffer->view->shape[i], returned_buffer->view->shape[i]);
        ck_assert_uint_eq(expected_buffer->view->strides[i], returned_buffer->view->strides[i]);
    }

    for (uint64_t i = 0; i < expected_buffer->n; ++i)
    {

        switch (expected_buffer->datatype)
        {
        case FLOAT32:
            if (isnanf(((float32_t *) expected_buffer->data)[i]))
            {
                ck_assert_float_nan(((float32_t *) returned_buffer->data)[i]);
            }
            else
            {
                ck_assert_float_eq_tol(((float32_t *) returned_buffer->data)[i],
                                       ((float32_t *) expected_buffer->data)[i], EPSILON);
            }
            break;
        case FLOAT64:
            if (isnan(((float64_t *) expected_buffer->data)[i]))
            {
                ck_assert_double_nan(((float64_t *) returned_buffer->data)[i]);
            }
            else
            {
                ck_assert_double_eq_tol(((float64_t *) returned_buffer->data)[i],
                                        ((float64_t *) expected_buffer->data)[i], EPSILON);
            }
        default:
            break;
        }
    }
}

START_TEST(test_exponential)
{
    for (int i = 0; i < CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::exp(tensors[i]);

        unary_error = view_create(&expected_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_exponential(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
    }
}
END_TEST

START_TEST(test_logarithm)
{
    for (int i = 0; i < CASES; ++i)
    {
        torch::Tensor expected_tensor = torch::log(tensors[i]);

        unary_error = view_create(&expected_views[i],
                                  (uint64_t) expected_tensor.storage_offset(),
                                  (uint64_t) expected_tensor.ndimension(),
                                  (uint64_t *) expected_tensor.sizes().data(),
                                  NULL);
        ck_assert_ptr_null(unary_error);
        unary_error = buffer_create(&expected_unary_buffers[i],
                                    unary_buffers[i]->runtime,
                                    unary_buffers[i]->datatype,
                                    expected_views[i],
                                    (void *) expected_tensor.data_ptr(),
                                    (uint64_t) expected_tensor.numel(),
                                    true);
        ck_assert_ptr_null(unary_error);

        unary_error = runtime_logarithm(unary_buffers[i], returned_unary_buffers[i]);
        ck_assert_ptr_null(unary_error);

        ck_assert_buffer_eq(returned_unary_buffers[i], expected_unary_buffers[i]);
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
    tcase_add_test(tc_unary, test_logarithm);
    suite_add_tcase(s, tc_unary);

    return s;
}

extern "C" int main(void)
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
