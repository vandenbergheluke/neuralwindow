#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <buffer.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <test_helper.h>
}
#include <torch/torch.h>

#define CASES 6

nw_error_t *error;

buffer_t *buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *returned_buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *expected_buffers[RUNTIMES][DATATYPES][CASES];

view_t *views[RUNTIMES][DATATYPES][CASES];
view_t *returned_views[RUNTIMES][DATATYPES][CASES];
view_t *expected_views[RUNTIMES][DATATYPES][CASES];

torch::Tensor tensors[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> shapes[CASES] = {
    {1},
    {1},
    {1},
    {1},
    {3, 4, 5},
    {3, 4, 5},
};

void setup(void)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                buffers[i][j][k] = NULL;
                returned_buffers[i][j][k] = NULL;
                expected_buffers[i][j][k] = NULL;

                views[i][j][k] = NULL;
                returned_views[i][j][k] = NULL;
                expected_views[i][j][k] = NULL;
            }

            
            for (int k = 0; k < CASES; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    tensors[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                    break;
                case FLOAT64:
                    tensors[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }

                error = view_create(&views[i][j][k], 
                                    (uint64_t) tensors[i][j][k].storage_offset(),
                                    (uint64_t) tensors[i][j][k].ndimension(),
                                    (uint64_t *) tensors[i][j][k].sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      views[i][j][k],
                                      (void *) tensors[i][j][k].data_ptr(),
                                      (uint64_t) tensors[i][j][k].numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = view_create(&returned_views[i][j][k],
                                    (uint64_t) tensors[i][j][k].storage_offset(),
                                    (uint64_t) tensors[i][j][k].ndimension(),
                                    (uint64_t *) tensors[i][j][k].sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&returned_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      returned_views[i][j][k],
                                      NULL,
                                      (uint64_t) tensors[i][j][k].numel(),
                                      true);
                ck_assert_ptr_null(error);
            }
        }
    }
}

void teardown(void)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_destroy_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                buffer_destroy(buffers[i][j][k]);
                buffer_destroy(returned_buffers[i][j][k]);
                buffer_destroy(expected_buffers[i][j][k]);
            }
        }
    }
    error_print(error);
    error_destroy(error);
}


START_TEST(test_exponential)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::exp(tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_exponential(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_logarithm)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::log(tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_logarithm(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_sine)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::sin(tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_sine(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_cosine)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::cos(tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_cosine(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_square_root)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::sqrt(tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_square_root(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_reciprocal)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::reciprocal(tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_reciprocal(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_copy)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::clone(tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_copy(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_contiguous)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = tensors[i][j][k].contiguous();

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_contiguous(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_negation)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::neg(tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_negation(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_rectified_linear)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::relu(tensors[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_rectified_linear(buffers[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

Suite *make_buffer_unary_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Buffer Unary Suite");

    // Unary Operations
    tc_unary = tcase_create("Buffer Unary Case");
    tcase_add_checked_fixture(tc_unary, setup, teardown);
    tcase_add_test(tc_unary, test_exponential);
    tcase_add_test(tc_unary, test_logarithm);
    tcase_add_test(tc_unary, test_sine);
    tcase_add_test(tc_unary, test_cosine);
    tcase_add_test(tc_unary, test_square_root);
    tcase_add_test(tc_unary, test_reciprocal);
    tcase_add_test(tc_unary, test_copy);
    tcase_add_test(tc_unary, test_contiguous);
    tcase_add_test(tc_unary, test_negation);
    tcase_add_test(tc_unary, test_rectified_linear);

    suite_add_tcase(s, tc_unary);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_unary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
