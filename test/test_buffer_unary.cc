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
#include <function.h>
}
#include <torch/torch.h>

#define CASES 6

nw_error_t *error;

buffer_t *buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *returned_buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *expected_buffers[RUNTIMES][DATATYPES][CASES];

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

                view_t *view;
                storage_t *storage;

                error = view_create(&view, 
                                    (uint64_t) tensors[i][j][k].storage_offset(),
                                    (uint64_t) tensors[i][j][k].ndimension(),
                                    (uint64_t *) tensors[i][j][k].sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = storage_create(&storage,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      tensors[i][j][k].nbytes() / 
                                      datatype_size((datatype_t) j),
                                      (void *) tensors[i][j][k].data_ptr());
                error = buffer_create(&buffers[i][j][k],
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                error = view_create(&view,
                                    (uint64_t) tensors[i][j][k].storage_offset(),
                                    (uint64_t) tensors[i][j][k].ndimension(),
                                    (uint64_t *) tensors[i][j][k].sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       tensors[i][j][k].nbytes() / 
                                       datatype_size((datatype_t) j),
                                       NULL);
                error = buffer_create(&returned_buffers[i][j][k],
                                      view,
                                      storage,
                                      false);
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

void test_unary(unary_operation_type_t unary_operation_type)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor;
                view_t *view;
                storage_t *storage;

                switch (unary_operation_type)
                {
                case EXPONENTIAL_OPERATION:
                    expected_tensor = torch::exp(tensors[i][j][k]);
                    break;
                case LOGARITHM_OPERATION:
                    expected_tensor = torch::log(tensors[i][j][k]);
                    break;
                case SINE_OPERATION:
                    expected_tensor = torch::sin(tensors[i][j][k]);
                    break;
                case COSINE_OPERATION:
                    expected_tensor = torch::cos(tensors[i][j][k]);
                    break;
                case SQUARE_ROOT_OPERATION:
                    expected_tensor = torch::sqrt(tensors[i][j][k]);
                    break;
                case RECIPROCAL_OPERATION:
                    expected_tensor = torch::reciprocal(tensors[i][j][k]);
                    break;
                case COPY_OPERATION:
                    expected_tensor = torch::clone(tensors[i][j][k]);
                    break;
                case NEGATION_OPERATION:
                    expected_tensor = torch::neg(tensors[i][j][k]);
                    break;
                case CONTIGUOUS_OPERATION:
                    expected_tensor = tensors[i][j][k].contiguous();
                    break;
                case RECTIFIED_LINEAR_OPERATION:
                    expected_tensor = torch::relu(tensors[i][j][k]);
                    break;
                default:
                    break;
                }

                error = view_create(&view,
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = storage_create(&storage,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_tensor.nbytes() / 
                                      datatype_size((datatype_t) j),
                                      (void *) expected_tensor.data_ptr());
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                switch (unary_operation_type)
                {
                case EXPONENTIAL_OPERATION:
                    error = runtime_exponential(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case LOGARITHM_OPERATION:
                    error = runtime_logarithm(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case SINE_OPERATION:
                    error = runtime_sine(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case COSINE_OPERATION:
                    error = runtime_cosine(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case SQUARE_ROOT_OPERATION:
                    error = runtime_square_root(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RECIPROCAL_OPERATION:
                    error = runtime_reciprocal(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case COPY_OPERATION:
                    error = runtime_copy(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case NEGATION_OPERATION:
                    error = runtime_negation(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case CONTIGUOUS_OPERATION:
                    error = runtime_contiguous(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                case RECTIFIED_LINEAR_OPERATION:
                    error = runtime_rectified_linear(buffers[i][j][k], returned_buffers[i][j][k]);
                    break;
                default:
                    break;
                }
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}

START_TEST(test_exponential)
{
    test_unary(EXPONENTIAL_OPERATION);
}
END_TEST

START_TEST(test_logarithm)
{
    test_unary(LOGARITHM_OPERATION);
}
END_TEST

START_TEST(test_sine)
{
    test_unary(SINE_OPERATION);
}
END_TEST

START_TEST(test_cosine)
{
    test_unary(COSINE_OPERATION);
}
END_TEST

START_TEST(test_square_root)
{
    test_unary(SQUARE_ROOT_OPERATION);
}
END_TEST

START_TEST(test_reciprocal)
{
    test_unary(RECIPROCAL_OPERATION);
}
END_TEST

START_TEST(test_copy)
{
    test_unary(COPY_OPERATION);
}
END_TEST

START_TEST(test_contiguous)
{
    test_unary(CONTIGUOUS_OPERATION);
}
END_TEST

START_TEST(test_negation)
{
    test_unary(NEGATION_OPERATION);
}
END_TEST

START_TEST(test_rectified_linear)
{
    test_unary(RECTIFIED_LINEAR_OPERATION);
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
