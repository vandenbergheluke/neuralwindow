#include <iostream>
extern "C"
{
#include <datatype.h>
#include <check.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <test_helper.h>
}
#include <torch/torch.h>

#define CASES 7

nw_error_t *error;

tensor_t *tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_gradients[RUNTIMES][DATATYPES][CASES];

torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> shapes[CASES] = {
    {1},
    {10},
    {10, 1},
    {10, 10},
    {3, 4, 5},
    {3, 4, 5},
    {2, 3, 4, 5},
};

std::vector<int64_t> expanded_shapes[CASES] = {
    {2},
    {10},
    {10, 1},
    {10, 10},
    {3, 4, 5},
    {3, 4, 5},
    {2, 3, 4, 5},
};

void setup(void)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; ++k)
            {
                tensors[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
                expected_gradients[i][j][k] = NULL;
            }

            
            for (int k = 0; k < CASES; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_tensors[i][j][k] = torch::randn(shapes[k], 
                                                          torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)
                                                          ).expand(expanded_shapes[k]);
                    break;
                case FLOAT64:
                    torch_tensors[i][j][k] = torch::randn(shapes[k],
                                                          torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true)
                                                          ).expand(expanded_shapes[k]);
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }
                torch_tensors[i][j][k].retain_grad();

                view_t *view;
                storage_t *storage;
                buffer_t *buffer;

                error = view_create(&view, 
                                    (uint64_t) torch_tensors[i][j][k].storage_offset(),
                                    (uint64_t) torch_tensors[i][j][k].ndimension(),
                                    (uint64_t *) torch_tensors[i][j][k].sizes().data(),
                                    (uint64_t *) torch_tensors[i][j][k].strides().data());
                ck_assert_ptr_null(error);

                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       (uint64_t) torch_tensors[i][j][k].storage().nbytes() / (uint64_t) datatype_size((datatype_t) j),
                                       (void *) torch_tensors[i][j][k].data_ptr());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                error = tensor_create(&tensors[i][j][k], buffer, NULL, NULL, true, true);
                ck_assert_ptr_null(error);

                error = tensor_create_default(&returned_tensors[i][j][k]);
                ck_assert_ptr_null(error);
                returned_tensors[i][j][k]->lock = true;
            }
        }
    }
}

void teardown(void)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        runtime_destroy_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                tensor_destroy(tensors[i][j][k]);
                tensor_destroy(expected_tensors[i][j][k]);
                tensor_destroy(returned_tensors[i][j][k]);
                tensor_destroy(expected_gradients[i][j][k]);
            }
        }
    }

    error_print(error);
    error_destroy(error);
}

void test_unary(unary_operation_type_t unary_operation_type)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                unary_operation_t *unary_operation = NULL;
                operation_t *operation = NULL;
                torch::Tensor expected_tensor;

                switch (unary_operation_type)
                {
                case EXPONENTIAL_OPERATION:
                    expected_tensor = torch::exp(torch_tensors[i][j][k]);
                    break;
                case LOGARITHM_OPERATION:
                    expected_tensor = torch::log(torch_tensors[i][j][k]);
                    break;
                case SINE_OPERATION:
                    expected_tensor = torch::sin(torch_tensors[i][j][k]);
                    break;
                case COSINE_OPERATION:
                    expected_tensor = torch::cos(torch_tensors[i][j][k]);
                    break;
                case SQUARE_ROOT_OPERATION:
                    expected_tensor = torch::sqrt(torch_tensors[i][j][k]);
                    break;
                case RECIPROCAL_OPERATION:
                    expected_tensor = torch::reciprocal(torch_tensors[i][j][k]);
                    break;
                case COPY_OPERATION:
                    expected_tensor = torch::clone(torch_tensors[i][j][k]);
                    break;
                case CONTIGUOUS_OPERATION:
                    expected_tensor = torch_tensors[i][j][k].contiguous();
                    break;
                case NEGATION_OPERATION:
                    expected_tensor = torch::neg(torch_tensors[i][j][k]);
                    break;
                case RECTIFIED_LINEAR_OPERATION:
                    expected_tensor = torch::relu(torch_tensors[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unknown unary type.");
                }
                expected_tensor.sum().backward();

                view_t *view;
                buffer_t *buffer;
                storage_t *storage;

                error = view_create(&view,
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    (uint64_t *) expected_tensor.strides().data());
                ck_assert_ptr_null(error);

                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       (uint64_t) expected_tensor.storage().nbytes() / (uint64_t) datatype_size((datatype_t) j),
                                       (void *) expected_tensor.data_ptr());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                error = tensor_create(&expected_tensors[i][j][k],
                                      buffer,
                                      NULL,
                                      NULL,
                                      expected_tensor.requires_grad(),
                                      false);
                ck_assert_ptr_null(error);

                error = unary_operation_create(&unary_operation,
                                               unary_operation_type,
                                               tensors[i][j][k],
                                               expected_tensors[i][j][k]);
                ck_assert_ptr_null(error);
                error = operation_create(&operation, UNARY_OPERATION, unary_operation);
                ck_assert_ptr_null(error);
                error = function_create(&expected_tensors[i][j][k]->context, operation, UNARY_OPERATION);
                ck_assert_ptr_null(error);

                switch (unary_operation_type)
                {
                case EXPONENTIAL_OPERATION:
                    error = tensor_exponential(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                case LOGARITHM_OPERATION:
                    error = tensor_logarithm(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                case SINE_OPERATION:
                    error = tensor_sine(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                case COSINE_OPERATION:
                    error = tensor_cosine(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                case SQUARE_ROOT_OPERATION:
                    error = tensor_square_root(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                case RECIPROCAL_OPERATION:
                    error = tensor_reciprocal(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                case COPY_OPERATION:
                    error = tensor_copy(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                case CONTIGUOUS_OPERATION:
                    error = tensor_contiguous(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                case NEGATION_OPERATION:
                    error = tensor_negation(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                case RECTIFIED_LINEAR_OPERATION:
                    error = tensor_rectified_linear(tensors[i][j][k], returned_tensors[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unknown unary type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(returned_tensors[i][j][k], expected_tensors[i][j][k]);

                // Initialize expected gradient
                error = view_create(&view,
                                    (uint64_t) torch_tensors[i][j][k].grad().storage_offset(),
                                    (uint64_t) torch_tensors[i][j][k].grad().ndimension(),
                                    (uint64_t *) torch_tensors[i][j][k].grad().sizes().data(),
                                    (uint64_t *) torch_tensors[i][j][k].grad().strides().data());
                ck_assert_ptr_null(error);

                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       (uint64_t) torch_tensors[i][j][k].grad().storage().nbytes() / 
                                       (uint64_t) datatype_size((datatype_t) j),
                                       (void *) torch_tensors[i][j][k].grad().data_ptr());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                error = tensor_create(&expected_gradients[i][j][k],
                                      buffer,
                                      NULL,
                                      NULL,
                                      false,
                                      false);
                ck_assert_ptr_null(error);
                // Back prop
                tensor_t *cost;
                error = tensor_create_default(&cost);
                ck_assert_ptr_null(error);
                error = tensor_summation(returned_tensors[i][j][k], cost, NULL, returned_tensors[i][j][k]->buffer->view->rank, false);
                ck_assert_ptr_null(error);
                error = tensor_backward(cost, NULL);
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(tensors[i][j][k]->gradient, expected_gradients[i][j][k]);
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

Suite *make_unary_suite(void)
{
    Suite *s;
    TCase *tc_unary;

    s = suite_create("Test Unary Tensor Suite");

    tc_unary = tcase_create("Test Unary Tensor Case");
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

    sr = srunner_create(make_unary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
