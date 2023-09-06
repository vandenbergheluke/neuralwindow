#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <datatype.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <test_helper.h>
}
#include <torch/torch.h>

#define CASES 9

nw_error_t *error;

tensor_t *tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_gradient[RUNTIMES][DATATYPES][CASES];

torch::Tensor torch_tensors[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> axis[CASES] = {
    {0},
    {0},
    {1},
    {1},
    {0, 2},
    {1},
    {0, 1, 2, 3},
    {0, 1, 2, 3},
    {0, 2, 3},
};

bool_t keep_dimension[CASES] = {
    false,
    true,
    false,
    true,
    true,
    false,
    false,
    false,
    true,
};

std::vector<int64_t> shapes[CASES] = {
    {1},
    {10},
    {10, 1},
    {10, 10},
    {3, 4, 5},
    {3, 4, 5},
    {2, 3, 4, 5},
    {5},
    {3, 1, 1},
};

std::vector<int64_t> expanded_shapes[CASES] = {
    {2},
    {10},
    {10, 1},
    {10, 10},
    {3, 4, 5},
    {3, 4, 5},
    {2, 3, 4, 5},
    {2, 3, 4, 5},
    {2, 3, 4, 5},
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
                tensors[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
                expected_gradient[i][j][k] = NULL;
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
                buffer_t *buffer;
                error = view_create(&view, 
                                    (uint64_t) torch_tensors[i][j][k].storage_offset(),
                                    (uint64_t) torch_tensors[i][j][k].ndimension(),
                                    (uint64_t *) torch_tensors[i][j][k].sizes().data(),
                                    (uint64_t *) torch_tensors[i][j][k].strides().data());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      view,
                                      (void *) torch_tensors[i][j][k].data_ptr(),
                                      (uint64_t) torch_tensors[i][j][k].storage().nbytes() / (uint64_t) datatype_size((datatype_t) j),
                                      true);
                ck_assert_ptr_null(error);

                error = tensor_create(&tensors[i][j][k], buffer, NULL, NULL, true, true);
                ck_assert_ptr_null(error);

                error = tensor_create_empty(&returned_tensors[i][j][k]);
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
                tensor_destroy(expected_gradient[i][j][k]);
            }
        }
    }

    error_print(error);
    error_destroy(error);
}

void test_reduction(reduction_operation_type_t reduction_operation_type)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                reduction_operation_t *reduction_operation = NULL;
                operation_t *operation = NULL;
                torch::Tensor expected_tensor;

                switch (reduction_operation_type)
                {
                case SUMMATION_OPERATION:
                    expected_tensor = torch::sum(torch_tensors[i][j][k], axis[k], keep_dimension[k]);
                    break;
                case MAXIMUM_OPERATION:
                    expected_tensor = torch::amax(torch_tensors[i][j][k], axis[k], keep_dimension[k]);
                    break;
                default:
                    ck_abort_msg("unknown reduction type.");
                }
                expected_tensor.sum().backward();

                view_t *view;
                buffer_t *buffer;

                error = view_create(&view,
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    (uint64_t *) expected_tensor.strides().data());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      view,
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.storage().nbytes() / 
                                      (uint64_t) datatype_size((datatype_t) j),
                                      true);
                ck_assert_ptr_null(error);

                error = tensor_create(&expected_tensors[i][j][k],
                                      buffer,
                                      NULL,
                                      NULL,
                                      true,
                                      false);
                ck_assert_ptr_null(error);

                error = reduction_operation_create(&reduction_operation,
                                                   reduction_operation_type,
                                                   tensors[i][j][k],
                                                   (uint64_t *) axis[k].data(),
                                                   (uint64_t) axis[k].size(), 
                                                   keep_dimension[k],
                                                   expected_tensors[i][j][k]);
                ck_assert_ptr_null(error);
                error = operation_create(&operation, REDUCTION_OPERATION, reduction_operation);
                ck_assert_ptr_null(error);
                error = function_create(&expected_tensors[i][j][k]->context, operation, REDUCTION_OPERATION);
                ck_assert_ptr_null(error);

                switch (reduction_operation_type)
                {
                case SUMMATION_OPERATION:
                    error = tensor_summation(tensors[i][j][k], 
                                             returned_tensors[i][j][k],
                                             (uint64_t *) axis[k].data(),
                                             (uint64_t) axis[k].size(),
                                             keep_dimension[k]);
                    break;
                case MAXIMUM_OPERATION:
                    error = tensor_maximum(tensors[i][j][k], 
                                           returned_tensors[i][j][k],
                                           (uint64_t *) axis[k].data(),
                                           (uint64_t) axis[k].size(),
                                           keep_dimension[k]);
                    break;
                default:
                    ck_abort_msg("unknown reduction type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_tensor_eq(returned_tensors[i][j][k], expected_tensors[i][j][k]);

                // Initialize expected gradient
                error = view_create(&view,
                                    (uint64_t) torch_tensors[i][j][k].grad().storage_offset(),
                                    (uint64_t) torch_tensors[i][j][k].grad().ndimension(),
                                    (uint64_t *) torch_tensors[i][j][k].grad().sizes().data(),
                                    (uint64_t *) torch_tensors[i][j][k].grad().strides().data());
                ck_assert_ptr_null(error);
                error = buffer_create(&buffer,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      view,
                                      (void *) torch_tensors[i][j][k].grad().data_ptr(),
                                      (uint64_t) torch_tensors[i][j][k].grad().storage().nbytes() / 
                                      (uint64_t) datatype_size((datatype_t) j),
                                      true);
                ck_assert_ptr_null(error);

                error = tensor_create(&expected_gradient[i][j][k],
                                      buffer,
                                      NULL,
                                      NULL,
                                      false,
                                      false);
                ck_assert_ptr_null(error);
                error = tensor_backward(returned_tensors[i][j][k], NULL);

                // Back prop
                tensor_t *cost;
                error = tensor_create_empty(&cost);
                ck_assert_ptr_null(error);
                error = tensor_summation(returned_tensors[i][j][k], cost, NULL, returned_tensors[i][j][k]->buffer->view->rank, false);
                ck_assert_ptr_null(error);
                error = tensor_backward(cost, NULL);
                ck_assert_ptr_null(error);

                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(tensors[i][j][k]->gradient, expected_gradient[i][j][k]);
            }
        }
    }
}

START_TEST(test_summation)
{
    test_reduction(SUMMATION_OPERATION);
}
END_TEST

START_TEST(test_maximum)
{
    test_reduction(MAXIMUM_OPERATION);
}
END_TEST

Suite *make_reduction_suite(void)
{
    Suite *s;
    TCase *tc_reduction;

    s = suite_create("Test Reduction Tensor Suite");

    tc_reduction = tcase_create("Test Reduction Tensor Case");
    tcase_add_checked_fixture(tc_reduction, setup, teardown);
    tcase_add_test(tc_reduction, test_summation);
    tcase_add_test(tc_reduction, test_maximum);

    suite_add_tcase(s, tc_reduction);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_reduction_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}