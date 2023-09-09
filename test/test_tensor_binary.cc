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

#define BINARY_ELEMENTWISE_CASES 5
#define BINARY_MATRIX_MULTIPLICATION_CASES 5
#define CASES BINARY_ELEMENTWISE_CASES + BINARY_MATRIX_MULTIPLICATION_CASES

nw_error_t *error;

tensor_t *tensors_x[RUNTIMES][DATATYPES][CASES];
tensor_t *tensors_y[RUNTIMES][DATATYPES][CASES];
tensor_t *returned_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_tensors[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_gradients_x[RUNTIMES][DATATYPES][CASES];
tensor_t *expected_gradients_y[RUNTIMES][DATATYPES][CASES];

torch::Tensor torch_tensors_x[RUNTIMES][DATATYPES][CASES];
torch::Tensor torch_tensors_y[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> shapes_x[CASES] = {
    // Binary Elementwise
    {1},
    {3},
    {5, 1, 3, 1},
    {4, 2},
    {3, 2, 1},
    // Binary Matrix Multiplication
    {4, 1},
    {4, 5},
    {2, 4, 5},
    {1, 2, 2, 3, 5},
    {1, 2, 2, 9, 2},
};

std::vector<int64_t> shapes_y[CASES] = {
    // Binary Elementwise
    {10},
    {4, 3},
    {5, 2, 3, 4},
    {4, 2},
    {1, 2, 3},
    // Binary Matrix Multiplication
    {1, 4},
    {5, 4},
    {5, 4},
    {5, 7},
    {1, 2, 2, 2},
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
                tensors_x[i][j][k] = NULL;
                returned_tensors[i][j][k] = NULL;
                expected_tensors[i][j][k] = NULL;
                expected_gradients_x[i][j][k] = NULL;
                expected_gradients_y[i][j][k] = NULL;
            }

            
            for (int k = 0; k < CASES; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x[k],
                                                            torch::TensorOptions()
                                                            .dtype(torch::kFloat32)
                                                            .requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y[k],
                                                            torch::TensorOptions().
                                                            dtype(torch::kFloat32).
                                                            requires_grad(true));
                    break;
                case FLOAT64:
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x[k],
                                                            torch::TensorOptions().
                                                            dtype(torch::kFloat64).
                                                            requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y[k],
                                                            torch::TensorOptions().
                                                            dtype(torch::kFloat64).
                                                            requires_grad(true));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }
                torch_tensors_x[i][j][k].retain_grad();
                torch_tensors_y[i][j][k].retain_grad();

                view_t *view;
                storage_t *storage;
                buffer_t *buffer;
                
                // Operand x
                error = view_create(&view, 
                                    (uint64_t) torch_tensors_x[i][j][k].storage_offset(),
                                    (uint64_t) torch_tensors_x[i][j][k].ndimension(),
                                    (uint64_t *) torch_tensors_x[i][j][k].sizes().data(),
                                    (uint64_t *) torch_tensors_x[i][j][k].strides().data());
                ck_assert_ptr_null(error);

                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       (uint64_t) torch_tensors_x[i][j][k].storage().nbytes() /
                                       (uint64_t) datatype_size((datatype_t) j),
                                       (void *) torch_tensors_x[i][j][k].data_ptr());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                error = tensor_create(&tensors_x[i][j][k], buffer, NULL, NULL, true, true);
                ck_assert_ptr_null(error);

                // Operand y
                error = view_create(&view, 
                                    (uint64_t) torch_tensors_y[i][j][k].storage_offset(),
                                    (uint64_t) torch_tensors_y[i][j][k].ndimension(),
                                    (uint64_t *) torch_tensors_y[i][j][k].sizes().data(),
                                    (uint64_t *) torch_tensors_y[i][j][k].strides().data());
                ck_assert_ptr_null(error);

                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       (uint64_t) torch_tensors_y[i][j][k].storage().nbytes() /
                                       (uint64_t) datatype_size((datatype_t) j),
                                       (void *) torch_tensors_y[i][j][k].data_ptr());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                error = tensor_create(&tensors_y[i][j][k], buffer, NULL, NULL, true, true);
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
                tensor_destroy(tensors_x[i][j][k]);
                tensor_destroy(tensors_y[i][j][k]);
                tensor_destroy(expected_tensors[i][j][k]);
                tensor_destroy(returned_tensors[i][j][k]);
                tensor_destroy(expected_gradients_x[i][j][k]);
                tensor_destroy(expected_gradients_y[i][j][k]);
            }
        }
    }

    error_print(error);
    error_destroy(error);
}

void test_binary(binary_operation_type_t binary_operation_type)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            int start = binary_operation_type == MATRIX_MULTIPLICATION_OPERATION ?
                        BINARY_ELEMENTWISE_CASES : 0;
            int end = binary_operation_type == MATRIX_MULTIPLICATION_OPERATION ?
                      CASES : BINARY_ELEMENTWISE_CASES;
            for (int k = start; k < end; ++k)
            {
                view_t *view;
                storage_t *storage;
                buffer_t *buffer;
                binary_operation_t *binary_operation;
                operation_t *operation;
                torch::Tensor expected_tensor;

                switch (binary_operation_type)
                {
                case ADDITION_OPERATION:
                    expected_tensor = torch::add(torch_tensors_x[i][j][k],
                                                 torch_tensors_y[i][j][k]);
                    break;
                case SUBTRACTION_OPERATION:
                    expected_tensor = torch::sub(torch_tensors_x[i][j][k],
                                                 torch_tensors_y[i][j][k]);
                    break;
                case MULTIPLICATION_OPERATION:
                    expected_tensor = torch::mul(torch_tensors_x[i][j][k],
                                                 torch_tensors_y[i][j][k]);
                    break;
                case DIVISION_OPERATION:
                    expected_tensor = torch::div(torch_tensors_x[i][j][k],
                                                 torch_tensors_y[i][j][k]);
                    break;
                case POWER_OPERATION:
                    expected_tensor = torch::pow(torch_tensors_x[i][j][k],
                                                 torch_tensors_y[i][j][k]);
                    break;
                case MATRIX_MULTIPLICATION_OPERATION:
                    expected_tensor = torch::matmul(torch_tensors_x[i][j][k],
                                                    torch_tensors_y[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unsupported binary operation type.");
                }
                expected_tensor.sum().backward();

                error = view_create(&view,
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    (uint64_t *) expected_tensor.strides().data());
                ck_assert_ptr_null(error);

                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       (uint64_t) expected_tensor.storage().nbytes() / 
                                       (uint64_t) datatype_size((datatype_t) j),
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

                error = binary_operation_create(&binary_operation,
                                                binary_operation_type,
                                                tensors_x[i][j][k],
                                                tensors_y[i][j][k],
                                                expected_tensors[i][j][k]);
                ck_assert_ptr_null(error);
                error = operation_create(&operation, BINARY_OPERATION, binary_operation);
                ck_assert_ptr_null(error);
                error = function_create(&expected_tensors[i][j][k]->context,
                                        operation,
                                        BINARY_OPERATION);
                ck_assert_ptr_null(error);

                switch (binary_operation_type)
                {
                case ADDITION_OPERATION:
                    error = tensor_addition(tensors_x[i][j][k],
                                            tensors_y[i][j][k],
                                            returned_tensors[i][j][k]);
                    break;
                case SUBTRACTION_OPERATION:
                    error = tensor_subtraction(tensors_x[i][j][k],
                                               tensors_y[i][j][k],
                                               returned_tensors[i][j][k]);
                    break;
                case MULTIPLICATION_OPERATION:
                    error = tensor_multiplication(tensors_x[i][j][k],
                                                  tensors_y[i][j][k],
                                                  returned_tensors[i][j][k]);
                    break;
                case DIVISION_OPERATION:
                    error = tensor_division(tensors_x[i][j][k],
                                            tensors_y[i][j][k],
                                            returned_tensors[i][j][k]);
                    break;
                case POWER_OPERATION:
                    error = tensor_power(tensors_x[i][j][k],
                                         tensors_y[i][j][k],
                                         returned_tensors[i][j][k]);
                    break;
                case MATRIX_MULTIPLICATION_OPERATION:
                    error = tensor_matrix_multiplication(tensors_x[i][j][k],
                                                         tensors_y[i][j][k],
                                                         returned_tensors[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unsupported binary operation type.");
                }
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(returned_tensors[i][j][k], expected_tensors[i][j][k]);

                // Initialize expected gradient operand x
                error = view_create(&view,
                                    (uint64_t) torch_tensors_x[i][j][k].grad().storage_offset(),
                                    (uint64_t) torch_tensors_x[i][j][k].grad().ndimension(),
                                    (uint64_t *) torch_tensors_x[i][j][k].grad().sizes().data(),
                                    (uint64_t *) torch_tensors_x[i][j][k].grad().strides().data());
                ck_assert_ptr_null(error);

                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       (uint64_t) torch_tensors_x[i][j][k].grad().storage().nbytes() / 
                                       (uint64_t) datatype_size((datatype_t) j),
                                       (void *) torch_tensors_x[i][j][k].grad().data_ptr());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                error = tensor_create(&expected_gradients_x[i][j][k],
                                      buffer,
                                      NULL,
                                      NULL,
                                      false,
                                      false);
                ck_assert_ptr_null(error);

                // Initialize expected gradient operand y
                error = view_create(&view,
                                    (uint64_t) torch_tensors_y[i][j][k].grad().storage_offset(),
                                    (uint64_t) torch_tensors_y[i][j][k].grad().ndimension(),
                                    (uint64_t *) torch_tensors_y[i][j][k].grad().sizes().data(),
                                    (uint64_t *) torch_tensors_y[i][j][k].grad().strides().data());
                ck_assert_ptr_null(error);

                error = storage_create(&storage,
                                       (runtime_t) i,
                                       (datatype_t) j,
                                       (uint64_t) torch_tensors_y[i][j][k].grad().storage().nbytes() / 
                                       (uint64_t) datatype_size((datatype_t) j),
                                       (void *) torch_tensors_y[i][j][k].grad().data_ptr());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      view,
                                      storage,
                                      false);
                ck_assert_ptr_null(error);

                error = tensor_create(&expected_gradients_y[i][j][k],
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
                error = tensor_summation(returned_tensors[i][j][k],
                                         cost,
                                         NULL,
                                         returned_tensors[i][j][k]->buffer->view->rank,
                                         false);
                ck_assert_ptr_null(error);
                error = tensor_backward(cost, NULL);
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(tensors_x[i][j][k]->gradient,
                                       expected_gradients_x[i][j][k]);
                ck_assert_tensor_equiv(tensors_y[i][j][k]->gradient,
                                       expected_gradients_y[i][j][k]);
            }
        }
    }
}

START_TEST(test_addition)
{
    test_binary(ADDITION_OPERATION);
}
END_TEST

START_TEST(test_subtraction)
{
    test_binary(SUBTRACTION_OPERATION);
}
END_TEST

START_TEST(test_multiplication)
{
    test_binary(MULTIPLICATION_OPERATION);
}
END_TEST

START_TEST(test_division)
{
    test_binary(DIVISION_OPERATION);
}
END_TEST

START_TEST(test_power)
{
    test_binary(POWER_OPERATION);
}
END_TEST

START_TEST(test_matrix_multiplication)
{
    test_binary(MATRIX_MULTIPLICATION_OPERATION);
}
END_TEST

Suite *make_binary_suite(void)
{
    Suite *s;
    TCase *tc_binary;

    s = suite_create("Test Binary Tensor Suite");

    tc_binary= tcase_create("Test Binary Tensor Case");
    tcase_add_checked_fixture(tc_binary, setup, teardown);
    tcase_add_test(tc_binary, test_addition);
    tcase_add_test(tc_binary, test_subtraction);
    tcase_add_test(tc_binary, test_multiplication);
    tcase_add_test(tc_binary, test_division);
    tcase_add_test(tc_binary, test_power);
    tcase_add_test(tc_binary, test_matrix_multiplication);

    suite_add_tcase(s, tc_binary);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_binary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}