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

#define CASES 4

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
    {1},
    {10},
    {10, 1},
    {10, 10},
    // {3, 2, 1},
};

std::vector<int64_t> shapes_y[CASES] = {
    {1},
    {10},
    {10, 1},
    {10, 10},
    // {1, 2, 3},
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
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x[k], torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y[k], torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
                    break;
                case FLOAT64:
                    torch_tensors_x[i][j][k] = torch::randn(shapes_x[k], torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
                    torch_tensors_y[i][j][k] = torch::randn(shapes_y[k], torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }
                torch_tensors_x[i][j][k].retain_grad();
                torch_tensors_y[i][j][k].retain_grad();

                view_t *view;
                buffer_t *buffer;
                
                // Operand x
                error = view_create(&view, 
                                    (uint64_t) torch_tensors_x[i][j][k].storage_offset(),
                                    (uint64_t) torch_tensors_x[i][j][k].ndimension(),
                                    (uint64_t *) torch_tensors_x[i][j][k].sizes().data(),
                                    (uint64_t *) torch_tensors_x[i][j][k].strides().data());
                ck_assert_ptr_null(error);

                error = buffer_create(&buffer,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      view,
                                      (void *) torch_tensors_x[i][j][k].data_ptr(),
                                      (uint64_t) torch_tensors_x[i][j][k].storage().nbytes() / (uint64_t) datatype_size((datatype_t) j),
                                      true);
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

                error = buffer_create(&buffer,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      view,
                                      (void *) torch_tensors_y[i][j][k].data_ptr(),
                                      (uint64_t) torch_tensors_y[i][j][k].storage().nbytes() / (uint64_t) datatype_size((datatype_t) j),
                                      true);
                ck_assert_ptr_null(error);

                error = tensor_create(&tensors_y[i][j][k], buffer, NULL, NULL, true, true);
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

void test_binary_elementwise(binary_operation_type_t binary_operation_type)
{
    for (int i = 0; i < RUNTIMES; i++)
    {
        for (int j = 0; j < DATATYPES; j++)
        {
            for (int k = 0; k < CASES; k++)
            {
                binary_operation_t *binary_operation = NULL;
                operation_t *operation = NULL;
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
                default:
                    ck_abort_msg("unsupported binary elementwise operation type.");
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
                error = function_create(&expected_tensors[i][j][k]->context, operation, BINARY_OPERATION);
                ck_assert_ptr_null(error);

                switch (binary_operation_type)
                {
                case ADDITION_OPERATION:
                    error = tensor_addition(tensors_x[i][j][k], tensors_y[i][j][k], returned_tensors[i][j][k]);
                    break;
                case SUBTRACTION_OPERATION:
                    error = tensor_subtraction(tensors_x[i][j][k], tensors_y[i][j][k], returned_tensors[i][j][k]);
                    break;
                case MULTIPLICATION_OPERATION:
                    error = tensor_multiplication(tensors_x[i][j][k], tensors_y[i][j][k], returned_tensors[i][j][k]);
                    break;
                case DIVISION_OPERATION:
                    error = tensor_division(tensors_x[i][j][k], tensors_y[i][j][k], returned_tensors[i][j][k]);
                    break;
                case POWER_OPERATION:
                    error = tensor_power(tensors_x[i][j][k], tensors_y[i][j][k], returned_tensors[i][j][k]);
                    break;
                default:
                    ck_abort_msg("unsupported binary elementwise operation type.");
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
                error = buffer_create(&buffer,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      view,
                                      (void *) torch_tensors_x[i][j][k].grad().data_ptr(),
                                      (uint64_t) torch_tensors_x[i][j][k].grad().storage().nbytes() / 
                                      (uint64_t) datatype_size((datatype_t) j),
                                      true);
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
                error = buffer_create(&buffer,
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      view,
                                      (void *) torch_tensors_y[i][j][k].grad().data_ptr(),
                                      (uint64_t) torch_tensors_y[i][j][k].grad().storage().nbytes() / 
                                      (uint64_t) datatype_size((datatype_t) j),
                                      true);
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
                error = tensor_create_empty(&cost);
                ck_assert_ptr_null(error);
                error = tensor_summation(returned_tensors[i][j][k], cost, NULL, returned_tensors[i][j][k]->buffer->view->rank, false);
                ck_assert_ptr_null(error);
                error = tensor_backward(cost, NULL);
                ck_assert_ptr_null(error);

                ck_assert_tensor_equiv(tensors_x[i][j][k]->gradient, expected_gradients_x[i][j][k]);
                ck_assert_tensor_equiv(tensors_y[i][j][k]->gradient, expected_gradients_y[i][j][k]);
            }
        }
    }
}

START_TEST(test_addition)
{
    test_binary_elementwise(ADDITION_OPERATION);
}
END_TEST

START_TEST(test_subtraction)
{
    test_binary_elementwise(SUBTRACTION_OPERATION);
}
END_TEST

START_TEST(test_multiplication)
{
    test_binary_elementwise(MULTIPLICATION_OPERATION);
}
END_TEST

START_TEST(test_division)
{
    test_binary_elementwise(DIVISION_OPERATION);
}
END_TEST

START_TEST(test_power)
{
    test_binary_elementwise(POWER_OPERATION);
}
END_TEST

Suite *make_binary_elementwise_suite(void)
{
    Suite *s;
    TCase *tc_binary_elementwise;

    s = suite_create("Test Binary Elementwise Tensor Suite");

    tc_binary_elementwise = tcase_create("Test Binary Elementwise Tensor Case");
    tcase_add_checked_fixture(tc_binary_elementwise, setup, teardown);
    tcase_add_test(tc_binary_elementwise, test_addition);
    tcase_add_test(tc_binary_elementwise, test_subtraction);
    tcase_add_test(tc_binary_elementwise, test_multiplication);
    tcase_add_test(tc_binary_elementwise, test_division);
    tcase_add_test(tc_binary_elementwise, test_power);

    suite_add_tcase(s, tc_binary_elementwise);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_binary_elementwise_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}